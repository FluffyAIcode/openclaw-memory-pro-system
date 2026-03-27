"""
Memory Security — unified security + privacy module for OpenClaw Memory Pro.

Components:
  A. Content Safety     — prompt injection detection (EN + ZH patterns)
  B. Trust Scoring      — multi-signal trust evaluation for recall items
  C. Security Gate      — pre-filters Context Composer input, blocks untrusted items
  D. PII Scanner        — detects 8 categories of sensitive personal information
  E. Privacy Gateway    — masks PII before external LLM calls, unmasks on return
  F. Sensitivity Class. — auto-classifies content sensitivity at ingestion time
  G. Audit Logger       — persistent JSONL log of all security/privacy events
  H. Auth Token         — shared-secret token management for memory_server API

Public API for write-side callers:
  check_content_safety(text) -> ContentSafetyResult
  classify_sensitivity(text) -> str
  generate_auth_token() -> str
  verify_auth_token(token) -> bool
"""

import hashlib
import json
import logging
import os
import re
import secrets
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).parent


# ── A. Content Safety: Prompt Injection Detection ─────────────

_INJECTION_PATTERNS_EN = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|directives?)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above|earlier)"
    r"|forget\s+(?:everything|all|your\s+(?:instructions?|rules?))"
    r"|you\s+are\s+now\b"
    r"|new\s+(?:role|persona|identity)\s*[:\-]"
    r"|(?:^|\n)\s*system\s*:\s*"
    r"|from\s+now\s+on\b"
    r"|override\s+(?:all\s+)?(?:instructions?|rules?|safety)"
    r"|(?:act|behave|pretend|roleplay)\s+as\b"
    r"|jailbreak"
    r"|developer\s+mode"
    r"|do\s+anything\s+now"
    r"|DAN\s+mode"
    r"|ignore\s+(?:safety|content)\s+(?:filters?|policies?|guidelines?)"
    r"|bypass\s+(?:safety|content|security|restrictions?)"
    r"|you\s+(?:must|should)\s+(?:always|never)\s+(?:obey|follow|listen)"
    r"|entering\s+(?:admin|root|sudo|god)\s+mode"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

_INJECTION_PATTERNS_ZH = re.compile(
    r"(?:"
    r"忽略.{0,4}(?:之前|所有|上面|以上|先前).{0,4}(?:指令|指示|规则|提示|要求)"
    r"|无视.{0,4}(?:之前|所有|上面|以上|先前)"
    r"|你现在是"
    r"|新的?角色\s*[：:]"
    r"|系统提示\s*[：:]"
    r"|从现在开始"
    r"|覆盖.{0,4}(?:指令|规则|安全|限制)"
    r"|假装(?:你是|自己是)"
    r"|越狱"
    r"|开发者模式"
    r"|忘记.{0,4}(?:所有|之前|一切)"
    r"|绕过.{0,4}(?:安全|内容|限制|过滤)"
    r"|进入.{0,4}(?:管理员|超级|上帝|调试)模式"
    r")",
)


@dataclass
class ContentSafetyResult:
    is_safe: bool
    score: float  # 1.0 = safe, 0.0 = injection detected
    matched_patterns: List[str] = field(default_factory=list)
    reason: str = ""


def check_content_safety(text: str) -> ContentSafetyResult:
    """Scan text for prompt injection patterns. Used by both read-side and write-side."""
    if not text:
        return ContentSafetyResult(is_safe=True, score=1.0)

    matches = []

    for m in _INJECTION_PATTERNS_EN.finditer(text):
        matches.append(m.group()[:80])
    for m in _INJECTION_PATTERNS_ZH.finditer(text):
        matches.append(m.group()[:80])

    if not matches:
        return ContentSafetyResult(is_safe=True, score=1.0)

    return ContentSafetyResult(
        is_safe=False,
        score=0.0,
        matched_patterns=matches[:5],
        reason=f"prompt injection detected: {matches[0][:60]}",
    )


# ── B. Trust Scoring ─────────────────────────────────────────

TRUSTED_SOURCES: Dict[str, float] = {
    "openclaw": 1.0, "cursor": 1.0, "agent": 1.0,
    "hub": 0.95, "daily_log": 0.95, "chronos": 0.95,
    "auto_ingest": 0.9, "memora": 0.9, "msa": 0.9,
    "skill": 0.9, "kg": 0.9,
    "msa_interleave": 0.9, "inference_engine": 0.9,
    "kg_conflict": 0.85, "contradiction": 0.85,
    "telegram": 0.8, "cli": 0.8,
}

UNKNOWN_SOURCE_SCORE = 0.3

BLOCK_THRESHOLD = 0.4
FLAG_THRESHOLD = 0.7

_ITEM_WEIGHTS = {
    "content_safety": 0.40,
    "source_trust": 0.25,
    "temporal": 0.20,
    "importance": 0.15,
}

_SKILL_WEIGHTS = {
    "content_safety": 0.30,
    "skill_trust": 0.40,
    "source_trust": 0.10,
    "temporal": 0.10,
    "importance": 0.10,
}


@dataclass
class TrustSignal:
    name: str
    score: float
    reason: str = ""


@dataclass
class TrustVerdict:
    trust_score: float
    action: str  # "allow" | "flag" | "block"
    signals: List[TrustSignal] = field(default_factory=list)
    pii_types: List[str] = field(default_factory=list)


class TrustScorer:
    """Computes weighted trust score from multiple signals."""

    def __init__(self):
        self._pii_scanner = PIIScanner()

    def evaluate(self, item: dict, query: str = "") -> TrustVerdict:
        """Evaluate trust for a generic recall item (evidence, KG, memora, msa)."""
        signals = []

        content = item.get("content", "")
        safety = check_content_safety(content)
        signals.append(TrustSignal("content_safety", safety.score, safety.reason))

        source = self._extract_source(item)
        source_score = TRUSTED_SOURCES.get(source, UNKNOWN_SOURCE_SCORE)
        signals.append(TrustSignal("source_trust", source_score,
                                   f"source='{source}'"))

        temporal_score = self._score_temporal(item)
        signals.append(TrustSignal("temporal", temporal_score[0], temporal_score[1]))

        imp_score = self._score_importance(item, source)
        signals.append(TrustSignal("importance", imp_score[0], imp_score[1]))

        if safety.score == 0.0:
            return TrustVerdict(
                trust_score=0.0, action="block", signals=signals,
                pii_types=[],
            )

        trust = sum(
            _ITEM_WEIGHTS[s.name] * s.score
            for s in signals if s.name in _ITEM_WEIGHTS
        )

        pii_matches = self._pii_scanner.scan(content)
        pii_types = list({m.pii_type for m in pii_matches})
        if pii_types and source_score < 0.8:
            trust *= 0.7

        action = self._action(trust)
        return TrustVerdict(
            trust_score=round(trust, 4),
            action=action,
            signals=signals,
            pii_types=pii_types,
        )

    def evaluate_skill(self, skill_data: dict) -> TrustVerdict:
        """Evaluate trust for a skill item (stricter checks)."""
        signals = []

        content = skill_data.get("content", "")
        procedures = skill_data.get("procedures", "")
        exec_prompt = skill_data.get("executable_prompt", "")
        scan_text = f"{content} {procedures} {exec_prompt}"
        safety = check_content_safety(scan_text)
        signals.append(TrustSignal("content_safety", safety.score, safety.reason))

        source_score = TRUSTED_SOURCES.get("skill", 0.9)
        signals.append(TrustSignal("source_trust", source_score, "source='skill'"))

        skill_score, skill_reason = self._score_skill_trust(skill_data)
        signals.append(TrustSignal("skill_trust", skill_score, skill_reason))

        temporal_score = self._score_temporal(skill_data)
        signals.append(TrustSignal("temporal", temporal_score[0], temporal_score[1]))

        imp_score = self._score_importance(skill_data, "skill")
        signals.append(TrustSignal("importance", imp_score[0], imp_score[1]))

        if safety.score == 0.0:
            return TrustVerdict(
                trust_score=0.0, action="block", signals=signals,
            )

        trust = sum(
            _SKILL_WEIGHTS.get(s.name, 0) * s.score
            for s in signals
        )

        action = self._action(trust)
        return TrustVerdict(
            trust_score=round(trust, 4),
            action=action,
            signals=signals,
        )

    @staticmethod
    def _extract_source(item: dict) -> str:
        source = item.get("system", "")
        if not source:
            source = item.get("metadata", {}).get("source", "")
        if not source:
            source = item.get("source", "")
        return source.lower().strip() if source else ""

    @staticmethod
    def _score_temporal(item: dict) -> Tuple[float, str]:
        ts = item.get("timestamp", "")
        if not ts:
            ts = item.get("created_at", "")
        if not ts:
            ts = item.get("metadata", {}).get("timestamp", "")
        if not ts:
            return 0.5, "no timestamp"

        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if dt > now + timedelta(minutes=5):
                return 0.0, f"future timestamp: {ts}"
            age_days = (now - dt).total_seconds() / 86400
            if age_days > 730:
                return 0.7, f"very old: {age_days:.0f} days"
            return 1.0, "valid"
        except (ValueError, TypeError):
            return 0.4, f"unparseable timestamp: {ts[:30]}"

    @staticmethod
    def _score_importance(item: dict, source: str) -> Tuple[float, str]:
        imp = item.get("importance", None)
        if imp is None:
            imp = item.get("metadata", {}).get("importance", None)
        if imp is None:
            return 1.0, "no importance field"
        try:
            imp = float(imp)
        except (ValueError, TypeError):
            return 0.8, f"non-numeric importance: {imp}"

        if imp <= 0.8:
            return 1.0, "plausible"
        trusted_high = {"openclaw", "agent", "cursor", "manual", "hub"}
        if source in trusted_high:
            return 0.9, f"high importance ({imp}) from trusted source"
        if imp >= 1.0 and source not in trusted_high:
            return 0.2, f"importance=1.0 from source='{source}'"
        return 0.6, f"high importance ({imp}) from source='{source}'"

    @staticmethod
    def _score_skill_trust(skill: dict) -> Tuple[float, str]:
        action_type = skill.get("action_type", "none")
        if action_type == "webhook":
            return 0.3, "webhook action_type (external HTTP call risk)"

        status = skill.get("status", "draft")
        total_uses = skill.get("total_uses", 0)
        utility = skill.get("utility_rate", 0.5)
        created_at = skill.get("created_at", "")

        if status == "active" and total_uses >= 3 and utility > 0.5:
            return 1.0, f"established skill (uses={total_uses}, utility={utility:.0%})"
        if status == "active" and total_uses >= 1:
            return 0.8, f"active with some usage (uses={total_uses})"

        if status == "active" and total_uses == 0 and created_at:
            try:
                ct = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - ct).total_seconds() / 3600
                if age_hours < 1:
                    return 0.4, f"suspicious: active with 0 uses, age={age_hours:.1f}h"
            except (ValueError, TypeError):
                pass
            return 0.6, "active with 0 uses"

        if status == "draft":
            return 0.6, "draft status"
        return 0.7, f"status={status}"

    @staticmethod
    def _action(score: float) -> str:
        if score < BLOCK_THRESHOLD:
            return "block"
        if score < FLAG_THRESHOLD:
            return "flag"
        return "allow"


# ── C. Security Gate ─────────────────────────────────────────

class SecurityGate:
    """Pre-filters raw recall data before Context Composer layer assembly."""

    def __init__(self):
        self._scorer = TrustScorer()
        self._audit = SecurityAuditLogger()

    def filter_raw(self, raw: dict, query: str = "") -> Tuple[dict, dict]:
        """Filter all lists in the raw recall dict.

        Returns:
            (filtered_raw, security_stats)
        """
        stats = {
            "total_scanned": 0,
            "blocked": 0,
            "flagged": 0,
            "allowed": 0,
            "blocked_items": [],
            "flagged_items": [],
            "pii_detected": 0,
        }

        filtered = dict(raw)

        filtered["skills"] = self._filter_skills(
            raw.get("skills", []), query, stats)
        filtered["evidence"] = self._filter_items(
            raw.get("evidence", []), query, stats)
        filtered["kg_relations"] = self._filter_kg(
            raw.get("kg_relations", []), query, stats)
        filtered["memora"] = self._filter_items(
            raw.get("memora", []), query, stats)
        filtered["msa"] = self._filter_items(
            raw.get("msa", []), query, stats)
        filtered["contradictions"] = self._filter_contradictions(
            raw.get("contradictions", []), query, stats)

        if stats["blocked"] > 0 or stats["flagged"] > 0:
            logger.warning(
                "SecurityGate: scanned=%d blocked=%d flagged=%d pii=%d for '%s'",
                stats["total_scanned"], stats["blocked"],
                stats["flagged"], stats["pii_detected"], query[:50],
            )

        return filtered, stats

    def _filter_items(self, items: list, query: str, stats: dict) -> list:
        result = []
        for item in items:
            stats["total_scanned"] += 1
            verdict = self._scorer.evaluate(item, query)
            if verdict.pii_types:
                stats["pii_detected"] += 1
            if verdict.action == "block":
                stats["blocked"] += 1
                preview = item.get("content", "")[:120]
                stats["blocked_items"].append(preview)
                self._audit.log_event("block", verdict, item, query)
                continue
            if verdict.action == "flag":
                stats["flagged"] += 1
                preview = item.get("content", "")[:120]
                stats["flagged_items"].append(preview)
                self._audit.log_event("flag", verdict, item, query)
            item["_trust_score"] = verdict.trust_score
            result.append(item)
        return result

    def _filter_skills(self, skills: list, query: str, stats: dict) -> list:
        result = []
        for skill in skills:
            stats["total_scanned"] += 1
            verdict = self._scorer.evaluate_skill(skill)
            if verdict.action == "block":
                stats["blocked"] += 1
                stats["blocked_items"].append(f"[Skill] {skill.get('name', '?')}")
                self._audit.log_event("block_skill", verdict, skill, query)
                continue
            if verdict.action == "flag":
                stats["flagged"] += 1
                stats["flagged_items"].append(f"[Skill] {skill.get('name', '?')}")
                self._audit.log_event("flag_skill", verdict, skill, query)
            skill["_trust_score"] = verdict.trust_score
            result.append(skill)
        return result

    def _filter_kg(self, relations: list, query: str, stats: dict) -> list:
        result = []
        for rel in relations:
            stats["total_scanned"] += 1
            desc = rel.get("description", "")
            src = rel.get("source_content", "")
            tgt = rel.get("target_content", "")
            synthetic = {
                "content": f"{desc} {src} {tgt}",
                "system": "kg",
                "metadata": rel.get("metadata", {}),
            }
            verdict = self._scorer.evaluate(synthetic, query)
            if verdict.action == "block":
                stats["blocked"] += 1
                stats["blocked_items"].append(f"[KG] {desc[:80]}")
                self._audit.log_event("block_kg", verdict, rel, query)
                continue
            if verdict.action == "flag":
                stats["flagged"] += 1
                stats["flagged_items"].append(f"[KG] {desc[:80]}")
                self._audit.log_event("flag_kg", verdict, rel, query)
            result.append(rel)
        return result

    def _filter_contradictions(self, contradictions: list, query: str,
                               stats: dict) -> list:
        result = []
        for cr in contradictions:
            stats["total_scanned"] += 1
            decision = cr.get("decision_content", "")
            contra_list = cr.get("contradicting", [])
            contra_text = contra_list[0].get("content", "") if contra_list else ""
            synthetic = {
                "content": f"{decision} {contra_text}",
                "system": cr.get("source", "contradiction"),
                "metadata": {},
            }
            verdict = self._scorer.evaluate(synthetic, query)
            if verdict.action == "block":
                stats["blocked"] += 1
                stats["blocked_items"].append(f"[Contradiction] {decision[:60]}")
                self._audit.log_event("block_contradiction", verdict, cr, query)
                continue
            result.append(cr)
        return result


# ── D. PII Scanner ───────────────────────────────────────────

@dataclass
class PIIMatch:
    pii_type: str
    value: str
    start: int
    end: int
    masked: str


_PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("china_id", re.compile(
        r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"
    ), "[ID_NUM]"),

    ("phone_cn", re.compile(
        r"(?<!\d)1[3-9]\d{9}(?!\d)"
    ), "[PHONE]"),

    ("email", re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ), "[EMAIL]"),

    ("bank_card", re.compile(
        r"(?<!\d)(?:6[0-9]{15,18}|4[0-9]{15}|5[1-5][0-9]{14}|3[47][0-9]{13})(?!\d)"
    ), "[BANK_CARD]"),

    ("api_key", re.compile(
        r"(?:"
        r"sk-[a-zA-Z0-9]{20,}"
        r"|ghp_[a-zA-Z0-9]{36,}"
        r"|gho_[a-zA-Z0-9]{36,}"
        r"|glpat-[a-zA-Z0-9\-]{20,}"
        r"|AKIA[0-9A-Z]{16}"
        r"|xox[bpsa]-[a-zA-Z0-9\-]{10,}"
        r"|key-[a-zA-Z0-9]{20,}"
        r"|Bearer\s+[a-zA-Z0-9._\-]{20,}"
        r")"
    ), "[API_KEY]"),

    ("ip_address", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
    ), "[IP_ADDR]"),

    ("password_context", re.compile(
        r"(?:password|passwd|密码|口令)\s*[：:=]\s*\S{4,40}",
        re.IGNORECASE,
    ), "[PASSWORD]"),

    ("secret_key", re.compile(
        r"(?:secret[_.]?key|secret|token|private[_.]?key|access[_.]?key)\s*[：:=]\s*['\"]?[a-zA-Z0-9+/=_\-]{16,}['\"]?",
        re.IGNORECASE,
    ), "[SECRET_KEY]"),
]

_LOCALHOST_IP = re.compile(r"\b(?:127\.0\.0\.\d{1,3}|0\.0\.0\.0)\b")


class PIIScanner:
    """Detects personally identifiable and sensitive information."""

    def scan(self, text: str) -> List[PIIMatch]:
        if not text:
            return []
        matches = []
        for pii_type, pattern, mask_label in _PII_PATTERNS:
            for m in pattern.finditer(text):
                if pii_type == "ip_address" and _LOCALHOST_IP.match(m.group()):
                    continue
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=m.group(),
                    start=m.start(),
                    end=m.end(),
                    masked=mask_label,
                ))
        return matches

    def has_pii(self, text: str) -> bool:
        return len(self.scan(text)) > 0


# ── E. Privacy Gateway ───────────────────────────────────────

class PrivacyGateway:
    """Masks PII before sending to external LLMs, unmasks on return."""

    def __init__(self):
        self._scanner = PIIScanner()
        self._audit = SecurityAuditLogger()

    def mask(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace PII with indexed placeholders.

        Returns:
            (masked_text, pii_map) where pii_map maps placeholder -> original.
        """
        if not text:
            return text, {}

        matches = self._scanner.scan(text)
        if not matches:
            return text, {}

        matches.sort(key=lambda m: m.start, reverse=True)

        pii_map: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        result = text

        for m in matches:
            pii_type = m.pii_type
            counters[pii_type] = counters.get(pii_type, 0) + 1
            placeholder = f"[{m.masked.strip('[]')}_{counters[pii_type]}]"
            pii_map[placeholder] = m.value
            result = result[:m.start] + placeholder + result[m.end:]

        if pii_map:
            type_counts = {}
            for m in matches:
                type_counts[m.pii_type] = type_counts.get(m.pii_type, 0) + 1
            logger.info("PrivacyGateway: masked %d PII items (%s)",
                        len(pii_map), type_counts)

        return result, pii_map

    def unmask(self, text: str, pii_map: Dict[str, str]) -> str:
        """Restore placeholders to original values."""
        if not text or not pii_map:
            return text
        result = text
        for placeholder, original in pii_map.items():
            result = result.replace(placeholder, original)
        return result


# ── F. Sensitivity Classification ────────────────────────────

VALID_SENSITIVITIES = {"public", "private", "confidential", "secret"}

_SENSITIVITY_MAP = {
    "api_key": "secret",
    "secret_key": "secret",
    "password_context": "secret",
    "china_id": "confidential",
    "bank_card": "confidential",
    "phone_cn": "private",
    "email": "private",
    "ip_address": "private",
}

_SENSITIVITY_RANK = {"public": 0, "private": 1, "confidential": 2, "secret": 3}

_scanner_singleton = PIIScanner()


def classify_sensitivity(content: str) -> str:
    """Auto-classify content sensitivity based on PII detection."""
    if not content:
        return "public"
    matches = _scanner_singleton.scan(content)
    if not matches:
        return "public"
    highest = "public"
    for m in matches:
        level = _SENSITIVITY_MAP.get(m.pii_type, "private")
        if _SENSITIVITY_RANK[level] > _SENSITIVITY_RANK[highest]:
            highest = level
    return highest


# ── G. Security Audit Logger ─────────────────────────────────

class SecurityAuditLogger:
    """Persistent JSONL audit log for security and privacy events."""

    def __init__(self, log_dir: Optional[Path] = None):
        self._dir = log_dir or (_WORKSPACE / "memory" / "security")
        self._file = self._dir / "audit.jsonl"

    def log_event(self, action: str, verdict: TrustVerdict,
                  item: dict, query: str = ""):
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            content = item.get("content", "")
            if not content:
                content = item.get("description", "")
            if not content:
                content = item.get("decision_content", "")

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "trust_score": verdict.trust_score,
                "signals": [
                    {"name": s.name, "score": s.score, "reason": s.reason}
                    for s in verdict.signals
                ],
                "pii_types": verdict.pii_types,
                "content_preview": content[:200],
                "item_system": item.get("system", ""),
                "query": query[:100],
            }
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("SecurityAuditLogger: failed to write: %s", e)

    def log_write_rejection(self, action: str, reason: str,
                            content_preview: str = "", source: str = ""):
        """Log a write-side rejection (bookmark, skill, ingest, etc.)."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "reason": reason,
                "content_preview": content_preview[:200],
                "source": source,
            }
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("SecurityAuditLogger: failed to write: %s", e)

    def log_privacy_event(self, event_type: str, pii_type_counts: dict,
                          caller: str = ""):
        """Log a privacy-related event (PII masking, sensitivity classification)."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": f"privacy_{event_type}",
                "pii_type_counts": pii_type_counts,
                "caller": caller,
            }
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("SecurityAuditLogger: failed to write: %s", e)


# ── H. Auth Token Management ────────────────────────────────

_AUTH_TOKEN_DIR = _WORKSPACE / "memory" / "security"
_AUTH_TOKEN_FILE = _AUTH_TOKEN_DIR / ".auth_token"


def generate_auth_token() -> str:
    """Generate a new auth token and save to disk with restricted permissions."""
    _AUTH_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    _AUTH_TOKEN_FILE.write_text(token, encoding="utf-8")
    try:
        os.chmod(_AUTH_TOKEN_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass
    logger.info("Auth token generated and saved to %s", _AUTH_TOKEN_FILE)
    return token


def load_auth_token() -> Optional[str]:
    """Load the auth token from disk. Returns None if not found."""
    if _AUTH_TOKEN_FILE.exists():
        try:
            return _AUTH_TOKEN_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


def ensure_auth_token() -> str:
    """Load existing token or generate a new one."""
    token = load_auth_token()
    if not token:
        token = generate_auth_token()
    return token


def verify_auth_token(provided: str) -> bool:
    """Verify a provided token against the stored one."""
    stored = load_auth_token()
    if not stored:
        return True  # no token configured = auth disabled
    return secrets.compare_digest(provided.strip(), stored.strip())
