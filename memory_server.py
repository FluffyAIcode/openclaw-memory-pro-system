#!/usr/bin/env python3
"""
Memory Server — persistent HTTP service for OpenClaw's memory systems.

Keeps the SentenceTransformer model loaded in memory, shared across
Memora and MSA. Agent calls via curl or memory_cli.py instead of
spawning a new python3 process each time.

Zero external dependencies beyond what's already installed.

Usage:
    python3 memory_server.py                   # foreground
    python3 memory_server.py --port 18790      # custom port
    python3 memory_server.py --daemon           # background (writes PID file)
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [memory-server] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memory-server")

_WORKSPACE = Path(__file__).parent
_START_TIME = time.time()
_EMBEDDER_TYPE = "unknown"


def _load_env():
    """Load .env from OpenClaw directory."""
    env_path = Path.home() / ".openclaw" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _setup_hf_offline():
    """Configure HuggingFace for offline/mirror mode."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _force_cpu_for_daemon():
    """MPS (Apple GPU) is unreliable after setsid(); force CPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_MPS_ENABLED"] = "0"
    try:
        import torch
        torch.set_default_device("cpu")
    except Exception:
        pass


def _load_shared_embedder():
    """Load SentenceTransformer once, set as shared embedder."""
    global _EMBEDDER_TYPE

    import shared_embedder

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model_name = "nomic-ai/nomic-embed-text-v1.5"
        logger.info("Loading SentenceTransformer: %s ...", model_name)
        device = "cpu" if os.environ.get("PYTORCH_MPS_ENABLED") == "0" else None
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        logger.info("Model loaded successfully")

        class SharedSentenceEmbedder:
            """Wraps SentenceTransformer with the embed/embed_batch interface."""
            def __init__(self, m, dim=768):
                self._model = m
                self.dimension = dim

            def embed(self, text: str):
                return self._model.encode(text, normalize_embeddings=True).tolist()

            def embed_batch(self, texts):
                return self._model.encode(texts, normalize_embeddings=True).astype(np.float32)

            def embed_np(self, text: str):
                return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

        emb = SharedSentenceEmbedder(model)
        shared_embedder.set(emb)
        _EMBEDDER_TYPE = "SentenceTransformer"
        logger.info("Shared embedder ready (SentenceTransformer)")
        return emb

    except Exception as e:
        logger.warning("SentenceTransformer load failed (%s), using MockEmbedder", e)
        import hashlib
        import numpy as np

        class SharedMockEmbedder:
            def __init__(self, dim=768):
                self.dimension = dim

            def embed(self, text: str):
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                raw = [int(digest[i:i+2], 16) / 255.0 for i in range(0, min(len(digest), self.dimension*2), 2)]
                while len(raw) < self.dimension:
                    raw.append(0.5)
                vec = np.array(raw[:self.dimension], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                return vec.tolist()

            def embed_batch(self, texts):
                return np.array([np.array(self.embed(t), dtype=np.float32) for t in texts])

            def embed_np(self, text: str):
                return np.array(self.embed(text), dtype=np.float32)

        emb = SharedMockEmbedder()
        shared_embedder.set(emb)
        _EMBEDDER_TYPE = "MockEmbedder"
        logger.info("Shared embedder ready (MockEmbedder fallback)")
        return emb


_hub = None

def _get_hub():
    global _hub
    if _hub is None:
        from memory_hub import hub
        _hub = hub
    return _hub


# ── Telegram Pusher ───────────────────────────────────────────

class TelegramPusher:
    """Sends messages to the user's Telegram via OpenClaw's bot."""

    def __init__(self):
        self._token: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._loaded = False

    def _load_config(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            oc_json = Path.home() / ".openclaw" / "openclaw.json"
            if oc_json.exists():
                data = json.loads(oc_json.read_text(encoding="utf-8"))
                tg = data.get("channels", {}).get("telegram", {})
                if not tg:
                    tg = data.get("telegram", {})
                self._token = tg.get("botToken")

            allow_from = Path.home() / ".openclaw" / "credentials" / "telegram-default-allowFrom.json"
            if allow_from.exists():
                ids = json.loads(allow_from.read_text(encoding="utf-8"))
                chat_ids = ids.get("allowFrom", ids) if isinstance(ids, dict) else ids
                if isinstance(chat_ids, list) and chat_ids:
                    self._chat_id = str(chat_ids[0])
        except Exception as e:
            logger.warning("TelegramPusher config load failed: %s", e)

    def push(self, text: str, parse_mode: str = "Markdown") -> bool:
        self._load_config()
        if not self._token or not self._chat_id:
            logger.debug("Telegram push skipped: no token or chat_id")
            return False
        try:
            from urllib.request import Request, urlopen
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = json.dumps({
                "chat_id": self._chat_id,
                "text": text[:4000],
                "parse_mode": parse_mode,
            }).encode("utf-8")
            req = Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
                if result.get("ok"):
                    logger.info("Telegram push OK (msg_id=%s)", result["result"]["message_id"])
                    return True
                logger.warning("Telegram API error: %s", result)
                return False
        except Exception as e:
            logger.warning("Telegram push failed: %s", e)
            return False


_telegram = TelegramPusher()


# ── Auto-Ingestor ─────────────────────────────────────────────

class AutoIngestor:
    """Background thread that watches daily .md files and auto-ingests new content.

    Dedup strategy:
      - Track processed line count per file (skip already-processed lines)
      - Track content hashes of ingested paragraphs (skip exact duplicates)
      - VectorStore.add() also deduplicates by prefix match
      - Ingest directly into subsystems instead of hub.remember()
        to avoid writing back to the daily file (which creates a feedback loop)
    """

    _STATE_FILE = "memory/ingestion_state.json"
    _HASHES_FILE = "memory/ingestion_hashes.json"
    _MSA_STATE_FILE = "memory/msa_ingestion_state.json"
    _INTERVAL = 1800  # 30 minutes
    _MSA_MIN_CHARS = 200

    def __init__(self):
        self._state_path = _WORKSPACE / self._STATE_FILE
        self._hashes_path = _WORKSPACE / self._HASHES_FILE
        self._msa_state_path = _WORKSPACE / self._MSA_STATE_FILE
        self._state: Dict[str, int] = {}
        self._seen_hashes: set = set()
        self._msa_state: Dict[str, int] = {}
        self._load_state()

    def _load_state(self):
        try:
            if self._state_path.exists():
                self._state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            self._state = {}
        try:
            if self._hashes_path.exists():
                self._seen_hashes = set(json.loads(self._hashes_path.read_text(encoding="utf-8")))
        except Exception:
            self._seen_hashes = set()
        try:
            if self._msa_state_path.exists():
                self._msa_state = json.loads(self._msa_state_path.read_text(encoding="utf-8"))
        except Exception:
            self._msa_state = {}

    def _save_state(self):
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._hashes_path.write_text(
                json.dumps(list(self._seen_hashes), ensure_ascii=False), encoding="utf-8"
            )
            self._msa_state_path.write_text(
                json.dumps(self._msa_state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("AutoIngestor: failed to save state: %s", e)

    @staticmethod
    def _content_hash(text: str) -> str:
        import hashlib
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:16]

    def scan_and_ingest(self):
        memory_dir = _WORKSPACE / "memory"
        if not memory_dir.exists():
            return

        md_files = sorted(memory_dir.glob("20??-??-??.md"))
        total_ingested = 0

        for md_file in md_files:
            fname = md_file.name
            try:
                lines = md_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            processed_up_to = self._state.get(fname, 0)
            if len(lines) <= processed_up_to:
                continue

            new_lines = lines[processed_up_to:]
            paragraphs = self._split_paragraphs(new_lines)

            for para in paragraphs:
                text = para.strip()
                if len(text) < 15:
                    continue
                if self._is_test_content(text):
                    continue

                h = self._content_hash(text)
                if h in self._seen_hashes:
                    continue

                try:
                    from memora.vectorstore import vector_store
                    if vector_store.contains(text):
                        self._seen_hashes.add(h)
                        continue

                    from memora.collector import collector
                    collector.collect(text, source="auto_ingest", importance=0.6)
                    vector_store.add(text, metadata={
                        "source": "auto_ingest",
                        "importance": 0.6,
                        "timestamp": datetime.now().isoformat(),
                    })
                    _kg_extract_async(text, 0.6)
                    self._seen_hashes.add(h)
                    total_ingested += 1
                except Exception as e:
                    logger.warning("AutoIngestor: ingest failed: %s", e)

            self._state[fname] = len(lines)

        if total_ingested > 0:
            logger.info("AutoIngestor: ingested %d new paragraphs to Memora", total_ingested)

        msa_ingested = self._sync_daily_to_msa(md_files)
        if msa_ingested > 0:
            logger.info("AutoIngestor: synced %d daily files to MSA", msa_ingested)

        self._save_state()

    def _sync_daily_to_msa(self, md_files: list) -> int:
        """Ingest each daily .md file as a whole MSA document for cross-day deep-recall.

        After MSA ingestion, also trigger KG extraction on each chunk so the
        knowledge graph can capture cross-day relationships.
        """
        synced = 0
        for md_file in md_files:
            fname = md_file.name
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            file_size = len(content)
            if file_size < self._MSA_MIN_CHARS:
                continue

            prev_size = self._msa_state.get(fname, 0)
            if file_size <= prev_size:
                continue

            doc_id = f"daily-{md_file.stem}"
            date_str = md_file.stem
            try:
                from msa.bridge import bridge as msa_bridge
                result = msa_bridge.ingest_and_save(
                    content,
                    source="daily_log",
                    doc_id=doc_id,
                    metadata={"title": f"Daily Log {date_str}", "source": "daily_log", "date": date_str},
                    cross_index=False,
                    write_daily=False,
                )
                self._msa_state[fname] = file_size
                synced += 1
                logger.info("AutoIngestor: MSA synced %s (%d chars)", fname, file_size)

                self._kg_extract_from_msa_doc(doc_id)
            except Exception as e:
                logger.warning("AutoIngestor: MSA sync failed for %s: %s", fname, e)

        return synced

    def _kg_extract_from_msa_doc(self, doc_id: str):
        """Run KG extraction on each chunk of an MSA document (non-blocking)."""
        try:
            from msa.system import msa_system
            chunks, _ = msa_system.memory_bank.load_document_content(doc_id)
            if not chunks:
                return
            for chunk in chunks:
                text = chunk.strip()
                if len(text) >= 50:
                    _kg_extract_async(text, 0.65)
            logger.info("AutoIngestor: queued KG extraction for %s (%d chunks)", doc_id, len(chunks))
        except Exception as e:
            logger.warning("AutoIngestor: KG extract from MSA doc %s failed: %s", doc_id, e)

    def _split_paragraphs(self, lines: list) -> list:
        """Split lines into paragraphs using ### HH:MM:SS headers as delimiters."""
        import re
        paragraphs = []
        current = []
        for line in lines:
            if re.match(r'^###\s+\d{2}:\d{2}:\d{2}', line):
                if current:
                    paragraphs.append("\n".join(current))
                current = []
            else:
                if line.strip():
                    current.append(line)
        if current:
            paragraphs.append("\n".join(current))
        return paragraphs

    def _is_test_content(self, text: str) -> bool:
        lower = text.lower()
        test_markers = [
            "test", "测试", "[chronos/test]", "[hub/test]",
            "test content", "test query",
        ]
        if len(text) < 30 and any(m in lower for m in test_markers):
            return True
        return False

    def start(self):
        def _loop():
            while True:
                try:
                    self.scan_and_ingest()
                except Exception as e:
                    logger.error("AutoIngestor error: %s", e, exc_info=True)
                time.sleep(self._INTERVAL)
        t = threading.Thread(target=_loop, daemon=True, name="auto-ingestor")
        t.start()
        logger.info("AutoIngestor started (interval=%ds)", self._INTERVAL)


# ── Scheduler ─────────────────────────────────────────────────

class Scheduler:
    """Built-in periodic task scheduler with Telegram push capability."""

    _STATE_FILE = "memory/scheduler_state.json"

    def __init__(self, pusher: TelegramPusher):
        self._pusher = pusher
        self._state_path = _WORKSPACE / self._STATE_FILE
        self._state: Dict[str, float] = {}
        self._load_state()
        self._tasks = {
            "morning_briefing": {
                "interval": 86400,
                "hour": 8,
                "fn": self._task_morning_briefing,
            },
            "collision": {
                "interval": 21600,
                "fn": self._task_collision,
            },
            "chronos_consolidate": {
                "interval": 21600,
                "fn": self._task_consolidate,
            },
            "digest": {
                "interval": 86400,
                "fn": self._task_digest,
            },
            "dormant_check": {
                "interval": 259200,
                "fn": self._task_dormant_check,
            },
            "kg_contradiction_scan": {
                "interval": 86400,
                "fn": self._task_contradiction_scan,
            },
            "blindspot_scan": {
                "interval": 604800,
                "fn": self._task_blindspot_scan,
            },
            "skill_proposal": {
                "interval": 86400,
                "fn": self._task_skill_proposal,
            },
        }

    def _load_state(self):
        try:
            if self._state_path.exists():
                self._state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            self._state = {}

    def _save_state(self):
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Scheduler: failed to save state: %s", e)

    def _should_run(self, task_name: str, interval: float, hour: int = None) -> bool:
        last = self._state.get(task_name, 0)
        now = time.time()
        if now - last < interval:
            return False
        if hour is not None:
            current_hour = datetime.now().hour
            if current_hour < hour:
                return False
        return True

    def tick(self):
        for name, cfg in self._tasks.items():
            try:
                if self._should_run(name, cfg["interval"], cfg.get("hour")):
                    logger.info("Scheduler: running %s", name)
                    cfg["fn"]()
                    self._state[name] = time.time()
                    self._save_state()
            except Exception as e:
                logger.error("Scheduler task %s failed: %s", name, e, exc_info=True)

    def _task_morning_briefing(self):
        try:
            from second_brain.bridge import bridge as sb_bridge
            data = sb_bridge.daily_briefing()
            text = self._humanize_briefing(data)
            self._pusher.push(text)
        except Exception as e:
            logger.error("Morning briefing failed: %s", e)

    def _humanize_briefing(self, data: dict) -> str:
        """Generate a warm, human briefing using LLM if available."""
        try:
            import llm_client
            if llm_client.is_available():
                raw_text = data.get("text", "")
                stats = (
                    f"记忆总量={data.get('total_memories', 0)}, "
                    f"灵感={data.get('insight_count', 0)}, "
                    f"沉睡={data.get('dormant_count', 0)}, "
                    f"趋势={data.get('trend_count', 0)}, "
                    f"高活力={data.get('vitality_distribution', {}).get('high', 0)}"
                )
                personality = self._load_personality_style()
                prompt = (
                    f"你是用户的 AI 记忆伙伴。把以下记忆系统数据转化为一段温暖、自然、简洁的中文问候。"
                    f"不要列数字，不要用表格，像朋友一样聊天。控制在 200 字以内。"
                    f"\n\n原始数据:\n{raw_text}\n\n统计: {stats}"
                )
                if personality:
                    prompt += f"\n\n用户沟通风格偏好: {personality}"

                humanized = llm_client.generate(
                    prompt=prompt,
                    system="你是一个有温度的 AI 伙伴，语言简洁温暖，像朋友而非机器人。",
                    max_tokens=400,
                    temperature=0.7,
                )
                if humanized:
                    today = datetime.now().strftime("%Y-%m-%d")
                    return f"🧠 {today}\n\n{humanized}"
        except Exception as e:
            logger.debug("Humanized briefing failed, using raw: %s", e)

        return data.get("text", "无法生成简报")

    def _load_personality_style(self) -> str:
        try:
            import yaml
            p_path = _WORKSPACE / "PERSONALITY.yaml"
            if p_path.exists():
                data = yaml.safe_load(p_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    style = data.get("communication_style", data.get("沟通风格", ""))
                    return str(style)[:200] if style else ""
        except Exception:
            pass
        return ""

    def _task_collision(self):
        try:
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.collide()
            insights = result.get("insights", [])
            high_novelty = [i for i in insights if i.get("novelty", 0) >= 4]
            if high_novelty:
                text = "💡 *灵感碰撞有新发现:*\n\n"
                for ins in high_novelty[:2]:
                    text += (
                        f"*[{ins.get('strategy', '?')}]* (新颖度 {ins.get('novelty', 0)})\n"
                        f"联系: {ins.get('connection', '')[:120]}\n"
                        f"灵感: {ins.get('ideas', '')[:120]}\n\n"
                    )
                self._pusher.push(text)
        except Exception as e:
            logger.error("Collision task failed: %s", e)

    def _task_consolidate(self):
        try:
            from chronos.bridge import bridge as chronos_bridge
            chronos_bridge.consolidate()
        except Exception as e:
            logger.error("Consolidate failed: %s", e)

    def _task_digest(self):
        try:
            from second_brain.digest import digest_memories
            digest_memories(days=7)
        except Exception as e:
            logger.error("Digest failed: %s", e)

    def _task_dormant_check(self):
        try:
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.list_dormant()
            memories = result.get("memories", [])
            if memories:
                text = f"💤 *{len(memories)} 条记忆已经沉睡了:*\n\n"
                for m in memories[:3]:
                    days = m.get("dormant_days", 0)
                    content = m.get("content", "")[:80]
                    text += f"• ({days}天未提及) {content}\n"
                text += "\n需要跟进哪个？"
                self._pusher.push(text)
        except Exception as e:
            logger.error("Dormant check failed: %s", e)

    def _task_contradiction_scan(self):
        try:
            from second_brain.inference import inference_engine
            reports = inference_engine.scan_contradictions()
            critical = [r for r in reports if r.risk_score > 0.5]
            if critical:
                text = f"⚠️ *发现 {len(critical)} 个高风险矛盾:*\n\n"
                for r in critical[:2]:
                    text += (
                        f"决策: {r.decision_content[:80]}\n"
                        f"风险: {r.risk_score:.0%}\n"
                        f"矛盾证据: {r.contradicting[0].get('content', '')[:80] if r.contradicting else '?'}\n\n"
                    )
                self._pusher.push(text)
        except Exception as e:
            logger.error("Contradiction scan failed: %s", e)

    def _task_blindspot_scan(self):
        try:
            from second_brain.inference import inference_engine
            reports = inference_engine.detect_all_blind_spots()
            if reports:
                text = f"🔍 *盲区扫描发现 {len(reports)} 个决策有未考虑的维度:*\n\n"
                for r in reports[:2]:
                    missing = r.missing_dimensions[:3] if hasattr(r, "missing_dimensions") else []
                    text += (
                        f"决策: {r.decision_content[:80]}\n"
                        f"遗漏: {', '.join(missing)}\n\n"
                    )
                self._pusher.push(text)
        except Exception as e:
            logger.error("Blindspot scan failed: %s", e)

    def _task_skill_proposal(self):
        try:
            from second_brain.skill_proposer import proposer
            proposals = proposer.scan_and_propose(days=7)
            if proposals:
                text = f"🎯 *技能提名: {len(proposals)} 个新 draft skill*\n\n"
                for p in proposals[:2]:
                    scores = ", ".join(f"{k}={v}" for k, v in p.sources.items())
                    text += f"*{p.title}*\n分数: {scores}\n\n"
                text += "用 `memory-cli skills` 查看，`memory-cli skill-on <id>` 激活。"
                self._pusher.push(text)
        except Exception as e:
            logger.error("Skill proposal failed: %s", e)

    def start(self):
        def _loop():
            time.sleep(60)
            while True:
                try:
                    self.tick()
                except Exception as e:
                    logger.error("Scheduler tick error: %s", e, exc_info=True)
                time.sleep(300)
        t = threading.Thread(target=_loop, daemon=True, name="scheduler")
        t.start()
        logger.info("Scheduler started (%d tasks configured)", len(self._tasks))


def _track_access_async(results: list, query: str):
    """Record access events for returned search results (non-blocking)."""
    import threading
    def _do_track():
        try:
            from second_brain.bridge import bridge as sb_bridge
            for r in results[:5]:
                content = r.get("content", "")
                sb_bridge.track_access(
                    memory_id=r.get("metadata", {}).get("source", ""),
                    content=content,
                    query=query,
                )
        except Exception:
            pass
    threading.Thread(target=_do_track, daemon=True).start()


def _kg_extract_async(content: str, importance: float):
    """Extract knowledge nodes/edges from new memory (non-blocking).

    If a contradiction is found during propagation, push to Telegram immediately.
    """
    import threading
    def _do_extract():
        try:
            from second_brain.relation_extractor import extractor
            result = extractor.extract(content, importance=importance)
            if not result.skipped:
                logger.info("KG extraction: %d nodes, %d edges, gain=%.2f",
                            len(result.new_nodes), len(result.new_edges),
                            result.structural_gain)
                from second_brain.skill_proposer import save_kg_score
                save_kg_score(result.structural_gain, content[:200])

                from second_brain.inference import inference_engine
                for node in result.new_nodes:
                    alerts = inference_engine.propagate(node.id)
                    for alert in alerts:
                        logger.info("KG propagation alert: %s", alert.message)
                        if "contradicts" in alert.message.lower():
                            _telegram.push(
                                f"⚡ *实时矛盾检测*\n\n{alert.message[:300]}"
                            )
        except Exception as e:
            logger.debug("KG extraction skipped: %s", e)
    threading.Thread(target=_do_extract, daemon=True).start()


class TaskManager:
    """In-memory async task queue for long-running operations."""

    def __init__(self, max_workers: int = 3):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, fn: Callable, *args, **kwargs) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "name": name,
                "status": "running",
                "submitted_at": time.time(),
                "result": None,
                "error": None,
                "completed_at": None,
            }

        def _run():
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._tasks[task_id]["status"] = "done"
                    self._tasks[task_id]["result"] = result
                    self._tasks[task_id]["completed_at"] = time.time()
                elapsed = self._tasks[task_id]["completed_at"] - self._tasks[task_id]["submitted_at"]
                logger.info("Task %s (%s) completed in %.1fs", task_id, name, elapsed)
            except Exception as e:
                with self._lock:
                    self._tasks[task_id]["status"] = "error"
                    self._tasks[task_id]["error"] = str(e)
                    self._tasks[task_id]["completed_at"] = time.time()
                logger.error("Task %s (%s) failed: %s", task_id, name, e, exc_info=True)

        self._pool.submit(_run)
        return task_id

    def get(self, task_id: str) -> Optional[dict]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            info = {
                "id": task["id"],
                "name": task["name"],
                "status": task["status"],
                "submitted_at": task["submitted_at"],
                "elapsed": round(
                    (task["completed_at"] or time.time()) - task["submitted_at"], 1
                ),
            }
            if task["status"] == "done":
                info["result"] = task["result"]
            elif task["status"] == "error":
                info["error"] = task["error"]
            return info

    def list_tasks(self) -> list:
        with self._lock:
            result = []
            for task in self._tasks.values():
                info = {
                    "id": task["id"],
                    "name": task["name"],
                    "status": task["status"],
                    "elapsed": round(
                        (task["completed_at"] or time.time()) - task["submitted_at"], 1
                    ),
                }
                if task["status"] == "error":
                    info["error"] = task["error"]
                result.append(info)
            return result

    def cleanup(self, max_age: float = 3600):
        now = time.time()
        with self._lock:
            expired = [
                tid for tid, t in self._tasks.items()
                if t["status"] in ("done", "error")
                and t["completed_at"]
                and now - t["completed_at"] > max_age
            ]
            for tid in expired:
                del self._tasks[tid]
        if expired:
            logger.info("Cleaned up %d expired tasks", len(expired))


_task_manager = TaskManager(max_workers=3)

_SLOW_ENDPOINTS = frozenset({
    "/second-brain/collide",
    "/second-brain/deep-collide",
    "/deep-recall",
    "/msa/interleave",
    "/chronos/consolidate",
    "/digest",
    "/kg/extract",
})


def _execute_endpoint(path: str, body: dict) -> dict:
    """Core dispatch logic. Stateless — safe to call from any thread."""
    hub = _get_hub()

    if path == "/remember":
        result = hub.remember(
            content=body.get("content", ""),
            source=body.get("source", "openclaw"),
            importance=body.get("importance", 0.7),
            tag=body.get("tag"),
            doc_id=body.get("doc_id"),
            title=body.get("title"),
            force_systems=body.get("force_systems"),
        )
        _kg_extract_async(body.get("content", ""), body.get("importance", 0.7))
        return result

    elif path == "/recall":
        result = hub.recall(
            query=body.get("query", ""),
            top_k=body.get("top_k", 8),
        )
        _track_access_async(result.get("merged", []), body.get("query", ""))
        return result

    elif path == "/deep-recall":
        return hub.deep_recall(
            query=body.get("query", ""),
            max_rounds=body.get("max_rounds", 3),
        )

    elif path == "/search":
        from memora.vectorstore import vector_store
        results = vector_store.search(
            query=body.get("query", ""),
            limit=body.get("limit", 8),
        )
        _track_access_async(results, body.get("query", ""))
        return {"results": results}

    elif path == "/add":
        from memora.collector import collector
        from memora.vectorstore import vector_store
        content = body.get("content", "")
        source = body.get("source", "cli")
        importance = body.get("importance", 0.7)
        entry = collector.collect(content, source=source, importance=importance)
        vector_store.add(content, metadata={
            "source": source,
            "importance": importance,
            "timestamp": entry["timestamp"],
        })
        return entry

    elif path == "/digest":
        from second_brain.digest import digest_memories
        days = body.get("days", 7)
        ok = digest_memories(days=days)
        return {"success": ok, "days": days}

    elif path == "/chronos/learn":
        from chronos.bridge import bridge as chronos_bridge
        result = chronos_bridge.learn_and_save(
            content=body.get("content", ""),
            source=body.get("source", "openclaw"),
            importance=body.get("importance", 0.75),
        )
        return {"importance": result.importance, "timestamp": result.timestamp.isoformat()}

    elif path == "/chronos/consolidate":
        from chronos.bridge import bridge as chronos_bridge
        return chronos_bridge.consolidate()

    elif path == "/msa/ingest":
        from msa.bridge import bridge as msa_bridge
        return msa_bridge.ingest_and_save(
            content=body.get("content", ""),
            source=body.get("source", "openclaw"),
            doc_id=body.get("doc_id"),
            metadata=body.get("metadata"),
        )

    elif path == "/msa/query":
        from msa.bridge import bridge as msa_bridge
        return msa_bridge.query_memory(
            question=body.get("query", ""),
            top_k=body.get("top_k"),
        )

    elif path == "/msa/interleave":
        from msa.bridge import bridge as msa_bridge
        return msa_bridge.interleave_query(
            question=body.get("query", ""),
            max_rounds=body.get("max_rounds"),
        )

    elif path == "/second-brain/collide":
        from second_brain.bridge import bridge as sb_bridge
        return sb_bridge.collide()

    elif path == "/second-brain/deep-collide":
        from second_brain.bridge import bridge as sb_bridge
        return sb_bridge.deep_collide(topic=body.get("topic", ""))

    elif path == "/inspect":
        from second_brain.bridge import bridge as sb_bridge
        return sb_bridge.memory_lifecycle(query=body.get("query", ""))

    elif path == "/second-brain/track":
        from second_brain.bridge import bridge as sb_bridge
        sb_bridge.track_access(
            memory_id=body.get("memory_id", ""),
            content=body.get("content", ""),
            query=body.get("query", ""),
        )
        return {"ok": True}

    elif path == "/kg/extract":
        from second_brain.relation_extractor import extractor
        result = extractor.extract(
            content=body.get("content", ""),
            importance=body.get("importance", 0.5),
            source_hash=body.get("source_hash", ""),
        )
        return {
            "skipped": result.skipped,
            "reason": result.reason,
            "new_nodes": len(result.new_nodes),
            "new_edges": len(result.new_edges),
        }

    elif path == "/insight/rate":
        from second_brain.strategy_weights import strategy_weights
        return strategy_weights.rate_insight(
            insight_id=body.get("insight_id", ""),
            strategy=body.get("strategy", ""),
            rating=int(body.get("rating", 3)),
            comment=body.get("comment", ""),
        )

    elif path == "/bookmark":
        return _save_bookmark(body.get("summary", ""), body.get("topics", []))

    elif path == "/skills/add":
        from skill_registry import registry
        skill = registry.add(
            name=body.get("name", ""),
            content=body.get("content", ""),
            tags=body.get("tags", []),
            source_memories=body.get("source_memories", []),
        )
        return skill.to_dict()

    elif path == "/skills/promote":
        from skill_registry import registry
        skill = registry.promote(body.get("skill_id", ""))
        return skill.to_dict() if skill else {"error": "Skill not found"}

    elif path == "/skills/deprecate":
        from skill_registry import registry
        skill = registry.deprecate(body.get("skill_id", ""))
        return skill.to_dict() if skill else {"error": "Skill not found"}

    elif path == "/skills/propose":
        from second_brain.skill_proposer import proposer
        days = body.get("days", 7)
        proposals = proposer.scan_and_propose(days=days)
        if proposals:
            return {
                "proposals": [
                    {"title": p.title, "sources": p.sources, "tags": p.tags}
                    for p in proposals
                ],
                "count": len(proposals),
            }
        kg_score = proposer._get_best_kg_score(days)
        digest_score = proposer._get_best_digest_score(days)
        collision_score = proposer._get_best_collision_score(days)
        return {
            "proposals": [],
            "scores": {
                "kg": kg_score,
                "digest": digest_score,
                "collision": collision_score,
            },
            "message": "需要任意两个分数达标才能提名技能",
        }

    elif path == "/skills/feedback":
        from skill_registry import registry
        skill = registry.record_feedback(
            skill_id=body.get("skill_id", ""),
            query=body.get("query", ""),
            outcome=body.get("outcome", "failure"),
            context=body.get("context", ""),
        )
        if skill:
            return {
                "skill_id": skill.id,
                "name": skill.name,
                "utility_rate": round(skill.utility_rate, 3),
                "total_uses": skill.total_uses,
                "version": skill.version,
            }
        return {"error": "Skill not found"}

    elif path == "/training/export":
        from chronos.distiller import distiller
        merged = distiller.prepare_merged()
        return {"dataset_path": str(merged)}

    else:
        raise ValueError(f"Unknown endpoint: {path}")


# ── Bookmark + Session Context ────────────────────────────────

def _save_bookmark(summary: str, topics: list = None) -> dict:
    """Save a conversation bookmark for session continuity."""
    bookmark_path = _WORKSPACE / "memory" / "bookmarks.jsonl"
    bookmark_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "summary": summary,
        "topics": topics or [],
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(bookmark_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"ok": True, "timestamp": entry["timestamp"]}


def _get_session_context() -> dict:
    """Build a pre-constructed context block for session startup.

    Only uses fast, local data lookups. Avoids any LLM or heavy embedding calls
    to keep response time <2s.
    """
    result: Dict[str, Any] = {}

    bookmark_path = _WORKSPACE / "memory" / "bookmarks.jsonl"
    if bookmark_path.exists():
        try:
            lines = bookmark_path.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                result["last_conversation_summary"] = last.get("summary", "")
                result["last_conversation_topics"] = last.get("topics", [])
                result["last_conversation_date"] = last.get("date", "")
        except Exception:
            pass

    try:
        from second_brain.knowledge_graph import kg
        stats = kg.stats()
        node_types = stats.get("node_types", {})
        threads_hint = []
        for ntype in ["goal", "question", "decision"]:
            count = node_types.get(ntype, 0)
            if count > 0:
                threads_hint.append({"type": ntype, "count": count})
        result["active_threads"] = threads_hint
    except Exception:
        result["active_threads"] = []

    try:
        from second_brain.tracker import tracker as sb_tracker
        trends = sb_tracker.find_trends()
        result["recent_focus"] = [
            f"{t.get('queries', ['?'])[0]} ({t['hits']}次)" for t in trends[:5]
        ]
    except Exception:
        result["recent_focus"] = []

    result["pending_contradictions"] = []

    personality_path = _WORKSPACE / "PERSONALITY.yaml"
    if personality_path.exists():
        try:
            import yaml
            pdata = yaml.safe_load(personality_path.read_text(encoding="utf-8"))
            if isinstance(pdata, dict):
                traits = []
                for k, v in pdata.items():
                    if isinstance(v, str) and len(v) < 100:
                        traits.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        traits.append(f"{k}: {', '.join(str(x) for x in v[:3])}")
                result["personality_traits"] = "; ".join(traits[:6])
            else:
                result["personality_traits"] = ""
        except Exception:
            result["personality_traits"] = ""
    else:
        result["personality_traits"] = ""

    result["dormant_reminders"] = []

    try:
        from skill_registry import registry
        active = registry.list_active()
        result["active_skills"] = [
            {"name": s.name, "id": s.id, "tags": s.tags}
            for s in active[:10]
        ]
    except Exception:
        result["active_skills"] = []

    result["milestones"] = _compute_milestones()

    return result


def _compute_milestones() -> dict:
    memory_dir = _WORKSPACE / "memory"
    milestones = {}
    if memory_dir.exists():
        md_files = sorted(memory_dir.glob("20??-??-??.md"))
        if md_files:
            try:
                first_date_str = md_files[0].stem
                first_date = datetime.strptime(first_date_str, "%Y-%m-%d")
                days = (datetime.now() - first_date).days
                milestones["days_since_first_memory"] = days
            except Exception:
                pass
            milestones["total_days_with_logs"] = len(md_files)

    bookmark_path = _WORKSPACE / "memory" / "bookmarks.jsonl"
    if bookmark_path.exists():
        try:
            lines = bookmark_path.read_text(encoding="utf-8").strip().splitlines()
            today = datetime.now().strftime("%Y-%m-%d")
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            this_week = sum(
                1 for l in lines
                if json.loads(l).get("date", "") >= week_ago
            )
            milestones["conversations_this_week"] = this_week
        except Exception:
            pass

    return milestones


class MemoryHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the memory API."""

    def do_POST(self):
        try:
            self._handle_post()
        except Exception:
            logger.exception("Unhandled POST error on %s", self.path)
            try:
                self._respond(500, {"error": "Internal server error"})
            except Exception:
                pass

    def _handle_post(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
        except (json.JSONDecodeError, ValueError):
            self._respond(400, {"error": "Invalid JSON"})
            return

        use_async = body.pop("async", False)

        if use_async and self.path in _SLOW_ENDPOINTS:
            self._dispatch_async(body)
            return

        self._dispatch_sync(body)

    def _dispatch_async(self, body: dict):
        """Submit the request to the task manager and return immediately."""
        path = self.path

        def _run():
            return _execute_endpoint(path, body)

        task_id = _task_manager.submit(path, _run)
        logger.info("Async task %s submitted for %s", task_id, path)
        self._respond(202, {
            "task_id": task_id,
            "status": "running",
            "endpoint": path,
            "poll": f"/task/{task_id}",
        })

    def _dispatch_sync(self, body: dict):
        """Execute the request synchronously and return the result."""
        try:
            result = _execute_endpoint(self.path, body)
        except ValueError as e:
            self._respond(404, {"error": str(e)})
            return
        self._respond(200, result)

    def do_GET(self):
        try:
            self._handle_get()
        except Exception:
            logger.exception("Unhandled GET error on %s", self.path)
            try:
                self._respond(500, {"error": "Internal server error"})
            except Exception:
                pass

    def _handle_get(self):
        if self.path == "/health":
            _task_manager.cleanup()
            self._respond(200, {
                "status": "ok",
                "uptime_seconds": round(time.time() - _START_TIME),
                "embedder": _EMBEDDER_TYPE,
                "pid": os.getpid(),
            })

        elif self.path == "/status":
            hub = _get_hub()
            result = hub.status()
            result["server"] = {
                "uptime_seconds": round(time.time() - _START_TIME),
                "embedder": _EMBEDDER_TYPE,
            }
            self._respond(200, result)

        elif self.path.startswith("/task/"):
            task_id = self.path.split("/task/", 1)[1]
            task = _task_manager.get(task_id)
            if task is None:
                self._respond(404, {"error": f"Task {task_id} not found"})
            else:
                self._respond(200, task)

        elif self.path == "/tasks":
            tasks = _task_manager.list_tasks()
            self._respond(200, {"tasks": tasks, "count": len(tasks)})

        elif self.path == "/briefing":
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.daily_briefing()
            self._respond(200, result)

        elif self.path == "/vitality":
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.vitality_list()
            self._respond(200, result)

        elif self.path == "/dormant":
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.list_dormant()
            self._respond(200, result)

        elif self.path == "/second-brain/report":
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.report()
            self._respond(200, result)

        elif self.path == "/second-brain/status":
            from second_brain.bridge import bridge as sb_bridge
            result = sb_bridge.status()
            self._respond(200, result)

        elif self.path == "/contradictions":
            from second_brain.inference import inference_engine
            reports = inference_engine.scan_contradictions()
            self._respond(200, {
                "count": len(reports),
                "reports": [r.to_dict() for r in reports],
            })

        elif self.path == "/blindspots":
            from second_brain.inference import inference_engine
            reports = inference_engine.detect_all_blind_spots()
            self._respond(200, {
                "count": len(reports),
                "reports": [r.to_dict() for r in reports],
            })

        elif self.path == "/threads":
            from second_brain.inference import inference_engine
            threads = inference_engine.discover_threads()
            self._respond(200, {
                "count": len(threads),
                "threads": [t.to_dict() for t in threads],
            })

        elif self.path == "/kg/status":
            from second_brain.knowledge_graph import kg
            self._respond(200, kg.stats())

        elif self.path == "/kg/internalization":
            from second_brain.internalization import internalization_manager
            self._respond(200, internalization_manager.status())

        elif self.path == "/insight/stats":
            from second_brain.strategy_weights import strategy_weights
            self._respond(200, strategy_weights.stats())

        elif self.path == "/session-context":
            self._respond(200, _get_session_context())

        elif self.path == "/skills":
            from skill_registry import registry
            skills = [s.to_dict() for s in registry.list_all()]
            self._respond(200, {"skills": skills, "count": len(skills)})

        elif self.path == "/skills/active":
            from skill_registry import registry
            skills = [s.to_dict() for s in registry.list_active()]
            self._respond(200, {"skills": skills, "count": len(skills)})

        elif self.path == "/skills/stats":
            from skill_registry import registry
            self._respond(200, registry.stats())

        elif self.path == "/skills/usage":
            from skill_registry import registry
            self._respond(200, {"usage": registry.get_usage_stats()})

        else:
            self._respond(404, {"error": f"Unknown endpoint: {self.path}"})

    def _respond(self, code: int, data: Any):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logger.info("%s %s", self.command, self.path)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _daemonize(pid_file: Path, log_dir: Path):
    """
    Proper Unix double-fork daemon.

    1st fork  — detach from parent terminal
    setsid    — become session leader
    2nd fork  — prevent re-acquiring a controlling terminal
    Redirect  — stdin→/dev/null, stdout/stderr→log files
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "server-stdout.log"
    stderr_log = log_dir / "server-stderr.log"

    pid = os.fork()
    if pid > 0:
        # Parent waits briefly for child to write PID, then exits.
        time.sleep(0.3)
        if pid_file.exists():
            child_pid = pid_file.read_text().strip()
            print(f"Memory server started (PID {child_pid}, port 18790)")
        else:
            print(f"Memory server forked (PID {pid})")
        sys.exit(0)

    os.setsid()

    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # Now in grandchild — the actual daemon process.
    # Redirect standard file descriptors.
    sys.stdin.close()
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)

    out_fd = os.open(str(stdout_log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    err_fd = os.open(str(stderr_log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(out_fd, 1)
    os.dup2(err_fd, 2)
    os.close(out_fd)
    os.close(err_fd)

    # Reattach Python file objects to new FDs.
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

    # Reconfigure logging to use the new stderr.
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [memory-server] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Memory Server")
    parser.add_argument("--port", type=int, default=18790, help="Listen port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--daemon", action="store_true", help="Run in background")
    args = parser.parse_args()

    _load_env()
    _setup_hf_offline()

    # Ignore SIGPIPE — prevents broken-pipe from killing the daemon
    # when a client disconnects mid-response.
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    pid_file = _WORKSPACE / "memory" / "server.pid"
    log_dir = _WORKSPACE / "memory"

    if args.daemon:
        _daemonize(pid_file, log_dir)
        _force_cpu_for_daemon()

    _server_ref = None

    def _shutdown(signum, frame):
        logger.info("Shutting down (signal %s)...", signum)
        if _server_ref is not None:
            _server_ref.shutdown()
        if pid_file.exists():
            pid_file.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    logger.info("Loading shared embedder...")
    _load_shared_embedder()

    ingestor = AutoIngestor()
    ingestor.start()

    scheduler = Scheduler(_telegram)
    scheduler.start()

    server = ThreadedServer((args.host, args.port), MemoryHandler)
    _server_ref = server
    logger.info("Memory Server listening on %s:%d (PID %d)",
                args.host, args.port, os.getpid())
    logger.info("Endpoints: /remember /recall /deep-recall /status /health /search /add /digest")
    logger.info("           /chronos/learn /chronos/consolidate")
    logger.info("           /msa/ingest /msa/query /msa/interleave")
    logger.info("           /second-brain/collide /second-brain/report /second-brain/status /second-brain/track")
    logger.info("           /session-context /bookmark")
    logger.info("           /task/<id> /tasks  (async: POST with {\"async\":true})")
    logger.info("Background: AutoIngestor (30min), Scheduler (7 tasks)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Fatal server error")
    finally:
        server.server_close()
        if pid_file.exists():
            pid_file.unlink()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
