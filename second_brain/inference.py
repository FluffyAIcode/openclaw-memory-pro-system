"""Inference Engine — three reasoning capabilities that RAG cannot provide.

1. Contradiction Detection: find decisions with conflicting evidence
2. Absence Reasoning: discover blind spots (what hasn't been considered)
3. Forward Propagation: trace impact of new facts through dependency chains

Also: automatic thread discovery via graph community detection.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from .config import load_config
from .knowledge_graph import KGEdgeType, KGNode, KGNodeType, KnowledgeGraph, kg

logger = logging.getLogger(__name__)
config = load_config()


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class ContradictionReport:
    decision: KGNode
    supporting: List[dict] = field(default_factory=list)
    contradicting: List[dict] = field(default_factory=list)
    risk_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision.id,
            "decision_content": self.decision.content[:200],
            "supporting_count": len(self.supporting),
            "contradicting_count": len(self.contradicting),
            "risk_score": round(self.risk_score, 3),
            "supporting": self.supporting[:3],
            "contradicting": self.contradicting[:3],
        }


@dataclass
class BlindSpotReport:
    decision: KGNode
    expected_dimensions: List[str] = field(default_factory=list)
    covered_dimensions: List[str] = field(default_factory=list)
    missing_dimensions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision.id,
            "decision_content": self.decision.content[:200],
            "expected": self.expected_dimensions,
            "covered": self.covered_dimensions,
            "missing": self.missing_dimensions,
            "coverage_ratio": (
                round(len(self.covered_dimensions) /
                      max(len(self.expected_dimensions), 1), 2)
            ),
        }


@dataclass
class PropagationAlert:
    new_node: KGNode
    affected_node: KGNode
    path_ids: List[str] = field(default_factory=list)
    has_contradiction: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "new_node_id": self.new_node.id,
            "new_node_content": self.new_node.content[:200],
            "affected_node_id": self.affected_node.id,
            "affected_node_content": self.affected_node.content[:200],
            "affected_node_type": self.affected_node.node_type.value,
            "path_length": len(self.path_ids),
            "has_contradiction": self.has_contradiction,
            "message": self.message,
        }


@dataclass
class Thread:
    title: str
    node_ids: List[str] = field(default_factory=list)
    node_count: int = 0
    dominant_type: str = ""
    status: str = "active"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "node_count": self.node_count,
            "dominant_type": self.dominant_type,
            "status": self.status,
            "node_ids": self.node_ids[:10],
        }


# ── Inference Engine ─────────────────────────────────────────────

class InferenceEngine:

    def __init__(self, knowledge_graph: KnowledgeGraph = None):
        self._kg = knowledge_graph or kg
        self._thread_name_cache: Dict[str, str] = {}

    # ── 1. Contradiction Detection ───────────────────────────────

    def scan_contradictions(self) -> List[ContradictionReport]:
        """Find decisions with conflicting evidence, ranked by risk."""
        reports = []
        for decision in self._kg.get_decisions():
            in_edges = self._kg.get_edges(decision.id, direction="in")

            supporting = []
            contradicting = []
            for src, tgt, data in in_edges:
                src_node = self._kg.get_node(src)
                if src_node is None:
                    continue
                entry = {
                    "node_id": src,
                    "content": src_node.content[:150],
                    "weight": data.get("weight", 0.5),
                    "evidence": data.get("evidence", ""),
                }
                if data.get("edge_type") == KGEdgeType.SUPPORTS.value:
                    supporting.append(entry)
                elif data.get("edge_type") == KGEdgeType.CONTRADICTS.value:
                    contradicting.append(entry)

            if not contradicting:
                continue

            support_w = sum(e["weight"] for e in supporting)
            contra_w = sum(e["weight"] for e in contradicting)
            risk = contra_w / (support_w + contra_w + 0.001)

            reports.append(ContradictionReport(
                decision=decision,
                supporting=supporting,
                contradicting=contradicting,
                risk_score=risk,
            ))

        reports.sort(key=lambda r: r.risk_score, reverse=True)
        return reports

    # ── 2. Absence Reasoning (Blind Spot Detection) ──────────────

    def detect_blind_spots(self, decision_node_id: str) -> Optional[BlindSpotReport]:
        """For a specific decision, find dimensions not yet considered."""
        decision = self._kg.get_node(decision_node_id)
        if decision is None or decision.node_type != KGNodeType.DECISION:
            return None

        expected = self._generate_expected_dimensions(decision)
        if not expected:
            return None

        covered = self._get_covered_dimensions(decision)
        missing = [d for d in expected if d not in covered]

        return BlindSpotReport(
            decision=decision,
            expected_dimensions=expected,
            covered_dimensions=covered,
            missing_dimensions=missing,
        )

    def detect_all_blind_spots(self) -> List[BlindSpotReport]:
        """Run absence reasoning on all decision nodes."""
        reports = []
        for decision in self._kg.get_decisions():
            report = self.detect_blind_spots(decision.id)
            if report and report.missing_dimensions:
                reports.append(report)
        reports.sort(key=lambda r: len(r.missing_dimensions), reverse=True)
        return reports

    def _generate_expected_dimensions(self, decision: KGNode) -> List[str]:
        """Use LLM to generate the dimensions that should be considered."""
        try:
            import llm_client
            if not llm_client.is_available():
                return self._heuristic_dimensions(decision)
        except ImportError:
            return self._heuristic_dimensions(decision)

        prompt = (
            f"对于以下决策，列出做这个决策时通常应该考虑的 5-8 个关键维度。\n\n"
            f"决策：{decision.content}\n\n"
            f"只输出 JSON 数组，每个元素是一个维度名称（2-4个字）。例如：\n"
            f'["成本", "风险", "时间", "可维护性", "替代方案"]\n'
        )

        raw = llm_client.generate(
            prompt=prompt,
            system="只输出 JSON 数组，不要其他内容。",
            max_tokens=200,
            temperature=0.3,
            model=llm_client.FAST_MODEL,
        )
        if not raw:
            return self._heuristic_dimensions(decision)

        try:
            text = raw.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return self._heuristic_dimensions(decision)

    def _heuristic_dimensions(self, decision: KGNode) -> List[str]:
        """Fallback: generic decision dimensions when LLM is unavailable."""
        return ["成本", "风险", "时间约束", "替代方案", "可维护性", "扩展性"]

    def _get_covered_dimensions(self, decision: KGNode) -> List[str]:
        """Check which dimensions have evidence (edges) in the graph."""
        neighbors = self._kg.get_neighbors(decision.id, direction="in")
        covered = set()
        for neighbor in neighbors:
            content_lower = neighbor.content.lower()
            for dim in ["成本", "cost", "价格", "预算"]:
                if dim in content_lower:
                    covered.add("成本")
            for dim in ["风险", "risk", "问题", "缺点", "劣势"]:
                if dim in content_lower:
                    covered.add("风险")
            for dim in ["时间", "time", "工期", "deadline", "进度"]:
                if dim in content_lower:
                    covered.add("时间约束")
            for dim in ["替代", "alternative", "备选", "其他方案", "另一个"]:
                if dim in content_lower:
                    covered.add("替代方案")
            for dim in ["维护", "maintain", "运维", "长期"]:
                if dim in content_lower:
                    covered.add("可维护性")
            for dim in ["扩展", "scale", "增长", "容量"]:
                if dim in content_lower:
                    covered.add("扩展性")
            for dim in ["安全", "security", "隐私", "加密"]:
                if dim in content_lower:
                    covered.add("安全性")
            for dim in ["性能", "performance", "速度", "延迟", "吞吐"]:
                if dim in content_lower:
                    covered.add("性能")
            for dim in ["用户体验", "ux", "易用", "交互"]:
                if dim in content_lower:
                    covered.add("用户体验")
        return list(covered)

    # ── 3. Forward Propagation ───────────────────────────────────

    def propagate(self, new_node_id: str) -> List[PropagationAlert]:
        """When a new node is added, trace its impact through the graph."""
        new_node = self._kg.get_node(new_node_id)
        if new_node is None:
            return []

        alerts = []
        try:
            descendants = self._kg.get_descendants(new_node_id)
        except Exception:
            return []

        for desc_id in descendants:
            desc_node = self._kg.get_node(desc_id)
            if desc_node is None:
                continue
            if desc_node.node_type not in (KGNodeType.DECISION, KGNodeType.GOAL):
                continue

            try:
                path = nx.shortest_path(self._kg._graph, new_node_id, desc_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            has_contradiction = self._path_has_contradiction(path)

            if has_contradiction or len(path) <= 3:
                message = self._build_alert_message(new_node, desc_node, path,
                                                     has_contradiction)
                alerts.append(PropagationAlert(
                    new_node=new_node,
                    affected_node=desc_node,
                    path_ids=path,
                    has_contradiction=has_contradiction,
                    message=message,
                ))

        alerts.sort(key=lambda a: (a.has_contradiction, -len(a.path_ids)),
                    reverse=True)
        return alerts

    def _path_has_contradiction(self, path: List[str]) -> bool:
        for i in range(len(path) - 1):
            edge_data = self._kg._graph.get_edge_data(path[i], path[i + 1])
            if edge_data and edge_data.get("edge_type") == KGEdgeType.CONTRADICTS.value:
                return True
        return False

    def _build_alert_message(self, new_node: KGNode, affected: KGNode,
                              path: List[str], has_contradiction: bool) -> str:
        if has_contradiction:
            return (f"新信息「{new_node.content[:50]}」与你关于"
                    f"「{affected.content[:50]}」的{affected.node_type.value}存在矛盾链路")
        return (f"新信息「{new_node.content[:50]}」可能影响"
                f"「{affected.content[:50]}」（{len(path)-1}步关联）")

    # ── 4. Thread Discovery ──────────────────────────────────────

    def discover_threads(self) -> List[Thread]:
        """Use community detection to auto-discover thought threads.

        F-09: batch LLM naming (10 communities per call) + cache + heuristic
        for tiny communities (< 3 nodes) to reduce ~650s → ~70s.
        """
        communities = self._kg.get_communities()
        valid = [c for c in communities if len(c) >= 2]

        needs_llm = []
        small = []
        for community in valid:
            cache_key = self._community_cache_key(community)
            if cache_key in self._thread_name_cache:
                continue
            if len(community) < 3:
                small.append(community)
            else:
                needs_llm.append(community)

        for c in small:
            key = self._community_cache_key(c)
            self._thread_name_cache[key] = self._heuristic_name(c)

        if needs_llm:
            self._name_threads_batch(needs_llm)

        threads = []
        for community in valid:
            title = self._thread_name_cache.get(
                self._community_cache_key(community),
                self._heuristic_name(community))
            type_counts: Dict[str, int] = {}
            for node in community:
                t = node.node_type.value
                type_counts[t] = type_counts.get(t, 0) + 1
            dominant = max(type_counts, key=type_counts.get) if type_counts else "fact"
            status = self._assess_thread_status(community)
            threads.append(Thread(
                title=title,
                node_ids=[n.id for n in community],
                node_count=len(community),
                dominant_type=dominant,
                status=status,
            ))
        threads.sort(key=lambda t: t.node_count, reverse=True)
        return threads

    @staticmethod
    def _community_cache_key(community: List[KGNode]) -> str:
        ids = sorted(n.id for n in community)
        return "|".join(ids[:5])

    @staticmethod
    def _heuristic_name(community: List[KGNode]) -> str:
        for node in community:
            if node.node_type in (KGNodeType.GOAL, KGNodeType.DECISION):
                return node.content[:30]
        return community[0].content[:30] if community else "未命名线索"

    def _name_threads_batch(self, communities: List[List[KGNode]],
                            batch_size: int = 10):
        """Batch-name communities via LLM — 10 per call (F-09)."""
        try:
            import llm_client
            if not llm_client.is_available():
                for c in communities:
                    self._thread_name_cache[self._community_cache_key(c)] = self._heuristic_name(c)
                return
        except ImportError:
            for c in communities:
                self._thread_name_cache[self._community_cache_key(c)] = self._heuristic_name(c)
            return

        for start in range(0, len(communities), batch_size):
            batch = communities[start:start + batch_size]
            prompt_parts = []
            for i, community in enumerate(batch, 1):
                nodes_text = "\n".join(
                    f"  - ({n.node_type.value}) {n.content[:60]}"
                    for n in community[:5]
                )
                prompt_parts.append(f"社区 {i} ({len(community)} 节点):\n{nodes_text}")

            prompt = (
                "为以下每个知识社区各取一个简短标题（5-10字）。\n"
                "严格按格式输出，每行一个：1. 标题\n2. 标题\n...\n\n"
                + "\n\n".join(prompt_parts)
            )

            try:
                raw = llm_client.generate(
                    prompt=prompt,
                    system="只输出编号+标题列表，不要其他内容。",
                    max_tokens=30 * len(batch),
                    temperature=0.3,
                    model=llm_client.FAST_MODEL,
                )
                titles = self._parse_batch_titles(raw, len(batch))
                for j, community in enumerate(batch):
                    key = self._community_cache_key(community)
                    if j < len(titles) and titles[j]:
                        self._thread_name_cache[key] = titles[j]
                    else:
                        self._thread_name_cache[key] = self._heuristic_name(community)
            except Exception as e:
                logger.warning("Batch naming failed: %s", e)
                for community in batch:
                    key = self._community_cache_key(community)
                    self._thread_name_cache[key] = self._heuristic_name(community)

    @staticmethod
    def _parse_batch_titles(raw: str, expected: int) -> List[str]:
        if not raw:
            return []
        titles = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            cleaned = line.lstrip("0123456789.-) ").strip().strip('"').strip("'")
            if cleaned:
                titles.append(cleaned[:30])
        while len(titles) < expected:
            titles.append("")
        return titles

    def _assess_thread_status(self, community: List[KGNode]) -> str:
        has_question = any(n.node_type == KGNodeType.QUESTION for n in community)
        has_decision = any(n.node_type == KGNodeType.DECISION for n in community)
        if has_question and not has_decision:
            return "exploring"
        if has_decision:
            return "decided"
        if len(community) < 3:
            return "nascent"
        return "developing"


inference_engine = InferenceEngine()
