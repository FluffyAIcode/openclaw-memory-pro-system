"""Relation Extractor — extracts typed knowledge nodes and edges from new memories.

Pipeline:
  1. Skip low-importance memories (save LLM cost)
  2. Embedding pre-filter: find top-K candidate nodes already in the KG
  3. LLM structured extraction: identify new knowledge units + relationships
  4. Write results into the KG

This is the bridge between flat memory storage (Memora/MSA) and the structured
knowledge graph. It transforms "a piece of text" into "facts, decisions,
preferences, and their logical relationships."
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .config import load_config
from .knowledge_graph import (
    KGEdge, KGEdgeType, KGNode, KGNodeType, KnowledgeGraph, kg,
)

logger = logging.getLogger(__name__)
config = load_config()

_EXTRACTION_PROMPT = """\
你是一个知识图谱构建助手。给定一条新记忆和已有的知识节点列表，请提取结构化信息。

## 新记忆
{content}

## 已有知识节点（最相关的前{candidate_count}个）
{candidates_formatted}

## 输出要求
请输出严格的 JSON（不要加 ```json 标记）：
{{{{
  "new_nodes": [
    {{{{"content": "提取出的知识单元（简洁准确）", "type": "fact|decision|preference|goal|question", "confidence": 0.9, "sentiment": "positive|negative|neutral|excited|frustrated|curious|determined|concerned"}}}}
  ],
  "edges": [
    {{{{"from_content": "新节点内容 或 已有节点ID", "to_content": "新节点内容 或 已有节点ID", "type": "supports|contradicts|extends|depends_on|alternative_to|addresses", "weight": 0.8, "evidence": "一句话理由"}}}}
  ]
}}}}

## 规则
- 只提取有信息量的知识单元，跳过寒暄和无意义的内容
- type 必须准确：fact=客观事实, decision=用户做的选择, preference=偏好/倾向, goal=目标/愿望, question=未解决的问题
- sentiment 表示这条知识携带的情感色彩：positive(积极), negative(消极), neutral(中性), excited(兴奋), frustrated(沮丧), curious(好奇), determined(坚定), concerned(担忧)
- 如果新记忆与任何已有节点存在矛盾（contradicts），必须标出——这是最有价值的关系
- depends_on 表示逻辑依赖（A 成立的前提是 B 成立）
- 如果没有有价值的知识，返回空数组: {{{{"new_nodes": [], "edges": []}}}}
- from_content / to_content: 如果引用已有节点，使用其 ID（如 "abc123def456"）；如果引用新节点，使用该新节点的 content 文本
"""


@dataclass
class ExtractionResult:
    new_nodes: List[KGNode] = field(default_factory=list)
    new_edges: List[KGEdge] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""
    structural_gain: float = 0.0

    @classmethod
    def empty(cls, reason: str = "") -> "ExtractionResult":
        return cls(skipped=True, reason=reason)


class RelationExtractor:
    """Extracts typed knowledge nodes and edges from raw memory text."""

    def __init__(self, knowledge_graph: KnowledgeGraph = None):
        self._kg = knowledge_graph or kg

    def extract(self, content: str, importance: float = 0.5,
                source_hash: str = "") -> ExtractionResult:
        if not content or len(content.strip()) < 10:
            return ExtractionResult.empty("content too short")

        if importance < config.relation_extraction_min_importance:
            return ExtractionResult.empty(
                f"importance {importance:.2f} below threshold "
                f"{config.relation_extraction_min_importance}"
            )

        candidates = self._find_candidate_nodes(content)
        result = self._llm_extract(content, candidates)

        if result.skipped:
            return result

        node_id_map = {}
        for node in result.new_nodes:
            if source_hash:
                node.source_hashes.append(source_hash)
            node_id = self._kg.add_node(node)
            node_id_map[node.content] = node_id

        for edge in result.new_edges:
            edge.source_id = self._resolve_node_id(edge.source_id, node_id_map)
            edge.target_id = self._resolve_node_id(edge.target_id, node_id_map)
            self._kg.add_edge(edge)

        result.structural_gain = self._calc_structural_gain(
            result.new_nodes, result.new_edges, node_id_map)

        logger.info("Extraction complete: %d nodes, %d edges, gain=%.2f from content (%.0f chars)",
                     len(result.new_nodes), len(result.new_edges),
                     result.structural_gain, len(content))
        return result

    def _calc_structural_gain(self, new_nodes: List[KGNode],
                              new_edges: List[KGEdge],
                              node_id_map: dict) -> float:
        """Pure graph-structural score: how much value did this extraction add?

        Components:
          - integration: fraction of new edges connecting to pre-existing nodes
          - contradiction: bonus if any contradicts edge was found
          - question_addressed: bonus if any edge addresses a question node
          - bridge: bonus if extraction connects previously disconnected communities
        """
        if not new_nodes and not new_edges:
            return 0.0

        new_ids = set(node_id_map.values())
        all_node_ids = set(self._kg._nodes.keys())
        existing_ids = all_node_ids - new_ids

        edges_to_existing = 0
        has_contradiction = False
        addresses_question = False

        for edge in new_edges:
            src, tgt = edge.source_id, edge.target_id
            if src in existing_ids or tgt in existing_ids:
                edges_to_existing += 1
            if edge.edge_type == KGEdgeType.CONTRADICTS:
                has_contradiction = True
            if edge.edge_type == KGEdgeType.ADDRESSES:
                other_id = tgt if src in new_ids else src
                node = self._kg.get_node(other_id)
                if node and node.node_type == KGNodeType.QUESTION:
                    addresses_question = True

        integration = (edges_to_existing / max(len(new_edges), 1)) if new_edges else 0.0

        bridges_communities = False
        try:
            import networkx as nx
            undirected = self._kg._graph.to_undirected()
            connected_existing = set()
            for nid in new_ids:
                for neighbor in undirected.neighbors(nid):
                    if neighbor in existing_ids:
                        for comp in nx.connected_components(undirected):
                            if neighbor in comp:
                                connected_existing.add(frozenset(comp))
                                break
            bridges_communities = len(connected_existing) >= 2
        except Exception:
            pass

        gain = (
            0.3 * integration
            + 0.3 * (1.0 if has_contradiction else 0.0)
            + 0.2 * (1.0 if addresses_question else 0.0)
            + 0.2 * (1.0 if bridges_communities else 0.0)
        )
        return round(gain, 4)

    def _find_candidate_nodes(self, content: str,
                              top_k: int = None) -> List[KGNode]:
        top_k = top_k or config.kg_candidate_top_k
        all_nodes = self._kg.get_all_nodes()
        if not all_nodes:
            return []

        try:
            import shared_embedder
            emb = shared_embedder.get()
            if emb is None:
                return self._fallback_candidates(content, all_nodes, top_k)

            import numpy as np
            query_vec = np.array(emb.embed(content), dtype=np.float32)
            scored = []
            for node in all_nodes:
                node_vec = np.array(emb.embed(node.content), dtype=np.float32)
                sim = float(np.dot(query_vec, node_vec))
                scored.append((sim, node))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [node for _, node in scored[:top_k]]
        except Exception as e:
            logger.warning("Embedding candidate search failed: %s", e)
            return self._fallback_candidates(content, all_nodes, top_k)

    def _fallback_candidates(self, content: str, all_nodes: List[KGNode],
                             top_k: int) -> List[KGNode]:
        """Simple keyword overlap when embeddings are unavailable."""
        content_chars = set(content.lower())
        scored = []
        for node in all_nodes:
            overlap = len(content_chars & set(node.content.lower()))
            scored.append((overlap, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def _llm_extract(self, content: str,
                     candidates: List[KGNode]) -> ExtractionResult:
        try:
            import llm_client
            if not llm_client.is_available():
                return ExtractionResult.empty("LLM unavailable")
        except ImportError:
            return ExtractionResult.empty("llm_client not installed")

        candidates_text = "（无已有节点）" if not candidates else "\n".join(
            f"- [{c.id}] ({c.node_type.value}) {c.content[:120]}"
            for c in candidates
        )

        prompt = _EXTRACTION_PROMPT.format(
            content=content[:2000],
            candidate_count=len(candidates),
            candidates_formatted=candidates_text,
        )

        raw = llm_client.generate(
            prompt=prompt,
            system="你是知识图谱构建助手。只输出 JSON，不要添加其他内容。",
            max_tokens=1500,
            temperature=0.3,
        )

        if not raw:
            return ExtractionResult.empty("LLM returned empty")

        return self._parse_llm_output(raw, candidates)

    def _parse_llm_output(self, raw: str,
                          candidates: List[KGNode]) -> ExtractionResult:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```"))

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM JSON output")
                    return ExtractionResult.empty("JSON parse failed")
            else:
                return ExtractionResult.empty("no JSON in LLM output")

        candidate_map = {c.id: c for c in candidates}
        new_nodes = []
        for nd in data.get("new_nodes", []):
            try:
                node = KGNode(
                    content=nd["content"],
                    node_type=nd.get("type", "fact"),
                    confidence=float(nd.get("confidence", 0.8)),
                    sentiment=nd.get("sentiment", ""),
                )
                new_nodes.append(node)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed node: %s", e)

        new_edges = []
        for ed in data.get("edges", []):
            try:
                edge = KGEdge(
                    source_id=str(ed.get("from_content", ed.get("from", ""))),
                    target_id=str(ed.get("to_content", ed.get("to", ""))),
                    edge_type=ed.get("type", "extends"),
                    weight=float(ed.get("weight", 0.5)),
                    evidence=ed.get("evidence", ""),
                )
                new_edges.append(edge)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed edge: %s", e)

        return ExtractionResult(new_nodes=new_nodes, new_edges=new_edges)

    def _resolve_node_id(self, ref: str, node_id_map: dict) -> str:
        """Resolve a node reference — either an existing node ID or a new node's content."""
        if ref in self._kg._nodes:
            return ref
        if ref in node_id_map:
            return node_id_map[ref]
        for content, nid in node_id_map.items():
            if ref.strip() == content.strip():
                return nid
        logger.warning("Could not resolve node reference: '%s'", ref[:60])
        return ref


extractor = RelationExtractor()
