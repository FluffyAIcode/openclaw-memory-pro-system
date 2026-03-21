"""Internalization Manager — the bridge between KG (explicit) and Chronos (implicit).

This module identifies high-maturity patterns in the knowledge graph and prepares
them for internalization:

1. Find "mature" nodes (high access count, high confidence, stable over time)
2. Extract structured patterns (preference + evidence, decision + rationale)
3. Generate training data for future LoRA fine-tuning
4. Provide patterns to Chronos consolidator for PERSONALITY.yaml generation

The KG is the "hippocampus" (fast explicit memory); internalization moves stable
patterns toward the "neocortex" (slow implicit memory encoded in model behavior).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config
from .knowledge_graph import KGEdgeType, KGNode, KGNodeType, KnowledgeGraph, kg

logger = logging.getLogger(__name__)
config = load_config()


@dataclass
class Pattern:
    """A structured pattern extracted from the KG for internalization."""
    core_node: KGNode
    supporting_nodes: List[KGNode] = field(default_factory=list)
    pattern_type: str = ""  # "preference", "decision_rationale", "knowledge_anchor"
    summary: str = ""
    maturity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "core_id": self.core_node.id,
            "core_content": self.core_node.content[:200],
            "core_type": self.core_node.node_type.value,
            "pattern_type": self.pattern_type,
            "summary": self.summary,
            "maturity": self.maturity,
            "supporting_count": len(self.supporting_nodes),
            "supporting": [
                {"id": n.id, "content": n.content[:100]}
                for n in self.supporting_nodes[:5]
            ],
        }


@dataclass
class TrainingPair:
    """An instruction/response pair for future LoRA fine-tuning."""
    instruction: str
    response: str
    source_pattern_id: str

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "source_pattern_id": self.source_pattern_id,
        }


class InternalizationManager:

    def __init__(self, knowledge_graph: KnowledgeGraph = None):
        self._kg = knowledge_graph or kg
        self._workspace = Path(__file__).parent.parent
        self._training_dir = self._workspace / "memory" / "training_data"

    def find_candidates(self,
                        min_maturity: float = None) -> List[KGNode]:
        """Find KG nodes mature enough for internalization."""
        threshold = min_maturity or config.kg_maturity_threshold
        candidates = self._kg.get_mature_patterns(threshold)
        candidates.sort(key=lambda n: n.maturity, reverse=True)
        return candidates

    def extract_patterns(self, candidates: List[KGNode] = None) -> List[Pattern]:
        """Extract structured patterns from mature nodes and their neighborhoods."""
        if candidates is None:
            candidates = self.find_candidates()

        patterns = []
        for node in candidates:
            pattern = self._build_pattern(node)
            if pattern:
                patterns.append(pattern)

        logger.info("Extracted %d patterns from %d candidates",
                     len(patterns), len(candidates))
        return patterns

    def _build_pattern(self, node: KGNode) -> Optional[Pattern]:
        neighbors = self._kg.get_neighbors(node.id, direction="in")

        if node.node_type == KGNodeType.PREFERENCE:
            supporting = [n for n in neighbors
                          if self._has_edge_type(n.id, node.id, KGEdgeType.SUPPORTS)]
            summary = f"偏好: {node.content}"
            if supporting:
                summary += f" (基于 {len(supporting)} 条支撑证据)"
            return Pattern(
                core_node=node,
                supporting_nodes=supporting,
                pattern_type="preference",
                summary=summary,
                maturity=node.maturity,
            )

        elif node.node_type == KGNodeType.DECISION:
            supporting = [n for n in neighbors
                          if self._has_edge_type(n.id, node.id, KGEdgeType.SUPPORTS)]
            contradicting = [n for n in neighbors
                             if self._has_edge_type(n.id, node.id, KGEdgeType.CONTRADICTS)]
            summary = f"决策: {node.content}"
            if supporting:
                summary += f" | 支撑: {len(supporting)}"
            if contradicting:
                summary += f" | 矛盾: {len(contradicting)}"
            return Pattern(
                core_node=node,
                supporting_nodes=supporting,
                pattern_type="decision_rationale",
                summary=summary,
                maturity=node.maturity,
            )

        elif node.node_type == KGNodeType.FACT:
            dependents = self._kg.get_neighbors(node.id, direction="out")
            summary = f"知识: {node.content}"
            if dependents:
                summary += f" (被 {len(dependents)} 个节点依赖)"
            return Pattern(
                core_node=node,
                supporting_nodes=dependents,
                pattern_type="knowledge_anchor",
                summary=summary,
                maturity=node.maturity,
            )

        return None

    def _has_edge_type(self, src_id: str, tgt_id: str,
                       edge_type: KGEdgeType) -> bool:
        edges = self._kg.get_edges(tgt_id, edge_type=edge_type, direction="in")
        return any(s == src_id for s, _, _ in edges)

    def generate_training_data(self, patterns: List[Pattern] = None) -> List[TrainingPair]:
        """Generate instruction/response pairs for future LoRA fine-tuning."""
        if patterns is None:
            patterns = self.extract_patterns()

        pairs = []
        for pattern in patterns:
            pair = self._pattern_to_training_pair(pattern)
            if pair:
                pairs.append(pair)

        if pairs:
            self._save_training_data(pairs)

        logger.info("Generated %d training pairs", len(pairs))
        return pairs

    def _pattern_to_training_pair(self, pattern: Pattern) -> Optional[TrainingPair]:
        core = pattern.core_node

        if pattern.pattern_type == "preference":
            instruction = f"用户的偏好是什么？关于: {core.content[:50]}"
            evidence = " ".join(n.content[:80] for n in pattern.supporting_nodes[:3])
            response = f"用户偏好 {core.content}。{evidence}" if evidence else core.content
            return TrainingPair(instruction, response, core.id)

        elif pattern.pattern_type == "decision_rationale":
            instruction = f"用户做了什么决策？关于: {core.content[:50]}"
            evidence = " ".join(n.content[:80] for n in pattern.supporting_nodes[:3])
            response = (
                f"用户决定了 {core.content}。"
                f"{'理由: ' + evidence if evidence else ''}"
            )
            return TrainingPair(instruction, response, core.id)

        elif pattern.pattern_type == "knowledge_anchor":
            instruction = f"关于 {core.content[:50]} 你知道什么？"
            response = core.content
            return TrainingPair(instruction, response, core.id)

        return None

    def _save_training_data(self, pairs: List[TrainingPair]):
        self._training_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = self._training_dir / f"pairs_{today}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Training data saved to %s (%d pairs)", filepath, len(pairs))

    def get_patterns_for_personality(self) -> List[dict]:
        """Return formatted patterns for Chronos personality profile generation.

        This is the key integration point: KG patterns feed into PERSONALITY.yaml.
        """
        patterns = self.extract_patterns()
        formatted = []
        for p in patterns:
            formatted.append({
                "type": p.pattern_type,
                "content": p.core_node.content,
                "confidence": p.core_node.confidence,
                "maturity": p.maturity,
                "evidence_count": len(p.supporting_nodes),
            })
        return formatted

    def status(self) -> dict:
        candidates = self.find_candidates()
        patterns = self.extract_patterns(candidates)
        return {
            "mature_candidates": len(candidates),
            "extracted_patterns": len(patterns),
            "pattern_types": {
                "preference": sum(1 for p in patterns if p.pattern_type == "preference"),
                "decision_rationale": sum(1 for p in patterns if p.pattern_type == "decision_rationale"),
                "knowledge_anchor": sum(1 for p in patterns if p.pattern_type == "knowledge_anchor"),
            },
            "kg_stats": self._kg.stats(),
        }


internalization_manager = InternalizationManager()
