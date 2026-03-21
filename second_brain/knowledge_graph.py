"""Knowledge Graph — typed nodes and edges over NetworkX DiGraph.

Nodes represent knowledge units (facts, decisions, preferences, goals, questions).
Edges represent logical relationships (supports, contradicts, extends, depends_on,
alternative_to, addresses).

This is fundamentally different from RAG:
  - RAG stores flat vectors and retrieves by cosine similarity.
  - The KG stores *typed logical relationships* and reasons by graph traversal.
  - It can find contradictions, blind spots, and causal chains that RAG cannot.

Persistence: memory/kg/nodes.jsonl + edges.jsonl
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()


class KGNodeType(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"
    GOAL = "goal"
    QUESTION = "question"
    INSIGHT = "insight"


class KGEdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    DEPENDS_ON = "depends_on"
    ALTERNATIVE_TO = "alternative_to"
    ADDRESSES = "addresses"


class KGNode:
    __slots__ = ("id", "content", "node_type", "importance", "created_at",
                 "updated_at", "access_count", "confidence", "maturity",
                 "source_hashes", "sentiment")

    def __init__(self, content: str, node_type: KGNodeType,
                 importance: float = 0.5, confidence: float = 0.8,
                 node_id: str = "", created_at: str = "",
                 updated_at: str = "", access_count: int = 0,
                 maturity: float = 0.0, source_hashes: list = None,
                 sentiment: str = ""):
        self.id = node_id or uuid.uuid4().hex[:12]
        self.content = content
        self.node_type = KGNodeType(node_type)
        self.importance = importance
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or self.created_at
        self.access_count = access_count
        self.confidence = confidence
        self.maturity = maturity
        self.source_hashes = source_hashes or []
        self.sentiment = sentiment

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type.value,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "confidence": self.confidence,
            "maturity": self.maturity,
            "source_hashes": self.source_hashes,
        }
        if self.sentiment:
            d["sentiment"] = self.sentiment
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KGNode":
        return cls(
            content=d["content"],
            node_type=d["node_type"],
            importance=d.get("importance", 0.5),
            confidence=d.get("confidence", 0.8),
            node_id=d["id"],
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            access_count=d.get("access_count", 0),
            maturity=d.get("maturity", 0.0),
            source_hashes=d.get("source_hashes", []),
            sentiment=d.get("sentiment", ""),
        )

    def update_maturity(self):
        """maturity = f(access_count, age_days, confidence)"""
        try:
            age_days = max(
                (datetime.now() - datetime.fromisoformat(self.created_at)).days, 1
            )
        except (ValueError, TypeError):
            age_days = 1
        age_factor = min(age_days / 30.0, 1.0)
        access_factor = min(self.access_count / 10.0, 1.0)
        self.maturity = round(
            0.4 * self.confidence + 0.3 * access_factor + 0.3 * age_factor, 4
        )

    def touch(self):
        self.access_count += 1
        self.updated_at = datetime.now().isoformat()
        self.update_maturity()


class KGEdge:
    __slots__ = ("source_id", "target_id", "edge_type", "weight",
                 "created_at", "evidence")

    def __init__(self, source_id: str, target_id: str, edge_type: KGEdgeType,
                 weight: float = 0.5, created_at: str = "", evidence: str = ""):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = KGEdgeType(edge_type)
        self.weight = weight
        self.created_at = created_at or datetime.now().isoformat()
        self.evidence = evidence

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "created_at": self.created_at,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KGEdge":
        return cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            edge_type=d["edge_type"],
            weight=d.get("weight", 0.5),
            created_at=d.get("created_at", ""),
            evidence=d.get("evidence", ""),
        )


class KnowledgeGraph:
    """NetworkX DiGraph wrapper with typed nodes/edges and JSONL persistence."""

    def __init__(self, kg_path: Path = None):
        self._path = kg_path or config.kg_path
        self._path.mkdir(parents=True, exist_ok=True)
        self._nodes_file = self._path / "nodes.jsonl"
        self._edges_file = self._path / "edges.jsonl"
        self._graph = nx.DiGraph()
        self._nodes: Dict[str, KGNode] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True

    # ── Node Operations ──────────────────────────────────────────

    def add_node(self, node: KGNode) -> str:
        self._ensure_loaded()
        self._nodes[node.id] = node
        self._graph.add_node(node.id)
        self._append_node(node)
        logger.info("KG: added node %s (%s) '%s'",
                     node.id, node.node_type.value, node.content[:60])
        return node.id

    def get_node(self, node_id: str) -> Optional[KGNode]:
        self._ensure_loaded()
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[KGNode]:
        self._ensure_loaded()
        return list(self._nodes.values())

    def get_nodes_by_type(self, node_type: KGNodeType) -> List[KGNode]:
        self._ensure_loaded()
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_decisions(self) -> List[KGNode]:
        return self.get_nodes_by_type(KGNodeType.DECISION)

    def touch_node(self, node_id: str):
        node = self.get_node(node_id)
        if node:
            node.touch()

    # ── Edge Operations ──────────────────────────────────────────

    def add_edge(self, edge: KGEdge) -> bool:
        self._ensure_loaded()
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            logger.warning("KG: edge references unknown node(s): %s -> %s",
                           edge.source_id, edge.target_id)
            return False
        self._graph.add_edge(
            edge.source_id, edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            evidence=edge.evidence,
            created_at=edge.created_at,
        )
        self._append_edge(edge)
        logger.info("KG: added edge %s -[%s]-> %s",
                     edge.source_id, edge.edge_type.value, edge.target_id)
        return True

    def get_edges(self, node_id: str, edge_type: Optional[KGEdgeType] = None,
                  direction: str = "both") -> List[Tuple[str, str, dict]]:
        """Get edges connected to a node.

        direction: 'in', 'out', or 'both'
        Returns list of (source, target, edge_data) tuples.
        """
        self._ensure_loaded()
        edges = []
        if direction in ("in", "both"):
            for src, tgt, data in self._graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    edges.append((src, tgt, data))
        if direction in ("out", "both"):
            for src, tgt, data in self._graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    edges.append((src, tgt, data))
        return edges

    def get_neighbors(self, node_id: str,
                      edge_type: Optional[KGEdgeType] = None,
                      direction: str = "both") -> List[KGNode]:
        """Get neighboring nodes, optionally filtered by edge type."""
        edges = self.get_edges(node_id, edge_type, direction)
        neighbor_ids = set()
        for src, tgt, _ in edges:
            neighbor_ids.add(src if tgt == node_id else tgt)
        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    # ── Graph Queries ────────────────────────────────────────────

    def find_paths(self, source_id: str, target_id: str,
                   max_depth: int = 4) -> List[List[str]]:
        self._ensure_loaded()
        try:
            paths = list(nx.all_simple_paths(
                self._graph, source_id, target_id, cutoff=max_depth
            ))
            return paths
        except (nx.NetworkXError, nx.NodeNotFound):
            return []

    def get_contradictions(self) -> List[Tuple[KGNode, KGNode, dict]]:
        """Find all contradiction edges in the graph."""
        self._ensure_loaded()
        results = []
        for src, tgt, data in self._graph.edges(data=True):
            if data.get("edge_type") == KGEdgeType.CONTRADICTS.value:
                src_node = self._nodes.get(src)
                tgt_node = self._nodes.get(tgt)
                if src_node and tgt_node:
                    results.append((src_node, tgt_node, data))
        return results

    def get_descendants(self, node_id: str) -> List[str]:
        self._ensure_loaded()
        try:
            return list(nx.descendants(self._graph, node_id))
        except nx.NetworkXError:
            return []

    def get_communities(self) -> List[List[KGNode]]:
        """Detect communities using greedy modularity (works on DiGraph)."""
        self._ensure_loaded()
        if len(self._graph) < 2:
            return [list(self._nodes.values())] if self._nodes else []

        undirected = self._graph.to_undirected()
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(undirected)
        except Exception:
            components = list(nx.connected_components(undirected))
            communities = components

        result = []
        for community in communities:
            nodes = [self._nodes[nid] for nid in community if nid in self._nodes]
            if nodes:
                result.append(nodes)
        return result

    def get_mature_patterns(self, min_maturity: float = 0.7) -> List[KGNode]:
        self._ensure_loaded()
        for node in self._nodes.values():
            node.update_maturity()
        return [n for n in self._nodes.values() if n.maturity >= min_maturity]

    # ── Statistics ────────────────────────────────────────────────

    def stats(self) -> dict:
        self._ensure_loaded()
        edge_type_counts = {}
        for _, _, data in self._graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        node_type_counts = {}
        for node in self._nodes.values():
            nt = node.node_type.value
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": self._graph.number_of_edges(),
            "node_types": node_type_counts,
            "edge_types": edge_type_counts,
            "connected_components": nx.number_weakly_connected_components(self._graph),
        }

    # ── Persistence ──────────────────────────────────────────────

    def _append_node(self, node: KGNode):
        with open(self._nodes_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

    def _append_edge(self, edge: KGEdge):
        with open(self._edges_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(edge.to_dict(), ensure_ascii=False) + "\n")

    def load(self):
        """Load graph from JSONL files."""
        self._nodes.clear()
        self._graph.clear()

        if self._nodes_file.exists():
            with open(self._nodes_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        node = KGNode.from_dict(json.loads(line))
                        self._nodes[node.id] = node
                        self._graph.add_node(node.id)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping malformed node: %s", e)

        if self._edges_file.exists():
            with open(self._edges_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d["source_id"] in self._nodes and d["target_id"] in self._nodes:
                            self._graph.add_edge(
                                d["source_id"], d["target_id"],
                                edge_type=d["edge_type"],
                                weight=d.get("weight", 0.5),
                                evidence=d.get("evidence", ""),
                                created_at=d.get("created_at", ""),
                            )
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping malformed edge: %s", e)

        logger.info("KG loaded: %d nodes, %d edges",
                     len(self._nodes), self._graph.number_of_edges())
        self._loaded = True

    def save(self):
        """Full rewrite of both JSONL files (for compaction after edits)."""
        self._ensure_loaded()
        with open(self._nodes_file, "w", encoding="utf-8") as f:
            for node in self._nodes.values():
                f.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

        with open(self._edges_file, "w", encoding="utf-8") as f:
            for src, tgt, data in self._graph.edges(data=True):
                edge_dict = {
                    "source_id": src,
                    "target_id": tgt,
                    "edge_type": data.get("edge_type", ""),
                    "weight": data.get("weight", 0.5),
                    "created_at": data.get("created_at", ""),
                    "evidence": data.get("evidence", ""),
                }
                f.write(json.dumps(edge_dict, ensure_ascii=False) + "\n")

        logger.info("KG saved: %d nodes, %d edges",
                     len(self._nodes), self._graph.number_of_edges())

    def find_node_by_content(self, content_fragment: str,
                             max_results: int = 5) -> List[KGNode]:
        """Simple substring search for nodes (used when embedding is unavailable)."""
        self._ensure_loaded()
        results = []
        fragment_lower = content_fragment.lower()
        for node in self._nodes.values():
            if fragment_lower in node.content.lower():
                results.append(node)
                if len(results) >= max_results:
                    break
        return results


kg = KnowledgeGraph()
