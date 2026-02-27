"""
core/attack_graph/models.py

Data models for the attack graph engine.

The attack graph represents the security posture of an AWS account as a
directed graph. Nodes are IAM entities and cloud resources. Edges are
relationships ("can assume", "has policy", "is public", etc.).

The analyzer traverses this graph to find paths from low-trust starting
points (public resources, external principals) to high-privilege targets
(admin roles, sensitive resources).

Design decisions:
  - Separate GraphNode / GraphEdge / AttackGraph from the analyzer logic
    so the data structures can be serialised, stored, and diffed over time
  - properties dict on nodes/edges is intentionally flexible — different
    providers will populate different keys
  - AttackGraph provides adjacency helpers so the analyzer never does
    raw list comprehensions in its traversal logic
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """
    Every entity in the graph is one of these types.
    This drives how the analyzer interprets the node during traversal.
    """
    USER             = "USER"             # IAM User
    ROLE             = "ROLE"             # IAM Role
    GROUP            = "GROUP"            # IAM Group
    POLICY           = "POLICY"           # IAM Policy (managed or inline)
    PUBLIC_RESOURCE  = "PUBLIC_RESOURCE"  # Publicly accessible resource (S3, etc.)
    PRIVATE_RESOURCE = "PRIVATE_RESOURCE" # Internal resource
    PERMISSION       = "PERMISSION"       # A specific IAM action (sts:AssumeRole)
    ACCOUNT          = "ACCOUNT"          # AWS Account (for cross-account paths)
    EXTERNAL         = "EXTERNAL"         # Internet / anonymous actor


class EdgeType(str, Enum):
    """
    Every relationship between nodes is one of these types.
    Edge types determine traversal rules:
      - CAN_ASSUME is only valid if trust policy conditions are met
      - IS_PUBLIC means any EXTERNAL node can reach it
      - ESCALATES_TO is a synthetic edge added by the analyzer
    """
    CAN_ASSUME   = "CAN_ASSUME"    # Entity → Role  (trust policy)
    HAS_POLICY   = "HAS_POLICY"    # Entity → Policy (attached policy)
    GRANTS       = "GRANTS"        # Policy → Permission (allowed action)
    IS_PUBLIC    = "IS_PUBLIC"     # Resource → EXTERNAL (publicly accessible)
    CAN_ACCESS   = "CAN_ACCESS"    # Role/User → Resource (resource policy)
    MEMBER_OF    = "MEMBER_OF"     # User → Group
    ESCALATES_TO = "ESCALATES_TO"  # Synthetic: computed privilege escalation edge
    CROSS_ACCOUNT = "CROSS_ACCOUNT" # Principal in account A → role in account B


class AttackPathSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


# ---------------------------------------------------------------------------
# Node & Edge models
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """
    A vertex in the attack graph.

    node_id is always the ARN where available, otherwise a deterministic
    synthetic ID (e.g. "external:internet") so graph merges are idempotent.

    Key properties dict keys (AWS):
        is_admin:        bool  — has AdministratorAccess or *:*
        is_public:       bool  — accessible without authentication
        mfa_enabled:     bool  — for USER nodes
        has_active_keys: bool  — for USER nodes
        trust_wildcard:  bool  — for ROLE nodes with wildcard principal
        actions:         list  — for POLICY/PERMISSION nodes
    """
    node_id:    str        # ARN or synthetic ID
    node_type:  NodeType
    label:      str        # Human-readable (username, role name, bucket name)
    account_id: str
    region:     str         = field(default="global")
    properties: Dict[str, Any] = field(default_factory=dict)

    # Populated by analyzer during traversal
    finding_ids: List[str] = field(default_factory=list)

    @property
    def is_admin(self) -> bool:
        return bool(self.properties.get("is_admin", False))

    @property
    def is_public(self) -> bool:
        return bool(self.properties.get("is_public", False))

    @property
    def is_high_value_target(self) -> bool:
        """
        A node is a high-value target if compromising it represents
        significant blast radius: admin access, cross-account roles,
        or critical data stores.
        """
        return self.is_admin or bool(self.properties.get("is_high_value", False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id":     self.node_id,
            "node_type":   self.node_type.value,
            "label":       self.label,
            "account_id":  self.account_id,
            "region":      self.region,
            "properties":  self.properties,
            "finding_ids": self.finding_ids,
        }


@dataclass
class GraphEdge:
    """
    A directed edge between two nodes.

    source_id → target_id

    The conditions dict captures IAM condition keys that constrain the
    relationship (e.g. MFA required, IP restriction). The analyzer uses
    this to determine whether a path is actually exploitable.
    """
    edge_id:   str        = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str        = field(default="")
    target_id: str        = field(default="")
    edge_type: EdgeType   = field(default=EdgeType.CAN_ASSUME)
    label:     str        = field(default="")  # Human-readable (e.g. "can assume via trust policy")
    conditions: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"MFARequired": True, "IpRestricted": False, "ExternalId": "abc123"}
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_constrained(self) -> bool:
        """
        True if the edge has conditions that reduce exploitability.
        An attacker can still follow constrained edges but they're harder.
        """
        return bool(self.conditions)

    @property
    def requires_mfa(self) -> bool:
        return bool(self.conditions.get("MFARequired", False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id":    self.edge_id,
            "source_id":  self.source_id,
            "target_id":  self.target_id,
            "edge_type":  self.edge_type.value,
            "label":      self.label,
            "conditions": self.conditions,
            "properties": self.properties,
        }


# ---------------------------------------------------------------------------
# Attack Path — a traversal result
# ---------------------------------------------------------------------------

@dataclass
class AttackPath:
    """
    A complete attack path from an entry point to a high-value target.

    Produced by the AttackGraphAnalyzer after DFS/BFS traversal.
    The nodes list is ordered: nodes[0] is the attacker's entry point,
    nodes[-1] is the target.

    Example narrative:
        "An unauthenticated actor can access the public S3 bucket
        'prod-config', which contains IAM credentials for 'deploy-user'.
        That user can assume 'AdminRole' which has AdministratorAccess,
        achieving full account compromise."
    """
    path_id:   str
    nodes:     List[GraphNode]   # Ordered: entry → ... → target
    edges:     List[GraphEdge]   # edges[i] connects nodes[i] → nodes[i+1]
    severity:  AttackPathSeverity
    title:     str               # Short: "Public S3 → AdminRole → Account Compromise"
    narrative: str               # Long-form explanation for the report
    finding_ids: List[str]       = field(default_factory=list)  # Findings that enable this path

    @property
    def entry_point(self) -> Optional[GraphNode]:
        return self.nodes[0] if self.nodes else None

    @property
    def target(self) -> Optional[GraphNode]:
        return self.nodes[-1] if self.nodes else None

    @property
    def hop_count(self) -> int:
        return len(self.edges)

    @property
    def chain_summary(self) -> str:
        """One-line chain: 'NodeA → NodeB → NodeC'"""
        return " → ".join(n.label for n in self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id":     self.path_id,
            "title":       self.title,
            "severity":    self.severity.value,
            "narrative":   self.narrative,
            "chain":       self.chain_summary,
            "hop_count":   self.hop_count,
            "finding_ids": self.finding_ids,
            "nodes":       [n.to_dict() for n in self.nodes],
            "edges":       [e.to_dict() for e in self.edges],
        }


# ---------------------------------------------------------------------------
# Attack Graph — the full graph for one account
# ---------------------------------------------------------------------------

@dataclass
class AttackGraph:
    """
    Directed graph representing the security topology of one cloud account.

    Provides adjacency helpers so the analyzer never touches raw data
    structures. Adding/removing nodes and edges always goes through these
    methods so invariants are maintained.

    The graph is intentionally NOT an adjacency matrix or adjacency list
    at construction time — we build those lazily via the helper methods
    so the raw nodes/edges stay serialisable.
    """
    account_id: str
    provider:   str = "aws"

    # Primary storage: node_id → GraphNode
    nodes: Dict[str, GraphNode]  = field(default_factory=dict)

    # Edges stored as a flat list. For large accounts (10k+ resources)
    # we'd switch to adjacency list — fine for CSPM scale.
    edges: List[GraphEdge] = field(default_factory=list)

    # Populated by the analyzer after traversal
    attack_paths: List[AttackPath] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> None:
        """Idempotent — adding the same node_id twice updates it."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """
        Add an edge. Validates that source and target nodes exist.
        Silently skips duplicate edges (same source/target/type).
        """
        if edge.source_id not in self.nodes:
            raise ValueError(
                f"Edge source '{edge.source_id}' not found in graph. "
                f"Add the node before adding edges."
            )
        if edge.target_id not in self.nodes:
            raise ValueError(
                f"Edge target '{edge.target_id}' not found in graph. "
                f"Add the node before adding edges."
            )
        # Deduplicate on (source, target, type)
        for existing in self.edges:
            if (existing.source_id == edge.source_id
                    and existing.target_id == edge.target_id
                    and existing.edge_type == edge.edge_type):
                return  # Already present
        self.edges.append(edge)

    # ------------------------------------------------------------------
    # Adjacency helpers for the analyzer
    # ------------------------------------------------------------------

    def get_outbound_edges(self, node_id: str) -> List[GraphEdge]:
        """All edges where this node is the source."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_inbound_edges(self, node_id: str) -> List[GraphEdge]:
        """All edges where this node is the target."""
        return [e for e in self.edges if e.target_id == node_id]

    def get_neighbors(self, node_id: str) -> List[GraphNode]:
        """All nodes reachable from node_id via one outbound edge."""
        neighbor_ids = {e.target_id for e in self.get_outbound_edges(node_id)}
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_admin_nodes(self) -> List[GraphNode]:
        """All nodes that represent administrative/full-access privileges."""
        return [n for n in self.nodes.values() if n.is_admin]

    def get_public_entry_points(self) -> List[GraphNode]:
        """
        Nodes that an unauthenticated external actor can reach directly.
        These are the starting points for attack path analysis.
        """
        return [
            n for n in self.nodes.values()
            if n.node_type == NodeType.PUBLIC_RESOURCE or n.is_public
        ]

    def get_edges_between(self, source_id: str, target_id: str) -> List[GraphEdge]:
        return [
            e for e in self.edges
            if e.source_id == source_id and e.target_id == target_id
        ]

    def node_exists(self, node_id: str) -> bool:
        return node_id in self.nodes

    def all_node_ids(self) -> Set[str]:
        return set(self.nodes.keys())

    # ------------------------------------------------------------------
    # Stats (useful for the report summary)
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        type_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            key = node.node_type.value
            type_counts[key] = type_counts.get(key, 0) + 1
        return {
            "total_nodes":    len(self.nodes),
            "total_edges":    len(self.edges),
            "attack_paths":   len(self.attack_paths),
            "public_entries": len(self.get_public_entry_points()),
            "admin_targets":  len(self.get_admin_nodes()),
            **{f"node_{k.lower()}": v for k, v in type_counts.items()},
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id":    self.account_id,
            "provider":      self.provider,
            "stats":         self.stats,
            "nodes":         [n.to_dict() for n in self.nodes.values()],
            "edges":         [e.to_dict() for e in self.edges],
            "attack_paths":  [p.to_dict() for p in self.attack_paths],
        }