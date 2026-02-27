# core/attack_graph/models.py
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict

class NodeType(str, Enum):
    USER            = "USER"            # IAM User
    ROLE            = "ROLE"            # IAM Role
    POLICY          = "POLICY"          # IAM Policy
    GROUP           = "GROUP"           # IAM Group
    PUBLIC_RESOURCE = "PUBLIC_RESOURCE" # S3 bucket, public API, etc.
    PERMISSION      = "PERMISSION"      # Specific action (e.g., sts:AssumeRole)
    ACCOUNT         = "ACCOUNT"         # AWS Account (for cross-account paths)

class EdgeType(str, Enum):
    CAN_ASSUME      = "CAN_ASSUME"      # Entity → Role
    HAS_POLICY      = "HAS_POLICY"      # Entity → Policy
    GRANTS          = "GRANTS"          # Policy → Permission
    IS_PUBLIC       = "IS_PUBLIC"       # Resource → Internet
    CAN_ACCESS      = "CAN_ACCESS"      # Role → Resource
    ESCALATES_TO    = "ESCALATES_TO"    # Computed: privilege escalation edge

@dataclass
class GraphNode:
    node_id: str                        # ARN or synthetic ID
    node_type: NodeType
    label: str                          # Human-readable (username, role name)
    account_id: str
    properties: Dict = field(default_factory=dict)
    # e.g., {"is_admin": True, "mfa_enabled": False, "is_public": True}

@dataclass
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    label: str                          # Human-readable edge description
    properties: Dict = field(default_factory=dict)
    # e.g., {"condition": "aws:MultiFactorAuthPresent", "actions": ["s3:*"]}

@dataclass
class AttackPath:
    path_id: str
    nodes: List[GraphNode]              # Ordered list of nodes in the path
    edges: List[GraphEdge]              # Edges connecting them
    severity: "Severity"
    title: str
    # e.g., "Public S3 Bucket → Privileged Role → AdministratorAccess"
    narrative: str
    # e.g., "An unauthenticated actor can read credentials from the public S3
    #         bucket 'prod-config', use them to assume RoleX, which has
    #         AdministratorAccess, achieving full account compromise."
    finding_ids: List[str]             # Findings that make up this path

@dataclass
class AttackGraph:
    account_id: str
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)

    def get_neighbors(self, node_id: str) -> List[GraphNode]:
        """Returns all nodes reachable from node_id via outbound edges."""
        neighbor_ids = [e.target_id for e in self.edges if e.source_id == node_id]
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]