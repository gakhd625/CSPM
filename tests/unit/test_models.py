"""
tests/unit/test_models.py

Unit tests for all Step 1 data models.
These tests serve as both validation AND documentation —
a new engineer can read these tests to understand how each model works.

Run with:
    pytest tests/unit/test_models.py -v
"""

import json
import sys
import os
from datetime import datetime, timezone
from typing import List

# Allow running from the cspm/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from core.models.finding import (
    Finding, Severity, FindingStatus, CloudProvider,
    ComplianceReference, RemediationStep,
)
from core.models.scan_result import (
    ScanResult, AccountResult, AccountScore, ComplianceSummary,
)
from core.attack_graph.models import (
    AttackGraph, GraphNode, GraphEdge, AttackPath,
    NodeType, EdgeType, AttackPathSeverity,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def make_finding(
    finding_id: str = "AWS-IAM-001",
    severity: Severity = Severity.HIGH,
    status: FindingStatus = FindingStatus.OPEN,
    resource_name: str = "test-user",
) -> Finding:
    return Finding(
        finding_id=finding_id,
        title="Test Finding",
        description="A test finding for unit tests.",
        severity=severity,
        status=status,
        provider=CloudProvider.AWS,
        account_id="123456789012",
        region="us-east-1",
        resource_type="AWS::IAM::User",
        resource_id=f"arn:aws:iam::123456789012:user/{resource_name}",
        resource_name=resource_name,
        remediation_summary="Fix it.",
    )


def make_graph_node(
    node_id: str = "arn:aws:iam::123456789012:user/test",
    node_type: NodeType = NodeType.USER,
    label: str = "test-user",
    is_admin: bool = False,
    is_public: bool = False,
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        account_id="123456789012",
        properties={"is_admin": is_admin, "is_public": is_public},
    )


# ===========================================================================
# Severity tests
# ===========================================================================

class TestSeverity:
    def test_score_impacts_are_correct(self):
        assert Severity.CRITICAL.score_impact == 40
        assert Severity.HIGH.score_impact     == 20
        assert Severity.MEDIUM.score_impact   == 10
        assert Severity.LOW.score_impact      ==  5
        assert Severity.INFO.score_impact     ==  0

    def test_sort_order_is_most_severe_first(self):
        severities = [Severity.LOW, Severity.CRITICAL, Severity.MEDIUM, Severity.HIGH]
        sorted_severities = sorted(severities, key=lambda s: s.sort_order)
        assert sorted_severities == [
            Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW
        ]

    def test_severity_is_string_comparable(self):
        # str inheritance means this works without .value
        assert Severity.CRITICAL == "CRITICAL"
        assert Severity.HIGH == "HIGH"

    def test_all_severities_have_display_colors(self):
        for s in Severity:
            assert s.display_color.startswith("#"), f"{s} missing color"


# ===========================================================================
# Finding tests
# ===========================================================================

class TestFinding:
    def test_basic_construction(self):
        f = make_finding()
        assert f.finding_id == "AWS-IAM-001"
        assert f.severity == Severity.HIGH
        assert f.status == FindingStatus.OPEN
        assert f.is_open is True
        assert f.is_critical is False

    def test_critical_finding(self):
        f = make_finding(severity=Severity.CRITICAL)
        assert f.is_critical is True

    def test_score_impact_zero_when_suppressed(self):
        f = make_finding(severity=Severity.CRITICAL, status=FindingStatus.SUPPRESSED)
        assert f.score_impact == 0

    def test_score_impact_reflects_severity_when_open(self):
        f = make_finding(severity=Severity.HIGH)
        assert f.score_impact == 20

    def test_instance_id_is_auto_generated(self):
        f1 = make_finding()
        f2 = make_finding()
        # Two findings of the same type on different resources get unique IDs
        assert f1.instance_id != f2.instance_id

    def test_timestamp_is_utc(self):
        f = make_finding()
        assert f.timestamp.tzinfo == timezone.utc

    def test_to_dict_is_json_serialisable(self):
        f = make_finding()
        d = f.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert "AWS-IAM-001" in json_str

    def test_to_dict_contains_all_required_keys(self):
        f = make_finding()
        d = f.to_dict()
        required_keys = [
            "finding_id", "title", "description", "severity", "status",
            "provider", "account_id", "region", "resource_type",
            "resource_id", "resource_name", "remediation_summary",
            "compliance", "timestamp", "raw_evidence",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_from_dict_roundtrip(self):
        original = make_finding()
        original.compliance.append(ComplianceReference(
            framework="CIS", control_id="1.4",
            control_title="Ensure no root access key",
            description="Root access key is dangerous."
        ))
        d = original.to_dict()
        restored = Finding.from_dict(d)
        assert restored.finding_id    == original.finding_id
        assert restored.severity      == original.severity
        assert restored.status        == original.status
        assert restored.account_id    == original.account_id

    def test_framework_names_deduplicates(self):
        f = make_finding()
        f.compliance = [
            ComplianceReference("CIS", "1.4", "Title A", "Desc"),
            ComplianceReference("CIS", "1.5", "Title B", "Desc"),
            ComplianceReference("PCI-DSS", "8.3", "Title C", "Desc"),
        ]
        names = f.framework_names
        assert sorted(names) == ["CIS", "PCI-DSS"]
        assert len(names) == 2  # No duplicates

    def test_repr_is_readable(self):
        f = make_finding(resource_name="deploy-user")
        r = repr(f)
        assert "AWS-IAM-001" in r
        assert "HIGH" in r
        assert "deploy-user" in r


# ===========================================================================
# ComplianceReference tests
# ===========================================================================

class TestComplianceReference:
    def test_to_dict(self):
        ref = ComplianceReference(
            framework="CIS",
            control_id="1.4",
            control_title="Root access key check",
            description="No root access keys.",
        )
        d = ref.to_dict()
        assert d["framework"] == "CIS"
        assert d["control_id"] == "1.4"


# ===========================================================================
# AccountScore tests
# ===========================================================================

class TestAccountScore:
    def test_perfect_score_no_findings(self):
        score = AccountScore.from_findings("123456789012", [])
        assert score.score == 100
        assert score.grade == "A"

    def test_score_deducts_for_open_findings(self):
        findings = [make_finding(severity=Severity.HIGH)]   # -20
        score = AccountScore.from_findings("123456789012", findings)
        assert score.score == 80
        assert score.grade == "B"

    def test_suppressed_findings_do_not_deduct(self):
        findings = [
            make_finding(severity=Severity.CRITICAL, status=FindingStatus.SUPPRESSED)
        ]
        score = AccountScore.from_findings("123456789012", findings)
        assert score.score == 100

    def test_score_clamps_to_zero(self):
        # 3 CRITICAL findings = -120 points → should clamp to 0
        findings = [make_finding(severity=Severity.CRITICAL) for _ in range(3)]
        score = AccountScore.from_findings("123456789012", findings)
        assert score.score == 0
        assert score.grade == "F"

    def test_grade_boundaries(self):
        assert AccountScore._grade_from_score(100) == "A"
        assert AccountScore._grade_from_score(90)  == "A"
        assert AccountScore._grade_from_score(89)  == "B"
        assert AccountScore._grade_from_score(75)  == "B"
        assert AccountScore._grade_from_score(74)  == "C"
        assert AccountScore._grade_from_score(60)  == "C"
        assert AccountScore._grade_from_score(59)  == "D"
        assert AccountScore._grade_from_score(40)  == "D"
        assert AccountScore._grade_from_score(39)  == "F"
        assert AccountScore._grade_from_score(0)   == "F"

    def test_finding_counts_by_severity(self):
        findings = [
            make_finding(severity=Severity.CRITICAL),
            make_finding(severity=Severity.CRITICAL),
            make_finding(severity=Severity.HIGH),
        ]
        score = AccountScore.from_findings("123456789012", findings)
        assert score.finding_counts["CRITICAL"] == 2
        assert score.finding_counts["HIGH"]     == 1
        assert score.finding_counts["MEDIUM"]   == 0


# ===========================================================================
# ScanResult tests
# ===========================================================================

class TestScanResult:
    def _make_account_result(
        self,
        account_id: str = "123456789012",
        findings: List[Finding] = None
    ) -> AccountResult:
        ar = AccountResult(account_id=account_id, account_name=f"Account-{account_id}")
        ar.findings = findings or []
        return ar

    def test_empty_scan_result(self):
        result = ScanResult()
        result.finalise()
        assert result.total_score == 100
        assert result.overall_grade == "A"
        assert result.account_count == 0

    def test_finalise_computes_per_account_scores(self):
        ar = self._make_account_result(findings=[make_finding(severity=Severity.HIGH)])
        result = ScanResult()
        result.account_results.append(ar)
        result.finalise()

        assert ar.score is not None
        assert ar.score.score == 80
        assert result.total_score == 80

    def test_all_findings_flattens_across_accounts(self):
        ar1 = self._make_account_result("111111111111", [make_finding("AWS-IAM-001")])
        ar2 = self._make_account_result("222222222222", [make_finding("AWS-S3-001")])
        result = ScanResult()
        result.account_results = [ar1, ar2]
        result.finalise()

        all_ids = {f.finding_id for f in result.all_findings}
        assert "AWS-IAM-001" in all_ids
        assert "AWS-S3-001" in all_ids
        assert len(result.all_findings) == 2

    def test_has_criticals_flag(self):
        ar = self._make_account_result(findings=[
            make_finding(severity=Severity.CRITICAL)
        ])
        result = ScanResult()
        result.account_results.append(ar)
        result.finalise()
        assert result.has_criticals is True

    def test_to_dict_is_json_serialisable(self):
        ar = self._make_account_result(findings=[make_finding()])
        result = ScanResult()
        result.account_results.append(ar)
        result.finalise()

        d = result.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert len(json_str) > 0

    def test_diff_detects_new_findings(self):
        # Previous scan: 1 finding
        prev = ScanResult(scan_id="prev")
        prev.account_results.append(
            self._make_account_result(findings=[make_finding("AWS-IAM-001")])
        )
        prev.finalise()

        # Current scan: same finding + 1 new one
        curr = ScanResult(scan_id="curr")
        curr.account_results.append(
            self._make_account_result(findings=[
                make_finding("AWS-IAM-001"),
                make_finding("AWS-S3-001", resource_name="my-bucket"),
            ])
        )
        curr.finalise()

        diff = curr.diff(prev)
        assert diff["new_finding_count"] == 1
        assert diff["resolved_count"] == 0
        assert diff["new_findings"][0]["finding_id"] == "AWS-S3-001"

    def test_diff_detects_resolved_findings(self):
        prev = ScanResult(scan_id="prev")
        prev.account_results.append(
            self._make_account_result(findings=[
                make_finding("AWS-IAM-001"),
                make_finding("AWS-S3-001", resource_name="my-bucket"),
            ])
        )
        prev.finalise()

        # Current scan: one finding resolved
        curr = ScanResult(scan_id="curr")
        curr.account_results.append(
            self._make_account_result(findings=[make_finding("AWS-IAM-001")])
        )
        curr.finalise()

        diff = curr.diff(prev)
        assert diff["resolved_count"] == 1
        assert diff["score_change"] > 0  # Score improved


# ===========================================================================
# AttackGraph tests
# ===========================================================================

class TestAttackGraph:
    def _make_public_bucket_node(self) -> GraphNode:
        return GraphNode(
            node_id="arn:aws:s3:::prod-config",
            node_type=NodeType.PUBLIC_RESOURCE,
            label="prod-config",
            account_id="123456789012",
            properties={"is_public": True},
        )

    def _make_admin_role_node(self) -> GraphNode:
        return GraphNode(
            node_id="arn:aws:iam::123456789012:role/AdminRole",
            node_type=NodeType.ROLE,
            label="AdminRole",
            account_id="123456789012",
            properties={"is_admin": True},
        )

    def test_add_node(self):
        graph = AttackGraph(account_id="123456789012")
        node = self._make_public_bucket_node()
        graph.add_node(node)
        assert node.node_id in graph.nodes

    def test_add_node_is_idempotent(self):
        graph = AttackGraph(account_id="123456789012")
        node = self._make_public_bucket_node()
        graph.add_node(node)
        graph.add_node(node)  # Second add should not duplicate
        assert len(graph.nodes) == 1

    def test_add_edge_validates_nodes_exist(self):
        graph = AttackGraph(account_id="123456789012")
        edge = GraphEdge(
            source_id="nonexistent-source",
            target_id="nonexistent-target",
            edge_type=EdgeType.CAN_ASSUME,
            label="test edge",
        )
        with pytest.raises(ValueError, match="source"):
            graph.add_edge(edge)

    def test_add_edge_deduplicates(self):
        graph = AttackGraph(account_id="123456789012")
        bucket = self._make_public_bucket_node()
        role = self._make_admin_role_node()
        graph.add_node(bucket)
        graph.add_node(role)

        edge = GraphEdge(
            source_id=bucket.node_id,
            target_id=role.node_id,
            edge_type=EdgeType.CAN_ACCESS,
            label="can access via public read",
        )
        graph.add_edge(edge)
        graph.add_edge(edge)  # Duplicate
        assert len(graph.edges) == 1

    def test_get_neighbors(self):
        graph = AttackGraph(account_id="123456789012")
        bucket = self._make_public_bucket_node()
        role = self._make_admin_role_node()
        graph.add_node(bucket)
        graph.add_node(role)
        graph.add_edge(GraphEdge(
            source_id=bucket.node_id, target_id=role.node_id,
            edge_type=EdgeType.CAN_ACCESS, label="test"
        ))

        neighbors = graph.get_neighbors(bucket.node_id)
        assert len(neighbors) == 1
        assert neighbors[0].node_id == role.node_id

    def test_get_public_entry_points(self):
        graph = AttackGraph(account_id="123456789012")
        public = self._make_public_bucket_node()
        private = make_graph_node("arn:aws:iam::123456789012:user/alice",
                                   NodeType.USER, "alice")
        graph.add_node(public)
        graph.add_node(private)

        entry_points = graph.get_public_entry_points()
        assert len(entry_points) == 1
        assert entry_points[0].node_id == public.node_id

    def test_get_admin_nodes(self):
        graph = AttackGraph(account_id="123456789012")
        admin = self._make_admin_role_node()
        non_admin = self._make_public_bucket_node()
        graph.add_node(admin)
        graph.add_node(non_admin)

        admin_nodes = graph.get_admin_nodes()
        assert len(admin_nodes) == 1
        assert admin_nodes[0].node_id == admin.node_id

    def test_stats_returns_correct_counts(self):
        graph = AttackGraph(account_id="123456789012")
        graph.add_node(self._make_public_bucket_node())
        graph.add_node(self._make_admin_role_node())
        graph.add_edge(GraphEdge(
            source_id="arn:aws:s3:::prod-config",
            target_id="arn:aws:iam::123456789012:role/AdminRole",
            edge_type=EdgeType.CAN_ACCESS,
            label="test",
        ))

        stats = graph.stats
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert stats["public_entries"] == 1
        assert stats["admin_targets"] == 1

    def test_to_dict_is_json_serialisable(self):
        graph = AttackGraph(account_id="123456789012")
        graph.add_node(self._make_public_bucket_node())
        graph.add_node(self._make_admin_role_node())

        d = graph.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert "123456789012" in json_str


# ===========================================================================
# AttackPath tests
# ===========================================================================

class TestAttackPath:
    def test_chain_summary(self):
        nodes = [
            make_graph_node("n1", NodeType.PUBLIC_RESOURCE, "Public S3"),
            make_graph_node("n2", NodeType.USER,            "deploy-user"),
            make_graph_node("n3", NodeType.ROLE,            "AdminRole", is_admin=True),
        ]
        path = AttackPath(
            path_id="path-001",
            nodes=nodes,
            edges=[],
            severity=AttackPathSeverity.CRITICAL,
            title="Public S3 → deploy-user → AdminRole",
            narrative="An attacker can...",
        )
        assert path.chain_summary == "Public S3 → deploy-user → AdminRole"
        assert path.entry_point.label == "Public S3"
        assert path.target.label == "AdminRole"
        assert path.hop_count == 0  # No edges in this test

    def test_to_dict_is_json_serialisable(self):
        path = AttackPath(
            path_id="path-001",
            nodes=[make_graph_node()],
            edges=[],
            severity=AttackPathSeverity.HIGH,
            title="Test Path",
            narrative="Test narrative.",
        )
        d = path.to_dict()
        json_str = json.dumps(d)
        assert "path-001" in json_str


# ===========================================================================
# GraphEdge tests
# ===========================================================================

class TestGraphEdge:
    def test_constrained_edge_detection(self):
        unconstrained = GraphEdge(
            source_id="a", target_id="b",
            edge_type=EdgeType.CAN_ASSUME,
            label="no conditions",
        )
        assert unconstrained.is_constrained is False

        constrained = GraphEdge(
            source_id="a", target_id="b",
            edge_type=EdgeType.CAN_ASSUME,
            label="mfa required",
            conditions={"MFARequired": True},
        )
        assert constrained.is_constrained is True
        assert constrained.requires_mfa is True


if __name__ == "__main__":
    # Allow running directly: python tests/unit/test_models.py
    pytest.main([__file__, "-v"])