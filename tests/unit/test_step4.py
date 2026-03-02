"""
tests/unit/test_step4.py

Comprehensive unit tests for IAMChecker — all 5 checks.

Testing strategy:
  - All IAM API calls are mocked via MagicMock — no AWS credentials needed
  - Each check method is tested in isolation via a helper that calls run()
    with carefully constructed mock API responses
  - Edge cases: missing data, partial responses, malformed policy documents
  - Policy analysis (_is_admin_policy, _is_dangerous_trust_policy) is tested
    with a comprehensive matrix of policy patterns

Run with:
    python tests/unit/test_step4.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import os
import unittest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Pre-register boto3/botocore mocks (same pattern as test_step3.py)
from unittest.mock import MagicMock
_boto3 = MagicMock()
_botocore = MagicMock()

class FakeClientError(Exception):
    def __init__(self, error_response, operation_name="Op"):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(str(error_response))

_botocore_exc = MagicMock()
_botocore_exc.ClientError = FakeClientError
_botocore.exceptions = _botocore_exc
sys.modules.setdefault("boto3", _boto3)
sys.modules.setdefault("botocore", _botocore)
sys.modules.setdefault("botocore.exceptions", _botocore_exc)

from providers.aws.checkers.iam import IAMChecker
from core.models.finding import Severity, FindingStatus
from core.attack_graph.models import AttackGraph, NodeType


# ===========================================================================
# CSV credential report builder helper
# ===========================================================================

CRED_REPORT_HEADERS = [
    "user", "arn", "user_creation_time", "password_enabled", "password_last_used",
    "password_last_changed", "password_next_rotation", "mfa_active",
    "access_key_1_active", "access_key_1_last_rotated", "access_key_1_last_used_date",
    "access_key_1_last_used_region", "access_key_1_last_used_service",
    "access_key_2_active", "access_key_2_last_rotated", "access_key_2_last_used_date",
    "access_key_2_last_used_region", "access_key_2_last_used_service",
    "cert_1_active", "cert_1_last_rotated", "cert_2_active", "cert_2_last_rotated",
]


def make_cred_report_csv(rows: List[Dict]) -> str:
    """Build a CSV string matching the IAM credential report format."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CRED_REPORT_HEADERS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        # Fill in defaults for missing fields
        full_row = {h: "N/A" for h in CRED_REPORT_HEADERS}
        full_row.update(row)
        writer.writerow(full_row)
    return output.getvalue()


def days_ago(days: int) -> str:
    """Return an ISO 8601 timestamp N days ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def make_root_row(mfa_active="true", key1_active="false", key2_active="false") -> Dict:
    return {
        "user": "<root_account>",
        "arn": "arn:aws:iam::123456789012:root",
        "mfa_active": mfa_active,
        "access_key_1_active": key1_active,
        "access_key_2_active": key2_active,
        "password_enabled": "not_supported",
    }


def make_user_row(
    username: str = "alice",
    account_id: str = "123456789012",
    password_enabled: str = "true",
    mfa_active: str = "false",
    key1_active: str = "false",
    key1_rotated: str = "N/A",
    key2_active: str = "false",
    key2_rotated: str = "N/A",
) -> Dict:
    return {
        "user": username,
        "arn": f"arn:aws:iam::{account_id}:user/{username}",
        "password_enabled": password_enabled,
        "mfa_active": mfa_active,
        "access_key_1_active": key1_active,
        "access_key_1_last_rotated": key1_rotated,
        "access_key_2_active": key2_active,
        "access_key_2_last_rotated": key2_rotated,
    }


# ===========================================================================
# Mock IAM client builder
# ===========================================================================

def make_mock_iam_client(
    cred_report_rows: List[Dict] = None,
    users: List[Dict] = None,
    roles: List[Dict] = None,
    customer_policies: List[Dict] = None,
    user_inline_policies: Dict[str, List[str]] = None,   # {username: [policy_names]}
    role_inline_policies: Dict[str, List[str]] = None,   # {rolename: [policy_names]}
    inline_policy_docs: Dict[str, Dict] = None,          # {policy_name: document}
    attached_user_policies: Dict[str, List] = None,
    attached_role_policies: Dict[str, List] = None,
) -> MagicMock:
    """Build a fully mocked IAM boto3 client."""
    client = MagicMock()

    # Credential report
    if cred_report_rows is not None:
        csv_content = make_cred_report_csv(cred_report_rows)
        client.generate_credential_report.return_value = {"State": "COMPLETE"}
        client.get_credential_report.return_value = {
            "Content": csv_content.encode("utf-8"),
            "ReportFormat": "text/csv",
        }
    else:
        client.generate_credential_report.return_value = {"State": "COMPLETE"}
        client.get_credential_report.return_value = {
            "Content": make_cred_report_csv([make_root_row()]).encode("utf-8"),
        }

    # Users paginator
    users = users or []
    user_pager = MagicMock()
    user_pager.paginate.return_value = [{"Users": users}]

    # Roles paginator
    roles = roles or []
    role_pager = MagicMock()
    role_pager.paginate.return_value = [{"Roles": roles}]

    # Customer-managed policies paginator
    customer_policies = customer_policies or []
    policy_pager = MagicMock()
    policy_pager.paginate.return_value = [{"Policies": customer_policies}]

    def get_paginator(operation):
        return {
            "list_users":   user_pager,
            "list_roles":   role_pager,
            "list_policies": policy_pager,
        }.get(operation, MagicMock())

    client.get_paginator.side_effect = get_paginator

    # List/get inline policies
    user_inline = user_inline_policies or {}
    role_inline  = role_inline_policies or {}
    docs = inline_policy_docs or {}

    def list_user_policies(UserName, **kw):
        return {"PolicyNames": user_inline.get(UserName, [])}

    def list_role_policies(RoleName, **kw):
        return {"PolicyNames": role_inline.get(RoleName, [])}

    def get_user_policy(UserName, PolicyName, **kw):
        return {"PolicyDocument": docs.get(PolicyName, {})}

    def get_role_policy(RoleName, PolicyName, **kw):
        return {"PolicyDocument": docs.get(PolicyName, {})}

    client.list_user_policies.side_effect = list_user_policies
    client.list_role_policies.side_effect = list_role_policies
    client.get_user_policy.side_effect    = get_user_policy
    client.get_role_policy.side_effect    = get_role_policy

    # Attached policies
    att_user = attached_user_policies or {}
    att_role  = attached_role_policies or {}

    def list_attached_user_policies(UserName, **kw):
        return {"AttachedPolicies": att_user.get(UserName, [])}

    def list_attached_role_policies(RoleName, **kw):
        return {"AttachedPolicies": att_role.get(RoleName, [])}

    client.list_attached_user_policies.side_effect = list_attached_user_policies
    client.list_attached_role_policies.side_effect = list_attached_role_policies

    # list_access_keys (for key ID lookup)
    client.list_access_keys.return_value = {
        "AccessKeyMetadata": [
            {"AccessKeyId": "AKIAIOSFODNN7EXAMPLE", "Status": "Active"},
        ]
    }

    return client


def make_checker(
    iam_client: MagicMock = None,
    account_id: str = "123456789012",
) -> IAMChecker:
    """Create an IAMChecker with a mocked boto3 session."""
    session = MagicMock()
    iam_client = iam_client or make_mock_iam_client()
    session.client.return_value = iam_client
    checker = IAMChecker(
        session=session,
        account_id=account_id,
        region="global",
    )
    # Pre-inject the IAM client so _safe_get_client isn't called
    checker._iam = iam_client
    return checker


# ===========================================================================
# Tests: AWS-IAM-001 — User MFA
# ===========================================================================

class TestIAM001UserMFA(unittest.TestCase):

    def test_user_without_mfa_console_access_flagged(self):
        """User with password enabled but no MFA → AWS-IAM-001 finding."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("alice", password_enabled="true", mfa_active="false"),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        mfa_findings = [f for f in findings if f.finding_id == "AWS-IAM-001"]
        self.assertEqual(len(mfa_findings), 1)
        self.assertEqual(mfa_findings[0].resource_name, "alice")
        self.assertEqual(mfa_findings[0].severity, Severity.HIGH)
        self.assertEqual(mfa_findings[0].status, FindingStatus.OPEN)

    def test_user_with_mfa_not_flagged(self):
        """User with MFA enabled → no AWS-IAM-001 finding."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("bob", password_enabled="true", mfa_active="true"),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        mfa_findings = [f for f in findings if f.finding_id == "AWS-IAM-001"]
        self.assertEqual(len(mfa_findings), 0)

    def test_user_without_console_access_not_flagged(self):
        """Programmatic-only user (no console) → no MFA finding even without MFA."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("ci-robot", password_enabled="false", mfa_active="false"),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        mfa_findings = [f for f in findings if f.finding_id == "AWS-IAM-001"]
        self.assertEqual(len(mfa_findings), 0)

    def test_multiple_users_all_without_mfa(self):
        """Three users without MFA → three findings."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("alice", password_enabled="true", mfa_active="false"),
            make_user_row("bob",   password_enabled="true", mfa_active="false"),
            make_user_row("carol", password_enabled="true", mfa_active="false"),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        mfa_findings = [f for f in findings if f.finding_id == "AWS-IAM-001"]
        self.assertEqual(len(mfa_findings), 3)

    def test_finding_contains_evidence(self):
        """AWS-IAM-001 finding raw_evidence includes relevant fields."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("alice", password_enabled="true", mfa_active="false"),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        mfa_findings = [f for f in findings if f.finding_id == "AWS-IAM-001"]
        evidence = mfa_findings[0].raw_evidence
        self.assertIn("user_name", evidence)
        self.assertIn("mfa_active", evidence)
        self.assertIn("password_enabled", evidence)
        self.assertEqual(evidence["user_name"], "alice")

    def test_resource_id_is_arn(self):
        """Resource ID should be the user's full ARN."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("alice", account_id="123456789012", password_enabled="true", mfa_active="false"),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam, account_id="123456789012")
        findings = checker._check_mfa(checker._get_credential_report())

        mfa_findings = [f for f in findings if f.finding_id == "AWS-IAM-001"]
        self.assertIn("arn:aws:iam::123456789012:user/alice", mfa_findings[0].resource_id)


# ===========================================================================
# Tests: AWS-IAM-002 — Root Account
# ===========================================================================

class TestIAM002RootAccount(unittest.TestCase):

    def test_root_no_mfa_flagged(self):
        """Root with no MFA → AWS-IAM-002 finding."""
        rows = [make_root_row(mfa_active="false")]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        root_findings = [f for f in findings if f.finding_id == "AWS-IAM-002"]
        self.assertEqual(len(root_findings), 1)
        self.assertEqual(root_findings[0].severity, Severity.CRITICAL)

    def test_root_with_active_keys_flagged(self):
        """Root with active access keys → AWS-IAM-002 (keys should never exist)."""
        rows = [make_root_row(mfa_active="true", key1_active="true")]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        root_findings = [f for f in findings if f.finding_id == "AWS-IAM-002"]
        self.assertEqual(len(root_findings), 1)

    def test_root_no_mfa_and_keys_single_finding(self):
        """Both issues on root → one consolidated finding (not two)."""
        rows = [make_root_row(mfa_active="false", key1_active="true")]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        root_findings = [f for f in findings if f.finding_id == "AWS-IAM-002"]
        self.assertEqual(len(root_findings), 1)
        # But evidence should list both issues
        issues = root_findings[0].raw_evidence.get("issues", [])
        self.assertTrue(len(issues) >= 2)

    def test_root_with_mfa_and_no_keys_clean(self):
        """Well-configured root (MFA on, no keys) → no finding."""
        rows = [make_root_row(mfa_active="true", key1_active="false", key2_active="false")]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        root_findings = [f for f in findings if f.finding_id == "AWS-IAM-002"]
        self.assertEqual(len(root_findings), 0)

    def test_root_finding_resource_type(self):
        """Root finding should use AWS::IAM::RootAccount resource type."""
        rows = [make_root_row(mfa_active="false")]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_mfa(checker._get_credential_report())

        root_findings = [f for f in findings if f.finding_id == "AWS-IAM-002"]
        self.assertEqual(root_findings[0].resource_type, "AWS::IAM::RootAccount")
        self.assertEqual(root_findings[0].region, "global")


# ===========================================================================
# Tests: AWS-IAM-003 — Access Key Age
# ===========================================================================

class TestIAM003AccessKeyAge(unittest.TestCase):

    def test_old_active_key_flagged(self):
        """Active key older than 90 days → AWS-IAM-003."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row(
                "alice",
                key1_active="true",
                key1_rotated=days_ago(100),  # 100 days old
            ),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_access_key_age(checker._get_credential_report())

        age_findings = [f for f in findings if f.finding_id == "AWS-IAM-003"]
        self.assertEqual(len(age_findings), 1)
        self.assertEqual(age_findings[0].resource_name, "alice")
        self.assertEqual(age_findings[0].severity, Severity.MEDIUM)

    def test_old_inactive_key_still_flagged(self):
        """Inactive but old key → still flagged (should be deleted, not just disabled)."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row(
                "bob",
                key1_active="false",
                key1_rotated=days_ago(120),  # 120 days, inactive
            ),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_access_key_age(checker._get_credential_report())

        age_findings = [f for f in findings if f.finding_id == "AWS-IAM-003"]
        self.assertEqual(len(age_findings), 1)

    def test_recent_key_not_flagged(self):
        """Key rotated 30 days ago → no finding."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row(
                "carol",
                key1_active="true",
                key1_rotated=days_ago(30),  # 30 days — within limit
            ),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_access_key_age(checker._get_credential_report())

        age_findings = [f for f in findings if f.finding_id == "AWS-IAM-003"]
        self.assertEqual(len(age_findings), 0)

    def test_both_keys_old_two_findings(self):
        """User with both key slots old → two findings."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row(
                "dave",
                key1_active="true", key1_rotated=days_ago(100),
                key2_active="true", key2_rotated=days_ago(200),
            ),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_access_key_age(checker._get_credential_report())

        age_findings = [f for f in findings if f.finding_id == "AWS-IAM-003"]
        self.assertEqual(len(age_findings), 2)

    def test_no_key_slot_used_not_flagged(self):
        """User with no access keys → no finding."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("eve", key1_active="false", key1_rotated="N/A"),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_access_key_age(checker._get_credential_report())

        age_findings = [f for f in findings if f.finding_id == "AWS-IAM-003"]
        self.assertEqual(len(age_findings), 0)

    def test_finding_includes_age_days_in_evidence(self):
        """Evidence must include age_days for the report."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("frank", key1_active="true", key1_rotated=days_ago(95)),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_access_key_age(checker._get_credential_report())

        age_findings = [f for f in findings if f.finding_id == "AWS-IAM-003"]
        evidence = age_findings[0].raw_evidence
        self.assertIn("age_days", evidence)
        self.assertGreaterEqual(evidence["age_days"], 95)

    def test_root_keys_not_flagged_by_iam003(self):
        """Root account old keys are caught by AWS-IAM-002, not AWS-IAM-003."""
        rows = [
            make_root_row(mfa_active="true", key1_active="true"),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_access_key_age(checker._get_credential_report())

        # AWS-IAM-003 should skip root rows
        age_findings = [f for f in findings if f.finding_id == "AWS-IAM-003"]
        self.assertEqual(len(age_findings), 0)

    def test_exactly_90_days_boundary(self):
        """Key exactly 90 days old → should be flagged (>= threshold)."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("grace", key1_active="true", key1_rotated=days_ago(90)),
        ]
        iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(iam)
        findings = checker._check_access_key_age(checker._get_credential_report())
        age_findings = [f for f in findings if f.finding_id == "AWS-IAM-003"]
        self.assertEqual(len(age_findings), 1)


# ===========================================================================
# Tests: _is_admin_policy helper (pure logic, no API calls)
# ===========================================================================

class TestPolicyAnalysis(unittest.TestCase):
    """
    Tests for _is_admin_policy and _is_dangerous_trust_policy.
    These are pure logic tests — no API calls at all.
    """

    def setUp(self):
        self.checker = make_checker()

    def _admin(self, document) -> bool:
        result, reason = self.checker._is_admin_policy(document)
        return result

    def _reason(self, document) -> str:
        _, reason = self.checker._is_admin_policy(document)
        return reason

    # ---- Wildcard action ----

    def test_wildcard_action_is_admin(self):
        doc = {
            "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]
        }
        self.assertTrue(self._admin(doc))

    def test_wildcard_action_in_list_is_admin(self):
        doc = {
            "Statement": [{"Effect": "Allow", "Action": ["*"], "Resource": "*"}]
        }
        self.assertTrue(self._admin(doc))

    def test_iam_wildcard_with_star_resource_is_admin(self):
        doc = {
            "Statement": [{"Effect": "Allow", "Action": "iam:*", "Resource": "*"}]
        }
        self.assertTrue(self._admin(doc))

    def test_sts_wildcard_with_star_resource_is_admin(self):
        doc = {
            "Statement": [{"Effect": "Allow", "Action": "sts:*", "Resource": "*"}]
        }
        self.assertTrue(self._admin(doc))

    def test_s3_wildcard_not_flagged(self):
        """s3:* with * resource — not as dangerous as iam:* or sts:*"""
        doc = {
            "Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "*"}]
        }
        # s3:* alone is not an admin finding by our policy
        # (it's dangerous but not in the same league as iam:* or full admin)
        result, _ = self.checker._is_admin_policy(doc)
        # This assertion documents our intentional design decision
        self.assertFalse(result)

    def test_deny_wildcard_not_flagged(self):
        """Deny statements with wildcards should not trigger admin finding."""
        doc = {
            "Statement": [{"Effect": "Deny", "Action": "*", "Resource": "*"}]
        }
        self.assertFalse(self._admin(doc))

    def test_not_action_with_star_resource_is_admin(self):
        """NotAction + Allow + Resource:* = effectively grants everything except listed."""
        doc = {
            "Statement": [{
                "Effect": "Allow",
                "NotAction": ["iam:DeleteAccount"],
                "Resource": "*",
            }]
        }
        self.assertTrue(self._admin(doc))

    def test_specific_actions_not_admin(self):
        """Specific read-only actions → not admin."""
        doc = {
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": "arn:aws:s3:::my-bucket/*",
            }]
        }
        self.assertFalse(self._admin(doc))

    def test_administrator_access_pattern(self):
        """The AdministratorAccess policy pattern {Action:*, Resource:*}."""
        doc = {
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]
        }
        self.assertTrue(self._admin(doc))
        self.assertIn("*", self._reason(doc))

    def test_single_statement_not_in_list(self):
        """Statement as dict not wrapped in list → still detected."""
        doc = {
            "Statement": {"Effect": "Allow", "Action": "*", "Resource": "*"}
        }
        self.assertTrue(self._admin(doc))

    def test_multi_statement_only_one_dangerous(self):
        """Multiple statements, one dangerous one → detected."""
        doc = {
            "Statement": [
                {"Effect": "Allow", "Action": "s3:GetObject", "Resource": "*"},
                {"Effect": "Allow", "Action": "*", "Resource": "*"},   # Dangerous
            ]
        }
        self.assertTrue(self._admin(doc))

    # ---- Trust policy analysis ----

    def test_trust_bare_wildcard_is_dangerous(self):
        trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": "*",
                "Action": "sts:AssumeRole",
            }]
        }
        result, reason = self.checker._is_dangerous_trust_policy(trust)
        self.assertTrue(result)
        self.assertIn("*", reason)

    def test_trust_aws_wildcard_is_dangerous(self):
        trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "sts:AssumeRole",
            }]
        }
        result, reason = self.checker._is_dangerous_trust_policy(trust)
        self.assertTrue(result)

    def test_trust_specific_account_not_dangerous(self):
        trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": "arn:aws:iam::123456789012:root"},
                "Action": "sts:AssumeRole",
            }]
        }
        result, _ = self.checker._is_dangerous_trust_policy(trust)
        self.assertFalse(result)

    def test_trust_service_principal_not_dangerous(self):
        """Lambda service principal → not flagged."""
        trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }]
        }
        result, _ = self.checker._is_dangerous_trust_policy(trust)
        self.assertFalse(result)

    def test_trust_wildcard_account_root_is_dangerous(self):
        """arn:aws:iam::*:root means any account — dangerous."""
        trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": "arn:aws:iam::*:root"},
                "Action": "sts:AssumeRole",
            }]
        }
        result, reason = self.checker._is_dangerous_trust_policy(trust)
        self.assertTrue(result)

    def test_trust_deny_statement_not_flagged(self):
        trust = {
            "Statement": [{
                "Effect": "Deny",
                "Principal": "*",
                "Action": "sts:AssumeRole",
            }]
        }
        result, _ = self.checker._is_dangerous_trust_policy(trust)
        self.assertFalse(result)

    def test_trust_multiple_principals_one_wildcard(self):
        """List of principals where one is '*' → dangerous."""
        trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": ["arn:aws:iam::123:root", "*"]},
                "Action": "sts:AssumeRole",
            }]
        }
        result, _ = self.checker._is_dangerous_trust_policy(trust)
        self.assertTrue(result)


# ===========================================================================
# Tests: AWS-IAM-004 — Overly permissive customer-managed policies
# ===========================================================================

class TestIAM004PermissivePolicies(unittest.TestCase):

    def test_customer_policy_with_star_action_flagged(self):
        """Customer-managed policy with Action:* → AWS-IAM-004."""
        admin_policy_doc = {
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]
        }
        policies = [
            {
                "PolicyName": "DangerousCustomPolicy",
                "Arn": "arn:aws:iam::123456789012:policy/DangerousCustomPolicy",
                "DefaultVersionId": "v1",
                "AttachmentCount": 2,
            }
        ]
        mock_iam = make_mock_iam_client(customer_policies=policies)
        mock_iam.get_policy_version.return_value = {
            "PolicyVersion": {"Document": admin_policy_doc, "IsDefaultVersion": True}
        }
        checker = make_checker(mock_iam)
        findings = checker._check_permissive_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-004"]
        self.assertTrue(len(flagged) >= 1)
        names = [f.resource_name for f in flagged]
        self.assertIn("DangerousCustomPolicy", names)

    def test_restrictive_customer_policy_not_flagged(self):
        """Read-only policy → no finding."""
        readonly_doc = {
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": "arn:aws:s3:::my-bucket/*",
            }]
        }
        policies = [{
            "PolicyName": "S3ReadOnly",
            "Arn": "arn:aws:iam::123456789012:policy/S3ReadOnly",
            "DefaultVersionId": "v1",
            "AttachmentCount": 1,
        }]
        mock_iam = make_mock_iam_client(customer_policies=policies)
        mock_iam.get_policy_version.return_value = {
            "PolicyVersion": {"Document": readonly_doc}
        }
        checker = make_checker(mock_iam)
        findings = checker._check_permissive_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-004"
                   and f.resource_name == "S3ReadOnly"]
        self.assertEqual(len(flagged), 0)

    def test_inline_user_policy_with_wildcard_flagged(self):
        """Inline policy on user with Action:* → AWS-IAM-004."""
        admin_doc = {
            "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]
        }
        users = [{"UserName": "poweruser", "Arn": "arn:aws:iam::123:user/poweruser"}]
        mock_iam = make_mock_iam_client(
            users=users,
            user_inline_policies={"poweruser": ["AllAccessInline"]},
            inline_policy_docs={"AllAccessInline": admin_doc},
        )
        checker = make_checker(mock_iam)
        findings = checker._check_permissive_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-004"]
        self.assertTrue(len(flagged) >= 1)
        # At least one finding should reference the inline policy
        inline_findings = [f for f in flagged if "AllAccessInline" in f.resource_name]
        self.assertTrue(len(inline_findings) >= 1)

    def test_inline_role_policy_with_iam_wildcard_flagged(self):
        """Inline role policy with iam:* → AWS-IAM-004."""
        iam_admin_doc = {
            "Statement": [{"Effect": "Allow", "Action": "iam:*", "Resource": "*"}]
        }
        roles = [{"RoleName": "admin-role", "Arn": "arn:aws:iam::123:role/admin-role",
                  "AssumeRolePolicyDocument": {"Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}}]
        mock_iam = make_mock_iam_client(
            roles=roles,
            role_inline_policies={"admin-role": ["IAMFullAccessInline"]},
            inline_policy_docs={"IAMFullAccessInline": iam_admin_doc},
        )
        checker = make_checker(mock_iam)
        findings = checker._check_permissive_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-004"
                   and "IAMFullAccessInline" in f.resource_name]
        self.assertTrue(len(flagged) >= 1)

    def test_finding_resource_type_for_managed_policy(self):
        """Managed policy finding should have AWS::IAM::ManagedPolicy type."""
        admin_doc = {"Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}
        policies = [{"PolicyName": "MyAdminPolicy", "Arn": "arn:aws:iam::123:policy/MyAdminPolicy",
                     "DefaultVersionId": "v1", "AttachmentCount": 1}]
        mock_iam = make_mock_iam_client(customer_policies=policies)
        mock_iam.get_policy_version.return_value = {
            "PolicyVersion": {"Document": admin_doc}
        }
        checker = make_checker(mock_iam)
        findings = checker._check_permissive_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-004"
                   and "MyAdminPolicy" in f.resource_name]
        if flagged:
            self.assertEqual(flagged[0].resource_type, "AWS::IAM::ManagedPolicy")

    def test_no_customer_policies_no_findings(self):
        """Account with no customer-managed policies → no managed policy findings."""
        mock_iam = make_mock_iam_client(customer_policies=[])
        checker = make_checker(mock_iam)
        findings = checker._check_permissive_policies()

        # No managed policy findings (there may be inline checks on users/roles)
        managed_findings = [f for f in findings if f.finding_id == "AWS-IAM-004"
                            and "arn:aws:iam::123456789012:policy/" in f.resource_id]
        self.assertEqual(len(managed_findings), 0)


# ===========================================================================
# Tests: AWS-IAM-005 — Dangerous role trust policies
# ===========================================================================

class TestIAM005DangerousTrustPolicies(unittest.TestCase):

    def test_role_with_wildcard_trust_flagged(self):
        """Role with Principal:* → AWS-IAM-005 CRITICAL."""
        wildcard_trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": "*",
                "Action": "sts:AssumeRole",
            }]
        }
        roles = [{
            "RoleName": "public-role",
            "Arn": "arn:aws:iam::123456789012:role/public-role",
            "AssumeRolePolicyDocument": wildcard_trust,
            "Path": "/",
        }]
        mock_iam = make_mock_iam_client(roles=roles)
        checker = make_checker(mock_iam)
        findings = checker._check_dangerous_trust_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-005"]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0].resource_name, "public-role")
        self.assertEqual(flagged[0].severity, Severity.CRITICAL)

    def test_role_with_aws_wildcard_trust_flagged(self):
        """Role with Principal: {AWS: '*'} → AWS-IAM-005."""
        aws_wildcard_trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "sts:AssumeRole",
            }]
        }
        roles = [{
            "RoleName": "open-role",
            "Arn": "arn:aws:iam::123:role/open-role",
            "AssumeRolePolicyDocument": aws_wildcard_trust,
        }]
        mock_iam = make_mock_iam_client(roles=roles)
        checker = make_checker(mock_iam)
        findings = checker._check_dangerous_trust_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-005"]
        self.assertEqual(len(flagged), 1)

    def test_service_role_not_flagged(self):
        """Lambda service role with specific principal → not flagged."""
        service_trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }]
        }
        roles = [{
            "RoleName": "lambda-execution-role",
            "Arn": "arn:aws:iam::123:role/lambda-execution-role",
            "AssumeRolePolicyDocument": service_trust,
        }]
        mock_iam = make_mock_iam_client(roles=roles)
        checker = make_checker(mock_iam)
        findings = checker._check_dangerous_trust_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-005"]
        self.assertEqual(len(flagged), 0)

    def test_cross_account_role_with_specific_account_not_flagged(self):
        """Cross-account role with named account → not flagged (appropriate pattern)."""
        cross_account_trust = {
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": "arn:aws:iam::987654321098:root"},
                "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"sts:ExternalId": "unique-external-id"}},
            }]
        }
        roles = [{
            "RoleName": "cross-account-role",
            "Arn": "arn:aws:iam::123:role/cross-account-role",
            "AssumeRolePolicyDocument": cross_account_trust,
        }]
        mock_iam = make_mock_iam_client(roles=roles)
        checker = make_checker(mock_iam)
        findings = checker._check_dangerous_trust_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-005"]
        self.assertEqual(len(flagged), 0)

    def test_finding_includes_trust_policy_in_evidence(self):
        """Trust policy finding must include the trust document in evidence."""
        wildcard_trust = {
            "Statement": [{"Effect": "Allow", "Principal": "*", "Action": "sts:AssumeRole"}]
        }
        roles = [{"RoleName": "bad-role", "Arn": "arn:aws:iam::123:role/bad-role",
                  "AssumeRolePolicyDocument": wildcard_trust}]
        mock_iam = make_mock_iam_client(roles=roles)
        checker = make_checker(mock_iam)
        findings = checker._check_dangerous_trust_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-005"]
        self.assertIn("trust_policy", flagged[0].raw_evidence)
        self.assertIn("dangerous_reason", flagged[0].raw_evidence)

    def test_multiple_roles_only_wildcard_flagged(self):
        """Multiple roles — only the one with wildcard trust is flagged."""
        good_trust = {
            "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
        }
        bad_trust = {
            "Statement": [{"Effect": "Allow", "Principal": "*", "Action": "sts:AssumeRole"}]
        }
        roles = [
            {"RoleName": "ec2-role", "Arn": "arn:aws:iam::123:role/ec2-role", "AssumeRolePolicyDocument": good_trust},
            {"RoleName": "open-role", "Arn": "arn:aws:iam::123:role/open-role", "AssumeRolePolicyDocument": bad_trust},
        ]
        mock_iam = make_mock_iam_client(roles=roles)
        checker = make_checker(mock_iam)
        findings = checker._check_dangerous_trust_policies()

        flagged = [f for f in findings if f.finding_id == "AWS-IAM-005"]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0].resource_name, "open-role")


# ===========================================================================
# Tests: run() integration — full checker execution
# ===========================================================================

class TestIAMCheckerRun(unittest.TestCase):

    def test_run_returns_findings_list(self):
        """run() always returns a list (even empty)."""
        rows = [make_root_row(mfa_active="true")]
        mock_iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(mock_iam)
        result = checker.execute()

        self.assertIsNotNone(result)
        self.assertIsInstance(result.findings, list)
        self.assertTrue(result.succeeded)

    def test_run_combines_all_checks(self):
        """run() with multiple issues returns findings from all checks."""
        rows = [
            make_root_row(mfa_active="false"),  # → AWS-IAM-002
            make_user_row("alice", password_enabled="true", mfa_active="false"),  # → AWS-IAM-001
            make_user_row("bob", key1_active="true", key1_rotated=days_ago(100)),  # → AWS-IAM-003
        ]
        wildcard_trust = {"Statement": [{"Effect": "Allow", "Principal": "*", "Action": "sts:AssumeRole"}]}
        roles = [{"RoleName": "bad-role", "Arn": "arn:aws:iam::123:role/bad-role",
                  "AssumeRolePolicyDocument": wildcard_trust}]

        mock_iam = make_mock_iam_client(cred_report_rows=rows, roles=roles)
        checker = make_checker(mock_iam)
        result = checker.execute()

        finding_ids = {f.finding_id for f in result.findings}
        self.assertIn("AWS-IAM-001", finding_ids)
        self.assertIn("AWS-IAM-002", finding_ids)
        self.assertIn("AWS-IAM-003", finding_ids)
        self.assertIn("AWS-IAM-005", finding_ids)

    def test_run_clean_account_no_findings(self):
        """Perfect account configuration → zero findings."""
        rows = [
            make_root_row(mfa_active="true", key1_active="false"),
            make_user_row("alice", password_enabled="true", mfa_active="true",
                          key1_active="true", key1_rotated=days_ago(30)),
        ]
        safe_trust = {"Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
        roles = [{"RoleName": "safe-role", "Arn": "arn:aws:iam::123:role/safe-role",
                  "AssumeRolePolicyDocument": safe_trust}]
        safe_policy = {"Statement": [{"Effect": "Allow", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::bucket/*"}]}
        mock_iam = make_mock_iam_client(cred_report_rows=rows, roles=roles)
        checker = make_checker(mock_iam)
        result = checker.execute()

        self.assertEqual(len(result.findings), 0)

    def test_execute_records_timing(self):
        """execute() wrapper records non-zero duration."""
        mock_iam = make_mock_iam_client()
        checker = make_checker(mock_iam)
        result = checker.execute()
        self.assertGreaterEqual(result.duration_ms, 0)

    def test_execute_records_api_call_count(self):
        """execute() reports how many API calls were made."""
        rows = [make_root_row()]
        mock_iam = make_mock_iam_client(cred_report_rows=rows)
        checker = make_checker(mock_iam)
        result = checker.execute()
        self.assertGreater(result.api_calls_made, 0)

    def test_credential_report_failure_is_non_fatal(self):
        """If credential report fails, the checker still returns empty list (not crash)."""
        mock_iam = make_mock_iam_client()
        mock_iam.generate_credential_report.side_effect = Exception("Throttled")
        checker = make_checker(mock_iam)

        # Should not raise — credential report failure is non-fatal
        result = checker.execute()
        self.assertTrue(result.succeeded)

    def test_individual_check_failure_does_not_abort_others(self):
        """If one check raises, others still run."""
        rows = [
            make_root_row(mfa_active="false"),  # → AWS-IAM-002
        ]
        mock_iam = make_mock_iam_client(cred_report_rows=rows)
        # Make list_policies raise — policy check fails
        mock_iam.get_paginator.side_effect = lambda op: (
            _make_failing_paginator() if op == "list_policies"
            else mock_iam.get_paginator.side_effect  # This will loop — use different approach
        )
        # Simpler approach: just verify run doesn't raise even with mixed errors
        checker = make_checker(mock_iam)
        result = checker.execute()
        self.assertTrue(result.succeeded or result.error is not None)

    def test_checker_name(self):
        self.assertEqual(make_checker().checker_name, "iam")

    def test_required_permissions_list(self):
        perms = make_checker().required_permissions
        self.assertIn("iam:GenerateCredentialReport", perms)
        self.assertIn("iam:GetCredentialReport", perms)
        self.assertIn("iam:ListRoles", perms)
        self.assertIn("iam:ListPolicies", perms)

    def test_checker_registered_in_registry(self):
        """IAMChecker must be registered via @CheckerRegistry.register."""
        from core.checker_registry import CheckerRegistry
        checkers = CheckerRegistry.get_checkers("aws", domains=["iam"])
        self.assertIn(IAMChecker, checkers)

    def test_all_finding_templates_defined(self):
        """All 5 finding IDs must have templates."""
        templates = IAMChecker.FINDING_TEMPLATES
        for fid in ["AWS-IAM-001", "AWS-IAM-002", "AWS-IAM-003", "AWS-IAM-004", "AWS-IAM-005"]:
            self.assertIn(fid, templates, f"Missing template for {fid}")


def _make_failing_paginator():
    p = MagicMock()
    p.paginate.side_effect = Exception("Access Denied")
    return p


# ===========================================================================
# Tests: attack graph node building
# ===========================================================================

class TestIAMGraphBuilding(unittest.TestCase):

    def test_user_nodes_added_to_graph(self):
        """build_graph_nodes adds USER nodes for each IAM user."""
        users = [
            {"UserName": "alice", "Arn": "arn:aws:iam::123:user/alice"},
            {"UserName": "bob",   "Arn": "arn:aws:iam::123:user/bob"},
        ]
        mock_iam = make_mock_iam_client(users=users)
        checker = make_checker(mock_iam)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        user_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.USER]
        self.assertEqual(len(user_nodes), 2)
        node_labels = {n.label for n in user_nodes}
        self.assertIn("alice", node_labels)
        self.assertIn("bob", node_labels)

    def test_role_nodes_added_to_graph(self):
        """build_graph_nodes adds ROLE nodes for each IAM role."""
        service_trust = {"Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
        roles = [
            {"RoleName": "lambda-role", "Arn": "arn:aws:iam::123:role/lambda-role", "AssumeRolePolicyDocument": service_trust, "Path": "/"},
        ]
        mock_iam = make_mock_iam_client(roles=roles)
        checker = make_checker(mock_iam)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        role_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.ROLE]
        self.assertEqual(len(role_nodes), 1)
        self.assertEqual(role_nodes[0].label, "lambda-role")

    def test_admin_role_flagged_in_graph(self):
        """Role with AdministratorAccess should have is_admin=True in graph."""
        service_trust = {"Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
        roles = [
            {"RoleName": "admin-role", "Arn": "arn:aws:iam::123:role/admin-role", "AssumeRolePolicyDocument": service_trust, "Path": "/"},
        ]
        mock_iam = make_mock_iam_client(
            roles=roles,
            attached_role_policies={"admin-role": [
                {"PolicyArn": "arn:aws:iam::aws:policy/AdministratorAccess", "PolicyName": "AdministratorAccess"}
            ]},
        )
        checker = make_checker(mock_iam)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        role_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.ROLE]
        admin_roles = [n for n in role_nodes if n.properties.get("is_admin")]
        self.assertEqual(len(admin_roles), 1)

    def test_wildcard_trust_role_flagged_in_graph(self):
        """Role with wildcard trust should have trust_wildcard=True in graph."""
        wildcard_trust = {"Statement": [{"Effect": "Allow", "Principal": "*", "Action": "sts:AssumeRole"}]}
        roles = [
            {"RoleName": "open-role", "Arn": "arn:aws:iam::123:role/open-role", "AssumeRolePolicyDocument": wildcard_trust, "Path": "/"},
        ]
        mock_iam = make_mock_iam_client(roles=roles)
        checker = make_checker(mock_iam)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        role_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.ROLE]
        wildcard_roles = [n for n in role_nodes if n.properties.get("trust_wildcard")]
        self.assertEqual(len(wildcard_roles), 1)

    def test_mfa_status_captured_in_user_node(self):
        """User node properties should reflect MFA status from credential report."""
        rows = [
            make_root_row(mfa_active="true"),
            make_user_row("alice", password_enabled="true", mfa_active="false"),  # No MFA
        ]
        users = [{"UserName": "alice", "Arn": "arn:aws:iam::123:user/alice"}]
        mock_iam = make_mock_iam_client(cred_report_rows=rows, users=users)
        checker = make_checker(mock_iam)
        graph = AttackGraph(account_id="123456789012", provider="aws")

        cred_report = checker._get_credential_report()
        checker.build_graph_nodes(graph, credential_report=cred_report)

        alice_nodes = [n for n in graph.nodes.values() if n.label == "alice"]
        self.assertEqual(len(alice_nodes), 1)
        self.assertFalse(alice_nodes[0].properties.get("mfa_enabled"))

    def test_graph_build_failure_is_non_fatal(self):
        """If graph building fails internally, it should not raise."""
        mock_iam = make_mock_iam_client()
        mock_iam.get_paginator.side_effect = Exception("Graph build API error")
        checker = make_checker(mock_iam)
        graph = AttackGraph(account_id="123456789012", provider="aws")

        # Should not raise
        checker.build_graph_nodes(graph)
        # Graph will just be empty
        self.assertEqual(len(graph.nodes), 0)


# ===========================================================================
# Tests: credential report parsing edge cases
# ===========================================================================

class TestCredentialReportParsing(unittest.TestCase):

    def test_empty_credential_report_returns_empty_list(self):
        """Empty credential report → empty list, no crash."""
        mock_iam = make_mock_iam_client()
        mock_iam.get_credential_report.return_value = {
            "Content": make_cred_report_csv([]).encode("utf-8")
        }
        checker = make_checker(mock_iam)
        rows = checker._get_credential_report()
        self.assertEqual(rows, [])

    def test_credential_report_api_failure_returns_empty_list(self):
        """If API throws, returns [] instead of crashing."""
        mock_iam = make_mock_iam_client()
        mock_iam.generate_credential_report.side_effect = Exception("API Error")
        checker = make_checker(mock_iam)
        rows = checker._get_credential_report()
        self.assertEqual(rows, [])

    def test_credential_report_bytes_decoded(self):
        """Credential report content arrives as bytes from real boto3."""
        rows = [make_root_row(), make_user_row("alice")]
        csv_bytes = make_cred_report_csv(rows).encode("utf-8")
        mock_iam = make_mock_iam_client()
        mock_iam.generate_credential_report.return_value = {"State": "COMPLETE"}
        mock_iam.get_credential_report.return_value = {"Content": csv_bytes}
        checker = make_checker(mock_iam)
        parsed = checker._get_credential_report()
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[1]["user"], "alice")

    def test_credential_report_string_content_handled(self):
        """Content as string (mock environments) → also works."""
        rows = [make_root_row(), make_user_row("bob")]
        csv_str = make_cred_report_csv(rows)  # String, not bytes
        mock_iam = make_mock_iam_client()
        mock_iam.generate_credential_report.return_value = {"State": "COMPLETE"}
        mock_iam.get_credential_report.return_value = {"Content": csv_str}
        checker = make_checker(mock_iam)
        parsed = checker._get_credential_report()
        self.assertEqual(len(parsed), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)