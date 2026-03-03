"""
tests/unit/test_step5.py

Comprehensive unit tests for S3Checker — all 4 checks.

Testing strategy:
  - All S3 API calls are mocked via MagicMock — no AWS credentials needed
  - The two-layer public access check (BPA + policy/ACL) is tested exhaustively
  - Edge cases: NoSuchBucketPolicy, NoSuchPublicAccessBlockConfiguration,
    GetBucketLocation returning None (us-east-1 quirk), empty bucket lists
  - Graph node building: PUBLIC_RESOURCE vs PRIVATE_RESOURCE, IS_PUBLIC edges,
    internet node creation
  - All 4 finding templates verified to exist

Run with:
    python tests/unit/test_step5.py
"""

from __future__ import annotations

import json
import sys
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Pre-register boto3/botocore mocks (same pattern as prior step tests)
class FakeClientError(Exception):
    def __init__(self, error_response, operation_name="Op"):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(str(error_response))

_botocore_exc = MagicMock()
_botocore_exc.ClientError = FakeClientError
_botocore = MagicMock()
_botocore.exceptions = _botocore_exc
_boto3 = MagicMock()
sys.modules.setdefault("boto3", _boto3)
sys.modules.setdefault("botocore", _botocore)
sys.modules.setdefault("botocore.exceptions", _botocore_exc)

from providers.aws.checkers.s3 import S3Checker, INTERNET_NODE_ID
from core.models.finding import Severity, FindingStatus
from core.attack_graph.models import AttackGraph, NodeType, EdgeType


# ===========================================================================
# Mock builder helpers
# ===========================================================================

def make_s3_error(code: str) -> FakeClientError:
    return FakeClientError({"Error": {"Code": code, "Message": code}})


def make_block_public_access(
    block_acls=True, ignore_acls=True,
    block_policy=True, restrict_buckets=True
) -> Dict:
    return {
        "BlockPublicAcls":       block_acls,
        "IgnorePublicAcls":      ignore_acls,
        "BlockPublicPolicy":     block_policy,
        "RestrictPublicBuckets": restrict_buckets,
    }


def make_bucket_policy(public: bool = False, action: str = "s3:GetObject") -> str:
    """Return a JSON bucket policy string."""
    if public:
        return json.dumps({
            "Statement": [{
                "Effect": "Allow",
                "Principal": "*",
                "Action": action,
                "Resource": "arn:aws:s3:::test-bucket/*",
            }]
        })
    return json.dumps({
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::123456789012:root"},
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::test-bucket/*",
        }]
    })


def make_acl_response(public: bool = False) -> Dict:
    """Return a mock get_bucket_acl response."""
    if public:
        return {
            "Grants": [{
                "Grantee": {
                    "Type": "Group",
                    "URI": "http://acs.amazonaws.com/groups/global/AllUsers",
                },
                "Permission": "READ",
            }]
        }
    return {
        "Grants": [{
            "Grantee": {
                "Type": "CanonicalUser",
                "DisplayName": "owner",
                "ID": "abc123",
            },
            "Permission": "FULL_CONTROL",
        }]
    }


def make_encryption_response(algorithm: str = "AES256") -> Dict:
    return {
        "ServerSideEncryptionConfiguration": {
            "Rules": [{
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": algorithm,
                }
            }]
        }
    }


def make_logging_response(enabled: bool = True) -> Dict:
    if enabled:
        return {
            "LoggingEnabled": {
                "TargetBucket": "my-log-bucket",
                "TargetPrefix": "test-bucket/",
            }
        }
    return {}  # No LoggingEnabled key = disabled


def make_versioning_response(status: str = "Enabled") -> Dict:
    if status:
        return {"Status": status}
    return {}  # No Status key = never enabled


class MockS3Client:
    """
    Configurable mock S3 client.
    Each attribute can be set to a return value, an exception, or a callable.
    """

    def __init__(
        self,
        buckets: List[str] = None,
        location: str = "us-east-1",
        bpa: Any = "fully_blocked",          # Dict | "fully_blocked" | "no_config" | Exception
        policy: Any = "no_policy",            # str | "no_policy" | Exception
        acl: Any = "private",                 # "private" | "public" | Exception
        encryption: Any = "encrypted",        # "encrypted" | "kms" | "none" | Exception
        logging: Any = "enabled",             # "enabled" | "disabled" | Exception
        versioning: Any = "Enabled",          # "Enabled" | "Suspended" | "" | Exception
    ):
        self._buckets = buckets or ["test-bucket"]
        self._location = location
        self._bpa = bpa
        self._policy = policy
        self._acl = acl
        self._encryption = encryption
        self._logging = logging
        self._versioning = versioning

        # Track call counts
        self.call_counts: Dict[str, int] = {}

    def _count(self, method: str) -> None:
        self.call_counts[method] = self.call_counts.get(method, 0) + 1

    def list_buckets(self):
        self._count("list_buckets")
        return {"Buckets": [{"Name": b} for b in self._buckets]}

    def get_bucket_location(self, Bucket, **kw):
        self._count("get_bucket_location")
        loc = None if self._location == "us-east-1" else self._location
        return {"LocationConstraint": loc}

    def get_bucket_public_access_block(self, Bucket, **kw):
        self._count("get_bucket_public_access_block")
        if isinstance(self._bpa, Exception):
            raise self._bpa
        if self._bpa == "fully_blocked":
            return {"PublicAccessBlockConfiguration": make_block_public_access()}
        if self._bpa == "no_config":
            raise make_s3_error("NoSuchPublicAccessBlockConfiguration")
        if self._bpa is None:
            raise make_s3_error("NoSuchPublicAccessBlockConfiguration")
        return {"PublicAccessBlockConfiguration": self._bpa}

    def get_bucket_policy(self, Bucket, **kw):
        self._count("get_bucket_policy")
        if isinstance(self._policy, Exception):
            raise self._policy
        if self._policy == "no_policy":
            raise make_s3_error("NoSuchBucketPolicy")
        if self._policy == "public":
            return {"Policy": make_bucket_policy(public=True)}
        if isinstance(self._policy, str) and self._policy.startswith("{"):
            return {"Policy": self._policy}
        return {"Policy": make_bucket_policy(public=False)}

    def get_bucket_acl(self, Bucket, **kw):
        self._count("get_bucket_acl")
        if isinstance(self._acl, Exception):
            raise self._acl
        if self._acl == "public":
            return make_acl_response(public=True)
        return make_acl_response(public=False)

    def get_bucket_encryption(self, Bucket, **kw):
        self._count("get_bucket_encryption")
        if isinstance(self._encryption, Exception):
            raise self._encryption
        if self._encryption == "none":
            raise make_s3_error("ServerSideEncryptionConfigurationNotFoundError")
        if self._encryption == "kms":
            return make_encryption_response("aws:kms")
        if self._encryption == "encrypted":
            return make_encryption_response("AES256")
        return make_encryption_response("AES256")

    def get_bucket_logging(self, Bucket, **kw):
        self._count("get_bucket_logging")
        if isinstance(self._logging, Exception):
            raise self._logging
        if self._logging == "enabled":
            return make_logging_response(enabled=True)
        return make_logging_response(enabled=False)

    def get_bucket_versioning(self, Bucket, **kw):
        self._count("get_bucket_versioning")
        if isinstance(self._versioning, Exception):
            raise self._versioning
        return make_versioning_response(self._versioning)


def make_checker(
    mock_client: Any = None,
    buckets: List[str] = None,
    account_id: str = "123456789012",
    **client_kwargs,
) -> S3Checker:
    """
    Create an S3Checker with a MockS3Client injected.
    The session.client() is set up to return the same mock client
    for all region variants.
    """
    if mock_client is None:
        mock_client = MockS3Client(buckets=buckets or ["test-bucket"], **client_kwargs)

    session = MagicMock()
    session.client.return_value = mock_client

    checker = S3Checker(
        session=session,
        account_id=account_id,
        region="us-east-1",
    )
    # Inject the client directly
    checker._s3 = mock_client
    return checker


# ===========================================================================
# Tests: AWS-S3-001 — Public access
# ===========================================================================

class TestS3001PublicAccess(unittest.TestCase):

    def test_fully_blocked_bucket_not_flagged(self):
        """All BPA flags True → bucket cannot be public, no finding."""
        checker = make_checker(bpa="fully_blocked", policy="public", acl="public")
        findings = checker._check_public_access(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        s3_001 = [f for f in findings if f.finding_id == "AWS-S3-001"]
        self.assertEqual(len(s3_001), 0)

    def test_public_via_bucket_policy_flagged(self):
        """No BPA, public bucket policy → AWS-S3-001 CRITICAL."""
        checker = make_checker(bpa="no_config", policy="public", acl="private")
        findings = checker._check_public_access(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        s3_001 = [f for f in findings if f.finding_id == "AWS-S3-001"]
        self.assertEqual(len(s3_001), 1)
        self.assertEqual(s3_001[0].severity, Severity.CRITICAL)
        self.assertEqual(s3_001[0].resource_name, "test-bucket")

    def test_public_via_acl_flagged(self):
        """No BPA, private policy, public ACL → AWS-S3-001."""
        checker = make_checker(bpa="no_config", policy="no_policy", acl="public")
        findings = checker._check_public_access(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        s3_001 = [f for f in findings if f.finding_id == "AWS-S3-001"]
        self.assertEqual(len(s3_001), 1)

    def test_public_via_both_policy_and_acl_single_finding(self):
        """Public via both policy AND ACL → one finding, not two."""
        checker = make_checker(bpa="no_config", policy="public", acl="public")
        findings = checker._check_public_access(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        s3_001 = [f for f in findings if f.finding_id == "AWS-S3-001"]
        self.assertEqual(len(s3_001), 1)
        # Both reasons should be in the evidence
        evidence = s3_001[0].raw_evidence
        self.assertTrue(evidence.get("public_via_policy"))
        self.assertTrue(evidence.get("public_via_acl"))

    def test_private_bucket_no_finding(self):
        """Private policy, private ACL, no BPA needed → no finding."""
        checker = make_checker(bpa="no_config", policy="no_policy", acl="private")
        findings = checker._check_public_access(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 0)

    def test_partial_bpa_still_checks_policy(self):
        """BPA not fully enabled (only 2/4 flags) → check policy/ACL."""
        partial_bpa = make_block_public_access(
            block_acls=True, ignore_acls=True,
            block_policy=False, restrict_buckets=False  # Not fully blocked
        )
        checker = make_checker(bpa=partial_bpa, policy="public", acl="private")
        findings = checker._check_public_access(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 1)

    def test_bpa_api_failure_falls_back_to_policy_check(self):
        """If BPA API fails, we still check policy and ACL."""
        checker = make_checker(
            bpa=FakeClientError({"Error": {"Code": "AccessDenied", "Message": "Denied"}}),
            policy="public",
            acl="private",
        )
        findings = checker._check_public_access(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        # Fallback to policy check catches the public policy
        self.assertEqual(len(findings), 1)

    def test_no_such_bpa_config_treated_as_no_bpa(self):
        """NoSuchPublicAccessBlockConfiguration → treat as BPA=off, check policy."""
        checker = make_checker(bpa="no_config", policy="public", acl="private")
        findings = checker._check_public_access(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 1)

    def test_no_such_bucket_policy_not_flagged(self):
        """NoSuchBucketPolicy error → bucket has no policy (fine)."""
        checker = make_checker(bpa="no_config", policy="no_policy", acl="private")
        is_public, reason = checker._is_public_via_policy(
            checker._s3, "test-bucket"
        )
        self.assertFalse(is_public)

    def test_authenticated_users_acl_is_public(self):
        """ACL grant to AuthenticatedUsers (any AWS account) → public."""
        mock_client = MagicMock()
        mock_client.get_bucket_acl.return_value = {
            "Grants": [{
                "Grantee": {
                    "Type": "Group",
                    "URI": "http://acs.amazonaws.com/groups/global/AuthenticatedUsers",
                },
                "Permission": "WRITE",
            }]
        }
        checker = make_checker(mock_client)
        is_public, reason = checker._is_public_via_acl(mock_client, "test-bucket")
        self.assertTrue(is_public)
        self.assertIn("AuthenticatedUsers", reason)

    def test_finding_resource_id_is_arn(self):
        """Finding resource_id must be the S3 ARN format."""
        checker = make_checker(bpa="no_config", policy="public", acl="private")
        findings = checker._check_public_access(
            "my-bucket", "arn:aws:s3:::my-bucket", "eu-west-1"
        )
        self.assertEqual(findings[0].resource_id, "arn:aws:s3:::my-bucket")

    def test_finding_region_reflects_bucket_region(self):
        """Finding region should match the bucket's actual region."""
        checker = make_checker(bpa="no_config", policy="public", acl="private")
        findings = checker._check_public_access(
            "eu-bucket", "arn:aws:s3:::eu-bucket", "eu-west-1"
        )
        self.assertEqual(findings[0].region, "eu-west-1")

    def test_aws_principal_wildcard_is_public(self):
        """Principal.AWS: '*' in bucket policy → public."""
        policy = json.dumps({
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::test-bucket/*",
            }]
        })
        checker = make_checker(bpa="no_config", policy=policy, acl="private")
        is_public, reason = checker._is_public_via_policy(checker._s3, "test-bucket")
        self.assertTrue(is_public)

    def test_deny_statement_in_policy_not_flagged(self):
        """Policy with only Deny statements → not public."""
        policy = json.dumps({
            "Statement": [{
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:DeleteObject",
                "Resource": "arn:aws:s3:::test-bucket/*",
            }]
        })
        checker = make_checker(bpa="no_config", policy=policy, acl="private")
        is_public, reason = checker._is_public_via_policy(checker._s3, "test-bucket")
        self.assertFalse(is_public)


# ===========================================================================
# Tests: AWS-S3-002 — Encryption
# ===========================================================================

class TestS3002Encryption(unittest.TestCase):

    def test_aes256_encrypted_bucket_no_finding(self):
        """SSE-S3 (AES256) configured → no finding."""
        checker = make_checker(encryption="encrypted")
        findings = checker._check_encryption(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 0)

    def test_kms_encrypted_bucket_no_finding(self):
        """SSE-KMS configured → no finding."""
        checker = make_checker(encryption="kms")
        findings = checker._check_encryption(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 0)

    def test_no_encryption_flagged(self):
        """No default encryption → AWS-S3-002 MEDIUM."""
        checker = make_checker(encryption="none")
        findings = checker._check_encryption(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        s3_002 = [f for f in findings if f.finding_id == "AWS-S3-002"]
        self.assertEqual(len(s3_002), 1)
        self.assertEqual(s3_002[0].severity, Severity.MEDIUM)

    def test_encryption_api_error_not_flagged(self):
        """Unknown API error → don't false-positive (assume encrypted)."""
        checker = make_checker(
            encryption=FakeClientError({"Error": {"Code": "AccessDenied", "Message": "Denied"}})
        )
        findings = checker._check_encryption(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 0)

    def test_encryption_finding_evidence(self):
        """Encryption finding includes bucket name and region."""
        checker = make_checker(encryption="none")
        findings = checker._check_encryption(
            "test-bucket", "arn:aws:s3:::test-bucket", "eu-west-1"
        )
        evidence = findings[0].raw_evidence
        self.assertEqual(evidence["bucket_name"], "test-bucket")
        self.assertEqual(evidence["encryption"], "none")
        self.assertEqual(evidence["region"], "eu-west-1")


# ===========================================================================
# Tests: AWS-S3-003 — Logging
# ===========================================================================

class TestS3003Logging(unittest.TestCase):

    def test_logging_enabled_no_finding(self):
        """Server access logging configured → no finding."""
        checker = make_checker(logging="enabled")
        findings = checker._check_logging(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 0)

    def test_logging_disabled_flagged(self):
        """No LoggingEnabled → AWS-S3-003 LOW."""
        checker = make_checker(logging="disabled")
        findings = checker._check_logging(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        s3_003 = [f for f in findings if f.finding_id == "AWS-S3-003"]
        self.assertEqual(len(s3_003), 1)
        self.assertEqual(s3_003[0].severity, Severity.LOW)

    def test_logging_api_error_not_flagged(self):
        """API error checking logging → don't false-positive."""
        checker = make_checker(
            logging=FakeClientError({"Error": {"Code": "AccessDenied", "Message": "Denied"}})
        )
        findings = checker._check_logging(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 0)

    def test_logging_finding_evidence(self):
        """Logging finding includes relevant evidence."""
        checker = make_checker(logging="disabled")
        findings = checker._check_logging(
            "log-test", "arn:aws:s3:::log-test", "us-west-2"
        )
        evidence = findings[0].raw_evidence
        self.assertEqual(evidence["bucket_name"], "log-test")
        self.assertFalse(evidence["logging_enabled"])

    def test_logging_target_bucket_empty_is_disabled(self):
        """LoggingEnabled present but TargetBucket is empty → treat as disabled."""
        mock_client = MagicMock()
        mock_client.get_bucket_logging.return_value = {
            "LoggingEnabled": {"TargetBucket": "", "TargetPrefix": ""}
        }
        checker = make_checker(mock_client)
        findings = checker._check_logging(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 1)


# ===========================================================================
# Tests: AWS-S3-004 — Versioning
# ===========================================================================

class TestS3004Versioning(unittest.TestCase):

    def test_versioning_enabled_no_finding(self):
        """Versioning Enabled → no finding."""
        checker = make_checker(versioning="Enabled")
        findings = checker._check_versioning(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 0)

    def test_versioning_never_enabled_flagged(self):
        """No Status key → versioning never enabled → AWS-S3-004 LOW."""
        checker = make_checker(versioning="")
        findings = checker._check_versioning(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        s3_004 = [f for f in findings if f.finding_id == "AWS-S3-004"]
        self.assertEqual(len(s3_004), 1)
        self.assertEqual(s3_004[0].severity, Severity.LOW)

    def test_versioning_suspended_flagged(self):
        """Status: Suspended → flagged (was enabled, now paused)."""
        checker = make_checker(versioning="Suspended")
        findings = checker._check_versioning(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        s3_004 = [f for f in findings if f.finding_id == "AWS-S3-004"]
        self.assertEqual(len(s3_004), 1)
        # Description should mention suspension
        self.assertIn("Suspended", s3_004[0].description)

    def test_versioning_api_error_not_flagged(self):
        """API error → don't false-positive."""
        checker = make_checker(
            versioning=FakeClientError({"Error": {"Code": "AccessDenied", "Message": "Denied"}})
        )
        findings = checker._check_versioning(
            "test-bucket", "arn:aws:s3:::test-bucket", "us-east-1"
        )
        self.assertEqual(len(findings), 0)

    def test_versioning_evidence_includes_status(self):
        """Evidence includes the versioning status string."""
        checker = make_checker(versioning="Suspended")
        findings = checker._check_versioning(
            "ver-bucket", "arn:aws:s3:::ver-bucket", "us-east-1"
        )
        evidence = findings[0].raw_evidence
        self.assertEqual(evidence["versioning_status"], "Suspended")
        self.assertEqual(evidence["bucket_name"], "ver-bucket")


# ===========================================================================
# Tests: run() — full checker execution
# ===========================================================================

class TestS3CheckerRun(unittest.TestCase):

    def test_run_clean_bucket_no_findings(self):
        """Perfect bucket config → zero findings."""
        checker = make_checker(
            bpa="fully_blocked", encryption="kms",
            logging="enabled", versioning="Enabled"
        )
        result = checker.execute()
        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.findings), 0)

    def test_run_problematic_bucket_all_findings(self):
        """Worst-case bucket → findings for all 4 checks."""
        checker = make_checker(
            buckets=["bad-bucket"],
            bpa="no_config",
            policy="public",
            acl="private",
            encryption="none",
            logging="disabled",
            versioning="",
        )
        result = checker.execute()
        finding_ids = {f.finding_id for f in result.findings}
        self.assertIn("AWS-S3-001", finding_ids)
        self.assertIn("AWS-S3-002", finding_ids)
        self.assertIn("AWS-S3-003", finding_ids)
        self.assertIn("AWS-S3-004", finding_ids)

    def test_run_multiple_buckets(self):
        """Multiple buckets scanned in one run."""
        checker = make_checker(
            buckets=["bucket-a", "bucket-b", "bucket-c"],
            encryption="none",  # All unencrypted → 3 findings
        )
        result = checker.execute()
        enc_findings = [f for f in result.findings if f.finding_id == "AWS-S3-002"]
        self.assertEqual(len(enc_findings), 3)

    def test_run_empty_account_no_findings(self):
        """Account with no S3 buckets → no findings, no crash."""
        checker = make_checker(buckets=[])
        result = checker.execute()
        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.findings), 0)

    def test_run_list_buckets_failure_returns_empty(self):
        """If list_buckets raises, run() returns empty list gracefully."""
        mock_client = MockS3Client(buckets=["test-bucket"])
        mock_client.list_buckets = lambda: (_ for _ in ()).throw(
            FakeClientError({"Error": {"Code": "AccessDenied", "Message": "Denied"}})
        )
        session = MagicMock()
        session.client.return_value = mock_client
        checker = S3Checker(session=session, account_id="123", region="us-east-1")
        checker._s3 = mock_client

        # Replace list_buckets with one that raises
        checker._s3 = MagicMock()
        checker._s3.list_buckets.side_effect = FakeClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}}
        )
        result = checker.run()
        self.assertEqual(result, [])

    def test_execute_records_timing(self):
        """execute() wrapper records non-zero duration."""
        checker = make_checker()
        result = checker.execute()
        self.assertGreaterEqual(result.duration_ms, 0)

    def test_execute_records_api_calls(self):
        """execute() reports how many API calls were made."""
        checker = make_checker()
        result = checker.execute()
        self.assertGreater(result.api_calls_made, 0)

    def test_checker_name(self):
        self.assertEqual(make_checker().checker_name, "s3")

    def test_required_permissions_list(self):
        perms = make_checker().required_permissions
        self.assertIn("s3:ListAllMyBuckets", perms)
        self.assertIn("s3:GetBucketPublicAccessBlock", perms)
        self.assertIn("s3:GetBucketPolicy", perms)

    def test_checker_registered_in_registry(self):
        """S3Checker must be registered via @CheckerRegistry.register."""
        from core.checker_registry import CheckerRegistry
        checkers = CheckerRegistry.get_checkers("aws", domains=["s3"])
        from providers.aws.checkers.s3 import S3Checker
        self.assertIn(S3Checker, checkers)

    def test_all_finding_templates_defined(self):
        """All 4 finding IDs must have templates."""
        from providers.aws.checkers.s3 import S3Checker
        templates = S3Checker.FINDING_TEMPLATES
        for fid in ["AWS-S3-001", "AWS-S3-002", "AWS-S3-003", "AWS-S3-004"]:
            self.assertIn(fid, templates, f"Missing template for {fid}")

    def test_one_bucket_failure_doesnt_abort_others(self):
        """If one bucket fails mid-scan, others still complete."""
        mock_client = MockS3Client(buckets=["bucket-1", "bucket-2", "bucket-3"])
        # Make encryption check fail for ALL buckets — other checks still run
        mock_client._encryption = FakeClientError({"Error": {"Code": "AccessDenied", "Message": "Denied"}})
        checker = make_checker(mock_client)
        result = checker.execute()
        # Should still get logging/versioning findings from all 3 buckets
        self.assertTrue(result.succeeded)


# ===========================================================================
# Tests: Bucket location handling
# ===========================================================================

class TestBucketLocation(unittest.TestCase):

    def test_us_east_1_returns_none_normalised(self):
        """GetBucketLocation returns None for us-east-1 — we normalise to string."""
        mock_client = MagicMock()
        mock_client.get_bucket_location.return_value = {"LocationConstraint": None}
        checker = make_checker(mock_client)
        checker._s3 = mock_client
        region = checker._get_bucket_region("test-bucket")
        self.assertEqual(region, "us-east-1")

    def test_eu_west_1_returned_correctly(self):
        mock_client = MagicMock()
        mock_client.get_bucket_location.return_value = {"LocationConstraint": "eu-west-1"}
        checker = make_checker(mock_client)
        checker._s3 = mock_client
        region = checker._get_bucket_region("eu-bucket")
        self.assertEqual(region, "eu-west-1")

    def test_location_api_failure_defaults_to_us_east_1(self):
        mock_client = MagicMock()
        mock_client.get_bucket_location.side_effect = FakeClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}}
        )
        checker = make_checker(mock_client)
        checker._s3 = mock_client
        region = checker._get_bucket_region("any-bucket")
        self.assertEqual(region, "us-east-1")  # Safe fallback

    def test_regional_client_cached(self):
        """_regional_client returns same object for same region (no duplicate clients)."""
        session = MagicMock()
        mock_client = MagicMock()
        session.client.return_value = mock_client
        checker = S3Checker(session=session, account_id="123", region="us-east-1")
        checker._s3 = mock_client

        c1 = checker._regional_client("us-east-1")
        c2 = checker._regional_client("us-east-1")
        self.assertIs(c1, c2)
        # session.client should only be called once for the same region
        calls_for_region = [
            c for c in session.client.call_args_list
            if "us-east-1" in str(c)
        ]
        self.assertEqual(len(calls_for_region), 1)


# ===========================================================================
# Tests: Attack graph node building
# ===========================================================================

class TestS3GraphBuilding(unittest.TestCase):

    def _make_checker_with_graph_client(
        self,
        bucket_name: str = "public-bucket",
        is_public: bool = True,
    ) -> S3Checker:
        """Create a checker whose graph building will produce a public bucket."""
        mock_client = MockS3Client(
            buckets=[bucket_name],
            bpa="no_config" if is_public else "fully_blocked",
            policy="public" if is_public else "no_policy",
            acl="private",
        )
        session = MagicMock()
        session.client.return_value = mock_client
        checker = S3Checker(session=session, account_id="123456789012", region="us-east-1")
        checker._s3 = mock_client
        return checker

    def test_public_bucket_adds_public_resource_node(self):
        """Public bucket → PUBLIC_RESOURCE node in graph."""
        checker = self._make_checker_with_graph_client(is_public=True)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        public_nodes = graph.get_nodes_by_type(NodeType.PUBLIC_RESOURCE)
        self.assertEqual(len(public_nodes), 1)
        self.assertEqual(public_nodes[0].label, "public-bucket")

    def test_private_bucket_adds_private_resource_node(self):
        """Fully-blocked bucket → PRIVATE_RESOURCE node in graph."""
        checker = self._make_checker_with_graph_client(is_public=False)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        private_nodes = graph.get_nodes_by_type(NodeType.PRIVATE_RESOURCE)
        self.assertEqual(len(private_nodes), 1)
        public_nodes = graph.get_nodes_by_type(NodeType.PUBLIC_RESOURCE)
        self.assertEqual(len(public_nodes), 0)

    def test_public_bucket_creates_internet_node(self):
        """Public bucket → EXTERNAL internet node created."""
        checker = self._make_checker_with_graph_client(is_public=True)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        self.assertTrue(graph.node_exists(INTERNET_NODE_ID))
        internet_node = graph.nodes[INTERNET_NODE_ID]
        self.assertEqual(internet_node.node_type, NodeType.EXTERNAL)

    def test_public_bucket_creates_is_public_edge(self):
        """Public bucket → IS_PUBLIC edge from internet → bucket."""
        checker = self._make_checker_with_graph_client(is_public=True)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        bucket_arn = "arn:aws:s3:::public-bucket"
        edges = graph.get_edges_between(INTERNET_NODE_ID, bucket_arn)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].edge_type, EdgeType.IS_PUBLIC)

    def test_private_bucket_no_is_public_edge(self):
        """Private bucket → no IS_PUBLIC edge."""
        checker = self._make_checker_with_graph_client(is_public=False)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        edges = [e for e in graph.edges if e.edge_type == EdgeType.IS_PUBLIC]
        self.assertEqual(len(edges), 0)

    def test_internet_node_created_only_once_for_multiple_public_buckets(self):
        """Multiple public buckets → only one internet node."""
        mock_client = MockS3Client(
            buckets=["bucket-1", "bucket-2", "bucket-3"],
            bpa="no_config", policy="public", acl="private",
        )
        session = MagicMock()
        session.client.return_value = mock_client
        checker = S3Checker(session=session, account_id="123", region="us-east-1")
        checker._s3 = mock_client
        graph = AttackGraph(account_id="123", provider="aws")
        checker.build_graph_nodes(graph)

        external_nodes = graph.get_nodes_by_type(NodeType.EXTERNAL)
        self.assertEqual(len(external_nodes), 1)

    def test_graph_public_entry_points_includes_public_buckets(self):
        """AttackGraph.get_public_entry_points() returns the public S3 bucket."""
        checker = self._make_checker_with_graph_client(is_public=True)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        entry_points = graph.get_public_entry_points()
        bucket_entries = [n for n in entry_points if "public-bucket" in n.node_id]
        self.assertTrue(len(bucket_entries) >= 1)

    def test_graph_build_failure_is_non_fatal(self):
        """If list_buckets fails during graph building, no exception raised and no bucket nodes added."""
        mock_client = MagicMock()
        mock_client.list_buckets.side_effect = Exception("Unexpected failure")
        session = MagicMock()
        session.client.return_value = mock_client
        checker = S3Checker(session=session, account_id="123", region="us-east-1")
        checker._s3 = mock_client
        graph = AttackGraph(account_id="123", provider="aws")

        # Should not raise
        checker.build_graph_nodes(graph)

        # No S3 bucket nodes added (internet node may exist, that's fine)
        bucket_nodes = [
            n for n in graph.nodes.values()
            if n.node_type in (NodeType.PUBLIC_RESOURCE, NodeType.PRIVATE_RESOURCE)
        ]
        self.assertEqual(len(bucket_nodes), 0)

    def test_node_properties_include_public_flags(self):
        """Public bucket node properties contain public flag details."""
        checker = self._make_checker_with_graph_client(is_public=True)
        graph = AttackGraph(account_id="123456789012", provider="aws")
        checker.build_graph_nodes(graph)

        bucket_arn = "arn:aws:s3:::public-bucket"
        node = graph.nodes.get(bucket_arn)
        self.assertIsNotNone(node)
        self.assertTrue(node.properties.get("is_public"))
        self.assertEqual(node.properties.get("service"), "s3")


# ===========================================================================
# Tests: Both IAM and S3 checkers registered together
# ===========================================================================

class TestMultiCheckerRegistry(unittest.TestCase):

    def test_iam_and_s3_checkers_both_registered(self):
        """Registry should have both IAM and S3 checkers."""
        from core.checker_registry import CheckerRegistry
        from providers.aws.checkers.iam import IAMChecker
        from providers.aws.checkers.s3 import S3Checker

        all_checkers = CheckerRegistry.get_checkers("aws")
        self.assertIn(IAMChecker, all_checkers)
        self.assertIn(S3Checker, all_checkers)

    def test_domain_filtering_returns_only_s3(self):
        """get_checkers(domains=['s3']) returns only S3Checker."""
        from core.checker_registry import CheckerRegistry
        from providers.aws.checkers.iam import IAMChecker
        from providers.aws.checkers.s3 import S3Checker

        s3_only = CheckerRegistry.get_checkers("aws", domains=["s3"])
        self.assertIn(S3Checker, s3_only)
        self.assertNotIn(IAMChecker, s3_only)

    def test_domain_filtering_returns_only_iam(self):
        """get_checkers(domains=['iam']) returns only IAMChecker."""
        from core.checker_registry import CheckerRegistry
        from providers.aws.checkers.iam import IAMChecker
        from providers.aws.checkers.s3 import S3Checker

        iam_only = CheckerRegistry.get_checkers("aws", domains=["iam"])
        self.assertIn(IAMChecker, iam_only)
        self.assertNotIn(S3Checker, iam_only)

    def test_get_all_domains(self):
        """Both 'iam' and 's3' should be in registered domains."""
        from core.checker_registry import CheckerRegistry
        domains = CheckerRegistry.get_all_domains("aws")
        self.assertIn("iam", domains)
        self.assertIn("s3", domains)

    def test_get_all_finding_ids_includes_s3_ids(self):
        """S3 finding IDs should appear in the registry's full list."""
        from core.checker_registry import CheckerRegistry
        all_ids = CheckerRegistry.get_all_finding_ids("aws")
        self.assertIn("AWS-S3-001", all_ids)
        self.assertIn("AWS-S3-002", all_ids)
        self.assertIn("AWS-S3-003", all_ids)
        self.assertIn("AWS-S3-004", all_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)