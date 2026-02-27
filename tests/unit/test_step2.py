"""
tests/unit/test_step2.py

Unit tests for Step 2: BaseChecker, BaseProvider, CheckerRegistry, CSPMEngine.

All AWS-specific code is mocked — these tests validate contracts and
orchestration logic only, with zero real cloud API calls.

Run with:
    python tests/unit/test_step2.py
"""

from __future__ import annotations

import sys
import os
import json
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.attack_graph.models import AttackGraph
from core.base_checker import BaseChecker, CheckerResult, FindingTemplate
from core.base_provider import (
    BaseProvider, ProviderAuthError, ProviderNotAuthenticatedError,
    AccountScanError,
)
from core.checker_registry import CheckerRegistry
from core.engine import CSPMEngine, ScanConfig, CSPMScanError, CSPMConfigError
from core.models.finding import CloudProvider, Finding, Severity, FindingStatus
from core.models.scan_result import AccountResult, ScanResult


# ===========================================================================
# Concrete test implementations of the abstract classes
# ===========================================================================

class ConcreteChecker(BaseChecker):
    """Minimal concrete checker for testing BaseChecker behaviour."""

    FINDING_TEMPLATES = {
        "TEST-001": FindingTemplate(
            finding_id="TEST-001",
            title="Test Finding One",
            description_template="Resource {resource_name} has issue.",
            severity=Severity.HIGH,
            remediation_summary="Fix the issue.",
            references=["https://docs.example.com/fix"],
        ),
        "TEST-002": FindingTemplate(
            finding_id="TEST-002",
            title="Test Finding Two",
            description_template="Critical issue detected.",
            severity=Severity.CRITICAL,
            remediation_summary="Fix immediately.",
        ),
    }

    def __init__(self, session=None, account_id="123", region="us-east-1",
                 provider=CloudProvider.AWS, findings_to_return=None,
                 should_raise=None):
        super().__init__(session, account_id, region, provider)
        self._findings_to_return = findings_to_return or []
        self._should_raise = should_raise

    @property
    def checker_name(self) -> str:
        return "test"

    @property
    def required_permissions(self) -> List[str]:
        return ["test:ListResources", "test:GetResource"]

    def run(self) -> List[Finding]:
        if self._should_raise:
            raise self._should_raise
        return self._findings_to_return


class ConcreteProvider(BaseProvider):
    """Minimal concrete provider for testing BaseProvider/CSPMEngine behaviour."""

    def __init__(self, config=None, accounts=None, findings=None, auth_fails=False):
        super().__init__(config or {})
        self._accounts     = accounts or ["123456789012"]
        self._findings     = findings or []
        self._auth_fails   = auth_fails
        self._authenticated_called = False
        self._pre_hook_calls  = []
        self._post_hook_calls = []

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def supported_checks(self) -> List[str]:
        return ["TEST-001", "TEST-002"]

    def authenticate(self) -> bool:
        self._authenticated_called = True
        if self._auth_fails:
            raise ProviderAuthError("Mock auth failure")
        self._authenticated = True
        return True

    def get_accounts(self) -> List[str]:
        return self._accounts

    def run_checks(self, account_id: str) -> AccountResult:
        ar = AccountResult(account_id=account_id)
        ar.findings = list(self._findings)
        return ar

    def build_resource_graph(self, account_id: str) -> AttackGraph:
        return AttackGraph(account_id=account_id, provider="test")

    def pre_scan_hook(self, account_id: str) -> None:
        self._pre_hook_calls.append(account_id)

    def post_scan_hook(self, account_id: str, result: AccountResult) -> None:
        self._post_hook_calls.append(account_id)


# ===========================================================================
# BaseChecker tests
# ===========================================================================

class TestBaseChecker(unittest.TestCase):

    def test_execute_returns_checker_result(self):
        finding = Finding("TEST-001", "t", "d", Severity.HIGH)
        checker = ConcreteChecker(findings_to_return=[finding])
        result = checker.execute()

        self.assertIsInstance(result, CheckerResult)
        self.assertEqual(result.checker_name, "test")
        self.assertEqual(len(result.findings), 1)
        self.assertTrue(result.succeeded)
        self.assertIsNone(result.error)

    def test_execute_captures_exception_as_error(self):
        checker = ConcreteChecker(should_raise=RuntimeError("boto3 exploded"))
        result = checker.execute()

        self.assertFalse(result.succeeded)
        self.assertIsNotNone(result.error)
        self.assertIn("boto3 exploded", result.error)
        self.assertEqual(len(result.findings), 0)

    def test_execute_records_timing(self):
        checker = ConcreteChecker()
        result = checker.execute()
        self.assertGreaterEqual(result.duration_ms, 0)

    def test_finding_factory_creates_correct_finding(self):
        checker = ConcreteChecker(account_id="999")
        finding = checker._finding(
            finding_id="TEST-001",
            resource_id="arn:aws:iam::999:user/alice",
            resource_name="alice",
            resource_type="AWS::IAM::User",
            raw_evidence={"user": "alice"},
        )

        self.assertEqual(finding.finding_id, "TEST-001")
        self.assertEqual(finding.title, "Test Finding One")
        self.assertEqual(finding.severity, Severity.HIGH)
        self.assertEqual(finding.account_id, "999")
        self.assertEqual(finding.resource_name, "alice")
        self.assertEqual(finding.raw_evidence, {"user": "alice"})
        self.assertEqual(finding.status, FindingStatus.OPEN)
        self.assertEqual(finding.references, ["https://docs.example.com/fix"])

    def test_finding_factory_raises_on_unknown_id(self):
        checker = ConcreteChecker()
        with self.assertRaises(ValueError) as ctx:
            checker._finding(
                finding_id="NONEXISTENT-999",
                resource_id="arn:...",
                resource_name="x",
                resource_type="AWS::IAM::User",
            )
        self.assertIn("NONEXISTENT-999", str(ctx.exception))

    def test_severity_override(self):
        checker = ConcreteChecker()
        finding = checker._finding(
            finding_id="TEST-001",
            resource_id="arn:...",
            resource_name="x",
            resource_type="AWS::IAM::User",
            severity_override=Severity.CRITICAL,
        )
        self.assertEqual(finding.severity, Severity.CRITICAL)

    def test_description_override(self):
        checker = ConcreteChecker()
        finding = checker._finding(
            finding_id="TEST-001",
            resource_id="arn:...",
            resource_name="x",
            resource_type="AWS::IAM::User",
            description_override="Custom description here.",
        )
        self.assertEqual(finding.description, "Custom description here.")

    def test_api_call_tracking(self):
        checker = ConcreteChecker()
        self.assertEqual(checker._api_call_count, 0)
        checker._track_api_call()
        checker._track_api_call()
        self.assertEqual(checker._api_call_count, 2)

    def test_repr(self):
        checker = ConcreteChecker(account_id="123", region="eu-west-1")
        r = repr(checker)
        self.assertIn("123", r)
        self.assertIn("eu-west-1", r)


# ===========================================================================
# BaseProvider tests
# ===========================================================================

class TestBaseProvider(unittest.TestCase):

    def test_require_auth_raises_before_authentication(self):
        provider = ConcreteProvider()
        with self.assertRaises(ProviderNotAuthenticatedError):
            provider._require_auth()

    def test_require_auth_passes_after_authentication(self):
        provider = ConcreteProvider()
        provider.authenticate()
        provider._require_auth()  # Should not raise

    def test_is_authenticated_property(self):
        provider = ConcreteProvider()
        self.assertFalse(provider.is_authenticated)
        provider.authenticate()
        self.assertTrue(provider.is_authenticated)

    def test_auth_failure_raises_provider_auth_error(self):
        provider = ConcreteProvider(auth_fails=True)
        with self.assertRaises(ProviderAuthError):
            provider.authenticate()

    def test_validate_config_default_returns_empty(self):
        provider = ConcreteProvider(config={"key": "value"})
        errors = provider.validate_config()
        self.assertEqual(errors, [])

    def test_repr(self):
        provider = ConcreteProvider()
        r = repr(provider)
        self.assertIn("test", r)
        self.assertIn("False", r)  # authenticated=False initially


# ===========================================================================
# CheckerRegistry tests
# ===========================================================================

class TestCheckerRegistry(unittest.TestCase):

    def setUp(self):
        # Clear registry between tests to prevent cross-test pollution
        CheckerRegistry.clear()

    def test_register_decorator_adds_checker(self):
        @CheckerRegistry.register(provider="test", domain="iam")
        class MyIAMChecker(ConcreteChecker):
            pass

        checkers = CheckerRegistry.get_checkers("test")
        self.assertIn(MyIAMChecker, checkers)

    def test_domain_filter_returns_only_matching_checkers(self):
        @CheckerRegistry.register(provider="test", domain="iam")
        class IAMChecker(ConcreteChecker):
            pass

        @CheckerRegistry.register(provider="test", domain="s3")
        class S3Checker(ConcreteChecker):
            pass

        iam_only = CheckerRegistry.get_checkers("test", domains=["iam"])
        self.assertIn(IAMChecker, iam_only)
        self.assertNotIn(S3Checker, iam_only)

    def test_provider_isolation(self):
        @CheckerRegistry.register(provider="aws", domain="iam")
        class AWSIAMChecker(ConcreteChecker):
            pass

        @CheckerRegistry.register(provider="azure", domain="iam")
        class AzureIAMChecker(ConcreteChecker):
            pass

        aws_checkers   = CheckerRegistry.get_checkers("aws")
        azure_checkers = CheckerRegistry.get_checkers("azure")

        self.assertIn(AWSIAMChecker, aws_checkers)
        self.assertNotIn(AzureIAMChecker, aws_checkers)

        self.assertIn(AzureIAMChecker, azure_checkers)
        self.assertNotIn(AWSIAMChecker, azure_checkers)

    def test_double_registration_is_idempotent(self):
        @CheckerRegistry.register(provider="test", domain="iam")
        class MyChecker(ConcreteChecker):
            pass

        # Register again — should not create a duplicate
        CheckerRegistry.register(provider="test", domain="iam")(MyChecker)

        checkers = CheckerRegistry.get_checkers("test", domains=["iam"])
        self.assertEqual(checkers.count(MyChecker), 1)

    def test_get_all_domains(self):
        @CheckerRegistry.register(provider="test", domain="iam")
        class C1(ConcreteChecker): pass

        @CheckerRegistry.register(provider="test", domain="s3")
        class C2(ConcreteChecker): pass

        @CheckerRegistry.register(provider="test", domain="logging")
        class C3(ConcreteChecker): pass

        domains = CheckerRegistry.get_all_domains("test")
        self.assertIn("iam", domains)
        self.assertIn("s3", domains)
        self.assertIn("logging", domains)

    def test_list_checkers_output(self):
        @CheckerRegistry.register(provider="test", domain="iam")
        class MyIAMChecker(ConcreteChecker):
            FINDING_TEMPLATES = {
                "T-001": FindingTemplate("T-001", "T", "D", Severity.HIGH, "R"),
            }

        listing = CheckerRegistry.list_checkers("test")
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["domain"], "iam")
        self.assertEqual(listing[0]["finding_count"], 1)
        self.assertIn("T-001", listing[0]["finding_ids"])

    def test_unknown_provider_returns_empty_list(self):
        checkers = CheckerRegistry.get_checkers("nonexistent_cloud")
        self.assertEqual(checkers, [])

    def tearDown(self):
        CheckerRegistry.clear()


# ===========================================================================
# CSPMEngine tests
# ===========================================================================

class TestCSPMEngine(unittest.TestCase):

    def setUp(self):
        CheckerRegistry.clear()

    def _make_finding(self, severity=Severity.HIGH):
        return Finding(
            finding_id="TEST-001",
            title="Test",
            description="Test finding",
            severity=severity,
            account_id="123456789012",
        )

    def test_basic_scan_returns_scan_result(self):
        provider = ConcreteProvider()
        engine   = CSPMEngine(provider)
        result   = engine.scan()

        self.assertIsInstance(result, ScanResult)
        self.assertTrue(result.is_finalised)
        self.assertEqual(result.provider, "test")

    def test_scan_calls_authenticate(self):
        provider = ConcreteProvider()
        engine   = CSPMEngine(provider)
        engine.scan()
        self.assertTrue(provider._authenticated_called)

    def test_scan_calls_pre_and_post_hooks(self):
        provider = ConcreteProvider(accounts=["111", "222"])
        engine   = CSPMEngine(provider)
        engine.scan()

        self.assertIn("111", provider._pre_hook_calls)
        self.assertIn("222", provider._pre_hook_calls)
        self.assertIn("111", provider._post_hook_calls)
        self.assertIn("222", provider._post_hook_calls)

    def test_scan_aggregates_findings_from_all_accounts(self):
        finding = self._make_finding()
        provider = ConcreteProvider(
            accounts=["111111111111", "222222222222"],
            findings=[finding],
        )
        engine = CSPMEngine(provider)
        result = engine.scan()

        # Each account has 1 finding → 2 total
        self.assertEqual(len(result.all_findings), 2)

    def test_scan_computes_score_per_account(self):
        finding = self._make_finding(severity=Severity.HIGH)  # -20 points
        provider = ConcreteProvider(findings=[finding])
        engine   = CSPMEngine(provider)
        result   = engine.scan()

        account_result = result.account_results[0]
        self.assertIsNotNone(account_result.score)
        self.assertEqual(account_result.score.score, 80)
        self.assertEqual(account_result.score.grade, "B")

    def test_scan_raises_on_auth_failure(self):
        provider = ConcreteProvider(auth_fails=True)
        engine   = CSPMEngine(provider)

        with self.assertRaises(CSPMScanError):
            engine.scan()

    def test_config_accounts_overrides_provider(self):
        """If ScanConfig.accounts is set, provider.get_accounts() is not called."""
        provider = ConcreteProvider(accounts=["should-not-scan"])
        config   = ScanConfig(accounts=["111111111111"])
        engine   = CSPMEngine(provider, config)
        result   = engine.scan()

        scanned_ids = [ar.account_id for ar in result.account_results]
        self.assertIn("111111111111", scanned_ids)
        self.assertNotIn("should-not-scan", scanned_ids)

    def test_skip_graph_flag(self):
        """When skip_graph=True, build_resource_graph should not be called."""
        provider = ConcreteProvider()
        provider.build_resource_graph = MagicMock(
            return_value=AttackGraph(account_id="123")
        )
        config = ScanConfig(skip_graph=True)
        engine = CSPMEngine(provider, config)
        engine.scan()

        provider.build_resource_graph.assert_not_called()

    def test_account_scan_error_is_captured_not_raised(self):
        """
        If run_checks raises AccountScanError for one account,
        the scan continues and records the error — it does NOT abort.
        """
        provider = ConcreteProvider(accounts=["good-account", "bad-account"])

        original_run_checks = provider.run_checks
        def run_checks_with_failure(account_id):
            if account_id == "bad-account":
                raise AccountScanError("bad-account", "assume-role failed")
            return original_run_checks(account_id)

        provider.run_checks = run_checks_with_failure
        engine = CSPMEngine(provider)
        result = engine.scan()  # Should NOT raise

        self.assertEqual(len(result.account_results), 2)
        bad = next(ar for ar in result.account_results if ar.account_id == "bad-account")
        self.assertTrue(len(bad.errors) > 0)

    def test_scan_result_is_finalised(self):
        provider = ConcreteProvider()
        engine   = CSPMEngine(provider)
        result   = engine.scan()
        self.assertTrue(result.is_finalised)
        self.assertIsNotNone(result.scan_end)

    def test_scan_result_is_json_serialisable(self):
        provider = ConcreteProvider(findings=[self._make_finding()])
        engine   = CSPMEngine(provider)
        result   = engine.scan()
        d = result.to_dict()
        json_str = json.dumps(d)  # Should not raise
        self.assertGreater(len(json_str), 0)

    def test_scan_config_defaults(self):
        config = ScanConfig()
        self.assertIsNone(config.accounts)
        self.assertIsNone(config.domains)
        self.assertFalse(config.parallel)
        self.assertFalse(config.skip_graph)
        self.assertEqual(config.triggered_by, "manual")

    def test_multi_account_overall_score_is_average(self):
        """Overall score = average of account scores."""
        # Account 1: 1x HIGH → 80 points
        # Account 2: 1x CRITICAL → 60 points
        # Average: 70 → grade C
        findings_account_1 = [self._make_finding(Severity.HIGH)]
        findings_account_2 = [self._make_finding(Severity.CRITICAL)]

        call_count = [0]
        provider = ConcreteProvider(accounts=["acct-1", "acct-2"])

        def run_checks(account_id):
            ar = AccountResult(account_id=account_id)
            if account_id == "acct-1":
                ar.findings = findings_account_1
            else:
                ar.findings = findings_account_2
            return ar

        provider.run_checks = run_checks
        engine = CSPMEngine(provider)
        result = engine.scan()

        self.assertEqual(result.total_score, 70)
        self.assertEqual(result.overall_grade, "C")

    def tearDown(self):
        CheckerRegistry.clear()


# ===========================================================================
# Integration: engine + checker registry working together
# ===========================================================================

class TestEngineWithRegistry(unittest.TestCase):

    def setUp(self):
        CheckerRegistry.clear()

    def test_engine_runs_registered_checkers_via_provider(self):
        """
        This test validates the full flow:
        CSPMEngine → AWSProvider.run_checks() → registered checkers
        """

        # Register a test checker
        @CheckerRegistry.register(provider="integration-test", domain="iam")
        class IntegrationIAMChecker(ConcreteChecker):
            def run(self):
                return [self._finding(
                    finding_id="TEST-001",
                    resource_id="arn:aws:iam::123:user/alice",
                    resource_name="alice",
                    resource_type="AWS::IAM::User",
                )]

        class IntegrationProvider(ConcreteProvider):
            @property
            def provider_name(self):
                return "integration-test"

            def run_checks(self, account_id):
                ar = AccountResult(account_id=account_id)
                checker_classes = CheckerRegistry.get_checkers("integration-test")
                for cls in checker_classes:
                    checker = cls(session=None, account_id=account_id)
                    result = checker.execute()
                    ar.findings.extend(result.findings)
                return ar

        provider = IntegrationProvider()
        engine   = CSPMEngine(provider)
        result   = engine.scan()

        self.assertEqual(len(result.all_findings), 1)
        self.assertEqual(result.all_findings[0].finding_id, "TEST-001")
        self.assertEqual(result.all_findings[0].resource_name, "alice")

    def tearDown(self):
        CheckerRegistry.clear()


# ===========================================================================
# Convenience runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)