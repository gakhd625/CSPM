"""
providers/aws/provider.py

AWSProvider implements BaseProvider for AWS.

This is the AWS-specific implementation of the provider contract.
The engine calls BaseProvider methods — it never imports this file directly.

Architecture:
    CSPMEngine
        ↓ calls BaseProvider interface
    AWSProvider
        ↓ delegates to
    [IAMChecker, S3Checker, LoggingChecker, ...]   ← wired in via CheckerRegistry
    AWSSession                                       ← handles credentials + assume-role
    AWSOrganizations                                 ← handles multi-account discovery

What lives here:
  - Authentication (validates boto3 session via sts:GetCallerIdentity)
  - Account discovery (single account or full Organizations enumeration)
  - Checker orchestration (runs all registered AWS checkers)
  - Attack graph construction (aggregates IAM + resource data)

What does NOT live here:
  - Any check logic (that's in providers/aws/checkers/)
  - Any scoring logic (that's in core/scoring/)
  - Any reporting logic (that's in reporting/)

Step completion status:
  - Step 2: Provider skeleton ← YOU ARE HERE
  - Step 3: AWSSession + AWSOrganizations
  - Step 4: IAMChecker wired in
  - Step 5: S3Checker wired in
  - Step 6: LoggingChecker wired in
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.attack_graph.models import AttackGraph, EdgeType, GraphEdge, GraphNode, NodeType
from core.base_checker import BaseChecker, CheckerResult
from core.base_provider import (
    AccountScanError, BaseProvider, ProviderAuthError, ProviderPermissionError,
)
from core.checker_registry import CheckerRegistry
from core.models.finding import CloudProvider
from core.models.scan_result import AccountResult

logger = logging.getLogger(__name__)


class AWSProvider(BaseProvider):
    """
    AWS implementation of BaseProvider.

    Config keys:
        profile      (str)  AWS profile name from ~/.aws/credentials
        regions      (list) Regions to scan. Default: ["us-east-1"]
        role_name    (str)  Role to assume in member accounts (default: SecurityAudit)
        external_id  (str)  External ID for assume-role (optional)
        scan_org     (bool) True = enumerate and scan all Organizations accounts
        account_ids  (list) Explicit account IDs to scan (overrides scan_org)
    """

    # The role assumed in member accounts during Organizations scans.
    # Standard convention — most AWS customers have this role.
    DEFAULT_AUDIT_ROLE = "SecurityAudit"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self._session = None          # Set by authenticate()
        self._identity = {}           # Result of sts:GetCallerIdentity
        self._account_list: List[str] = []

        # Lazy imports — boto3 only required when actually running AWS scans
        self._boto3 = None

    # ------------------------------------------------------------------
    # BaseProvider: required methods
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "aws"

    @property
    def supported_checks(self) -> List[str]:
        return CheckerRegistry.get_all_finding_ids("aws")

    def validate_config(self) -> List[str]:
        errors = []
        regions = self.config.get("regions", [])
        if regions and not isinstance(regions, list):
            errors.append("config.regions must be a list of region strings")
        return errors

    def authenticate(self) -> bool:
        """
        Validate AWS credentials using sts:GetCallerIdentity.
        Raises ProviderAuthError on hard failure.

        Uses the profile from config if provided, otherwise falls back
        to the default boto3 credential chain (env vars, instance profile, etc.)
        """
        try:
            boto3 = self._import_boto3()
            profile = self.config.get("profile")

            if profile:
                self._session = boto3.Session(profile_name=profile)
                self.logger.info(f"Using AWS profile: {profile!r}")
            else:
                self._session = boto3.Session()
                self.logger.info("Using default AWS credential chain")

            # Validate credentials with a lightweight API call
            sts = self._session.client("sts")
            self._identity = sts.get_caller_identity()

            account_id = self._identity["Account"]
            arn = self._identity["Arn"]
            self.logger.info(
                f"AWS authentication successful: "
                f"account={account_id}, arn={arn}"
            )

            self._authenticated = True
            return True

        except ImportError:
            raise ProviderAuthError(
                "boto3 is not installed. Run: pip install boto3"
            )
        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code in ("InvalidClientTokenId", "AuthFailure", "ExpiredToken"):
                raise ProviderAuthError(
                    f"AWS credentials are invalid or expired: {e}"
                )
            raise ProviderAuthError(f"AWS authentication failed: {e}") from e

    def get_accounts(self) -> List[str]:
        """
        Return the list of accounts to scan.

        Behaviour:
          - config.account_ids set → use that explicit list
          - config.scan_org = True → enumerate all Organizations accounts
          - Neither → scan only the authenticated account
        """
        self._require_auth()

        # Explicit list takes priority
        if self.config.get("account_ids"):
            self.logger.info(
                f"Using explicit account list: {self.config['account_ids']}"
            )
            self._account_list = self.config["account_ids"]
            return self._account_list

        # Organizations enumeration
        if self.config.get("scan_org"):
            self.logger.info("Organizations scan enabled — enumerating accounts...")
            # AWSOrganizations implemented in Step 3
            try:
                from providers.aws.organizations import AWSOrganizations
                orgs = AWSOrganizations(self._session)
                self._account_list = orgs.list_accounts()
                self.logger.info(
                    f"Found {len(self._account_list)} accounts in Organizations"
                )
                return self._account_list
            except Exception as e:
                self.logger.warning(
                    f"Organizations enumeration failed: {e}. "
                    f"Falling back to single-account scan."
                )

        # Default: scan only the authenticated account
        current_account = self._identity["Account"]
        self.logger.info(f"Single-account scan: {current_account}")
        self._account_list = [current_account]
        return self._account_list

    def run_checks(self, account_id: str) -> AccountResult:
        """
        Run all registered AWS checkers for one account.

        Returns AccountResult populated with findings.
        Non-fatal checker errors are recorded in AccountResult.errors.
        """
        self._require_auth()

        account_result = AccountResult(account_id=account_id)
        session = self._get_session_for_account(account_id)

        # Get all registered checker classes for AWS
        domains = self.config.get("domains")  # None = all domains
        checker_classes = CheckerRegistry.get_checkers(
            provider="aws", domains=domains
        )

        if not checker_classes:
            self.logger.warning(
                "No AWS checkers registered. "
                "Have you imported the checker modules?"
            )
            return account_result

        self.logger.info(
            f"Running {len(checker_classes)} checker(s) "
            f"for account {account_id}..."
        )

        for checker_cls in checker_classes:
            checker = self._instantiate_checker(
                checker_cls, session, account_id
            )
            if checker is None:
                continue

            result: CheckerResult = checker.execute()

            # Findings: add to account result
            account_result.findings.extend(result.findings)

            # Errors and warnings: surface to account result
            if result.error:
                account_result.add_error(
                    service=checker.checker_name,
                    message=result.error,
                )
            if result.warning:
                account_result.add_error(
                    service=checker.checker_name,
                    message=f"Warning: {result.warning}",
                )

        self.logger.info(
            f"All checks complete for {account_id}: "
            f"{len(account_result.findings)} total findings, "
            f"{len(account_result.errors)} errors/warnings"
        )
        return account_result

    def build_resource_graph(self, account_id: str) -> AttackGraph:
        """
        Construct the IAM/resource graph for attack path analysis.

        STUB — populated progressively as checkers are implemented:
          Step 4 (IAM checker):      adds USER, ROLE, GROUP, POLICY nodes + edges
          Step 5 (S3 checker):       adds PUBLIC_RESOURCE nodes
          Step 6 (Logging checker):  adds metadata to existing nodes
          Step 7 (Attack graph):     full graph build + traversal

        For now: returns an empty graph so the engine pipeline works.
        """
        self.logger.debug(
            f"build_resource_graph called for {account_id} "
            f"— STUB: returning empty graph until Step 4+"
        )
        return AttackGraph(account_id=account_id, provider="aws")

    # ------------------------------------------------------------------
    # BaseProvider: optional overrides
    # ------------------------------------------------------------------

    def get_account_name(self, account_id: str) -> str:
        """
        Try to get a human-readable account name.
        For Organizations scans: use the Organizations account name.
        For single-account scans: use the IAM account alias if set.
        """
        # Step 3 will populate this from Organizations/IAM account aliases
        return account_id

    def get_regions(self) -> List[str]:
        """
        Return configured regions, or a sensible default.
        Step 4+ will use this when checkers need to iterate regions.
        """
        return self.config.get("regions", ["us-east-1"])

    def supports_multi_account(self) -> bool:
        return bool(self.config.get("scan_org") or self.config.get("account_ids"))

    def pre_scan_hook(self, account_id: str) -> None:
        """
        If this is an Organizations scan and account_id != our own account,
        assume the cross-account role. Step 3 will implement assume-role.
        """
        current = self._identity.get("Account", "")
        if account_id != current and self.config.get("scan_org"):
            self.logger.info(
                f"Cross-account scan: will assume role in {account_id} "
                f"(assume-role implemented in Step 3)"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session_for_account(self, account_id: str):
        """
        Return a boto3 session configured for the target account.
        - Same account as authenticated: return self._session
        - Different account: assume cross-account role (Step 3)
        """
        current_account = self._identity.get("Account", "")
        if account_id == current_account:
            return self._session

        # Cross-account: assume role (Step 3 implements AWSSession.assume_role)
        self.logger.info(
            f"Cross-account session for {account_id} — "
            f"assume-role implemented in Step 3"
        )
        return self._session  # Temporary: use current session until Step 3

    def _instantiate_checker(
        self,
        checker_cls,
        session,
        account_id: str,
    ) -> Optional[BaseChecker]:
        """
        Safely instantiate a checker class.
        Returns None if instantiation fails (don't abort the whole scan).
        """
        try:
            return checker_cls(
                session=session,
                account_id=account_id,
                region=self.get_regions()[0],  # Primary region
                provider=CloudProvider.AWS,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to instantiate {checker_cls.__name__}: {e}"
            )
            return None

    def _import_boto3(self):
        """Lazy boto3 import with a helpful error message."""
        if self._boto3 is None:
            try:
                import boto3
                self._boto3 = boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for the AWS provider. "
                    "Install it with: pip install boto3"
                )
        return self._boto3