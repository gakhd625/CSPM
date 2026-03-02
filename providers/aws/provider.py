"""
providers/aws/provider.py

AWSProvider implements BaseProvider for AWS.

Step 3 update: AWSSession and AWSOrganizations are now fully wired in.
  - authenticate() delegates to AWSSession.initialise()
  - get_accounts() uses AWSOrganizations for org-wide scans
  - pre_scan_hook() uses AWSSession.get_session_for_account() to assume roles
  - get_account_name() uses AWSOrganizations cache for friendly names
  - get_regions() enumerates enabled regions via AWSSession

Step completion status:
  - Step 2: Provider skeleton
  - Step 3: AWSSession + AWSOrganizations wired in <- YOU ARE HERE
  - Step 4: IAMChecker wired in
  - Step 5: S3Checker wired in
  - Step 6: LoggingChecker wired in
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.attack_graph.models import AttackGraph
from core.base_checker import BaseChecker, CheckerResult
from core.base_provider import (
    AccountScanError, BaseProvider, ProviderAuthError,
)
from core.checker_registry import CheckerRegistry
from core.models.finding import CloudProvider
from core.models.scan_result import AccountResult

from providers.aws.session import AWSSession, AWSSessionError, AssumeRoleError
from providers.aws.organizations import AWSOrganizations, OrganizationsError

logger = logging.getLogger(__name__)


class AWSProvider(BaseProvider):
    """
    AWS implementation of BaseProvider.

    Config keys:
        profile          (str)   AWS profile name from ~/.aws/credentials
        regions          (list)  Regions to scan. None = all enabled regions.
        role_name        (str)   Role to assume in member accounts
                                 Default: "SecurityAudit"
        external_id      (str)   External ID for assume-role (optional, recommended)
        session_duration (int)   Seconds for assumed role sessions (default: 3600)
        scan_org         (bool)  True = enumerate and scan all Organizations accounts
        account_ids      (list)  Explicit account IDs to scan (overrides scan_org)
        include_suspended (bool) Include suspended accounts in org scans (default: False)
        domains          (list)  Check domains to run. None = all. e.g. ["iam", "s3"]
        mfa_serial       (str)   MFA device ARN (dev/manual use)
        mfa_token        (str)   Current MFA TOTP code (dev/manual use)
    """

    DEFAULT_AUDIT_ROLE = "SecurityAudit"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self._aws_session: Optional[AWSSession] = None
        self._organizations: Optional[AWSOrganizations] = None
        self._account_names: Dict[str, str] = {}

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
        duration = self.config.get("session_duration")
        if duration is not None:
            if not isinstance(duration, int) or not (900 <= duration <= 43200):
                errors.append(
                    "config.session_duration must be an integer between 900 and 43200 seconds"
                )
        return errors

    def authenticate(self) -> bool:
        """
        Initialise AWSSession and validate credentials via sts:GetCallerIdentity.
        Delegates all credential logic to AWSSession.
        """
        try:
            self._aws_session = AWSSession(config=self.config)
            identity = self._aws_session.initialise()
            self.logger.info(
                f"AWS authentication successful: "
                f"account={identity['Account']}, arn={identity['Arn']}"
            )
            self._authenticated = True
            return True
        except AWSSessionError as e:
            raise ProviderAuthError(str(e)) from e

    def get_accounts(self) -> List[str]:
        """
        Return the list of account IDs to scan.

        Priority:
          1. config.account_ids (explicit list)
          2. config.scan_org = True -> enumerate via AWSOrganizations
          3. Default -> scan only the authenticated account
        """
        self._require_auth()

        if self.config.get("account_ids"):
            account_ids = self.config["account_ids"]
            self.logger.info(f"Using explicit account list: {account_ids}")
            return account_ids

        if self.config.get("scan_org"):
            return self._get_org_accounts()

        home_account = self._aws_session.home_account_id
        self.logger.info(f"Single-account scan: {home_account}")
        return [home_account]

    def run_checks(self, account_id: str) -> AccountResult:
        """Run all registered AWS checkers for one account."""
        self._require_auth()

        account_result = AccountResult(account_id=account_id)
        session = self._get_session_for_account(account_id)
        domains = self.config.get("domains")
        checker_classes = CheckerRegistry.get_checkers(provider="aws", domains=domains)

        if not checker_classes:
            self.logger.warning("No AWS checkers registered.")
            return account_result

        self.logger.info(
            f"Running {len(checker_classes)} checker(s) for account {account_id}..."
        )

        for checker_cls in checker_classes:
            checker = self._instantiate_checker(checker_cls, session, account_id)
            if checker is None:
                continue
            result: CheckerResult = checker.execute()
            account_result.findings.extend(result.findings)
            if result.error:
                account_result.add_error(service=checker.checker_name, message=result.error)
            if result.warning:
                account_result.add_error(
                    service=checker.checker_name, message=f"Warning: {result.warning}"
                )

        self.logger.info(
            f"Checks complete for {account_id}: "
            f"{len(account_result.findings)} findings, "
            f"{len(account_result.errors)} errors/warnings"
        )
        return account_result

    def build_resource_graph(self, account_id: str) -> AttackGraph:
        """STUB - fully implemented in Step 7."""
        return AttackGraph(account_id=account_id, provider="aws")

    # ------------------------------------------------------------------
    # BaseProvider: optional overrides
    # ------------------------------------------------------------------

    def get_account_name(self, account_id: str) -> str:
        """Return a human-readable account name from Organizations or IAM alias."""
        if account_id in self._account_names:
            return self._account_names[account_id]
        if self._organizations:
            name = self._organizations.get_account_name(account_id)
            if name != account_id:
                self._account_names[account_id] = name
                return name
        if self._aws_session and account_id == self._aws_session.home_account_id:
            alias = self._aws_session.get_account_alias()
            if alias:
                self._account_names[account_id] = alias
                return alias
        return account_id

    def get_regions(self) -> List[str]:
        """Return configured regions, or enumerate all enabled regions."""
        configured = self.config.get("regions")
        if configured:
            return configured
        if self._aws_session and self._aws_session._base_session:
            return self._aws_session.get_enabled_regions()
        return ["us-east-1"]

    def supports_multi_account(self) -> bool:
        return bool(self.config.get("scan_org") or self.config.get("account_ids"))

    def pre_scan_hook(self, account_id: str) -> None:
        """Assume cross-account role before scanning a member account."""
        if not self._aws_session:
            return
        home = self._aws_session.home_account_id
        if account_id == home:
            return
        self.logger.info(f"Pre-scan: assuming role in {account_id}")
        try:
            self._aws_session.get_session_for_account(account_id)
            self.logger.info(f"Successfully assumed role in {account_id}")
        except AssumeRoleError as e:
            raise AccountScanError(account_id, str(e)) from e
        except Exception as e:
            raise AccountScanError(account_id, f"Unexpected error in pre_scan_hook: {e}") from e

    def post_scan_hook(self, account_id: str, result: AccountResult) -> None:
        self.logger.debug(f"Post-scan for {account_id}: {len(result.findings)} findings")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_org_accounts(self) -> List[str]:
        """Enumerate accounts via AWSOrganizations."""
        try:
            if not self._organizations:
                self._organizations = AWSOrganizations(
                    session=self._aws_session._base_session
                )
            if not self._organizations.is_enabled():
                home = self._aws_session.home_account_id
                self.logger.warning(
                    "scan_org=True but Organizations not enabled. "
                    f"Falling back to single-account: {home}"
                )
                return [home]

            include_suspended = self.config.get("include_suspended", False)
            accounts = self._organizations.list_accounts(
                include_suspended=include_suspended
            )
            for org_acct in self._organizations.list_account_objects(
                include_suspended=include_suspended
            ):
                self._account_names[org_acct.account_id] = org_acct.name

            self.logger.info(f"Organizations: {len(accounts)} accounts to scan")
            return accounts
        except OrganizationsError as e:
            home = self._aws_session.home_account_id
            self.logger.error(f"Org enumeration failed: {e}. Falling back to single-account.")
            return [home]

    def _get_session_for_account(self, account_id: str) -> Any:
        """Return a boto3 Session for the target account (uses cache)."""
        if not self._aws_session:
            raise RuntimeError("AWSSession not initialised")
        try:
            return self._aws_session.get_session_for_account(account_id)
        except AssumeRoleError as e:
            raise AccountScanError(account_id, str(e)) from e

    def _instantiate_checker(
        self, checker_cls, session: Any, account_id: str
    ) -> Optional[BaseChecker]:
        """Safely instantiate a checker class."""
        try:
            return checker_cls(
                session=session,
                account_id=account_id,
                region=self.get_regions()[0],
                provider=CloudProvider.AWS,
            )
        except Exception as e:
            self.logger.error(f"Failed to instantiate {checker_cls.__name__}: {e}")
            return None

    @property
    def organizations_info(self) -> Optional[Any]:
        """Return OrganizationInfo if an org scan was run, else None."""
        if self._organizations:
            try:
                return self._organizations.describe()
            except Exception:
                return None
        return None

    def __repr__(self) -> str:
        home = self._aws_session.home_account_id if self._aws_session else "not-initialised"
        return (
            f"AWSProvider(account={home!r}, "
            f"authenticated={self._authenticated}, "
            f"org_scan={self.config.get('scan_org', False)})"
        )