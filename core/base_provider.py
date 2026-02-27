"""
core/base_provider.py

BaseProvider is the contract that every cloud provider (AWS, Azure, GCP)
must satisfy. The CSPMEngine only ever calls methods on BaseProvider —
it never imports boto3, azure-sdk, or google-cloud directly.

This is the primary abstraction boundary in the entire system.
If this interface is right, adding Azure later is just:
    1. Create providers/azure/provider.py
    2. Implement AzureProvider(BaseProvider)
    3. Done — the engine, scorer, and reporters need zero changes.

Design principles:
  - Methods are coarse-grained (run_checks, build_resource_graph) so the
    provider owns the parallelism and retry strategy internally
  - authenticate() is separate from __init__ so we can construct the
    provider, validate config, then authenticate as a distinct step —
    useful for dry-run mode and testing
  - get_accounts() returns ["single-account-id"] for non-Org scans and
    a full list for Organizations scans — the engine treats them identically
  - All methods that can partially fail take account_id so the engine can
    isolate failures per account
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.attack_graph.models import AttackGraph
from core.models.finding import Finding
from core.models.scan_result import AccountResult

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Abstract base class for all cloud providers.

    Concrete implementations: AWSProvider, AzureProvider (future), GCPProvider (future)

    The engine interacts exclusively with this interface.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Provider-specific configuration dict.
                    For AWS: {"profile": "prod", "regions": ["us-east-1"], ...}
                    For Azure: {"subscription_id": "...", "tenant_id": "..."}
                    For GCP: {"project_id": "...", "credentials_file": "..."}
        """
        self.config = config
        self._authenticated = False
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Required: every provider MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def authenticate(self) -> bool:
        """
        Validate credentials and establish sessions.

        Called once before any scanning begins. Should raise
        ProviderAuthError on hard failures (invalid creds).
        Returns True on success, False on soft failures (partial access).

        AWS implementation: validates boto3 session, calls sts:GetCallerIdentity
        Azure implementation: validates service principal / managed identity
        GCP implementation: validates application default credentials
        """
        ...

    @abstractmethod
    def get_accounts(self) -> List[str]:
        """
        Return the list of account IDs to scan.

        Single-account scan: returns [current_account_id]
        AWS Organizations scan: returns all member account IDs
        Azure: returns subscription IDs
        GCP: returns project IDs

        This is always called before run_checks so the engine knows
        how many accounts it's dealing with.
        """
        ...

    @abstractmethod
    def run_checks(self, account_id: str) -> AccountResult:
        """
        Execute all security checks for one account.

        The provider is responsible for:
          - Iterating over all registered checkers
          - Handling per-checker exceptions (don't let one checker kill the scan)
          - Setting up the correct session/credentials for this account
          - Populating AccountResult.errors for non-fatal failures

        Returns an AccountResult with all findings populated.
        The score field is NOT populated here — the engine does that.
        """
        ...

    @abstractmethod
    def build_resource_graph(self, account_id: str) -> AttackGraph:
        """
        Construct the attack graph for one account.

        This is a separate method from run_checks because:
          1. Graph building requires data from MULTIPLE services (IAM + S3 + etc.)
          2. It may run after all checks complete (needs finding data to annotate nodes)
          3. Future: graph building could be skipped with a --no-graph flag

        The provider populates nodes and edges. The AttackGraphAnalyzer
        (called by the engine) does the traversal and finds paths.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Lowercase provider identifier: "aws", "azure", "gcp"
        Used in Finding.provider and report headers.
        """
        ...

    @property
    @abstractmethod
    def supported_checks(self) -> List[str]:
        """
        List of check IDs this provider implements.
        Used for capability reporting and selective scanning (--checks iam,s3).
        Example: ["AWS-IAM-001", "AWS-IAM-002", "AWS-S3-001", ...]
        """
        ...

    # ------------------------------------------------------------------
    # Optional: providers CAN override these for richer behaviour
    # ------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        """
        Validate the config dict before authentication.
        Returns a list of error strings (empty = valid).

        Override this to add provider-specific config validation.
        The engine calls this before authenticate().
        """
        return []

    def get_account_name(self, account_id: str) -> str:
        """
        Return a human-readable name for an account ID.
        Default: return the account ID itself.
        AWS override: look up the account alias or Organizations account name.
        """
        return account_id

    def get_regions(self) -> List[str]:
        """
        Return the list of regions to scan.
        Default: provider-specific default regions.
        AWS override: reads config["regions"] or all enabled regions.
        """
        return self.config.get("regions", [])

    def supports_multi_account(self) -> bool:
        """
        True if this provider can scan multiple accounts in one run.
        AWS override: True when Organizations is enabled.
        """
        return False

    def pre_scan_hook(self, account_id: str) -> None:
        """
        Called by the engine before scanning each account.
        Use for: assuming cross-account roles, switching subscriptions, etc.
        Default: no-op.
        """
        pass

    def post_scan_hook(self, account_id: str, result: AccountResult) -> None:
        """
        Called by the engine after scanning each account.
        Use for: session cleanup, rate limit backoff, progress logging.
        Default: no-op.
        """
        pass

    # ------------------------------------------------------------------
    # Concrete helpers available to all providers
    # ------------------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    def _require_auth(self) -> None:
        """Call at the start of any method that needs credentials."""
        if not self._authenticated:
            raise ProviderNotAuthenticatedError(
                f"{self.provider_name} provider is not authenticated. "
                f"Call authenticate() before running checks."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider_name!r}, "
            f"authenticated={self._authenticated})"
        )


# ---------------------------------------------------------------------------
# Provider-specific exceptions
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base exception for all provider errors."""
    pass


class ProviderAuthError(ProviderError):
    """Raised when authentication fails hard (invalid credentials)."""
    pass


class ProviderNotAuthenticatedError(ProviderError):
    """Raised when a method is called before authenticate()."""
    pass


class ProviderPermissionError(ProviderError):
    """
    Raised when credentials lack permission for a specific API call.
    This is NON-FATAL — the scan continues but records this as an error.
    """
    def __init__(self, service: str, action: str, message: str = ""):
        self.service = service
        self.action = action
        super().__init__(
            f"Permission denied: {service}:{action}. {message}"
        )


class AccountScanError(ProviderError):
    """Raised when an entire account scan fails (e.g. assume-role failed)."""
    def __init__(self, account_id: str, message: str):
        self.account_id = account_id
        super().__init__(f"Failed to scan account {account_id}: {message}")