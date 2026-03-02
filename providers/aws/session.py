"""
providers/aws/session.py

AWSSession manages everything to do with boto3 credentials and sessions.

Responsibilities:
  1. Build the initial boto3 Session from config (profile, env vars, instance profile)
  2. Assume cross-account roles for Organizations scans
  3. Cache assumed-role sessions so we don't call sts:AssumeRole on every checker
  4. Refresh sessions when STS credentials expire (1-hour lifetime)
  5. Provide per-region clients (checkers request clients, not sessions)
  6. Handle credential chain errors with clear, actionable messages

Why a dedicated session module?
  - assume-role logic is complex (external IDs, MFA serials, duration, refresh)
  - Without caching, an org scan of 50 accounts × 10 checkers = 500 AssumeRole calls
  - Centralising session creation means credential logic is never duplicated in checkers
  - Makes unit testing possible — inject a mock AWSSession, don't touch real AWS

Credential chain (in priority order):
  1. Explicit profile (config["profile"])
  2. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables
  3. AWS credential file (~/.aws/credentials)
  4. IAM instance profile (EC2 / Lambda)
  5. ECS task role
  6. Web identity token (EKS / IRSA)

Cross-account flow:
  Scanner account   →  sts:AssumeRole  →  SecurityAudit role in target account
  [Management acct]                        [Member acct 1, 2, 3 ...]
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credential cache entry
# ---------------------------------------------------------------------------

@dataclass
class CachedCredentials:
    """
    A cached set of temporary STS credentials for one assumed role.

    STS credentials expire after the requested duration (default: 1 hour).
    We refresh proactively when < REFRESH_BUFFER_SECONDS remain, so
    running checkers never hit an expiry mid-scan.
    """

    # Non-default fields must come first in a dataclass
    access_key:    str
    secret_key:    str
    session_token: str
    expiry:        datetime   # UTC

    # Class-level constant (not a field — use ClassVar to prevent dataclass treating it as a field)
    REFRESH_BUFFER_SECONDS: ClassVar[int] = 300  # 5 minutes

    @property
    def is_expired(self) -> bool:
        now = datetime.now(timezone.utc)
        return now >= self.expiry

    @property
    def needs_refresh(self) -> bool:
        """True when close to expiry or already expired."""
        now = datetime.now(timezone.utc)
        remaining = (self.expiry - now).total_seconds()
        return remaining < self.REFRESH_BUFFER_SECONDS

    @property
    def as_boto3_credentials(self) -> Dict[str, str]:
        """Return as kwargs for boto3.Session()."""
        return {
            "aws_access_key_id":     self.access_key,
            "aws_secret_access_key": self.secret_key,
            "aws_session_token":     self.session_token,
        }


# ---------------------------------------------------------------------------
# AWSSession
# ---------------------------------------------------------------------------

class AWSSession:
    """
    Manages boto3 sessions and credentials for the CSPM scanner.

    Thread-safe: the session cache uses a lock so concurrent checker
    threads (future Step parallel scanning) don't race on assume-role.

    Usage:
        # Initialise once per scan
        aws_session = AWSSession(config={"profile": "management-account"})
        aws_session.initialise()

        # Get a session for any account — cached after first call
        session = aws_session.get_session_for_account("123456789012")
        iam_client = session.client("iam")

        # Get a regional client directly
        s3 = aws_session.get_client("s3", account_id="123456789012", region="eu-west-1")
    """

    # Default STS session duration for assumed roles (in seconds)
    # 3600 = 1 hour (AWS minimum). Max is 12h if role allows.
    DEFAULT_SESSION_DURATION = 3600

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Provider config dict. Relevant keys:
                profile        (str)  AWS profile name
                role_name      (str)  Role to assume in member accounts
                                      Default: "SecurityAudit"
                external_id    (str)  STS ExternalId condition value
                session_duration (int) Seconds for assumed role sessions
                mfa_serial     (str)  MFA device ARN (for manual/dev use)
                mfa_token      (str)  Current MFA TOTP code (for manual/dev use)
        """
        self.config = config
        self._base_session = None       # The primary boto3 Session
        self._identity: Dict = {}       # sts:GetCallerIdentity result
        self._home_account_id: str = "" # Account ID of the scanner itself

        # Cache: account_id → CachedCredentials
        # Protected by _lock for thread safety
        self._credential_cache: Dict[str, CachedCredentials] = {}
        self._lock = threading.Lock()

        # Track all accounts we've successfully assumed roles into
        self._assumed_accounts: List[str] = []

        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self) -> Dict[str, Any]:
        """
        Build the base session and validate credentials.
        Must be called before any other method.

        Returns the sts:GetCallerIdentity response dict:
            {"Account": "...", "UserId": "...", "Arn": "..."}

        Raises:
            AWSSessionError: If credentials are invalid or unavailable.
        """
        try:
            import boto3
            import botocore.exceptions
        except ImportError:
            raise AWSSessionError(
                "boto3 is not installed. Install it with: pip install boto3"
            )

        profile = self.config.get("profile")
        try:
            if profile:
                self._base_session = boto3.Session(profile_name=profile)
                self.logger.info(f"boto3 session initialised with profile: {profile!r}")
            else:
                self._base_session = boto3.Session()
                self.logger.info(
                    "boto3 session initialised with default credential chain"
                )

            # Verify credentials are valid NOW (fail fast)
            sts = self._base_session.client("sts")
            self._identity = sts.get_caller_identity()
            self._home_account_id = self._identity["Account"]

            self.logger.info(
                f"Credentials verified: account={self._home_account_id}, "
                f"arn={self._identity['Arn']}"
            )
            return self._identity

        except botocore.exceptions.NoCredentialsError:
            raise AWSSessionError(
                "No AWS credentials found. Configure credentials via one of:\n"
                "  - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables\n"
                "  - ~/.aws/credentials profile\n"
                "  - IAM instance profile (EC2/Lambda)\n"
                "  - ECS task role or EKS IRSA\n"
                "  - 'aws configure' CLI command"
            )
        except botocore.exceptions.ProfileNotFound:
            raise AWSSessionError(
                f"AWS profile {profile!r} not found in ~/.aws/credentials. "
                f"Available profiles: run 'aws configure list-profiles'"
            )
        except Exception as e:
            error_code = getattr(
                getattr(e, "response", {}), "get", lambda k, d=None: d
            )("Error", {}).get("Code", "")
            if error_code in ("InvalidClientTokenId", "ExpiredToken", "AuthFailure"):
                raise AWSSessionError(
                    f"AWS credentials are invalid or expired (code={error_code}). "
                    f"Run 'aws sts get-caller-identity' to diagnose."
                )
            raise AWSSessionError(f"Failed to initialise AWS session: {e}") from e

    # ------------------------------------------------------------------
    # Session and client access
    # ------------------------------------------------------------------

    def get_session_for_account(self, account_id: str) -> Any:
        """
        Return a boto3 Session configured for the target account.

        - Home account (same as scanner): returns base session directly
        - Member account: assumes SecurityAudit role, caches credentials,
          returns a new Session using the temporary credentials

        This is the primary method checkers use indirectly via the provider.
        """
        if not self._base_session:
            raise AWSSessionError(
                "AWSSession not initialised. Call initialise() first."
            )

        # Same account as the scanner — no assume-role needed
        if account_id == self._home_account_id:
            return self._base_session

        # Check cache (with refresh if needed)
        cached = self._get_cached_credentials(account_id)
        if cached and not cached.needs_refresh:
            return self._session_from_credentials(cached)

        # Assume role and cache the credentials
        credentials = self._assume_role(account_id)
        self._cache_credentials(account_id, credentials)
        return self._session_from_credentials(credentials)

    def get_client(
        self,
        service:    str,
        account_id: str,
        region:     str = "us-east-1",
    ) -> Any:
        """
        Get a boto3 client for a specific service, account, and region.

        Convenience wrapper around get_session_for_account() for checkers
        that only need a single client.

        Usage in a checker:
            iam_client = self.aws_session.get_client("iam", account_id, "global")
            s3_client  = self.aws_session.get_client("s3", account_id, "us-east-1")
        """
        session = self.get_session_for_account(account_id)
        # IAM and some global services don't need a region
        if region == "global":
            return session.client(service)
        return session.client(service, region_name=region)

    # ------------------------------------------------------------------
    # Assume-role
    # ------------------------------------------------------------------

    def _assume_role(self, account_id: str) -> CachedCredentials:
        """
        Perform sts:AssumeRole into a member account.

        Role ARN format:
            arn:aws:iam::{account_id}:role/{role_name}

        The role must:
          1. Exist in the target account
          2. Have a trust policy allowing the scanner's principal to assume it
          3. Have the SecurityAudit managed policy attached (read-only)

        Common assume-role failure causes:
          - Role doesn't exist in target account → NoSuchEntity
          - Trust policy doesn't allow scanner → AccessDenied
          - ExternalId mismatch → AccessDenied
          - MFA required → AccessDenied (if mfa_serial not configured)
        """
        role_name   = self.config.get("role_name", "SecurityAudit")
        external_id = self.config.get("external_id")
        duration    = self.config.get("session_duration", self.DEFAULT_SESSION_DURATION)
        role_arn    = f"arn:aws:iam::{account_id}:role/{role_name}"

        self.logger.info(
            f"Assuming role: {role_arn} "
            f"[duration={duration}s, external_id={'set' if external_id else 'not set'}]"
        )

        # Build the AssumeRole request
        assume_kwargs: Dict[str, Any] = {
            "RoleArn":         role_arn,
            "RoleSessionName": f"cspm-scanner-{int(time.time())}",
            "DurationSeconds": duration,
        }
        if external_id:
            assume_kwargs["ExternalId"] = external_id

        # MFA support (for developer/manual runs — Lambda uses instance profiles)
        mfa_serial = self.config.get("mfa_serial")
        mfa_token  = self.config.get("mfa_token")
        if mfa_serial and mfa_token:
            assume_kwargs["SerialNumber"] = mfa_serial
            assume_kwargs["TokenCode"]    = mfa_token
            self.logger.info(f"Using MFA device: {mfa_serial}")

        try:
            sts = self._base_session.client("sts")
            response = sts.assume_role(**assume_kwargs)
        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            raise AssumeRoleError(
                account_id=account_id,
                role_name=role_name,
                error_code=error_code,
                message=str(e),
            ) from e

        creds = response["Credentials"]
        expiry = creds["Expiration"]

        # boto3 returns expiry as a datetime — ensure it's timezone-aware
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        cached = CachedCredentials(
            access_key=creds["AccessKeyId"],
            secret_key=creds["SecretAccessKey"],
            session_token=creds["SessionToken"],
            expiry=expiry,
        )

        self.logger.info(
            f"Successfully assumed {role_arn} "
            f"[expires={expiry.isoformat()}, "
            f"remaining={(expiry - datetime.now(timezone.utc)).seconds // 60}m]"
        )
        self._assumed_accounts.append(account_id)
        return cached

    def _session_from_credentials(self, credentials: CachedCredentials) -> Any:
        """Build a boto3 Session from temporary STS credentials."""
        try:
            import boto3
            return boto3.Session(**credentials.as_boto3_credentials)
        except ImportError:
            raise AWSSessionError("boto3 is not installed.")

    # ------------------------------------------------------------------
    # Credential cache
    # ------------------------------------------------------------------

    def _get_cached_credentials(
        self, account_id: str
    ) -> Optional[CachedCredentials]:
        """Thread-safe cache read."""
        with self._lock:
            return self._credential_cache.get(account_id)

    def _cache_credentials(
        self, account_id: str, credentials: CachedCredentials
    ) -> None:
        """Thread-safe cache write."""
        with self._lock:
            self._credential_cache[account_id] = credentials

    def invalidate_cache(self, account_id: Optional[str] = None) -> None:
        """
        Invalidate cached credentials.
        - account_id=None: invalidate all cached credentials
        - account_id set: invalidate only that account's credentials
        """
        with self._lock:
            if account_id:
                self._credential_cache.pop(account_id, None)
                self.logger.debug(f"Cache invalidated for account {account_id}")
            else:
                self._credential_cache.clear()
                self.logger.debug("All cached credentials invalidated")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def home_account_id(self) -> str:
        """Account ID of the scanner's own credentials."""
        return self._home_account_id

    @property
    def identity(self) -> Dict[str, Any]:
        """Full sts:GetCallerIdentity response."""
        return self._identity

    @property
    def assumed_accounts(self) -> List[str]:
        """List of account IDs we've successfully assumed roles into."""
        return list(self._assumed_accounts)

    def get_enabled_regions(self, service: str = "ec2") -> List[str]:
        """
        Return all enabled regions for the given service.
        Used by checkers that need to iterate regions (CloudTrail, GuardDuty).

        Args:
            service: AWS service name (default "ec2" — covers most regional checks)
        """
        if not self._base_session:
            raise AWSSessionError("AWSSession not initialised.")

        try:
            ec2 = self._base_session.client("ec2", region_name="us-east-1")
            response = ec2.describe_regions(Filters=[
                {"Name": "opt-in-status", "Values": ["opt-in-not-required", "opted-in"]}
            ])
            regions = [r["RegionName"] for r in response["Regions"]]
            self.logger.debug(f"Found {len(regions)} enabled regions")
            return sorted(regions)
        except Exception as e:
            self.logger.warning(
                f"Could not enumerate enabled regions: {e}. "
                f"Falling back to us-east-1."
            )
            return ["us-east-1"]

    def get_account_alias(self) -> Optional[str]:
        """
        Return the IAM account alias for the current account, if set.
        Many organisations set account aliases like 'acme-production'.
        """
        if not self._base_session:
            return None
        try:
            iam = self._base_session.client("iam")
            response = iam.list_account_aliases()
            aliases = response.get("AccountAliases", [])
            return aliases[0] if aliases else None
        except Exception as e:
            self.logger.debug(f"Could not retrieve account alias: {e}")
            return None

    def __repr__(self) -> str:
        return (
            f"AWSSession("
            f"account={self._home_account_id!r}, "
            f"cached_accounts={len(self._credential_cache)}, "
            f"initialised={self._base_session is not None})"
        )


# ---------------------------------------------------------------------------
# AWSSession-specific exceptions
# ---------------------------------------------------------------------------

class AWSSessionError(Exception):
    """Base exception for session and credential failures."""
    pass


class AssumeRoleError(AWSSessionError):
    """
    Raised when sts:AssumeRole fails for a member account.

    This is intentionally NON-FATAL at the engine level —
    AWSProvider.pre_scan_hook() catches this and records it as an
    account-level error so the scan continues with other accounts.
    """

    # Friendly messages for common error codes
    ERROR_MESSAGES = {
        "AccessDenied": (
            "Access denied. Verify that:\n"
            "  1. The role '{role_name}' exists in account {account_id}\n"
            "  2. The trust policy allows principal '{scanner_arn}' to assume it\n"
            "  3. The ExternalId matches (if configured)\n"
            "  4. The role has not been deleted or renamed"
        ),
        "NoSuchEntity": (
            "Role '{role_name}' does not exist in account {account_id}.\n"
            "Create the role with: aws iam create-role --role-name {role_name} ..."
        ),
        "RegionDisabledException": (
            "STS is not available in this region for account {account_id}."
        ),
        "ExpiredToken": (
            "The base session credentials have expired. "
            "Refresh your credentials and try again."
        ),
    }

    def __init__(
        self,
        account_id:  str,
        role_name:   str,
        error_code:  str,
        message:     str,
        scanner_arn: str = "",
    ):
        self.account_id  = account_id
        self.role_name   = role_name
        self.error_code  = error_code
        self.scanner_arn = scanner_arn

        template = self.ERROR_MESSAGES.get(error_code, "Unexpected error: {message}")
        friendly = template.format(
            account_id=account_id,
            role_name=role_name,
            scanner_arn=scanner_arn or "unknown",
            message=message,
        )
        super().__init__(
            f"AssumeRole failed for account {account_id} "
            f"[role={role_name}, code={error_code}]:\n{friendly}"
        )