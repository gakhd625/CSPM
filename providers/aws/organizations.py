"""
providers/aws/organizations.py

AWSOrganizations handles everything related to AWS Organizations:
  1. Detect whether Organizations is enabled in this management account
  2. Enumerate all member accounts (with status filtering)
  3. Map accounts to their Organizational Units (OUs) for grouping in reports
  4. Retrieve Service Control Policies (SCPs) — future: factor into attack graph
  5. Provide account metadata (name, email, tags) for report display

Why does this need its own module?
  - Organizations pagination is non-trivial (can have 1000s of accounts)
  - OU hierarchy traversal is recursive
  - SCP retrieval requires separate API calls per policy
  - Account metadata enriches the report significantly (name vs just account ID)

Key Organizations concepts:
  - Management account: the account that owns the Org (was: "master account")
  - Member accounts: all other accounts in the Org
  - Root: the top-level container (always exactly one)
  - OU (Organizational Unit): a grouping container below Root
  - SCP (Service Control Policy): guardrails applied to OUs/accounts
  - Account status: ACTIVE, SUSPENDED, PENDING_CLOSURE

IAM permissions required:
  - organizations:DescribeOrganization
  - organizations:ListAccounts
  - organizations:ListAccountsForParent
  - organizations:ListOrganizationalUnitsForParent
  - organizations:ListRoots
  - organizations:ListPolicies
  - organizations:DescribePolicy (for SCP content)
  - organizations:ListTargetsForPolicy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models for Organizations entities
# ---------------------------------------------------------------------------

@dataclass
class OrgAccount:
    """
    Represents one AWS account within the Organization.
    Enriched with OU path for report grouping.
    """
    account_id:   str
    name:         str
    email:        str
    status:       str         # "ACTIVE", "SUSPENDED", "PENDING_CLOSURE"
    joined_method: str        # "INVITED" or "CREATED"
    joined_timestamp: Optional[str] = None
    ou_path:      str         = ""     # e.g. "/Root/Production/Web"
    ou_id:        str         = ""     # Direct parent OU ID
    tags:         Dict[str, str] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"

    @property
    def display_name(self) -> str:
        """'{name} ({account_id})' — used in reports and log output."""
        return f"{self.name} ({self.account_id})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id":       self.account_id,
            "name":             self.name,
            "email":            self.email,
            "status":           self.status,
            "joined_method":    self.joined_method,
            "joined_timestamp": self.joined_timestamp,
            "ou_path":          self.ou_path,
            "ou_id":            self.ou_id,
            "tags":             self.tags,
        }


@dataclass
class OrganizationInfo:
    """
    Top-level information about the AWS Organization itself.
    Returned by AWSOrganizations.describe() and included in reports.
    """
    org_id:              str
    master_account_id:   str
    master_account_email: str
    feature_set:         str   # "ALL" (full) or "CONSOLIDATED_BILLING" (limited)
    total_accounts:      int   = 0
    active_accounts:     int   = 0
    suspended_accounts:  int   = 0

    @property
    def is_all_features(self) -> bool:
        """
        True if Organizations is in 'ALL' features mode.
        SCPs and tag policies only work in ALL features mode.
        If False, the scanner can still enumerate accounts but SCPs don't apply.
        """
        return self.feature_set == "ALL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id":              self.org_id,
            "master_account_id":   self.master_account_id,
            "master_account_email": self.master_account_email,
            "feature_set":         self.feature_set,
            "is_all_features":     self.is_all_features,
            "total_accounts":      self.total_accounts,
            "active_accounts":     self.active_accounts,
            "suspended_accounts":  self.suspended_accounts,
        }


@dataclass
class ServiceControlPolicy:
    """A Service Control Policy — a guardrail on what accounts can do."""
    policy_id:   str
    name:        str
    description: str
    aws_managed: bool
    content:     Dict[str, Any] = field(default_factory=dict)  # Parsed JSON document
    targets:     List[str]      = field(default_factory=list)  # OU/account IDs


# ---------------------------------------------------------------------------
# AWSOrganizations
# ---------------------------------------------------------------------------

class AWSOrganizations:
    """
    Interacts with the AWS Organizations API.

    Must be called from the management account (or a delegated admin account).
    Member accounts can call DescribeOrganization but NOT ListAccounts.

    Usage:
        orgs = AWSOrganizations(session=boto3_session)

        if not orgs.is_enabled():
            print("Organizations not in use")
        else:
            info = orgs.describe()
            accounts = orgs.list_accounts(include_suspended=False)
    """

    def __init__(self, session: Any, region: str = "us-east-1"):
        """
        Args:
            session: A boto3 Session (must be for the management account)
            region:  Organizations is a global service but API calls go to a
                     specific region. us-east-1 is the conventional choice.
        """
        self._session = session
        self._region  = region
        self._client  = None   # Lazy — created on first use
        self._org_info: Optional[OrganizationInfo] = None
        self._account_cache: Optional[List[OrgAccount]] = None
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """
        Return True if this account is part of an AWS Organization.

        This is the safe first call — it returns False (not raises) if:
          - Organizations is not enabled
          - This is a standalone account
          - The caller lacks organizations:DescribeOrganization permission
        """
        try:
            self._get_client().describe_organization()
            return True
        except Exception as e:
            error_code = self._error_code(e)
            if error_code in ("AWSOrganizationsNotInUseException",
                              "AccessDeniedException"):
                self.logger.info(
                    f"AWS Organizations is not enabled or not accessible: "
                    f"{error_code}"
                )
                return False
            # Unexpected error — log and treat as "not enabled" to avoid aborting
            self.logger.warning(
                f"Unexpected error checking Organizations: {e}"
            )
            return False

    # ------------------------------------------------------------------
    # Organization details
    # ------------------------------------------------------------------

    def describe(self) -> OrganizationInfo:
        """
        Return metadata about the Organization itself.
        Cached after first call.

        Raises:
            OrganizationsError: If not in an Organization or permissions missing.
        """
        if self._org_info:
            return self._org_info

        try:
            response = self._get_client().describe_organization()
            org = response["Organization"]

            self._org_info = OrganizationInfo(
                org_id=org["Id"],
                master_account_id=org["MasterAccountId"],
                master_account_email=org["MasterAccountEmail"],
                feature_set=org["FeatureSet"],
            )
            self.logger.info(
                f"Organization: id={self._org_info.org_id}, "
                f"management={self._org_info.master_account_id}, "
                f"features={self._org_info.feature_set}"
            )
            return self._org_info

        except Exception as e:
            raise OrganizationsError(
                f"Failed to describe organization: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Account enumeration
    # ------------------------------------------------------------------

    def list_accounts(
        self,
        include_suspended: bool = False,
    ) -> List[str]:
        """
        Return a list of account IDs in the Organization.

        This is the primary method called by AWSProvider.get_accounts().
        Returns account IDs (strings) only — for rich account objects use
        list_account_objects().

        Args:
            include_suspended: If False (default), skip SUSPENDED accounts.
                               Suspended accounts have no active resources
                               but their data still exists.

        Returns:
            List of account ID strings, e.g. ["123456789012", "234567890123"]

        Raises:
            OrganizationsError: If enumeration fails.
        """
        accounts = self.list_account_objects(include_suspended=include_suspended)
        ids = [a.account_id for a in accounts]
        self.logger.info(
            f"Found {len(ids)} {'(including suspended) ' if include_suspended else ''}"
            f"accounts in organization"
        )
        return ids

    def list_account_objects(
        self,
        include_suspended: bool = False,
    ) -> List[OrgAccount]:
        """
        Return full OrgAccount objects for all accounts.
        Cached after first call (subsequent calls return cached data).

        Uses pagination to handle organizations with 1000+ accounts.
        """
        if self._account_cache is not None:
            if include_suspended:
                return self._account_cache
            return [a for a in self._account_cache if a.is_active]

        self.logger.info("Enumerating all accounts in organization...")
        accounts: List[OrgAccount] = []

        try:
            paginator = self._get_client().get_paginator("list_accounts")
            for page in paginator.paginate():
                for raw_account in page["Accounts"]:
                    account = OrgAccount(
                        account_id=raw_account["Id"],
                        name=raw_account["Name"],
                        email=raw_account["Email"],
                        status=raw_account["Status"],
                        joined_method=raw_account["JoinedMethod"],
                        joined_timestamp=raw_account.get("JoinedTimestamp", ""),
                    )
                    accounts.append(account)

        except Exception as e:
            error_code = self._error_code(e)
            if error_code == "AccessDeniedException":
                raise OrganizationsError(
                    "Access denied to organizations:ListAccounts. "
                    "This API must be called from the management account or a "
                    "delegated administrator account. "
                    "Current account may be a member account."
                ) from e
            raise OrganizationsError(
                f"Failed to list organization accounts: {e}"
            ) from e

        # Enrich with OU paths (best-effort — don't fail if this errors)
        try:
            self._enrich_with_ou_paths(accounts)
        except Exception as e:
            self.logger.warning(
                f"Could not retrieve OU paths (non-fatal): {e}"
            )

        self._account_cache = accounts

        active    = sum(1 for a in accounts if a.is_active)
        suspended = sum(1 for a in accounts if not a.is_active)
        self.logger.info(
            f"Account enumeration complete: "
            f"{active} active, {suspended} suspended"
        )

        # Update org info counts if available
        if self._org_info:
            self._org_info.total_accounts    = len(accounts)
            self._org_info.active_accounts   = active
            self._org_info.suspended_accounts = suspended

        if include_suspended:
            return accounts
        return [a for a in accounts if a.is_active]

    def get_account(self, account_id: str) -> Optional[OrgAccount]:
        """Return a single account by ID, or None if not found."""
        all_accounts = self.list_account_objects(include_suspended=True)
        for account in all_accounts:
            if account.account_id == account_id:
                return account
        return None

    def get_account_name(self, account_id: str) -> str:
        """
        Return the friendly name for an account ID.
        Falls back to the account ID itself if not found.
        """
        account = self.get_account(account_id)
        return account.name if account else account_id

    # ------------------------------------------------------------------
    # OU hierarchy
    # ------------------------------------------------------------------

    def _enrich_with_ou_paths(self, accounts: List[OrgAccount]) -> None:
        """
        Populate the ou_path field on each account by traversing the OU tree.
        This is expensive for large orgs (N accounts × depth API calls).
        We build a full OU→path map first, then look up each account.
        """
        self.logger.debug("Building OU path map...")

        # Build: account_id → (ou_id, ou_path)
        account_to_ou = self._build_account_ou_map()

        for account in accounts:
            if account.account_id in account_to_ou:
                ou_id, ou_path = account_to_ou[account.account_id]
                account.ou_id   = ou_id
                account.ou_path = ou_path

    def _build_account_ou_map(self) -> Dict[str, Tuple[str, str]]:
        """
        Walk the OU tree top-down and build account_id → (ou_id, path).
        Returns a dict so account lookups are O(1).
        """
        result: Dict[str, Tuple[str, str]] = {}

        try:
            # Get the root
            roots_response = self._get_client().list_roots()
            roots = roots_response.get("Roots", [])
            if not roots:
                return result

            root_id = roots[0]["Id"]

            # BFS/DFS through the OU tree
            # Stack: (parent_id, parent_path)
            stack = [(root_id, "/Root")]

            while stack:
                parent_id, parent_path = stack.pop()

                # Get accounts directly under this parent
                try:
                    acct_paginator = self._get_client().get_paginator(
                        "list_accounts_for_parent"
                    )
                    for page in acct_paginator.paginate(ParentId=parent_id):
                        for acct in page["Accounts"]:
                            result[acct["Id"]] = (parent_id, parent_path)
                except Exception as e:
                    self.logger.debug(
                        f"Could not list accounts for {parent_id}: {e}"
                    )

                # Get child OUs and recurse
                try:
                    ou_paginator = self._get_client().get_paginator(
                        "list_organizational_units_for_parent"
                    )
                    for page in ou_paginator.paginate(ParentId=parent_id):
                        for ou in page["OrganizationalUnits"]:
                            ou_path = f"{parent_path}/{ou['Name']}"
                            stack.append((ou["Id"], ou_path))
                except Exception as e:
                    self.logger.debug(
                        f"Could not list OUs for {parent_id}: {e}"
                    )

        except Exception as e:
            self.logger.warning(f"OU tree traversal failed: {e}")

        return result

    # ------------------------------------------------------------------
    # Service Control Policies
    # ------------------------------------------------------------------

    def list_scps(self) -> List[ServiceControlPolicy]:
        """
        Return all Service Control Policies in the Organization.

        SCPs are only available in "ALL" features mode.
        Future use: the attack graph engine will use SCPs to determine
        whether a privilege escalation path is actually exploitable
        (an SCP denying sts:AssumeRole blocks the path).

        Returns empty list if SCPs are not accessible or not enabled.
        """
        try:
            org_info = self.describe()
            if not org_info.is_all_features:
                self.logger.info(
                    "Organization is in CONSOLIDATED_BILLING mode — "
                    "SCPs are not available."
                )
                return []

            scps = []
            paginator = self._get_client().get_paginator("list_policies")
            for page in paginator.paginate(Filter="SERVICE_CONTROL_POLICY"):
                for raw_policy in page["Policies"]:
                    scp = ServiceControlPolicy(
                        policy_id=raw_policy["Id"],
                        name=raw_policy["Name"],
                        description=raw_policy.get("Description", ""),
                        aws_managed=raw_policy["AwsManaged"],
                    )
                    scps.append(scp)

            self.logger.info(f"Found {len(scps)} Service Control Policies")
            return scps

        except Exception as e:
            self.logger.warning(f"Could not retrieve SCPs: {e}")
            return []

    def get_scp_content(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the JSON document for a specific SCP.
        Returns None if retrieval fails.
        """
        try:
            import json
            response = self._get_client().describe_policy(PolicyId=policy_id)
            content_str = response["Policy"]["Content"]
            return json.loads(content_str)
        except Exception as e:
            self.logger.warning(f"Could not retrieve SCP content for {policy_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Summary for reporting
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a report-ready summary of the Organization.
        Used by the HTML reporter to render the Organizations section.
        """
        if not self.is_enabled():
            return {"enabled": False}

        org_info = self.describe()
        accounts = self.list_account_objects(include_suspended=True)
        scps     = self.list_scps()

        # Group accounts by OU path for the report
        ou_groups: Dict[str, List[str]] = {}
        for account in accounts:
            path = account.ou_path or "/Root"
            ou_groups.setdefault(path, []).append(account.display_name)

        return {
            "enabled":      True,
            "org_info":     org_info.to_dict(),
            "accounts":     [a.to_dict() for a in accounts],
            "ou_groups":    ou_groups,
            "scp_count":    len(scps),
            "scps":         [
                {"id": s.policy_id, "name": s.name, "aws_managed": s.aws_managed}
                for s in scps
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazy-create the organizations boto3 client."""
        if self._client is None:
            # Organizations is a global service — us-east-1 is correct
            self._client = self._session.client(
                "organizations", region_name="us-east-1"
            )
        return self._client

    @staticmethod
    def _error_code(exception: Exception) -> str:
        """Extract the AWS error code from a botocore exception."""
        response = getattr(exception, "response", {}) or {}
        return response.get("Error", {}).get("Code", "")

    def __repr__(self) -> str:
        account_count = (
            len(self._account_cache) if self._account_cache is not None else "?"
        )
        return (
            f"AWSOrganizations("
            f"org_id={self._org_info.org_id if self._org_info else 'not-loaded'!r}, "
            f"accounts={account_count})"
        )


# ---------------------------------------------------------------------------
# Organizations-specific exceptions
# ---------------------------------------------------------------------------

class OrganizationsError(Exception):
    """Raised when Organizations API calls fail in a non-recoverable way."""
    pass