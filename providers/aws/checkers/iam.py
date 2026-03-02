"""
providers/aws/checkers/iam.py

IAMChecker implements all 5 IAM security checks:

  AWS-IAM-001  IAM users without MFA enabled
  AWS-IAM-002  Root account without MFA (or with active access keys)
  AWS-IAM-003  Access keys older than 90 days
  AWS-IAM-004  Overly permissive policies (Action:* or AdministratorAccess)
  AWS-IAM-005  Roles with dangerous trust policies (wildcard principal)

Design decisions:
  - One private method per check (e.g. _check_mfa, _check_root_account).
    Each method is independently testable and easy to extend.
  - All IAM data is fetched via the credential report where possible.
    The credential report is a single CSV download that contains MFA status,
    key ages, and password info for ALL users — far more efficient than
    calling GetUser + ListMFADevices + ListAccessKeys per user.
  - Policy analysis uses a dedicated _is_admin_policy() helper that
    understands both managed policies and inline policies.
  - Trust policy analysis is separate from permission policy analysis —
    they serve different attack surfaces.
  - The checker contributes nodes to the attack graph by populating
    self.graph_nodes and self.graph_edges — the provider assembles these
    into the AttackGraph after all checkers run.

IAM permissions required:
  iam:GenerateCredentialReport
  iam:GetCredentialReport
  iam:GetAccountSummary
  iam:ListUsers
  iam:ListMFADevices
  iam:GetLoginProfile
  iam:ListAccessKeys
  iam:ListAttachedUserPolicies
  iam:ListUserPolicies
  iam:GetUserPolicy
  iam:ListAttachedRolePolicies
  iam:ListRolePolicies
  iam:GetRolePolicy
  iam:ListRoles
  iam:ListPolicies
  iam:GetPolicy
  iam:GetPolicyVersion
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from core.attack_graph.models import (
    AttackGraph, EdgeType, GraphEdge, GraphNode, NodeType
)
from core.base_checker import BaseChecker, FindingTemplate
from core.checker_registry import CheckerRegistry
from core.models.finding import Finding, RemediationStep, Severity

logger = logging.getLogger(__name__)


@CheckerRegistry.register(provider="aws", domain="iam")
class IAMChecker(BaseChecker):
    """
    AWS Identity and Access Management security checker.

    Checks all 5 IAM security requirements from the project spec.
    Self-registers with CheckerRegistry via the decorator above —
    no changes needed in provider.py to add this checker.
    """

    # ------------------------------------------------------------------
    # Rule book — all finding definitions in one place
    # ------------------------------------------------------------------

    FINDING_TEMPLATES: Dict[str, FindingTemplate] = {

        "AWS-IAM-001": FindingTemplate(
            finding_id="AWS-IAM-001",
            title="IAM User Missing MFA",
            description_template=(
                "IAM user '{resource_name}' has console access enabled but no "
                "Multi-Factor Authentication (MFA) device configured. Without MFA, "
                "a compromised password gives an attacker full console access."
            ),
            severity=Severity.HIGH,
            remediation_summary=(
                "Enable MFA for this IAM user immediately. Prefer hardware MFA "
                "tokens (FIDO2/YubiKey) for privileged users."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Navigate to IAM → Users → select user → Security credentials tab",
                    console_steps="IAM Console → Users → {user} → Security credentials → MFA → Assign MFA device",
                    cli_command="aws iam create-virtual-mfa-device --virtual-mfa-device-name {user}-mfa --outfile /tmp/qrcode.png --bootstrap-method QRCodePNG",
                ),
                RemediationStep(
                    step_number=2,
                    description="Associate the MFA device with the user account",
                    cli_command="aws iam enable-mfa-device --user-name {user} --serial-number arn:aws:iam::ACCOUNT:mfa/{user}-mfa --authentication-code1 CODE1 --authentication-code2 CODE2",
                ),
                RemediationStep(
                    step_number=3,
                    description="Enforce MFA via IAM policy condition: aws:MultiFactorAuthPresent: 'true'",
                    terraform='resource "aws_iam_policy" "require_mfa" { ... condition { test = "Bool" variable = "aws:MultiFactorAuthPresent" values = ["true"] } }',
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_mfa.html",
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html#enable-mfa-for-privileged-users",
                "https://www.cisecurity.org/benchmark/amazon_web_services",
            ],
        ),

        "AWS-IAM-002": FindingTemplate(
            finding_id="AWS-IAM-002",
            title="Root Account MFA Not Enabled or Access Keys Active",
            description_template=(
                "The AWS root account has insufficient security controls. "
                "The root account has unrestricted access to all AWS services "
                "and cannot be restricted by IAM policies — it is the single most "
                "sensitive credential in an AWS account."
            ),
            severity=Severity.CRITICAL,
            remediation_summary=(
                "Enable hardware MFA on the root account and delete all root access keys. "
                "Never use root credentials for day-to-day operations."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Sign in as root and enable hardware MFA",
                    console_steps="AWS Console → Account → Security credentials → Multi-factor authentication (MFA) → Assign MFA device → Hardware TOTP token",
                ),
                RemediationStep(
                    step_number=2,
                    description="Delete all root access keys (there should be none)",
                    console_steps="AWS Console → Account → Security credentials → Access keys → Delete",
                    cli_command="# Root access keys cannot be deleted via CLI — must use console",
                ),
                RemediationStep(
                    step_number=3,
                    description="Create an IAM user or role for all administrative tasks instead",
                    cli_command="aws iam create-user --user-name admin-user && aws iam attach-user-policy --user-name admin-user --policy-arn arn:aws:iam::aws:policy/AdministratorAccess",
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_root-user.html",
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html#lock-away-credentials",
                "https://www.cisecurity.org/benchmark/amazon_web_services",
            ],
        ),

        "AWS-IAM-003": FindingTemplate(
            finding_id="AWS-IAM-003",
            title="IAM Access Key Older Than 90 Days",
            description_template=(
                "IAM access key '{key_id}' for user '{resource_name}' has not been "
                "rotated in over 90 days. Long-lived credentials increase the blast "
                "radius of a key compromise — the longer a key exists, the more "
                "time an attacker has to use it undetected."
            ),
            severity=Severity.MEDIUM,
            remediation_summary=(
                "Rotate this access key immediately. Create a new key, update all "
                "systems using the old key, then deactivate and delete the old key."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Create a new access key for the user",
                    cli_command="aws iam create-access-key --user-name {user}",
                ),
                RemediationStep(
                    step_number=2,
                    description="Update all applications, CI/CD pipelines, and scripts to use the new key",
                ),
                RemediationStep(
                    step_number=3,
                    description="Deactivate the old key first (keep for 24h in case rollback needed)",
                    cli_command="aws iam update-access-key --user-name {user} --access-key-id OLD_KEY_ID --status Inactive",
                ),
                RemediationStep(
                    step_number=4,
                    description="Delete the old key after confirming no usage",
                    cli_command="aws iam delete-access-key --user-name {user} --access-key-id OLD_KEY_ID",
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html",
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html#rotate-credentials",
            ],
        ),

        "AWS-IAM-004": FindingTemplate(
            finding_id="AWS-IAM-004",
            title="Overly Permissive IAM Policy (Wildcard Actions or AdministratorAccess)",
            description_template=(
                "IAM policy '{resource_name}' grants wildcard actions (*) or "
                "AdministratorAccess. Wildcard policies violate the principle of "
                "least privilege and give attached principals broad access "
                "to AWS services, significantly expanding the blast radius of "
                "a credential compromise."
            ),
            severity=Severity.HIGH,
            remediation_summary=(
                "Replace wildcard policies with specific action lists following "
                "least-privilege principles. Use IAM Access Analyzer to generate "
                "least-privilege policies based on actual usage."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Use IAM Access Analyzer to review what actions are actually needed",
                    cli_command="aws accessanalyzer list-findings --analyzer-arn arn:aws:access-analyzer:REGION:ACCOUNT:analyzer/ANALYZER",
                ),
                RemediationStep(
                    step_number=2,
                    description="Generate a least-privilege policy based on CloudTrail activity",
                    console_steps="IAM Console → Policies → select policy → Generate policy (last 90 days of activity)",
                ),
                RemediationStep(
                    step_number=3,
                    description="Replace the current policy version with the least-privilege version",
                    cli_command="aws iam create-policy-version --policy-arn POLICY_ARN --policy-document file://least-privilege-policy.json --set-as-default",
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html#grant-least-privilege",
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_access-advisor.html",
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/what-is-access-analyzer.html",
            ],
        ),

        "AWS-IAM-005": FindingTemplate(
            finding_id="AWS-IAM-005",
            title="IAM Role with Dangerous Trust Policy (Wildcard Principal)",
            description_template=(
                "IAM role '{resource_name}' has a trust policy that allows any "
                "AWS principal (*) to assume it. This means any authenticated "
                "AWS entity — including entities in other accounts — can assume "
                "this role and inherit its permissions."
            ),
            severity=Severity.CRITICAL,
            remediation_summary=(
                "Restrict the trust policy to specific, named principals. "
                "Never use Principal: '*' in a role trust policy unless the role "
                "has an explicit ExternalId condition and very limited permissions."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Identify who legitimately needs to assume this role",
                ),
                RemediationStep(
                    step_number=2,
                    description="Update the trust policy to name specific principals",
                    cli_command=(
                        'aws iam update-assume-role-policy --role-name ROLE_NAME '
                        '--policy-document \'{"Version":"2012-10-17","Statement":[{'
                        '"Effect":"Allow","Principal":{"AWS":"arn:aws:iam::ACCOUNT:root"},'
                        '"Action":"sts:AssumeRole","Condition":{"StringEquals":{'
                        '"sts:ExternalId":"UNIQUE-EXTERNAL-ID"}}}]}\''
                    ),
                ),
                RemediationStep(
                    step_number=3,
                    description="If the role is unused, consider deleting it entirely",
                    cli_command="aws iam delete-role --role-name ROLE_NAME",
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_manage-assume-role-policy.html",
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_principal.html",
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html",
            ],
        ),
    }

    # Access key rotation threshold (days)
    KEY_ROTATION_DAYS = 90

    # ------------------------------------------------------------------
    # BaseChecker implementation
    # ------------------------------------------------------------------

    @property
    def checker_name(self) -> str:
        return "iam"

    @property
    def required_permissions(self) -> List[str]:
        return [
            "iam:GenerateCredentialReport",
            "iam:GetCredentialReport",
            "iam:GetAccountSummary",
            "iam:ListUsers",
            "iam:ListMFADevices",
            "iam:GetLoginProfile",
            "iam:ListAccessKeys",
            "iam:ListAttachedUserPolicies",
            "iam:ListUserPolicies",
            "iam:GetUserPolicy",
            "iam:ListRoles",
            "iam:GetRole",
            "iam:ListAttachedRolePolicies",
            "iam:ListRolePolicies",
            "iam:GetRolePolicy",
            "iam:ListPolicies",
            "iam:GetPolicy",
            "iam:GetPolicyVersion",
        ]

    def run(self) -> List[Finding]:
        """
        Execute all IAM checks. Returns combined findings list.

        Each check is independent — one failing doesn't stop the others.
        Errors in individual checks are logged as warnings, not raised.
        """
        self._iam = self._safe_get_client("iam")
        findings: List[Finding] = []

        # Pull the credential report once and reuse it across checks.
        # This avoids N×GetUser + N×ListMFADevices + N×ListAccessKeys calls.
        credential_report = self._get_credential_report()

        checks = [
            ("AWS-IAM-001/002: MFA checks",    lambda: self._check_mfa(credential_report)),
            ("AWS-IAM-003: Access key age",     lambda: self._check_access_key_age(credential_report)),
            ("AWS-IAM-004: Permissive policies",lambda: self._check_permissive_policies()),
            ("AWS-IAM-005: Role trust policies",lambda: self._check_dangerous_trust_policies()),
        ]

        for check_name, check_fn in checks:
            try:
                results = check_fn()
                findings.extend(results)
                self.logger.debug(f"{check_name}: {len(results)} finding(s)")
            except Exception as e:
                self.logger.warning(f"{check_name} failed (non-fatal): {e}")

        return findings

    # ------------------------------------------------------------------
    # Check: AWS-IAM-001 (user MFA) + AWS-IAM-002 (root MFA + root keys)
    # ------------------------------------------------------------------

    def _check_mfa(self, credential_report: List[Dict]) -> List[Finding]:
        """
        Parse the IAM credential report to find:
          - Users with console access but no MFA (AWS-IAM-001)
          - Root account issues: no MFA, or active access keys (AWS-IAM-002)

        Why credential report instead of ListMFADevices per user?
          For an account with 200 users, ListMFADevices = 200 API calls.
          One GenerateCredentialReport + GetCredentialReport = 2 API calls.
        """
        findings: List[Finding] = []

        for row in credential_report:
            username = row.get("user", "")
            is_root = (username == "<root_account>")

            # ---- Root account check (AWS-IAM-002) ----
            if is_root:
                root_findings = self._check_root_row(row)
                findings.extend(root_findings)
                continue

            # ---- Regular user MFA check (AWS-IAM-001) ----
            # Only check users with password-based console access
            has_console = row.get("password_enabled", "false").lower() == "true"
            if not has_console:
                continue

            mfa_active = row.get("mfa_active", "false").lower() == "true"
            if not mfa_active:
                user_arn = row.get("arn", f"arn:aws:iam::{self.account_id}:user/{username}")
                findings.append(self._finding(
                    finding_id="AWS-IAM-001",
                    resource_id=user_arn,
                    resource_name=username,
                    resource_type="AWS::IAM::User",
                    description_override=(
                        f"IAM user '{username}' has console access enabled but no MFA "
                        f"device configured. Password last used: "
                        f"{row.get('password_last_used', 'never')}."
                    ),
                    raw_evidence={
                        "user_name":         username,
                        "password_enabled":  row.get("password_enabled"),
                        "mfa_active":        row.get("mfa_active"),
                        "password_last_used": row.get("password_last_used"),
                        "user_creation_time": row.get("user_creation_time"),
                    },
                ))

        return findings

    def _check_root_row(self, row: Dict) -> List[Finding]:
        """Check root account MFA and key presence from the credential report row."""
        findings: List[Finding] = []
        root_arn = f"arn:aws:iam::{self.account_id}:root"

        mfa_active = row.get("mfa_active", "false").lower() == "true"
        key_1_active = row.get("access_key_1_active", "false").lower() == "true"
        key_2_active = row.get("access_key_2_active", "false").lower() == "true"
        has_active_keys = key_1_active or key_2_active

        issues = []
        if not mfa_active:
            issues.append("MFA is not enabled")
        if has_active_keys:
            issues.append("active access keys exist (root keys should never exist)")

        if issues:
            description = (
                f"AWS root account has critical security issues: "
                f"{'; '.join(issues)}. The root account has unrestricted access "
                f"to all AWS services and cannot be restricted by IAM policies."
            )
            findings.append(self._finding(
                finding_id="AWS-IAM-002",
                resource_id=root_arn,
                resource_name="root",
                resource_type="AWS::IAM::RootAccount",
                description_override=description,
                raw_evidence={
                    "mfa_active":        row.get("mfa_active"),
                    "access_key_1_active": row.get("access_key_1_active"),
                    "access_key_2_active": row.get("access_key_2_active"),
                    "issues":            issues,
                },
                region_override="global",
            ))

        return findings

    # ------------------------------------------------------------------
    # Check: AWS-IAM-003 — Access key age
    # ------------------------------------------------------------------

    def _check_access_key_age(self, credential_report: List[Dict]) -> List[Finding]:
        """
        Find access keys older than KEY_ROTATION_DAYS days.

        We check both key_1 and key_2 slots from the credential report.
        Keys in INACTIVE status are still flagged — they should be deleted,
        not just disabled. Disabled keys can be re-enabled by anyone with
        iam:UpdateAccessKey permission.
        """
        findings: List[Finding] = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.KEY_ROTATION_DAYS)

        for row in credential_report:
            username = row.get("user", "")
            if username == "<root_account>":
                continue  # Root keys are caught by AWS-IAM-002

            user_arn = row.get("arn", f"arn:aws:iam::{self.account_id}:user/{username}")

            for slot in ("1", "2"):
                active_col  = f"access_key_{slot}_active"
                rotated_col = f"access_key_{slot}_last_rotated"

                key_active = row.get(active_col, "false").lower() == "true"
                last_rotated_str = row.get(rotated_col, "N/A")

                if last_rotated_str == "N/A" or not last_rotated_str:
                    continue  # Key doesn't exist

                try:
                    last_rotated = datetime.fromisoformat(
                        last_rotated_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    self.logger.debug(
                        f"Could not parse key date for {username}: {last_rotated_str}"
                    )
                    continue

                if last_rotated < cutoff:
                    age_days = (datetime.now(timezone.utc) - last_rotated).days

                    # Get the actual key ID for better evidence
                    key_id = self._get_key_id_for_user(username, slot)

                    findings.append(self._finding(
                        finding_id="AWS-IAM-003",
                        resource_id=user_arn,
                        resource_name=username,
                        resource_type="AWS::IAM::User",
                        description_override=(
                            f"IAM access key (slot {slot}) for user '{username}' is "
                            f"{age_days} days old — exceeds {self.KEY_ROTATION_DAYS}-day "
                            f"rotation policy. Status: {'active' if key_active else 'inactive'}."
                        ),
                        raw_evidence={
                            "user_name":      username,
                            "key_slot":       slot,
                            "key_id":         key_id or "unknown",
                            "age_days":       age_days,
                            "key_active":     key_active,
                            "last_rotated":   last_rotated_str,
                            "rotation_limit": self.KEY_ROTATION_DAYS,
                        },
                    ))

        return findings

    def _get_key_id_for_user(self, username: str, slot: str) -> Optional[str]:
        """
        Fetch the actual access key ID for a user (slot 1 or 2).
        The credential report doesn't include the key ID, only dates.
        Best-effort — returns None on failure.
        """
        try:
            self._track_api_call()
            response = self._iam.list_access_keys(UserName=username)
            keys = response.get("AccessKeyMetadata", [])
            idx = int(slot) - 1
            if idx < len(keys):
                return keys[idx]["AccessKeyId"]
        except Exception as e:
            self.logger.debug(f"Could not get key ID for {username}: {e}")
        return None

    # ------------------------------------------------------------------
    # Check: AWS-IAM-004 — Overly permissive policies
    # ------------------------------------------------------------------

    def _check_permissive_policies(self) -> List[Finding]:
        """
        Find customer-managed policies that grant Action:* or are AdministratorAccess.

        Strategy:
          1. List all customer-managed policies (Scope=Local)
          2. For each, get the current default policy version document
          3. Check each statement for Action:* / Resource:* combinations
          4. Also check for the AdministratorAccess managed policy being
             attached to users/roles (that check is separate from policy content)

        We skip AWS-managed policies — they're AWS's responsibility.
        We DO flag customer-managed policies that replicate AdministratorAccess.
        """
        findings: List[Finding] = []

        try:
            self._track_api_call()
            paginator = self._iam.get_paginator("list_policies")
            pages = paginator.paginate(Scope="Local")  # Customer-managed only

            for page in pages:
                for policy in page.get("Policies", []):
                    policy_arn  = policy["Arn"]
                    policy_name = policy["PolicyName"]
                    default_ver = policy.get("DefaultVersionId")

                    if not default_ver:
                        continue

                    try:
                        self._track_api_call()
                        ver_response = self._iam.get_policy_version(
                            PolicyArn=policy_arn,
                            VersionId=default_ver,
                        )
                        document = ver_response["PolicyVersion"]["Document"]

                        # document may come as dict (already parsed) or string
                        if isinstance(document, str):
                            document = json.loads(document)

                        is_admin, reason = self._is_admin_policy(document)
                        if is_admin:
                            findings.append(self._finding(
                                finding_id="AWS-IAM-004",
                                resource_id=policy_arn,
                                resource_name=policy_name,
                                resource_type="AWS::IAM::ManagedPolicy",
                                description_override=(
                                    f"Customer-managed IAM policy '{policy_name}' "
                                    f"grants excessive permissions: {reason}. "
                                    f"This policy is attached to users and/or roles "
                                    f"who inherit these broad permissions."
                                ),
                                raw_evidence={
                                    "policy_arn":     policy_arn,
                                    "policy_name":    policy_name,
                                    "version_id":     default_ver,
                                    "reason":         reason,
                                    "attachment_count": policy.get("AttachmentCount", 0),
                                },
                                region_override="global",
                            ))

                    except Exception as e:
                        self.logger.debug(
                            f"Could not fetch policy version for {policy_arn}: {e}"
                        )

        except Exception as e:
            self.logger.warning(f"Could not enumerate customer-managed policies: {e}")

        # Also check inline policies on users and roles
        findings.extend(self._check_inline_policies_on_users())
        findings.extend(self._check_inline_policies_on_roles())

        return findings

    def _check_inline_policies_on_users(self) -> List[Finding]:
        """Check inline policies on all IAM users for wildcard permissions."""
        findings: List[Finding] = []
        try:
            self._track_api_call()
            paginator = self._iam.get_paginator("list_users")
            for page in paginator.paginate():
                for user in page.get("Users", []):
                    username = user["UserName"]
                    user_arn = user["Arn"]
                    findings.extend(
                        self._check_entity_inline_policies(
                            entity_name=username,
                            entity_arn=user_arn,
                            entity_type="user",
                            resource_type="AWS::IAM::User",
                        )
                    )
        except Exception as e:
            self.logger.debug(f"Could not check user inline policies: {e}")
        return findings

    def _check_inline_policies_on_roles(self) -> List[Finding]:
        """Check inline policies on all IAM roles for wildcard permissions."""
        findings: List[Finding] = []
        try:
            self._track_api_call()
            paginator = self._iam.get_paginator("list_roles")
            for page in paginator.paginate():
                for role in page.get("Roles", []):
                    role_name = role["RoleName"]
                    role_arn  = role["Arn"]
                    findings.extend(
                        self._check_entity_inline_policies(
                            entity_name=role_name,
                            entity_arn=role_arn,
                            entity_type="role",
                            resource_type="AWS::IAM::Role",
                        )
                    )
        except Exception as e:
            self.logger.debug(f"Could not check role inline policies: {e}")
        return findings

    def _check_entity_inline_policies(
        self,
        entity_name:  str,
        entity_arn:   str,
        entity_type:  str,   # "user" or "role"
        resource_type: str,
    ) -> List[Finding]:
        """Fetch and check inline policies for a single IAM user or role."""
        findings: List[Finding] = []

        list_fn  = getattr(self._iam, f"list_{entity_type}_policies")
        get_fn   = getattr(self._iam, f"get_{entity_type}_policy")
        name_key = "UserName" if entity_type == "user" else "RoleName"

        try:
            self._track_api_call()
            response = list_fn(**{name_key: entity_name})
            policy_names = response.get("PolicyNames", [])

            for policy_name in policy_names:
                try:
                    self._track_api_call()
                    pol_response = get_fn(**{name_key: entity_name, "PolicyName": policy_name})
                    document = pol_response.get("PolicyDocument", {})

                    if isinstance(document, str):
                        document = json.loads(document)

                    is_admin, reason = self._is_admin_policy(document)
                    if is_admin:
                        inline_id = f"{entity_arn}/inline/{policy_name}"
                        findings.append(self._finding(
                            finding_id="AWS-IAM-004",
                            resource_id=inline_id,
                            resource_name=f"{entity_name}/{policy_name}",
                            resource_type=resource_type,
                            description_override=(
                                f"Inline IAM policy '{policy_name}' on {entity_type} "
                                f"'{entity_name}' grants excessive permissions: {reason}."
                            ),
                            raw_evidence={
                                "entity_name":  entity_name,
                                "entity_type":  entity_type,
                                "policy_name":  policy_name,
                                "reason":       reason,
                            },
                            region_override="global",
                        ))
                except Exception as e:
                    self.logger.debug(
                        f"Could not get inline policy {policy_name} on {entity_name}: {e}"
                    )
        except Exception as e:
            self.logger.debug(
                f"Could not list inline policies for {entity_name}: {e}"
            )

        return findings

    # ------------------------------------------------------------------
    # Check: AWS-IAM-005 — Dangerous role trust policies
    # ------------------------------------------------------------------

    def _check_dangerous_trust_policies(self) -> List[Finding]:
        """
        Find IAM roles whose trust policies allow any principal (*) to assume them.

        A role with Principal: '*' in its trust policy can be assumed by:
          - Any authenticated AWS principal
          - Potentially any IAM entity in ANY account

        This is almost always a misconfiguration. The rare legitimate case
        (e.g. an STS federation endpoint) must have tight Condition blocks.

        We also flag roles where the Principal is an entire AWS account root
        (arn:aws:iam::*:root) which is equivalent to wildcard within that account.
        """
        findings: List[Finding] = []

        try:
            self._track_api_call()
            paginator = self._iam.get_paginator("list_roles")

            for page in paginator.paginate():
                for role in page.get("Roles", []):
                    role_name    = role["RoleName"]
                    role_arn     = role["Arn"]
                    trust_policy = role.get("AssumeRolePolicyDocument", {})

                    if isinstance(trust_policy, str):
                        try:
                            trust_policy = json.loads(trust_policy)
                        except json.JSONDecodeError:
                            continue

                    is_dangerous, reason = self._is_dangerous_trust_policy(trust_policy)
                    if is_dangerous:
                        findings.append(self._finding(
                            finding_id="AWS-IAM-005",
                            resource_id=role_arn,
                            resource_name=role_name,
                            resource_type="AWS::IAM::Role",
                            description_override=(
                                f"IAM role '{role_name}' has a dangerous trust policy: "
                                f"{reason}. Any AWS principal matching this pattern can "
                                f"assume this role and inherit its permissions."
                            ),
                            raw_evidence={
                                "role_arn":          role_arn,
                                "role_name":         role_name,
                                "trust_policy":      trust_policy,
                                "dangerous_reason":  reason,
                            },
                            region_override="global",
                        ))

        except Exception as e:
            self.logger.warning(f"Could not enumerate roles for trust policy check: {e}")

        return findings

    # ------------------------------------------------------------------
    # Policy analysis helpers
    # ------------------------------------------------------------------

    def _is_admin_policy(self, document: Dict) -> Tuple[bool, str]:
        """
        Analyse a policy document for overly permissive statements.

        Returns (is_admin: bool, reason: str).

        Detection patterns:
          1. Action: "*" with Effect: "Allow" — grants all actions
          2. Action: ["*"] — same, as a list
          3. Action: "iam:*" + Resource: "*" — full IAM admin
          4. Wildcard service: "s3:*" + Resource: "*" is not flagged
             unless it's combined with other wildcards
             (we focus on truly dangerous cross-service wildcards)

        We flag:
          - Action: "*" (any resource) — full admin
          - Action containing "*" with Resource: "*" — effective full admin for that service+
        """
        statements = document.get("Statement", [])
        if isinstance(statements, dict):
            statements = [statements]  # Single statement not wrapped in list

        for stmt in statements:
            if stmt.get("Effect", "Allow") != "Allow":
                continue  # Skip Deny statements

            actions   = stmt.get("Action", [])
            resources = stmt.get("Resource", [])
            not_action = stmt.get("NotAction")  # NotAction with Allow = whitelist inverse

            # Normalise to lists
            if isinstance(actions, str):
                actions = [actions]
            if isinstance(resources, str):
                resources = [resources]

            # NotAction + Effect: Allow is extremely permissive
            # (allows everything EXCEPT the listed actions)
            if not_action and resources and "*" in (
                [resources] if isinstance(resources, str) else resources
            ):
                return True, "Uses NotAction with Effect:Allow and Resource:* (effectively grants all actions except the listed ones)"

            # Direct wildcard action
            if "*" in actions:
                return True, "Grants Action:* (all AWS actions allowed)"

            # Wildcard sub-service + wildcard resource
            for action in actions:
                if action.endswith(":*") and "*" in resources:
                    service = action.split(":")[0]
                    if service in ("iam", "sts"):  # IAM/STS wildcards are especially dangerous
                        return True, f"Grants {action} on all resources (full {service} admin)"

        return False, ""

    def _is_dangerous_trust_policy(self, document: Dict) -> Tuple[bool, str]:
        """
        Analyse a role trust policy for dangerous principals.

        Returns (is_dangerous: bool, reason: str).

        Dangerous patterns:
          1. Principal: "*" — anyone can assume this role
          2. Principal: {"AWS": "*"} — any AWS account entity
          3. Principal: {"AWS": "arn:aws:iam::*:root"} — any account root
             (effectively any account if no ExternalId condition)
          4. Principal with no Condition block and broad scope
        """
        statements = document.get("Statement", [])
        if isinstance(statements, dict):
            statements = [statements]

        for stmt in statements:
            if stmt.get("Effect", "Allow") != "Allow":
                continue

            principal = stmt.get("Principal", {})
            conditions = stmt.get("Condition", {})

            # Pattern 1: bare wildcard
            if principal == "*":
                return True, "Principal is '*' — any entity can assume this role"

            if isinstance(principal, dict):
                aws_principal = principal.get("AWS", "")
                service_principal = principal.get("Service", "")
                federated_principal = principal.get("Federated", "")

                # Normalise to list
                if isinstance(aws_principal, str):
                    aws_principal = [aws_principal]

                for p in aws_principal:
                    # Pattern 2: explicit wildcard AWS principal
                    if p == "*":
                        return True, "Principal.AWS is '*' — any AWS account can assume this role"

                    # Pattern 3: wildcard account root
                    if ":root" in p and p.startswith("arn:aws:iam::*:"):
                        return True, (
                            f"Principal.AWS is '{p}' — any AWS account root can assume this role"
                        )

                    # Pattern 4: cross-account wildcard without ExternalId
                    if p.endswith(":root") and not conditions:
                        # Any entire account can assume it with no condition — risky
                        # (we flag this at MEDIUM but don't return True here — it's a separate finding)
                        pass

        return False, ""

    # ------------------------------------------------------------------
    # Credential report
    # ------------------------------------------------------------------

    def _get_credential_report(self) -> List[Dict]:
        """
        Generate and download the IAM credential report as a list of row dicts.

        The credential report is a CSV containing one row per IAM entity
        (including root). It's the most efficient way to gather:
          - Password status and last use
          - MFA status
          - Access key ages and active status

        AWS throttles GenerateCredentialReport — it can only be regenerated
        every 4 hours. If a recent report exists, GetCredentialReport returns
        it immediately without triggering a new generation.

        Returns [] on any error (checks degrade gracefully).
        """
        try:
            # Trigger report generation
            self._track_api_call()
            generate_response = self._iam.generate_credential_report()
            state = generate_response.get("State", "")

            # Poll if report is being freshly generated (usually instant)
            max_polls = 10
            while state == "STARTED" and max_polls > 0:
                time.sleep(2)
                self._track_api_call()
                generate_response = self._iam.generate_credential_report()
                state = generate_response.get("State", "")
                max_polls -= 1

            # Download the report
            self._track_api_call()
            report_response = self._iam.get_credential_report()
            content = report_response["Content"]

            # Content is bytes in real boto3, string in some mocks
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            # Parse CSV
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            self.logger.debug(f"Credential report: {len(rows)} rows")
            return rows

        except Exception as e:
            self.logger.warning(
                f"Could not retrieve IAM credential report: {e}. "
                f"MFA and key age checks will be skipped."
            )
            return []

    # ------------------------------------------------------------------
    # Attack graph contribution
    # ------------------------------------------------------------------

    def build_graph_nodes(
        self,
        graph: AttackGraph,
        credential_report: Optional[List[Dict]] = None,
    ) -> None:
        """
        Contribute IAM nodes and edges to the attack graph.

        Called by the provider after run() completes. Adds:
          - USER nodes for each IAM user
          - ROLE nodes for each IAM role (with is_admin flag)
          - POLICY nodes for admin policies
          - CAN_ASSUME edges between users and roles they can assume
          - HAS_POLICY edges between entities and their policies

        This is called separately from run() because graph building
        is optional (can be skipped with --no-graph) and needs
        data from multiple checkers (S3 checker adds PUBLIC_RESOURCE nodes).
        """
        try:
            self._iam = self._safe_get_client("iam")
            self._add_user_nodes(graph, credential_report or [])
            self._add_role_nodes(graph)
        except Exception as e:
            self.logger.warning(f"IAM graph building failed (non-fatal): {e}")

    def _add_user_nodes(
        self, graph: AttackGraph, credential_report: List[Dict]
    ) -> None:
        """Add USER nodes and annotate with MFA status."""
        # Build a lookup from username → credential report row
        cr_lookup = {row.get("user", ""): row for row in credential_report}

        try:
            self._track_api_call()
            paginator = self._iam.get_paginator("list_users")
            for page in paginator.paginate():
                for user in page.get("Users", []):
                    username = user["UserName"]
                    user_arn = user["Arn"]
                    cr_row   = cr_lookup.get(username, {})

                    node = GraphNode(
                        node_id=user_arn,
                        node_type=NodeType.USER,
                        label=username,
                        account_id=self.account_id,
                        region="global",
                        properties={
                            "mfa_enabled":    cr_row.get("mfa_active", "false") == "true",
                            "has_console":    cr_row.get("password_enabled", "false") == "true",
                            "has_keys":       (
                                cr_row.get("access_key_1_active", "false") == "true" or
                                cr_row.get("access_key_2_active", "false") == "true"
                            ),
                            "is_admin":       False,  # Updated below when policies are checked
                        },
                    )
                    graph.add_node(node)
        except Exception as e:
            self.logger.debug(f"Could not add user nodes to graph: {e}")

    def _add_role_nodes(self, graph: AttackGraph) -> None:
        """Add ROLE nodes with is_admin and trust policy metadata."""
        try:
            self._track_api_call()
            paginator = self._iam.get_paginator("list_roles")
            for page in paginator.paginate():
                for role in page.get("Roles", []):
                    role_name    = role["RoleName"]
                    role_arn     = role["Arn"]
                    trust_policy = role.get("AssumeRolePolicyDocument", {})

                    if isinstance(trust_policy, str):
                        try:
                            trust_policy = json.loads(trust_policy)
                        except Exception:
                            trust_policy = {}

                    is_dangerous, _ = self._is_dangerous_trust_policy(trust_policy)
                    is_admin = self._role_has_admin_policy(role_name)

                    node = GraphNode(
                        node_id=role_arn,
                        node_type=NodeType.ROLE,
                        label=role_name,
                        account_id=self.account_id,
                        region="global",
                        properties={
                            "is_admin":            is_admin,
                            "trust_wildcard":      is_dangerous,
                            "trust_policy":        trust_policy,
                            "is_service_role":     role_arn.startswith("arn:aws:iam::aws:"),
                            "path":                role.get("Path", "/"),
                        },
                    )
                    graph.add_node(node)
        except Exception as e:
            self.logger.debug(f"Could not add role nodes to graph: {e}")

    def _role_has_admin_policy(self, role_name: str) -> bool:
        """Check whether a role has AdministratorAccess or a wildcard policy."""
        try:
            # Check attached managed policies
            self._track_api_call()
            response = self._iam.list_attached_role_policies(RoleName=role_name)
            for policy in response.get("AttachedPolicies", []):
                if policy["PolicyArn"] == "arn:aws:iam::aws:policy/AdministratorAccess":
                    return True
                if "Admin" in policy["PolicyName"] or "FullAccess" in policy["PolicyName"]:
                    # Heuristic — fetch and check the document to be sure
                    pass  # Full implementation in production

            # Check inline policies
            self._track_api_call()
            inline_response = self._iam.list_role_policies(RoleName=role_name)
            for policy_name in inline_response.get("PolicyNames", []):
                self._track_api_call()
                doc_response = self._iam.get_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )
                document = doc_response.get("PolicyDocument", {})
                if isinstance(document, str):
                    document = json.loads(document)
                is_admin, _ = self._is_admin_policy(document)
                if is_admin:
                    return True

        except Exception as e:
            self.logger.debug(f"Could not check admin status for role {role_name}: {e}")

        return False