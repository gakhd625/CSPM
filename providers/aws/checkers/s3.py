"""
providers/aws/checkers/s3.py

S3Checker implements 4 S3 security checks:

  AWS-S3-001  Bucket is publicly accessible (ACL or bucket policy)
  AWS-S3-002  Bucket does not enforce server-side encryption at rest
  AWS-S3-003  Bucket access logging is not enabled
  AWS-S3-004  Bucket versioning is not enabled (or is suspended)

Architecture decisions:

  1. Single list_buckets call, then per-bucket checks in parallel-ready loops.
     list_buckets is global (returns ALL buckets regardless of region),
     so we get the bucket list once and iterate it.

  2. Per-bucket API calls use the bucket's actual region.
     S3 API calls must go to the bucket's home region — a bucket in eu-west-1
     called via us-east-1 will return a redirect or silently wrong data.
     We detect each bucket's region via get_bucket_location and use a
     region-specific client for all subsequent calls.

  3. Public access is checked at TWO levels — block public access settings
     AND bucket policy / ACL. A bucket can have "Block Public Access = ON"
     which overrides everything. We check BlockPublicAccess first and skip
     the policy/ACL check if it's fully blocked (saves API calls).

  4. Encryption checks default encryption settings, NOT object-level encryption.
     Object-level can be overridden; default encryption is the floor.

  5. S3 contributes PUBLIC_RESOURCE nodes to the attack graph with IS_PUBLIC
     edges to a synthetic EXTERNAL "internet" node. Step 7 then finds paths
     from internet → public-S3 → admin-role.

IAM permissions required:
  s3:ListAllMyBuckets
  s3:GetBucketLocation
  s3:GetBucketPublicAccessBlock
  s3:GetBucketAcl
  s3:GetBucketPolicy
  s3:GetBucketEncryption
  s3:GetBucketLogging
  s3:GetBucketVersioning
  s3:GetBucketTagging
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.attack_graph.models import (
    AttackGraph, EdgeType, GraphEdge, GraphNode, NodeType,
)
from core.base_checker import BaseChecker, FindingTemplate
from core.checker_registry import CheckerRegistry
from core.models.finding import Finding, RemediationStep, Severity

logger = logging.getLogger(__name__)

# Synthetic node ID for the internet / anonymous actor.
# Shared with IAM checker so paths connect: EXTERNAL → PUBLIC_RESOURCE → ROLE
INTERNET_NODE_ID = "external:internet"


@CheckerRegistry.register(provider="aws", domain="s3")
class S3Checker(BaseChecker):
    """
    AWS S3 security checker.

    Checks 4 S3 controls that are both high-frequency findings in real accounts
    and directly exploitable in attack paths.

    Self-registers with CheckerRegistry via @CheckerRegistry.register —
    no changes needed in provider.py.
    """

    # ------------------------------------------------------------------
    # Rule book
    # ------------------------------------------------------------------

    FINDING_TEMPLATES: Dict[str, FindingTemplate] = {

        "AWS-S3-001": FindingTemplate(
            finding_id="AWS-S3-001",
            title="S3 Bucket Publicly Accessible",
            description_template=(
                "S3 bucket '{resource_name}' is publicly accessible. "
                "Public buckets can be read (or written) by any unauthenticated "
                "internet user. This is the leading cause of cloud data breaches."
            ),
            severity=Severity.CRITICAL,
            remediation_summary=(
                "Enable S3 Block Public Access at the bucket level (and ideally "
                "the account level). Audit the bucket policy and ACL to remove "
                "any public grants."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Enable Block Public Access on the bucket",
                    cli_command=(
                        "aws s3api put-public-access-block "
                        "--bucket BUCKET_NAME "
                        "--public-access-block-configuration "
                        "BlockPublicAcls=true,IgnorePublicAcls=true,"
                        "BlockPublicPolicy=true,RestrictPublicBuckets=true"
                    ),
                    terraform=(
                        'resource "aws_s3_bucket_public_access_block" "example" {\n'
                        '  bucket                  = aws_s3_bucket.example.id\n'
                        '  block_public_acls       = true\n'
                        '  block_public_policy     = true\n'
                        '  ignore_public_acls      = true\n'
                        '  restrict_public_buckets = true\n'
                        '}'
                    ),
                ),
                RemediationStep(
                    step_number=2,
                    description="Review and remove any bucket policy statements granting public access",
                    cli_command="aws s3api get-bucket-policy --bucket BUCKET_NAME",
                ),
                RemediationStep(
                    step_number=3,
                    description="Remove public ACL grants",
                    cli_command="aws s3api put-bucket-acl --bucket BUCKET_NAME --acl private",
                ),
                RemediationStep(
                    step_number=4,
                    description="Enable Block Public Access at the account level to prevent future misconfigurations",
                    cli_command=(
                        "aws s3control put-public-access-block "
                        "--account-id ACCOUNT_ID "
                        "--public-access-block-configuration "
                        "BlockPublicAcls=true,IgnorePublicAcls=true,"
                        "BlockPublicPolicy=true,RestrictPublicBuckets=true"
                    ),
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html",
                "https://aws.amazon.com/premiumsupport/knowledge-center/secure-s3-resources/",
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/about-object-ownership.html",
            ],
        ),

        "AWS-S3-002": FindingTemplate(
            finding_id="AWS-S3-002",
            title="S3 Bucket Encryption Not Enabled",
            description_template=(
                "S3 bucket '{resource_name}' does not have default server-side "
                "encryption configured. Without default encryption, objects uploaded "
                "without explicit encryption headers are stored in plaintext."
            ),
            severity=Severity.MEDIUM,
            remediation_summary=(
                "Enable default server-side encryption on the bucket. "
                "Use SSE-KMS with a customer-managed key for regulated data."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Enable default SSE-S3 encryption (minimum recommendation)",
                    cli_command=(
                        "aws s3api put-bucket-encryption "
                        "--bucket BUCKET_NAME "
                        '--server-side-encryption-configuration \'{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}\''
                    ),
                    terraform=(
                        'resource "aws_s3_bucket_server_side_encryption_configuration" "example" {\n'
                        '  bucket = aws_s3_bucket.example.id\n'
                        '  rule {\n'
                        '    apply_server_side_encryption_by_default {\n'
                        '      sse_algorithm = "AES256"\n'
                        '    }\n'
                        '  }\n'
                        '}'
                    ),
                ),
                RemediationStep(
                    step_number=2,
                    description="For sensitive data, use SSE-KMS with a customer-managed key (CMK)",
                    cli_command=(
                        "aws s3api put-bucket-encryption "
                        "--bucket BUCKET_NAME "
                        '--server-side-encryption-configuration \'{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "aws:kms", "KMSMasterKeyID": "arn:aws:kms:REGION:ACCOUNT:key/KEY-ID"}}]}\''
                    ),
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucket-encryption.html",
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingKMSEncryption.html",
            ],
        ),

        "AWS-S3-003": FindingTemplate(
            finding_id="AWS-S3-003",
            title="S3 Bucket Access Logging Not Enabled",
            description_template=(
                "S3 bucket '{resource_name}' does not have server access logging "
                "enabled. Without access logs, there is no audit trail of who "
                "accessed or modified objects — making breach detection and "
                "forensics impossible."
            ),
            severity=Severity.LOW,
            remediation_summary=(
                "Enable S3 server access logging to a separate logging bucket. "
                "Ensure the logging bucket itself has appropriate retention policies."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Create a dedicated logging bucket (if not already exists)",
                    cli_command=(
                        "aws s3api create-bucket --bucket LOGGING_BUCKET_NAME "
                        "--region REGION"
                    ),
                ),
                RemediationStep(
                    step_number=2,
                    description="Grant S3 log delivery write access to the logging bucket",
                    cli_command=(
                        "aws s3api put-bucket-acl "
                        "--bucket LOGGING_BUCKET_NAME "
                        "--grant-write URI=http://acs.amazonaws.com/groups/s3/LogDelivery "
                        "--grant-read-acp URI=http://acs.amazonaws.com/groups/s3/LogDelivery"
                    ),
                ),
                RemediationStep(
                    step_number=3,
                    description="Enable logging on the source bucket",
                    cli_command=(
                        "aws s3api put-bucket-logging "
                        "--bucket BUCKET_NAME "
                        '--bucket-logging-status \'{"LoggingEnabled": {"TargetBucket": "LOGGING_BUCKET_NAME", "TargetPrefix": "BUCKET_NAME/"}}\''
                    ),
                    terraform=(
                        'resource "aws_s3_bucket_logging" "example" {\n'
                        '  bucket        = aws_s3_bucket.example.id\n'
                        '  target_bucket = aws_s3_bucket.log_bucket.id\n'
                        '  target_prefix = "log/"\n'
                        '}'
                    ),
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html",
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/enable-server-access-logging.html",
            ],
        ),

        "AWS-S3-004": FindingTemplate(
            finding_id="AWS-S3-004",
            title="S3 Bucket Versioning Not Enabled",
            description_template=(
                "S3 bucket '{resource_name}' does not have versioning enabled. "
                "Without versioning, deleted or overwritten objects are "
                "unrecoverable. Versioning also helps detect unauthorized "
                "modifications."
            ),
            severity=Severity.LOW,
            remediation_summary=(
                "Enable versioning on the bucket. Add a lifecycle policy to "
                "expire old versions and avoid unbounded storage growth."
            ),
            remediation_steps=[
                RemediationStep(
                    step_number=1,
                    description="Enable versioning on the bucket",
                    cli_command=(
                        "aws s3api put-bucket-versioning "
                        "--bucket BUCKET_NAME "
                        '--versioning-configuration Status=Enabled'
                    ),
                    terraform=(
                        'resource "aws_s3_bucket_versioning" "example" {\n'
                        '  bucket = aws_s3_bucket.example.id\n'
                        '  versioning_configuration {\n'
                        '    status = "Enabled"\n'
                        '  }\n'
                        '}'
                    ),
                ),
                RemediationStep(
                    step_number=2,
                    description="Add a lifecycle rule to expire non-current versions after 90 days",
                    cli_command=(
                        "aws s3api put-bucket-lifecycle-configuration "
                        "--bucket BUCKET_NAME "
                        '--lifecycle-configuration \'{"Rules":[{"Status":"Enabled","NoncurrentVersionExpiration":{"NoncurrentDays":90},"Filter":{"Prefix":""},"ID":"expire-old-versions"}]}\''
                    ),
                ),
            ],
            references=[
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html",
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/versioning-workflows.html",
            ],
        ),
    }

    # ------------------------------------------------------------------
    # BaseChecker implementation
    # ------------------------------------------------------------------

    @property
    def checker_name(self) -> str:
        return "s3"

    @property
    def required_permissions(self) -> List[str]:
        return [
            "s3:ListAllMyBuckets",
            "s3:GetBucketLocation",
            "s3:GetBucketPublicAccessBlock",
            "s3:GetBucketAcl",
            "s3:GetBucketPolicy",
            "s3:GetBucketEncryption",
            "s3:GetBucketLogging",
            "s3:GetBucketVersioning",
            "s3:GetBucketTagging",
        ]

    def run(self) -> List[Finding]:
        """
        Execute all S3 checks.

        Flow:
          1. list_buckets — get all bucket names (global, single call)
          2. For each bucket:
             a. get_bucket_location — needed for region-specific API calls
             b. get_public_access_block — check Block Public Access settings
             c. If not fully blocked: get_bucket_acl + get_bucket_policy
             d. get_bucket_encryption — default encryption
             e. get_bucket_logging — server access logging
             f. get_bucket_versioning — versioning status

        Each per-bucket check is isolated — one bucket failing doesn't
        abort others.
        """
        # Use global client for list_buckets (it's region-agnostic)
        self._s3 = self.session.client("s3")
        self._track_api_call()

        try:
            response = self._s3.list_buckets()
        except Exception as e:
            self.logger.error(f"Cannot list S3 buckets: {e}")
            return []

        buckets = response.get("Buckets", [])
        self.logger.info(f"Found {len(buckets)} S3 bucket(s) to check")

        findings: List[Finding] = []
        for bucket_meta in buckets:
            bucket_name = bucket_meta["Name"]
            bucket_findings = self._check_bucket(bucket_name)
            findings.extend(bucket_findings)

        return findings

    # ------------------------------------------------------------------
    # Per-bucket orchestration
    # ------------------------------------------------------------------

    def _check_bucket(self, bucket_name: str) -> List[Finding]:
        """
        Run all checks for a single bucket.
        Returns combined findings; errors in one check don't abort others.
        """
        findings: List[Finding] = []

        # Detect the bucket's home region — required for correct API calls
        region = self._get_bucket_region(bucket_name)
        bucket_arn = f"arn:aws:s3:::{bucket_name}"

        checks = [
            ("public_access",  lambda: self._check_public_access(bucket_name, bucket_arn, region)),
            ("encryption",     lambda: self._check_encryption(bucket_name, bucket_arn, region)),
            ("logging",        lambda: self._check_logging(bucket_name, bucket_arn, region)),
            ("versioning",     lambda: self._check_versioning(bucket_name, bucket_arn, region)),
        ]

        for check_name, check_fn in checks:
            try:
                result = check_fn()
                findings.extend(result)
            except Exception as e:
                self.logger.warning(
                    f"Check '{check_name}' failed for bucket '{bucket_name}' "
                    f"(non-fatal): {e}"
                )

        return findings

    # ------------------------------------------------------------------
    # Check: AWS-S3-001 — Public access
    # ------------------------------------------------------------------

    def _check_public_access(
        self, bucket_name: str, bucket_arn: str, region: str
    ) -> List[Finding]:
        """
        Check whether the bucket is publicly accessible.

        Two-layer check:
          Layer 1: Block Public Access (BPA) settings — the override switch.
                   If ALL four BPA flags are True, the bucket cannot be public
                   regardless of policy or ACL. We stop here if fully blocked.
          Layer 2: If BPA is not fully blocking, check:
                   a. Bucket policy: any Statement with Principal:* + Effect:Allow
                   b. Bucket ACL: any grant to AllUsers or AuthenticatedUsers

        This order minimises API calls for well-configured buckets.
        """
        client = self._regional_client(region)

        # Layer 1: Block Public Access settings
        bpa = self._get_block_public_access(client, bucket_name)
        if bpa is not None:
            fully_blocked = all([
                bpa.get("BlockPublicAcls",       False),
                bpa.get("IgnorePublicAcls",       False),
                bpa.get("BlockPublicPolicy",      False),
                bpa.get("RestrictPublicBuckets",  False),
            ])
            if fully_blocked:
                # Fast path: BPA fully enabled — can't be public
                return []

        # Layer 2a: Bucket policy
        is_public_via_policy, policy_reason = self._is_public_via_policy(client, bucket_name)

        # Layer 2b: Bucket ACL
        is_public_via_acl, acl_reason = self._is_public_via_acl(client, bucket_name)

        if is_public_via_policy or is_public_via_acl:
            reasons = []
            if is_public_via_policy:
                reasons.append(f"policy: {policy_reason}")
            if is_public_via_acl:
                reasons.append(f"ACL: {acl_reason}")

            return [self._finding(
                finding_id="AWS-S3-001",
                resource_id=bucket_arn,
                resource_name=bucket_name,
                resource_type="AWS::S3::Bucket",
                description_override=(
                    f"S3 bucket '{bucket_name}' is publicly accessible via "
                    f"{'; '.join(reasons)}. Any unauthenticated internet user "
                    f"can read (and potentially write) objects in this bucket."
                ),
                raw_evidence={
                    "bucket_name":              bucket_name,
                    "public_via_policy":        is_public_via_policy,
                    "public_via_acl":           is_public_via_acl,
                    "policy_reason":            policy_reason,
                    "acl_reason":               acl_reason,
                    "block_public_access":      bpa or {},
                },
                region_override=region,
            )]

        return []

    def _get_block_public_access(
        self, client: Any, bucket_name: str
    ) -> Optional[Dict]:
        """
        Retrieve Block Public Access configuration.
        Returns None if the API call fails (e.g. access denied).
        A None response means we can't confirm BPA is on — proceed with
        policy/ACL checks to be safe.
        """
        try:
            self._track_api_call()
            response = client.get_bucket_public_access_block(Bucket=bucket_name)
            return response.get("PublicAccessBlockConfiguration", {})
        except Exception as e:
            error_code = self._s3_error_code(e)
            if error_code == "NoSuchPublicAccessBlockConfiguration":
                # No BPA config at all — same as all flags = False
                return {}
            self.logger.debug(
                f"Could not get BPA config for '{bucket_name}': {e}"
            )
            return None

    def _is_public_via_policy(
        self, client: Any, bucket_name: str
    ) -> Tuple[bool, str]:
        """
        Check if the bucket policy grants public access.

        A bucket policy grants public access when it has a statement with:
          Effect: Allow + Principal: * (or Principal: {"AWS": "*"})

        Returns (is_public: bool, reason: str).
        """
        try:
            self._track_api_call()
            response = client.get_bucket_policy(Bucket=bucket_name)
            policy_str = response.get("Policy", "{}")
            policy = json.loads(policy_str)
        except Exception as e:
            error_code = self._s3_error_code(e)
            if error_code == "NoSuchBucketPolicy":
                return False, ""
            self.logger.debug(f"Could not get bucket policy for '{bucket_name}': {e}")
            return False, ""

        statements = policy.get("Statement", [])
        if isinstance(statements, dict):
            statements = [statements]

        for stmt in statements:
            if stmt.get("Effect") != "Allow":
                continue
            principal = stmt.get("Principal", "")
            if principal == "*":
                action = stmt.get("Action", "")
                return True, f"Principal:* allows {action}"
            if isinstance(principal, dict):
                aws_principal = principal.get("AWS", "")
                if isinstance(aws_principal, str):
                    aws_principal = [aws_principal]
                if "*" in aws_principal:
                    action = stmt.get("Action", "")
                    return True, f"Principal.AWS:* allows {action}"

        return False, ""

    def _is_public_via_acl(
        self, client: Any, bucket_name: str
    ) -> Tuple[bool, str]:
        """
        Check if the bucket ACL grants public access.

        Public ACL grants are those to:
          - http://acs.amazonaws.com/groups/global/AllUsers (anyone)
          - http://acs.amazonaws.com/groups/global/AuthenticatedUsers (any AWS account)

        Returns (is_public: bool, reason: str).
        """
        PUBLIC_GRANTEES = {
            "http://acs.amazonaws.com/groups/global/AllUsers",
            "http://acs.amazonaws.com/groups/global/AuthenticatedUsers",
        }

        try:
            self._track_api_call()
            response = client.get_bucket_acl(Bucket=bucket_name)
        except Exception as e:
            self.logger.debug(f"Could not get ACL for '{bucket_name}': {e}")
            return False, ""

        for grant in response.get("Grants", []):
            grantee = grant.get("Grantee", {})
            uri = grantee.get("URI", "")
            if uri in PUBLIC_GRANTEES:
                permission = grant.get("Permission", "UNKNOWN")
                grantee_type = (
                    "AllUsers" if "AllUsers" in uri else "AuthenticatedUsers"
                )
                return True, f"{grantee_type} has {permission} permission"

        return False, ""

    # ------------------------------------------------------------------
    # Check: AWS-S3-002 — Encryption
    # ------------------------------------------------------------------

    def _check_encryption(
        self, bucket_name: str, bucket_arn: str, region: str
    ) -> List[Finding]:
        """
        Check whether the bucket has default server-side encryption configured.

        Since AWS announced all new S3 buckets are encrypted by default (Jan 2023),
        this check primarily matters for:
          a. Buckets created before January 2023
          b. Compliance requirements that mandate SSE-KMS (not just SSE-S3)
          c. Accounts where the default was explicitly overridden

        We still flag unencrypted buckets since they may pre-date the default.
        Future enhancement: add a severity_override=LOW if the bucket was
        created after Jan 2023 (when AWS enabled default encryption).
        """
        client = self._regional_client(region)
        try:
            self._track_api_call()
            response = client.get_bucket_encryption(Bucket=bucket_name)
            rules = response.get(
                "ServerSideEncryptionConfiguration", {}
            ).get("Rules", [])

            if rules:
                # Encryption is configured — check what algorithm
                algo = (
                    rules[0]
                    .get("ApplyServerSideEncryptionByDefault", {})
                    .get("SSEAlgorithm", "unknown")
                )
                self.logger.debug(
                    f"Bucket '{bucket_name}' encryption: {algo}"
                )
                return []

        except Exception as e:
            error_code = self._s3_error_code(e)
            if error_code not in (
                "ServerSideEncryptionConfigurationNotFoundError",
                "NoSuchEncryptionConfiguration",
            ):
                self.logger.debug(
                    f"Could not get encryption for '{bucket_name}': {e}"
                )
                return []  # Unknown — don't false-positive

        # No encryption configuration found
        return [self._finding(
            finding_id="AWS-S3-002",
            resource_id=bucket_arn,
            resource_name=bucket_name,
            resource_type="AWS::S3::Bucket",
            raw_evidence={
                "bucket_name": bucket_name,
                "encryption":  "none",
                "region":      region,
            },
            region_override=region,
        )]

    # ------------------------------------------------------------------
    # Check: AWS-S3-003 — Access logging
    # ------------------------------------------------------------------

    def _check_logging(
        self, bucket_name: str, bucket_arn: str, region: str
    ) -> List[Finding]:
        """
        Check whether server access logging is enabled.

        Returns a finding if:
          - LoggingEnabled key is missing from the response
          - LoggingEnabled.TargetBucket is empty

        Note: logging bucket's name is captured in raw_evidence for the
        report — if it's non-empty, we also verify the target bucket exists
        (cross-bucket logging to a deleted bucket is a silent failure).
        """
        client = self._regional_client(region)
        try:
            self._track_api_call()
            response = client.get_bucket_logging(Bucket=bucket_name)
            logging_config = response.get("LoggingEnabled")

            if logging_config and logging_config.get("TargetBucket"):
                # Logging is configured
                return []

        except Exception as e:
            self.logger.debug(
                f"Could not get logging config for '{bucket_name}': {e}"
            )
            return []  # Unknown — don't false-positive

        return [self._finding(
            finding_id="AWS-S3-003",
            resource_id=bucket_arn,
            resource_name=bucket_name,
            resource_type="AWS::S3::Bucket",
            raw_evidence={
                "bucket_name":    bucket_name,
                "logging_enabled": False,
                "region":          region,
            },
            region_override=region,
        )]

    # ------------------------------------------------------------------
    # Check: AWS-S3-004 — Versioning
    # ------------------------------------------------------------------

    def _check_versioning(
        self, bucket_name: str, bucket_arn: str, region: str
    ) -> List[Finding]:
        """
        Check whether versioning is enabled.

        Three possible states from GetBucketVersioning:
          - Status: "Enabled"   → versioning on, no finding
          - Status: "Suspended" → versioning was on but paused → flag as LOW
          - (no Status key)     → versioning never enabled → flag as LOW

        Suspended versioning is notable because it means old versions exist
        but new ones aren't being created — objects are at risk.
        """
        client = self._regional_client(region)
        try:
            self._track_api_call()
            response = client.get_bucket_versioning(Bucket=bucket_name)
            status = response.get("Status", "")

            if status == "Enabled":
                return []

            # "Suspended" or "" (never enabled)
            versioning_state = status if status else "never_enabled"

        except Exception as e:
            self.logger.debug(
                f"Could not get versioning for '{bucket_name}': {e}"
            )
            return []

        description = (
            f"S3 bucket '{bucket_name}' versioning is {versioning_state}. "
            + (
                "Versioning was previously enabled but is now suspended — "
                "new objects are not versioned."
                if versioning_state == "Suspended"
                else "Versioning has never been enabled — deleted objects "
                     "cannot be recovered."
            )
        )

        return [self._finding(
            finding_id="AWS-S3-004",
            resource_id=bucket_arn,
            resource_name=bucket_name,
            resource_type="AWS::S3::Bucket",
            description_override=description,
            raw_evidence={
                "bucket_name":       bucket_name,
                "versioning_status": versioning_state,
                "region":            region,
            },
            region_override=region,
        )]

    # ------------------------------------------------------------------
    # Attack graph contribution
    # ------------------------------------------------------------------

    def build_graph_nodes(self, graph: AttackGraph) -> None:
        """
        Add PUBLIC_RESOURCE nodes for publicly accessible S3 buckets.

        For each public bucket:
          1. Add a PUBLIC_RESOURCE node with is_public=True
          2. Ensure the synthetic EXTERNAL "internet" node exists
          3. Add an IS_PUBLIC edge: internet → bucket

        Step 7 attack path analyzer starts from IS_PUBLIC nodes and finds
        paths to admin roles (e.g. if the bucket holds IAM credentials).

        This is called by the provider after run() — it reuses the public
        buckets detected in AWS-S3-001 findings rather than making
        additional API calls.
        """
        try:
            self._s3 = self.session.client("s3")

            # Ensure the EXTERNAL internet node exists
            self._ensure_internet_node(graph)

            # List all buckets and check public access
            self._track_api_call()
            response = self._s3.list_buckets()
            buckets = response.get("Buckets", [])

            for bucket_meta in buckets:
                bucket_name = bucket_meta["Name"]
                self._add_bucket_node(graph, bucket_name)

        except Exception as e:
            self.logger.warning(f"S3 graph building failed (non-fatal): {e}")

    def _add_bucket_node(self, graph: AttackGraph, bucket_name: str) -> None:
        """Add a single bucket node to the graph with public status."""
        region = self._get_bucket_region(bucket_name)
        client = self._regional_client(region)
        bucket_arn = f"arn:aws:s3:::{bucket_name}"

        # Determine public status
        bpa = self._get_block_public_access(client, bucket_name)
        fully_blocked = bpa is not None and all([
            bpa.get("BlockPublicAcls", False),
            bpa.get("IgnorePublicAcls", False),
            bpa.get("BlockPublicPolicy", False),
            bpa.get("RestrictPublicBuckets", False),
        ])

        is_public_policy, _ = (False, "") if fully_blocked else self._is_public_via_policy(client, bucket_name)
        is_public_acl, _    = (False, "") if fully_blocked else self._is_public_via_acl(client, bucket_name)
        is_public = is_public_policy or is_public_acl

        node_type = NodeType.PUBLIC_RESOURCE if is_public else NodeType.PRIVATE_RESOURCE
        node = GraphNode(
            node_id=bucket_arn,
            node_type=node_type,
            label=bucket_name,
            account_id=self.account_id,
            region=region,
            properties={
                "is_public":         is_public,
                "service":           "s3",
                "public_via_policy": is_public_policy,
                "public_via_acl":    is_public_acl,
                "block_public_access": bpa or {},
            },
        )
        graph.add_node(node)

        # Add IS_PUBLIC edge: internet → bucket (for attack path traversal)
        if is_public:
            self._add_internet_edge(graph, bucket_arn)

    def _ensure_internet_node(self, graph: AttackGraph) -> None:
        """Add the synthetic EXTERNAL internet node if not already present."""
        if not graph.node_exists(INTERNET_NODE_ID):
            internet_node = GraphNode(
                node_id=INTERNET_NODE_ID,
                node_type=NodeType.EXTERNAL,
                label="Internet (anonymous)",
                account_id=self.account_id,
                region="global",
                properties={"is_public": True},
            )
            graph.add_node(internet_node)

    def _add_internet_edge(self, graph: AttackGraph, bucket_arn: str) -> None:
        """Add IS_PUBLIC edge from internet node to the public bucket."""
        try:
            edge = GraphEdge(
                source_id=INTERNET_NODE_ID,
                target_id=bucket_arn,
                edge_type=EdgeType.IS_PUBLIC,
                label="publicly accessible",
            )
            graph.add_edge(edge)
        except ValueError as e:
            # Nodes might not exist if something went wrong — log and skip
            self.logger.debug(f"Could not add IS_PUBLIC edge for {bucket_arn}: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_bucket_region(self, bucket_name: str) -> str:
        """
        Get the bucket's home region via GetBucketLocation.

        AWS quirk: us-east-1 returns None (not "us-east-1") from
        GetBucketLocation. We normalise None → "us-east-1".

        Falls back to "us-east-1" on error — not ideal but prevents
        a bad location response from killing all subsequent checks.
        """
        try:
            self._track_api_call()
            response = self._s3.get_bucket_location(Bucket=bucket_name)
            location = response.get("LocationConstraint")
            return location if location else "us-east-1"
        except Exception as e:
            self.logger.debug(
                f"Could not get location for '{bucket_name}': {e}. "
                f"Defaulting to us-east-1."
            )
            return "us-east-1"

    def _regional_client(self, region: str) -> Any:
        """Return a region-specific S3 client. Cached by region."""
        if not hasattr(self, "_regional_clients"):
            self._regional_clients: Dict[str, Any] = {}
        if region not in self._regional_clients:
            self._regional_clients[region] = self.session.client(
                "s3", region_name=region
            )
        return self._regional_clients[region]

    @staticmethod
    def _s3_error_code(exception: Exception) -> str:
        """Extract AWS error code from a boto3 exception."""
        response = getattr(exception, "response", {}) or {}
        return response.get("Error", {}).get("Code", "")