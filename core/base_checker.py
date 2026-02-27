"""
core/base_checker.py

BaseChecker is the contract that every service-domain checker must satisfy.
Each checker owns exactly one service domain: IAM, S3, CloudTrail, etc.

Why separate checkers rather than one big scan() method?
  1. Isolation — an IAM API error doesn't abort the S3 scan
  2. Testability — mock one boto3 client, test one checker in isolation
  3. Extensibility — add a new checker by dropping a file in checkers/
  4. Selective scanning — run only IAM checks with --checks iam
  5. Parallelism — checkers are stateless so they can run concurrently

Design decisions:
  - Checkers are stateless — they receive a session in __init__ and
    return findings from run(). No side effects, no shared mutable state.
  - Each checker has a FINDINGS registry (class-level) that maps
    finding_id → metadata. This is the single source of truth for
    what each finding_id means, its default severity, and its remediation.
  - _make_finding() is a convenience factory that fills in all the
    boilerplate so checker code focuses on detection logic only.
  - execute() wraps run() with timing, error handling, and logging —
    providers call execute(), not run() directly.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from core.models.finding import (
    CloudProvider, Finding, FindingStatus, RemediationStep, Severity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Finding template — defined at checker class level
# ---------------------------------------------------------------------------

@dataclass
class FindingTemplate:
    """
    Static definition of a finding type.

    Stored in BaseChecker.FINDING_TEMPLATES at the class level.
    Used by _make_finding() to avoid repeating boilerplate in every check.

    Think of this as the "rule definition" and Finding as "rule hit".
    """
    finding_id:          str
    title:               str
    description_template: str   # May contain {resource_name}, {detail} placeholders
    severity:            Severity
    remediation_summary: str
    remediation_steps:   List[RemediationStep] = field(default_factory=list)
    references:          List[str]             = field(default_factory=list)


# ---------------------------------------------------------------------------
# Checker execution result — wraps findings with timing & error info
# ---------------------------------------------------------------------------

@dataclass
class CheckerResult:
    """
    Output of one checker's execution.
    Returned by execute() (not run()) so callers get metadata too.
    """
    checker_name:   str
    findings:       List[Finding]
    duration_ms:    float         = 0.0
    error:          Optional[str] = None
    warning:        Optional[str] = None    # Partial success (e.g. some regions failed)
    api_calls_made: int           = 0       # For cost/rate-limit tracking

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def finding_count(self) -> int:
        return len(self.findings)


# ---------------------------------------------------------------------------
# BaseChecker
# ---------------------------------------------------------------------------

class BaseChecker(ABC):
    """
    Abstract base class for all service-domain security checkers.

    Concrete implementations:
        providers/aws/checkers/iam.py      → IAMChecker
        providers/aws/checkers/s3.py       → S3Checker
        providers/aws/checkers/logging.py  → LoggingChecker

    Usage pattern inside a provider:
        checker = IAMChecker(session=boto3_session, account_id="123", region="us-east-1")
        result = checker.execute()
        findings.extend(result.findings)

    Writing a new checker:
        1. Subclass BaseChecker
        2. Define FINDING_TEMPLATES at class level
        3. Implement run() — call self._finding() for each issue found
        4. Register the checker in the provider's checker list
    """

    # Override in subclasses: {finding_id: FindingTemplate}
    # This is the "rule book" for this checker.
    FINDING_TEMPLATES: Dict[str, FindingTemplate] = {}

    def __init__(
        self,
        session: Any,               # boto3 Session, azure Credential, etc.
        account_id: str,
        region: str        = "global",
        provider: CloudProvider = CloudProvider.AWS,
    ):
        self.session    = session
        self.account_id = account_id
        self.region     = region
        self.provider   = provider
        self.logger     = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self._api_call_count = 0

    # ------------------------------------------------------------------
    # Abstract — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> List[Finding]:
        """
        Execute all checks in this domain. Return all findings.

        Implementation guidelines:
          - Use self._finding() to create Finding objects
          - Catch service-specific exceptions and log warnings
          - Never let one failed API call abort the whole checker
          - Document which IAM permissions each check requires
          - Keep individual check methods small and focused

        Example structure:
            def run(self) -> List[Finding]:
                findings = []
                findings.extend(self._check_mfa_enabled())
                findings.extend(self._check_access_keys_age())
                return findings
        """
        ...

    @property
    @abstractmethod
    def checker_name(self) -> str:
        """
        Short identifier: "iam", "s3", "cloudtrail", "guardduty"
        Used in CheckerResult and log output.
        """
        ...

    @property
    @abstractmethod
    def required_permissions(self) -> List[str]:
        """
        IAM actions this checker needs to operate fully.
        Used to generate the least-privilege scanner IAM policy
        and to warn when permissions are missing.

        Example: ["iam:ListUsers", "iam:GetLoginProfile", "iam:ListMFADevices"]
        """
        ...

    # ------------------------------------------------------------------
    # Concrete: execute() wraps run() with timing + error handling
    # ------------------------------------------------------------------

    def execute(self) -> CheckerResult:
        """
        Called by the provider — NOT run() directly.

        Provides:
          - Exception isolation (checker crash doesn't kill the scan)
          - Wall-clock timing
          - Structured logging
          - api_calls_made tracking
        """
        self.logger.info(
            f"Starting {self.checker_name} checks "
            f"[account={self.account_id}, region={self.region}]"
        )
        start = time.perf_counter()

        try:
            findings = self.run()
            duration_ms = (time.perf_counter() - start) * 1000

            self.logger.info(
                f"Completed {self.checker_name}: "
                f"{len(findings)} findings in {duration_ms:.0f}ms "
                f"({self._api_call_count} API calls)"
            )
            return CheckerResult(
                checker_name=self.checker_name,
                findings=findings,
                duration_ms=duration_ms,
                api_calls_made=self._api_call_count,
            )

        except PermissionError as e:
            # Non-fatal: record and continue
            duration_ms = (time.perf_counter() - start) * 1000
            msg = f"Permission denied in {self.checker_name}: {e}"
            self.logger.warning(msg)
            return CheckerResult(
                checker_name=self.checker_name,
                findings=[],
                duration_ms=duration_ms,
                warning=msg,
                api_calls_made=self._api_call_count,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            msg = f"Checker {self.checker_name} failed: {type(e).__name__}: {e}"
            self.logger.error(msg, exc_info=True)
            return CheckerResult(
                checker_name=self.checker_name,
                findings=[],
                duration_ms=duration_ms,
                error=msg,
                api_calls_made=self._api_call_count,
            )

    # ------------------------------------------------------------------
    # Convenience factory — use in run() implementations
    # ------------------------------------------------------------------

    def _finding(
        self,
        finding_id:     str,
        resource_id:    str,
        resource_name:  str,
        resource_type:  str,
        description_override: Optional[str] = None,
        severity_override:    Optional[Severity] = None,
        raw_evidence:   Optional[Dict[str, Any]] = None,
        region_override: Optional[str] = None,
        extra_remediation_steps: Optional[List[RemediationStep]] = None,
        tags:           Optional[Dict[str, str]] = None,
    ) -> Finding:
        """
        Create a Finding from a FindingTemplate.

        This is the ONLY way checker implementations should create findings.
        Centralises all boilerplate so check methods stay lean.

        Args:
            finding_id:    Must match a key in self.FINDING_TEMPLATES
            resource_id:   ARN or unique identifier of the affected resource
            resource_name: Human-friendly name
            resource_type: CloudFormation resource type (e.g. "AWS::IAM::User")
            description_override: Replace template description (use sparingly)
            severity_override:    Override template severity (use sparingly)
            raw_evidence:  Dict of raw API data that proves the issue
            region_override: Override self.region for global resources
            extra_remediation_steps: Append steps beyond the template's
            tags:          Resource tags from the cloud provider

        Example:
            return self._finding(
                finding_id="AWS-IAM-001",
                resource_id="arn:aws:iam::123:user/alice",
                resource_name="alice",
                resource_type="AWS::IAM::User",
                raw_evidence={"user_name": "alice", "mfa_devices": []},
            )
        """
        template = self.FINDING_TEMPLATES.get(finding_id)
        if template is None:
            raise ValueError(
                f"No FindingTemplate registered for '{finding_id}' "
                f"in {self.__class__.__name__}. "
                f"Available IDs: {list(self.FINDING_TEMPLATES.keys())}"
            )

        description = description_override or template.description_template
        severity    = severity_override    or template.severity
        region      = region_override      or self.region

        steps = list(template.remediation_steps)
        if extra_remediation_steps:
            steps.extend(extra_remediation_steps)

        return Finding(
            finding_id=finding_id,
            title=template.title,
            description=description,
            severity=severity,
            status=FindingStatus.OPEN,
            provider=self.provider,
            account_id=self.account_id,
            region=region,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            remediation_summary=template.remediation_summary,
            remediation_steps=steps,
            references=template.references,
            raw_evidence=raw_evidence or {},
            tags=tags or {},
        )

    def _track_api_call(self) -> None:
        """
        Call this once per boto3/SDK API call to track usage.
        Used for cost monitoring and rate-limit debugging.

        Example:
            self._track_api_call()
            response = self.iam_client.list_users()
        """
        self._api_call_count += 1

    def _safe_get_client(self, service: str) -> Any:
        """
        Helper for AWS checkers to get a boto3 client from the session.
        Override in non-AWS checkers if needed.
        """
        try:
            return self.session.client(service, region_name=self.region)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {service} client "
                f"[account={self.account_id}, region={self.region}]: {e}"
            ) from e

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"account={self.account_id!r}, "
            f"region={self.region!r})"
        )