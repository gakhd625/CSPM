"""
core/models/finding.py

The Finding is the atomic unit of output for the entire CSPM tool.
Every checker produces a list of Findings. The scorer consumes them.
The compliance mapper annotates them. The reporter renders them.

Design decisions:
  - dataclass for clean instantiation and repr
  - Enums for severity/status so values are never raw strings (typo-safe)
  - raw_evidence stores the exact API response that triggered the finding
    (critical for audit trails and IR investigations)
  - compliance list is populated AFTER scanning, by the ComplianceMapper
  - attack_paths is populated AFTER graph analysis, by the AttackGraphAnalyzer
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """
    Severity levels with associated score impact.
    Inheriting from str means Severity.CRITICAL == "CRITICAL" is True,
    which makes JSON serialisation and comparisons effortless.
    """
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"

    @property
    def score_impact(self) -> int:
        """Points deducted from the account security score."""
        return {
            Severity.CRITICAL: 40,
            Severity.HIGH:     20,
            Severity.MEDIUM:   10,
            Severity.LOW:       5,
            Severity.INFO:      0,
        }[self]

    @property
    def display_color(self) -> str:
        """HTML hex color for the HTML reporter."""
        return {
            Severity.CRITICAL: "#C0392B",
            Severity.HIGH:     "#E67E22",
            Severity.MEDIUM:   "#F1C40F",
            Severity.LOW:      "#3498DB",
            Severity.INFO:     "#95A5A6",
        }[self]

    @property
    def sort_order(self) -> int:
        """Lower = more severe. Used to sort findings in reports."""
        return {
            Severity.CRITICAL: 0,
            Severity.HIGH:     1,
            Severity.MEDIUM:   2,
            Severity.LOW:      3,
            Severity.INFO:     4,
        }[self]


class FindingStatus(str, Enum):
    """
    Lifecycle state of a finding.
    SUPPRESSED = intentionally accepted risk (requires justification in prod).
    RESOLVED   = remediated, kept for audit history.
    """
    OPEN       = "OPEN"
    SUPPRESSED = "SUPPRESSED"
    RESOLVED   = "RESOLVED"


class CloudProvider(str, Enum):
    """Supported cloud providers. Extend as new providers are added."""
    AWS   = "aws"
    AZURE = "azure"
    GCP   = "gcp"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

@dataclass
class ComplianceReference:
    """
    Maps a single finding to a specific control in a compliance framework.
    One finding can map to multiple controls across multiple frameworks.

    Example:
        ComplianceReference(
            framework="CIS",
            control_id="1.4",
            control_title="Ensure no root account access key exists",
            description="The root account is the most privileged user..."
        )
    """
    framework:     str   # "CIS", "PCI-DSS", "HIPAA", "SOC2"
    control_id:    str   # "1.4", "8.3.6", "164.312(a)(2)(i)"
    control_title: str
    description:   str

    def to_dict(self) -> Dict[str, str]:
        return {
            "framework":     self.framework,
            "control_id":    self.control_id,
            "control_title": self.control_title,
            "description":   self.description,
        }


@dataclass
class RemediationStep:
    """
    Structured remediation guidance.
    Having both console and CLI steps means any engineer can fix the issue
    regardless of their preferred workflow.
    """
    step_number:   int
    description:   str
    console_steps: Optional[str] = None   # AWS Console instructions
    cli_command:   Optional[str] = None   # AWS CLI command
    terraform:     Optional[str] = None   # IaC fix

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "step":          self.step_number,
            "description":   self.description,
            "console":       self.console_steps,
            "cli":           self.cli_command,
            "terraform":     self.terraform,
        }.items() if v is not None}


# ---------------------------------------------------------------------------
# Core Finding model
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """
    The central data model. One Finding = one misconfiguration detected.

    Fields are grouped by concern:
      1. Identity       - what check generated this
      2. Severity       - how bad is it
      3. Cloud context  - where does it live
      4. Guidance       - how to fix it
      5. Compliance     - which frameworks care about it
      6. Metadata       - timestamps, evidence, internal tracking
      7. Graph linkage  - populated by the attack graph engine

    Usage:
        finding = Finding(
            finding_id="AWS-IAM-001",
            title="IAM User Missing MFA",
            description="User 'deploy-bot' has no MFA device configured.",
            severity=Severity.HIGH,
            provider=CloudProvider.AWS,
            account_id="123456789012",
            resource_type="AWS::IAM::User",
            resource_id="arn:aws:iam::123456789012:user/deploy-bot",
            resource_name="deploy-bot",
            remediation_summary="Enable MFA for this IAM user.",
        )
    """

    # ------------------------------------------------------------------
    # 1. Identity — what check is this
    # ------------------------------------------------------------------
    finding_id:   str   # e.g. "AWS-IAM-001" — must be unique per check type
    title:        str   # Short, scannable title
    description:  str   # Full human-readable explanation of the issue

    # ------------------------------------------------------------------
    # 2. Severity & Status
    # ------------------------------------------------------------------
    severity: Severity
    status:   FindingStatus = field(default=FindingStatus.OPEN)

    # ------------------------------------------------------------------
    # 3. Cloud context — where does the affected resource live
    # ------------------------------------------------------------------
    provider:       CloudProvider = field(default=CloudProvider.AWS)
    account_id:     str           = field(default="")
    region:         str           = field(default="global")
    resource_type:  str           = field(default="")   # CloudFormation resource type
    resource_id:    str           = field(default="")   # ARN or resource identifier
    resource_name:  str           = field(default="")   # Human-friendly name

    # ------------------------------------------------------------------
    # 4. Guidance — how to fix it
    # ------------------------------------------------------------------
    remediation_summary: str              = field(default="")
    remediation_steps:   List[RemediationStep] = field(default_factory=list)
    references:          List[str]        = field(default_factory=list)  # Doc URLs

    # ------------------------------------------------------------------
    # 5. Compliance — populated by ComplianceMapper post-scan
    # ------------------------------------------------------------------
    compliance: List[ComplianceReference] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 6. Metadata
    # ------------------------------------------------------------------
    # Auto-generate a unique instance ID so two findings of the same type
    # on different resources are always distinguishable in storage/DB
    instance_id:  str      = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scan_id:      str      = field(default="")  # Links to the parent ScanResult

    # Raw API response snippet that triggered this finding.
    # NEVER include secrets. Truncate large responses.
    raw_evidence: Dict[str, Any] = field(default_factory=dict)

    # Arbitrary k/v tags (e.g. resource tags from AWS, team ownership)
    tags: Dict[str, str] = field(default_factory=dict)

    # Suppression justification — required when status == SUPPRESSED
    suppression_reason: Optional[str] = field(default=None)

    # ------------------------------------------------------------------
    # 7. Attack graph linkage — populated by AttackGraphAnalyzer
    # ------------------------------------------------------------------
    is_attack_node:  bool       = field(default=False)
    attack_path_ids: List[str]  = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self.status == FindingStatus.OPEN

    @property
    def is_critical(self) -> bool:
        return self.severity == Severity.CRITICAL

    @property
    def framework_names(self) -> List[str]:
        """Unique list of compliance frameworks this finding maps to."""
        return list({c.framework for c in self.compliance})

    @property
    def score_impact(self) -> int:
        """Score points deducted. 0 if suppressed."""
        if self.status == FindingStatus.SUPPRESSED:
            return 0
        return self.severity.score_impact

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Full serialisation to a plain dict — used by JSON reporter and
        Lambda S3 storage. All values are JSON-safe primitives.
        """
        return {
            # Identity
            "finding_id":   self.finding_id,
            "instance_id":  self.instance_id,
            "title":        self.title,
            "description":  self.description,
            # Severity
            "severity":     self.severity.value,
            "status":       self.status.value,
            "score_impact": self.score_impact,
            # Cloud context
            "provider":       self.provider.value,
            "account_id":     self.account_id,
            "region":         self.region,
            "resource_type":  self.resource_type,
            "resource_id":    self.resource_id,
            "resource_name":  self.resource_name,
            # Guidance
            "remediation_summary": self.remediation_summary,
            "remediation_steps":   [s.to_dict() for s in self.remediation_steps],
            "references":          self.references,
            # Compliance
            "compliance": [c.to_dict() for c in self.compliance],
            # Metadata
            "timestamp":            self.timestamp.isoformat(),
            "scan_id":              self.scan_id,
            "raw_evidence":         self.raw_evidence,
            "tags":                 self.tags,
            "suppression_reason":   self.suppression_reason,
            # Attack graph
            "is_attack_node":  self.is_attack_node,
            "attack_path_ids": self.attack_path_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        """Deserialise a Finding from a dict (e.g. loading from stored JSON)."""
        return cls(
            finding_id=data["finding_id"],
            title=data["title"],
            description=data["description"],
            severity=Severity(data["severity"]),
            status=FindingStatus(data.get("status", "OPEN")),
            provider=CloudProvider(data.get("provider", "aws")),
            account_id=data.get("account_id", ""),
            region=data.get("region", "global"),
            resource_type=data.get("resource_type", ""),
            resource_id=data.get("resource_id", ""),
            resource_name=data.get("resource_name", ""),
            remediation_summary=data.get("remediation_summary", ""),
            references=data.get("references", []),
            instance_id=data.get("instance_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            scan_id=data.get("scan_id", ""),
            raw_evidence=data.get("raw_evidence", {}),
            tags=data.get("tags", {}),
            suppression_reason=data.get("suppression_reason"),
            is_attack_node=data.get("is_attack_node", False),
            attack_path_ids=data.get("attack_path_ids", []),
        )

    def __repr__(self) -> str:
        return (
            f"Finding(id={self.finding_id!r}, severity={self.severity.value}, "
            f"resource={self.resource_name or self.resource_id!r}, "
            f"status={self.status.value})"
        )