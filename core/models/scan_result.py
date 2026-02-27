"""
core/models/scan_result.py

ScanResult is the top-level output of one complete CSPM scan run.
It aggregates findings from all accounts, attack paths from the graph
engine, compliance mappings, and the final security score.

This is what gets serialised to JSON, rendered as HTML, stored in S3,
and diffed between scan runs to detect regressions or improvements.

Design decisions:
  - account_results is a list of AccountResult (one per AWS account)
    so multi-account scans keep per-account data intact
  - top-level aggregated fields are derived properties that compute
    on demand from account_results — no duplication, no sync bugs
  - diff() method lets you compare two scans to show what changed
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.models.finding import Finding, Severity, FindingStatus
from core.attack_graph.models import AttackGraph, AttackPath


# ---------------------------------------------------------------------------
# Per-account score
# ---------------------------------------------------------------------------

@dataclass
class AccountScore:
    """
    Security score for a single AWS account.

    Score starts at 100 and findings deduct points based on severity.
    Grade is assigned from the final numeric score.
    """
    account_id:   str
    account_name: str = ""
    raw_score:    int = 100     # Before clamping to [0, 100]
    score:        int = 100     # Final clamped score
    grade:        str = "A"
    finding_counts: Dict[str, int] = field(default_factory=dict)
    # e.g. {"CRITICAL": 2, "HIGH": 5, "MEDIUM": 8, "LOW": 3, "INFO": 1}

    @classmethod
    def from_findings(
        cls,
        account_id: str,
        findings: List[Finding],
        account_name: str = ""
    ) -> "AccountScore":
        """
        Calculate the score for an account from its findings.
        Only OPEN findings affect the score (SUPPRESSED are excluded).
        """
        open_findings = [f for f in findings if f.is_open]

        # Deduct points for each open finding
        deduction = sum(f.score_impact for f in open_findings)
        raw_score = 100 - deduction
        score = max(0, min(100, raw_score))

        # Count findings by severity for the summary
        counts: Dict[str, int] = {s.value: 0 for s in Severity}
        for f in open_findings:
            counts[f.severity.value] += 1

        return cls(
            account_id=account_id,
            account_name=account_name,
            raw_score=raw_score,
            score=score,
            grade=cls._grade_from_score(score),
            finding_counts=counts,
        )

    @staticmethod
    def _grade_from_score(score: int) -> str:
        """
        A:  90–100  (excellent posture, minor issues only)
        B:  75–89   (good posture, some medium findings)
        C:  60–74   (moderate risk, high findings present)
        D:  40–59   (significant risk, criticals likely)
        F:  0–39    (critical failures, account at risk)
        """
        if score >= 90: return "A"
        if score >= 75: return "B"
        if score >= 60: return "C"
        if score >= 40: return "D"
        return "F"

    @property
    def grade_color(self) -> str:
        return {
            "A": "#27AE60",
            "B": "#2ECC71",
            "C": "#F39C12",
            "D": "#E67E22",
            "F": "#C0392B",
        }.get(self.grade, "#95A5A6")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id":     self.account_id,
            "account_name":   self.account_name,
            "score":          self.score,
            "grade":          self.grade,
            "grade_color":    self.grade_color,
            "finding_counts": self.finding_counts,
        }


# ---------------------------------------------------------------------------
# Per-account result
# ---------------------------------------------------------------------------

@dataclass
class AccountResult:
    """
    The complete scan result for a single AWS account.
    Multi-account scans produce one AccountResult per account.
    """
    account_id:   str
    account_name: str          = ""
    region:       str          = "us-east-1"
    scan_start:   datetime     = field(default_factory=lambda: datetime.now(timezone.utc))
    scan_end:     Optional[datetime] = field(default=None)

    findings:     List[Finding]     = field(default_factory=list)
    attack_graph: Optional[AttackGraph] = field(default=None)
    score:        Optional[AccountScore] = field(default=None)

    # Scan errors — non-fatal errors (e.g. permission denied on one service)
    errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def attack_paths(self) -> List[AttackPath]:
        if self.attack_graph:
            return self.attack_graph.attack_paths
        return []

    @property
    def open_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.is_open]

    @property
    def critical_findings(self) -> List[Finding]:
        return [f for f in self.open_findings if f.is_critical]

    @property
    def scan_duration_seconds(self) -> Optional[float]:
        if self.scan_end:
            return (self.scan_end - self.scan_start).total_seconds()
        return None

    def add_error(self, service: str, message: str, exception: str = "") -> None:
        self.errors.append({
            "service":   service,
            "message":   message,
            "exception": exception,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id":        self.account_id,
            "account_name":      self.account_name,
            "scan_start":        self.scan_start.isoformat(),
            "scan_end":          self.scan_end.isoformat() if self.scan_end else None,
            "duration_seconds":  self.scan_duration_seconds,
            "score":             self.score.to_dict() if self.score else None,
            "finding_count":     len(self.findings),
            "open_finding_count": len(self.open_findings),
            "findings":          [f.to_dict() for f in self.findings],
            "attack_paths":      [p.to_dict() for p in self.attack_paths],
            "attack_graph_stats": self.attack_graph.stats if self.attack_graph else {},
            "errors":            self.errors,
        }


# ---------------------------------------------------------------------------
# Compliance summary
# ---------------------------------------------------------------------------

@dataclass
class ComplianceSummary:
    """
    Aggregated compliance posture across all findings.

    For each framework (CIS, PCI-DSS, HIPAA), this summarises:
      - which controls have failing findings
      - how many findings map to each control
      - overall compliance percentage
    """
    framework:      str
    total_controls: int
    failed_controls: int
    control_failures: Dict[str, List[str]]  = field(default_factory=dict)
    # control_id → [finding_ids]

    @property
    def passing_controls(self) -> int:
        return self.total_controls - self.failed_controls

    @property
    def compliance_percentage(self) -> float:
        if self.total_controls == 0:
            return 100.0
        return round((self.passing_controls / self.total_controls) * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework":              self.framework,
            "total_controls":         self.total_controls,
            "failed_controls":        self.failed_controls,
            "passing_controls":       self.passing_controls,
            "compliance_percentage":  self.compliance_percentage,
            "control_failures":       self.control_failures,
        }


# ---------------------------------------------------------------------------
# Top-level ScanResult
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """
    The complete output of one CSPM scan run — potentially spanning
    multiple AWS accounts (via Organizations).

    This is the object passed to the reporter. Everything the reporter
    needs to generate JSON or HTML output is available here.

    Usage:
        result = ScanResult(scan_id="run-20240101", provider="aws")
        result.account_results.append(account_result)
        result.finalise()   # Computes aggregated score and compliance summary
    """
    scan_id:    str   = field(default_factory=lambda: str(uuid.uuid4()))
    provider:   str   = "aws"
    scan_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scan_end:   Optional[datetime] = field(default=None)

    account_results:    List[AccountResult]   = field(default_factory=list)
    compliance_summary: List[ComplianceSummary] = field(default_factory=list)

    # Aggregated values — populated by finalise()
    total_score:  int  = 0
    overall_grade: str = "F"
    is_finalised:  bool = False

    # Scan-level metadata
    scanner_version: str = "1.0.0"
    triggered_by:    str = "manual"  # "manual", "lambda", "ci"

    # ------------------------------------------------------------------
    # Aggregated properties (derived from account_results)
    # ------------------------------------------------------------------

    @property
    def all_findings(self) -> List[Finding]:
        """Flat list of all findings across all accounts."""
        return [f for ar in self.account_results for f in ar.findings]

    @property
    def all_open_findings(self) -> List[Finding]:
        return [f for f in self.all_findings if f.is_open]

    @property
    def all_attack_paths(self) -> List[AttackPath]:
        return [p for ar in self.account_results for p in ar.attack_paths]

    @property
    def critical_findings(self) -> List[Finding]:
        return [f for f in self.all_open_findings if f.is_critical]

    @property
    def has_criticals(self) -> bool:
        return len(self.critical_findings) > 0

    @property
    def account_count(self) -> int:
        return len(self.account_results)

    @property
    def finding_counts_by_severity(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s.value: 0 for s in Severity}
        for f in self.all_open_findings:
            counts[f.severity.value] += 1
        return counts

    @property
    def scan_duration_seconds(self) -> Optional[float]:
        if self.scan_end:
            return (self.scan_end - self.scan_start).total_seconds()
        return None

    # ------------------------------------------------------------------
    # Finalise — call after all accounts have been scanned
    # ------------------------------------------------------------------

    def finalise(self) -> None:
        """
        Compute and cache aggregated score and grade.
        Must be called before passing to a reporter.
        """
        self.scan_end = datetime.now(timezone.utc)

        # Compute per-account scores if not already done
        for ar in self.account_results:
            if ar.score is None:
                ar.score = AccountScore.from_findings(
                    ar.account_id, ar.findings, ar.account_name
                )
            if ar.scan_end is None:
                ar.scan_end = self.scan_end

        # Overall score = average of account scores
        if self.account_results:
            scores = [ar.score.score for ar in self.account_results if ar.score]
            self.total_score = round(sum(scores) / len(scores)) if scores else 0
        else:
            self.total_score = 100  # No accounts = nothing wrong

        self.overall_grade = AccountScore._grade_from_score(self.total_score)
        self.is_finalised = True

    # ------------------------------------------------------------------
    # Diff — compare two scan runs
    # ------------------------------------------------------------------

    def diff(self, previous: "ScanResult") -> Dict[str, Any]:
        """
        Compare this scan result to a previous one.
        Returns new findings, resolved findings, and score change.

        Use this for the Lambda function to send SNS alerts only on
        net-new criticals rather than alerting on every scan.
        """
        current_ids  = {f.finding_id + f.resource_id for f in self.all_open_findings}
        previous_ids = {f.finding_id + f.resource_id for f in previous.all_open_findings}

        new_finding_keys      = current_ids - previous_ids
        resolved_finding_keys = previous_ids - current_ids

        new_findings = [
            f for f in self.all_open_findings
            if (f.finding_id + f.resource_id) in new_finding_keys
        ]
        resolved_findings = [
            f for f in previous.all_open_findings
            if (f.finding_id + f.resource_id) in resolved_finding_keys
        ]

        return {
            "score_change":       self.total_score - previous.total_score,
            "new_findings":       [f.to_dict() for f in new_findings],
            "resolved_findings":  [f.to_dict() for f in resolved_findings],
            "new_criticals":      [f.to_dict() for f in new_findings if f.is_critical],
            "new_finding_count":  len(new_findings),
            "resolved_count":     len(resolved_findings),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        if not self.is_finalised:
            self.finalise()

        return {
            # Scan metadata
            "scan_id":         self.scan_id,
            "provider":        self.provider,
            "scan_start":      self.scan_start.isoformat(),
            "scan_end":        self.scan_end.isoformat() if self.scan_end else None,
            "duration_seconds": self.scan_duration_seconds,
            "scanner_version": self.scanner_version,
            "triggered_by":    self.triggered_by,

            # Top-level summary
            "overall_score":         self.total_score,
            "overall_grade":         self.overall_grade,
            "account_count":         self.account_count,
            "total_findings":        len(self.all_findings),
            "open_findings":         len(self.all_open_findings),
            "finding_counts_by_severity": self.finding_counts_by_severity,
            "has_criticals":         self.has_criticals,
            "total_attack_paths":    len(self.all_attack_paths),

            # Per-account results
            "accounts": [ar.to_dict() for ar in self.account_results],

            # Compliance
            "compliance": [c.to_dict() for c in self.compliance_summary],
        }

    def __repr__(self) -> str:
        return (
            f"ScanResult(id={self.scan_id!r}, accounts={self.account_count}, "
            f"findings={len(self.all_findings)}, score={self.total_score}, "
            f"grade={self.overall_grade!r})"
        )