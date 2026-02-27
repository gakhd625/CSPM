# core/models/finding.py
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime

class Severity(str, Enum):
    CRITICAL = "CRITICAL"   # 40 points deducted
    HIGH     = "HIGH"       # 20 points deducted
    MEDIUM   = "MEDIUM"     # 10 points deducted
    LOW      = "LOW"        #  5 points deducted
    INFO     = "INFO"       #  0 points deducted

class FindingStatus(str, Enum):
    OPEN       = "OPEN"
    SUPPRESSED = "SUPPRESSED"   # intentionally ignored
    RESOLVED   = "RESOLVED"

@dataclass
class ComplianceReference:
    framework: str          # "CIS", "PCI-DSS", "HIPAA"
    control_id: str         # "1.4", "8.3.6", "164.312(a)"
    control_title: str
    description: str

@dataclass
class Finding:
    # Identity
    finding_id: str                     # e.g., "AWS-IAM-001"
    title: str                          # "IAM User Missing MFA"
    description: str                    # Human-readable explanation

    # Severity & Status
    severity: Severity
    status: FindingStatus = FindingStatus.OPEN

    # Cloud context
    provider: str = "aws"               # "aws" | "azure" | "gcp"
    account_id: str = ""
    region: str = "global"
    resource_type: str = ""             # "AWS::IAM::User"
    resource_id: str = ""               # ARN or resource name
    resource_name: str = ""

    # Guidance
    remediation: str = ""               # Step-by-step fix
    references: List[str] = field(default_factory=list)   # AWS docs URLs

    # Compliance
    compliance: List[ComplianceReference] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_evidence: Dict = field(default_factory=dict)  # Raw API response snippet
    tags: Dict[str, str] = field(default_factory=dict)

    # Attack graph linkage
    is_attack_node: bool = False        # True if this finding is a graph node
    attack_paths: List[str] = field(default_factory=list)  # Path IDs this appears in