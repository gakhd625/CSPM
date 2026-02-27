# core/base_checker.py
from abc import ABC, abstractmethod
from typing import List
from core.models.finding import Finding

class BaseChecker(ABC):
    """
    Each checker owns one service domain (IAM, S3, etc.)
    It receives a session/client object and returns raw findings.
    Checkers are stateless — no side effects, pure scan → findings.
    """

    def __init__(self, session):
        self.session = session
        self.findings: List[Finding] = []

    @abstractmethod
    def run(self) -> List[Finding]: ...

    @property
    @abstractmethod
    def checker_name(self) -> str: ...