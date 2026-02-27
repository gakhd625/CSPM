# core/base_provider.py
from abc import ABC, abstractmethod
from typing import List
from core.models.finding import Finding

class BaseProvider(ABC):
    """
    Every cloud provider (AWS, Azure, GCP) must implement this interface.
    The engine only ever talks to this contract — never to boto3 directly.
    """

    @abstractmethod
    def authenticate(self) -> bool: ...

    @abstractmethod
    def get_accounts(self) -> List[str]: ...

    @abstractmethod
    def run_checks(self, account_id: str) -> List[Finding]: ...

    @abstractmethod
    def build_resource_graph(self, account_id: str) -> "AttackGraph": ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...  # "aws", "azure", "gcp"