"""
core/checker_registry.py

CheckerRegistry is a self-registering system that lets checkers declare
themselves rather than the provider having a hardcoded list.

Why a registry?
  - Adding a new checker requires zero changes to the provider or engine
  - Selective scanning (--checks iam,s3) is implemented here, not in 10 places
  - The registry can report what checks are available (useful for --list-checks)
  - Checkers can declare dependencies (e.g. "I need IAM data from X first")

Usage:
    # In a checker module (e.g. providers/aws/checkers/iam.py):
    @CheckerRegistry.register(provider="aws", domain="iam")
    class IAMChecker(BaseChecker):
        ...

    # In the provider:
    checkers = CheckerRegistry.get_checkers(provider="aws", domains=["iam", "s3"])
    for checker_cls in checkers:
        checker = checker_cls(session=session, account_id=account_id)
        result = checker.execute()

Design: we use a class-level dict keyed by (provider, domain) so the registry
is a singleton without needing module-level state gymnastics.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)


class CheckerRegistry:
    """
    Central registry for all checker classes.

    Checkers self-register via the @CheckerRegistry.register decorator.
    The provider calls get_checkers() to get the checker classes it needs.
    """

    # Registry storage: (provider_name, domain) → [CheckerClass, ...]
    # e.g. ("aws", "iam") → [IAMChecker]
    _registry: Dict[Tuple[str, str], List[type]] = {}

    # All registered (provider, domain) combinations for introspection
    _all_entries: List[Dict] = []

    @classmethod
    def register(cls, provider: str, domain: str):
        """
        Decorator that registers a checker class.

        Usage:
            @CheckerRegistry.register(provider="aws", domain="iam")
            class IAMChecker(BaseChecker):
                ...
        """
        def decorator(checker_cls: type) -> type:
            key = (provider.lower(), domain.lower())
            if key not in cls._registry:
                cls._registry[key] = []

            # Avoid double-registration (e.g. if module is imported twice)
            if checker_cls not in cls._registry[key]:
                cls._registry[key].append(checker_cls)
                cls._all_entries.append({
                    "provider":  provider.lower(),
                    "domain":    domain.lower(),
                    "class":     checker_cls,
                    "class_name": checker_cls.__name__,
                })
                logger.debug(
                    f"Registered checker: {checker_cls.__name__} "
                    f"[provider={provider}, domain={domain}]"
                )
            return checker_cls
        return decorator

    @classmethod
    def get_checkers(
        cls,
        provider: str,
        domains: Optional[List[str]] = None,
    ) -> List[type]:
        """
        Get all registered checker classes for a provider.

        Args:
            provider: "aws", "azure", "gcp"
            domains:  Optional filter. None = return all domains.
                      ["iam", "s3"] = return only IAM and S3 checkers.

        Returns list of checker CLASSES (not instances).
        The caller instantiates them with the appropriate session.
        """
        result = []
        provider = provider.lower()

        for (reg_provider, reg_domain), checker_classes in cls._registry.items():
            if reg_provider != provider:
                continue
            if domains is not None and reg_domain not in [d.lower() for d in domains]:
                continue
            result.extend(checker_classes)

        if not result:
            logger.warning(
                f"No checkers found for provider={provider!r}, domains={domains!r}. "
                f"Did you forget to import the checker modules?"
            )

        return result

    @classmethod
    def get_all_domains(cls, provider: str) -> List[str]:
        """Return all registered domains for a provider."""
        return sorted({
            domain
            for (prov, domain) in cls._registry
            if prov == provider.lower()
        })

    @classmethod
    def get_all_finding_ids(cls, provider: str) -> List[str]:
        """
        Return all finding IDs registered across all checkers for a provider.
        Useful for validating compliance mappings and generating docs.
        """
        ids = []
        for checker_cls in cls.get_checkers(provider):
            ids.extend(checker_cls.FINDING_TEMPLATES.keys())
        return sorted(ids)

    @classmethod
    def list_checkers(cls, provider: Optional[str] = None) -> List[Dict]:
        """
        Return a human-readable list of all registered checkers.
        Used by: cli --list-checks, documentation generation.
        """
        entries = cls._all_entries
        if provider:
            entries = [e for e in entries if e["provider"] == provider.lower()]
        return [
            {
                "provider":   e["provider"],
                "domain":     e["domain"],
                "class_name": e["class_name"],
                "finding_count": len(e["class"].FINDING_TEMPLATES),
                "finding_ids":   list(e["class"].FINDING_TEMPLATES.keys()),
            }
            for e in entries
        ]

    @classmethod
    def auto_discover(cls, providers_package: str = "providers") -> None:
        """
        Walk the providers package and import all checker modules so their
        @register decorators fire. Call this once at startup.

        This means adding a new checker module to providers/aws/checkers/
        automatically makes it available — no manifest to update.

        Args:
            providers_package: Dotted package name to walk.
                               Default "providers" works when running from
                               the cspm/ root directory.
        """
        try:
            package = importlib.import_module(providers_package)
        except ImportError as e:
            logger.warning(f"Could not import {providers_package}: {e}")
            return

        package_path = getattr(package, "__path__", None)
        if not package_path:
            return

        for finder, module_name, is_pkg in pkgutil.walk_packages(
            package_path,
            prefix=providers_package + ".",
        ):
            if "checkers" in module_name:
                try:
                    importlib.import_module(module_name)
                    logger.debug(f"Auto-discovered checker module: {module_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to import checker module {module_name}: {e}"
                    )

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registrations. Only used in tests to prevent
        state leaking between test cases.
        """
        cls._registry.clear()
        cls._all_entries.clear()