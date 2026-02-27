"""
core/engine.py

CSPMEngine is the central orchestrator. It drives the full scan lifecycle:
    1. Validate provider config
    2. Authenticate
    3. Discover accounts (single or Organizations)
    4. For each account:
        a. Run all security checkers
        b. Build the resource graph
        c. Analyse attack paths
        d. Score the account
    5. Map findings to compliance frameworks
    6. Finalise the ScanResult
    7. Pass to reporter(s)

Design principles:
  - The engine knows NOTHING about AWS, boto3, or any specific cloud.
    It only talks to BaseProvider, BaseReporter, and core models.
  - Account scans run sequentially by default. The engine exposes a
    parallel=True option for future concurrency — but sequential first
    because it's easier to debug.
  - Every step is logged at INFO level so operators can watch progress.
  - Partial failures (one account fails) are captured and the scan
    continues — never let one bad account kill the whole run.
  - The engine is the ONLY place that calls provider.authenticate(),
    provider.get_accounts(), and result.finalise(). Callers just do:
        engine = CSPMEngine(provider, config)
        result = engine.scan()
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Type

from core.attack_graph.analyzer import AttackGraphAnalyzer
from core.base_provider import (
    AccountScanError, BaseProvider, ProviderAuthError, ProviderError
)
from core.checker_registry import CheckerRegistry
from core.models.finding import Finding
from core.models.scan_result import AccountResult, AccountScore, ScanResult

logger = logging.getLogger(__name__)


class ScanConfig:
    """
    Configuration for a single scan run.

    Separating config from the engine makes it easy to build different
    invocation patterns (CLI, Lambda, API) that all produce ScanConfig
    and hand it to CSPMEngine.
    """

    def __init__(
        self,
        # Scope
        accounts:          Optional[List[str]] = None,   # None = all accounts
        domains:           Optional[List[str]] = None,   # None = all domains
        regions:           Optional[List[str]] = None,   # None = all regions

        # Behaviour
        parallel:          bool = False,   # Future: concurrent account scans
        skip_graph:        bool = False,   # Skip attack path analysis
        skip_compliance:   bool = False,   # Skip compliance mapping
        max_findings:      int  = 10_000,  # Safety valve per account

        # Output
        triggered_by:      str  = "manual",  # "manual", "lambda", "ci"
        scanner_version:   str  = "1.0.0",

        # Filtering
        severity_threshold: Optional[str] = None,  # Only report >= this severity
    ):
        self.accounts           = accounts
        self.domains            = domains
        self.regions            = regions
        self.parallel           = parallel
        self.skip_graph         = skip_graph
        self.skip_compliance    = skip_compliance
        self.max_findings       = max_findings
        self.triggered_by       = triggered_by
        self.scanner_version    = scanner_version
        self.severity_threshold = severity_threshold


class CSPMEngine:
    """
    The central orchestrator for CSPM scans.

    The engine is intentionally thin — it delegates all real work to:
      - provider:    knows how to scan a specific cloud
      - analyzer:    knows how to find attack paths in a graph
      - mapper:      knows how to map findings to compliance controls
      - reporter(s): know how to render results

    The engine's only job is to call them in the right order and
    aggregate the results correctly.

    Usage:
        provider = AWSProvider(config={"profile": "prod"})
        engine   = CSPMEngine(provider=provider)
        result   = engine.scan()

        json_reporter = JSONReporter(output_dir="/tmp/reports")
        json_reporter.render(result)
    """

    def __init__(
        self,
        provider:  BaseProvider,
        config:    Optional[ScanConfig] = None,
    ):
        self.provider  = provider
        self.config    = config or ScanConfig()
        self.logger    = logging.getLogger(__name__)

        # These are imported lazily to avoid circular imports and
        # to allow future dependency injection in tests
        self._analyzer: Optional[AttackGraphAnalyzer] = None
        self._compliance_mapper = None   # Step 8
        self._reporters: List[Any] = []

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def scan(self) -> ScanResult:
        """
        Execute a complete CSPM scan. This is the only public method
        callers need.

        Returns a fully populated, finalised ScanResult.
        Raises CSPMScanError only on unrecoverable failures
        (auth failure, config error). Per-account failures are
        captured inside AccountResult.errors.
        """
        scan_id = str(uuid.uuid4())
        self.logger.info(
            f"=== CSPM Scan Starting === "
            f"[scan_id={scan_id}, provider={self.provider.provider_name}, "
            f"triggered_by={self.config.triggered_by}]"
        )

        result = ScanResult(
            scan_id=scan_id,
            provider=self.provider.provider_name,
            triggered_by=self.config.triggered_by,
            scanner_version=self.config.scanner_version,
        )

        try:
            # Step 1: Validate config
            self._validate_provider_config()

            # Step 2: Authenticate
            self._authenticate()

            # Step 3: Discover accounts to scan
            accounts = self._resolve_accounts()
            self.logger.info(f"Scanning {len(accounts)} account(s): {accounts}")

            # Step 4: Scan each account
            for account_id in accounts:
                account_result = self._scan_account(
                    account_id=account_id,
                    scan_id=scan_id,
                )
                result.account_results.append(account_result)

            # Step 5: Compliance mapping (post-scan, cross-account)
            if not self.config.skip_compliance:
                self._apply_compliance_mapping(result)

            # Step 6: Finalise (compute scores, seal timestamps)
            result.finalise()

            self.logger.info(
                f"=== CSPM Scan Complete === "
                f"[scan_id={scan_id}, accounts={len(accounts)}, "
                f"findings={len(result.all_findings)}, "
                f"score={result.total_score}, grade={result.overall_grade}]"
            )
            return result

        except (ProviderAuthError, CSPMConfigError) as e:
            # These are unrecoverable — abort the scan
            self.logger.error(f"Scan aborted: {e}")
            raise CSPMScanError(str(e)) from e

    # ------------------------------------------------------------------
    # Per-account scan orchestration
    # ------------------------------------------------------------------

    def _scan_account(self, account_id: str, scan_id: str) -> AccountResult:
        """
        Orchestrate the full scan for a single account.

        Failure modes:
          - assume-role fails → return AccountResult with error, continue
          - one checker fails → CheckerResult.error set, continue other checkers
          - graph build fails → log warning, continue without attack paths
        """
        self.logger.info(f"--- Scanning account: {account_id} ---")
        start = time.perf_counter()

        # Provider sets up credentials for this specific account
        # (e.g. assumes cross-account role for AWS Organizations scans)
        try:
            self.provider.pre_scan_hook(account_id)
        except Exception as e:
            self.logger.error(
                f"pre_scan_hook failed for {account_id}: {e}. "
                f"Skipping account."
            )
            ar = AccountResult(account_id=account_id)
            ar.add_error("pre_scan", f"Account setup failed: {e}")
            return ar

        # Run all security checks
        try:
            account_result = self.provider.run_checks(account_id)
        except AccountScanError as e:
            self.logger.error(str(e))
            ar = AccountResult(account_id=account_id)
            ar.add_error("scan", str(e))
            return ar
        except Exception as e:
            self.logger.error(
                f"Unexpected error scanning {account_id}: {e}", exc_info=True
            )
            ar = AccountResult(account_id=account_id)
            ar.add_error("scan", f"Unexpected error: {e}")
            return ar

        # Tag all findings with the scan_id for traceability
        for finding in account_result.findings:
            finding.scan_id = scan_id

        self.logger.info(
            f"Checks complete for {account_id}: "
            f"{len(account_result.findings)} findings "
            f"({len(account_result.errors)} errors)"
        )

        # Build attack graph and analyse paths
        if not self.config.skip_graph:
            self._analyse_attack_paths(account_id, account_result)

        # Score this account
        account_name = self.provider.get_account_name(account_id)
        account_result.account_name = account_name
        account_result.score = AccountScore.from_findings(
            account_id=account_id,
            findings=account_result.findings,
            account_name=account_name,
        )

        # Post-scan hook (session cleanup, progress reporting, etc.)
        try:
            self.provider.post_scan_hook(account_id, account_result)
        except Exception as e:
            self.logger.warning(f"post_scan_hook failed for {account_id}: {e}")

        duration = (time.perf_counter() - start)
        self.logger.info(
            f"Account {account_id} complete: "
            f"score={account_result.score.score} ({account_result.score.grade}) "
            f"in {duration:.1f}s"
        )
        return account_result

    # ------------------------------------------------------------------
    # Attack path analysis
    # ------------------------------------------------------------------

    def _analyse_attack_paths(
        self,
        account_id: str,
        account_result: AccountResult,
    ) -> None:
        """
        Build the resource graph for the account and run DFS traversal
        to find attack paths. Populates account_result.attack_graph.
        """
        try:
            self.logger.info(f"Building attack graph for {account_id}...")
            graph = self.provider.build_resource_graph(account_id)

            self.logger.info(
                f"Graph built: {graph.stats['total_nodes']} nodes, "
                f"{graph.stats['total_edges']} edges"
            )

            # Annotate graph nodes with finding data
            # (so report can link from attack path → finding)
            self._annotate_graph_with_findings(graph, account_result.findings)

            # Run traversal
            analyzer = self._get_analyzer()
            paths = analyzer.find_attack_paths(graph)
            graph.attack_paths = paths

            account_result.attack_graph = graph

            if paths:
                self.logger.warning(
                    f"ATTACK PATHS FOUND in {account_id}: {len(paths)} path(s)"
                )
                for path in paths:
                    self.logger.warning(f"  [{path.severity.value}] {path.chain_summary}")
            else:
                self.logger.info(f"No attack paths found in {account_id}")

        except Exception as e:
            self.logger.error(
                f"Attack graph analysis failed for {account_id}: {e}",
                exc_info=True,
            )
            account_result.add_error(
                "attack_graph", f"Graph analysis failed: {e}"
            )

    def _annotate_graph_with_findings(self, graph, findings: List[Finding]) -> None:
        """
        Link findings to their corresponding graph nodes.
        So the HTML report can show: "This node has 3 findings → click to expand."
        """
        for finding in findings:
            # Match by resource_id (ARN) — the most reliable identifier
            if finding.resource_id and graph.node_exists(finding.resource_id):
                node = graph.nodes[finding.resource_id]
                if finding.instance_id not in node.finding_ids:
                    node.finding_ids.append(finding.instance_id)
                finding.is_attack_node = True

    # ------------------------------------------------------------------
    # Compliance mapping
    # ------------------------------------------------------------------

    def _apply_compliance_mapping(self, result: ScanResult) -> None:
        """
        Annotate findings with compliance framework references.
        Stub for now — implemented fully in Step 8.
        """
        # The ComplianceMapper will be implemented in Step 8.
        # For now, log that it would run here.
        self.logger.info(
            "Compliance mapping: will be applied in Step 8 "
            "(ComplianceMapper not yet implemented)"
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _validate_provider_config(self) -> None:
        errors = self.provider.validate_config()
        if errors:
            raise CSPMConfigError(
                f"Provider config invalid for {self.provider.provider_name}: "
                + "; ".join(errors)
            )
        self.logger.debug("Provider config validated.")

    def _authenticate(self) -> None:
        self.logger.info(
            f"Authenticating with {self.provider.provider_name}..."
        )
        try:
            success = self.provider.authenticate()
            if not success:
                raise ProviderAuthError(
                    f"Authentication returned False for "
                    f"{self.provider.provider_name}. Check credentials."
                )
            self.logger.info("Authentication successful.")
        except ProviderAuthError:
            raise
        except Exception as e:
            raise ProviderAuthError(
                f"Authentication failed for {self.provider.provider_name}: {e}"
            ) from e

    def _resolve_accounts(self) -> List[str]:
        """
        Determine which accounts to scan.
        If config.accounts is set, use that list.
        Otherwise ask the provider (may return 1 or many).
        """
        if self.config.accounts:
            self.logger.info(
                f"Account list from config: {self.config.accounts}"
            )
            return self.config.accounts

        accounts = self.provider.get_accounts()
        if not accounts:
            raise CSPMScanError(
                f"Provider {self.provider.provider_name} returned no accounts to scan."
            )
        return accounts

    def _get_analyzer(self) -> AttackGraphAnalyzer:
        """Lazy-initialise the analyzer (avoids import at module level)."""
        if self._analyzer is None:
            self._analyzer = AttackGraphAnalyzer()
        return self._analyzer

    def add_reporter(self, reporter: Any) -> None:
        """Register a reporter. Reporters are invoked by the caller after scan()."""
        self._reporters.append(reporter)

    def __repr__(self) -> str:
        return (
            f"CSPMEngine("
            f"provider={self.provider.provider_name!r}, "
            f"skip_graph={self.config.skip_graph})"
        )


# ---------------------------------------------------------------------------
# Engine-level exceptions
# ---------------------------------------------------------------------------

class CSPMError(Exception):
    """Base exception for all engine-level errors."""
    pass


class CSPMConfigError(CSPMError):
    """Raised when the engine or provider config is invalid."""
    pass


class CSPMScanError(CSPMError):
    """Raised on unrecoverable scan failures (auth, config)."""
    pass