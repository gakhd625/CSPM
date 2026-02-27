# core/engine.py — the orchestrator
class CSPMEngine:
    def __init__(self, provider: BaseProvider, config: dict):
        self.provider = provider
        self.config = config
        self.graph_analyzer = AttackGraphAnalyzer()
        self.compliance_mapper = ComplianceMapper()
        self.scorer = Scorer()

    def scan(self) -> ScanResult:
        accounts = self.provider.get_accounts()
        all_findings = []

        for account in accounts:
            findings = self.provider.run_checks(account)
            graph = self.provider.build_resource_graph(account)
            attack_paths = self.graph_analyzer.analyze(graph)
            all_findings.extend(findings)

        scored = self.scorer.score(all_findings)
        mapped = self.compliance_mapper.map(all_findings)

        return ScanResult(
            findings=all_findings,
            attack_paths=attack_paths,
            score=scored,
            compliance=mapped
        )