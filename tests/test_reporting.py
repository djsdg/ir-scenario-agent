from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ir_agent.domain import (
    AgentResult,
    ResolutionCandidate,
    ScenarioResolution,
    ToolCallRecord,
)
from ir_agent.library import ScenarioLibrary
from ir_agent.reporting import build_analysis_report, save_run_report


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        library_path = Path(self.temp_dir.name) / "scenario_library.json"
        library_path.write_bytes(Path("data/scenario_library.json").read_bytes())
        self.library = ScenarioLibrary(library_path)

    def _result(self) -> AgentResult:
        requirement = self.library.get_requirement("IR-XXXX-001")
        match = self.library.match_ir(requirement, top_k=5)
        top_scenario = match.scenario_matches[0]
        top_use_case = match.use_case_matches[0]
        resolution = ScenarioResolution(
            status="matched",
            decision=match.decision,
            ir_id=requirement.id,
            request_summary=requirement.title,
            candidates=[
                ResolutionCandidate(
                    scenario_id=top_scenario.scenario.id,
                    score=top_scenario.score,
                    matched_terms=top_scenario.matched_terms,
                    matched_dimensions=top_scenario.matched_dimensions,
                    gaps=top_scenario.gaps,
                    reason="测试匹配",
                )
            ],
            selected_scenario_ids=[top_scenario.scenario.id],
            use_case_ids=[top_use_case.use_case.id],
            created_scenario_id=None,
            created_use_case_ids=[],
            confidence=match.confidence,
            missing_required_fields=[],
            gaps=[],
            next_steps=[],
        )
        return AgentResult(
            output_text=resolution.model_dump_json(),
            resolution=resolution,
            tool_calls=[
                ToolCallRecord(
                    name="match_ir_requirement",
                    result={"ok": True, "match": match.model_dump(mode="json")},
                )
            ],
        )

    def test_report_contains_field_evidence_and_parent_uc(self) -> None:
        report = build_analysis_report(self._result(), self.library)
        self.assertTrue(report["scenarios"]["matches"])
        self.assertTrue(report["scenarios"]["matches"][0]["matched_fields"])
        self.assertTrue(report["use_cases"]["matches"])
        parent_id = report["use_cases"]["matches"][0]["parent_scenario_id"]
        self.assertEqual(parent_id, report["scenarios"]["matches"][0]["id"])
        self.assertTrue(report["use_cases"]["by_scenario"])

    def test_report_writes_separate_scenario_and_use_case_folders(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            result_path = save_run_report(
                self._result(),
                output_dir,
                session_id="report-test",
                input_text="IR 测试",
                library=self.library,
            )
            run_root = result_path.parent
            self.assertTrue((run_root / "result.json").exists())
            self.assertTrue((run_root / "report.md").exists())
            self.assertTrue((run_root / "scenarios" / "matches.json").exists())
            self.assertTrue((run_root / "use_cases" / "matches.json").exists())
            self.assertTrue((run_root / "use_cases" / "by_scenario").is_dir())
            markdown = (run_root / "report.md").read_text(encoding="utf-8")
            self.assertIn("命中部分", markdown)
            self.assertIn("SC → UC 关系", markdown)

    def test_report_captures_scenario_and_use_case_updates(self) -> None:
        result = self._result()
        scenario = self.library.get_scenario("SCN-XXXX-001")
        use_case = self.library.get_use_case(scenario.use_case_ids[0])
        result.tool_calls.extend(
            [
                ToolCallRecord(
                    name="update_scenario",
                    arguments={"scenario_id": scenario.id},
                    approved=True,
                    result={"ok": True, "updated": True, "scenario": scenario.model_dump(mode="json")},
                ),
                ToolCallRecord(
                    name="update_use_case",
                    arguments={"use_case_id": use_case.id},
                    approved=True,
                    result={"ok": True, "updated": True, "use_case": use_case.model_dump(mode="json")},
                ),
            ]
        )
        report = build_analysis_report(result, self.library)
        self.assertEqual(report["scenarios"]["updated"][0]["id"], scenario.id)
        self.assertEqual(report["use_cases"]["updated"][0]["id"], use_case.id)
        self.assertTrue(report["library"]["updated_by_this_run"])


if __name__ == "__main__":
    unittest.main()
