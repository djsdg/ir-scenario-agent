from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ir_agent.domain import (
    AgentResult,
    IRRequirementInput,
    ResolutionCandidate,
    ScenarioResolution,
    ToolCallRecord,
)
from ir_agent.library import ScenarioLibrary
from ir_agent.reporting import apply_human_review, build_analysis_report, save_run_report


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        library_path = Path(self.temp_dir.name) / "scenario_library.json"
        library_path.write_bytes(Path("data/scenario_library.json").read_bytes())
        self.library = ScenarioLibrary(library_path)

    def _result(self, *, top_k: int = 5) -> AgentResult:
        requirement = self.library.get_requirement("IR-XXXX-001")
        match = self.library.match_ir(requirement, top_k=top_k)
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
        self.assertIn("matching", report)
        self.assertIn("confidence_label", report["matching"])
        self.assertIn("evidence_completeness", report["matching"])
        self.assertTrue(report["field_comparison"])
        self.assertIn("dimension_scores", report["scenarios"]["matches"][0])
        self.assertIn("fit_score", report["use_cases"]["matches"][0])

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
            self.assertTrue((run_root / "evaluation" / "match_summary.csv").exists())
            self.assertTrue((run_root / "evaluation" / "field_comparison.csv").exists())
            self.assertTrue((run_root / "evaluation" / "human_review_template.csv").exists())
            self.assertTrue((run_root / "evaluation" / "human_review_matrix.csv").exists())
            self.assertTrue((run_root / "evaluation" / "review_candidates.json").exists())
            self.assertTrue((run_root / "evaluation" / "scenario_fit.csv").exists())
            summary_csv = (run_root / "evaluation" / "match_summary.csv").read_text(
                encoding="utf-8-sig"
            )
            field_csv = (run_root / "evaluation" / "field_comparison.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("目标/行为_score", summary_csv)
            self.assertIn("fit_score", summary_csv)
            self.assertIn("evidence_completeness", summary_csv)
            self.assertIn("ai_consistency_hint", field_csv)
            self.assertIn("candidate_rank", field_csv)
            self.assertIn("review_status", field_csv)
            markdown = (run_root / "report.md").read_text(encoding="utf-8")
            self.assertIn("命中部分", markdown)
            self.assertIn("可用证据匹配度", markdown)
            self.assertIn("人工复核字段表", markdown)
            self.assertIn("候选总览", markdown)
            self.assertIn("SC → UC 关系", markdown)

    def test_human_review_can_be_edited_and_applied_without_library_write(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            result_path = save_run_report(
                self._result(),
                output_dir,
                session_id="review-edit-test",
                input_text="IR 测试",
                library=self.library,
            )
            run_root = result_path.parent
            review_csv = run_root / "evaluation" / "human_review_template.csv"
            with review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            rows[0]["human_value"] = "人工确认：本系统"
            rows[0]["consistency"] = "一致"
            rows[0]["review_status"] = "已确认"
            rows[0]["human_decision"] = "复用"
            rows[0]["human_notes"] = "字段与 IR 的 Who 一致"
            edited_csv = Path(output_dir) / "edited_review.csv"
            with edited_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            report = build_analysis_report(self._result(), self.library)
            reviewed = apply_human_review(report, edited_csv)

            first_row = reviewed["field_comparison"][0]
            self.assertEqual(first_row["human_value"], "人工确认：本系统")
            self.assertEqual(first_row["consistency"], "一致")
            self.assertEqual(first_row["review_status"], "已确认")
            self.assertEqual(first_row["human_decision"], "复用")
            self.assertEqual(reviewed["review"]["reviewed_rows"], 1)
            self.assertFalse(reviewed["library"].get("updated_by_this_review", False))
            self.assertEqual(
                reviewed["review"]["candidate_summary"][0]["review_status"],
                "已确认",
            )

    def test_human_review_uses_top_two_candidates_even_when_not_selected(self) -> None:
        ir = IRRequirementInput(
            title="Quantum Clock",
            description="A remote device performs quantum clock synchronization for orbital networks.",
            who="external system",
            what="synchronize quantum clock signals",
        )
        match = self.library.match_ir(ir, top_k=1)
        self.assertEqual(match.decision, "create_scenario_and_uc")
        result = AgentResult(
            output_text="",
            tool_calls=[
                ToolCallRecord(
                    name="match_ir_requirement",
                    result={"ok": True, "match": match.model_dump(mode="json")},
                )
            ],
            resolution=ScenarioResolution(
                status="no_match",
                decision="create_scenario_and_uc",
                ir_id=None,
                request_summary="测试不复用场景时的人工复核候选",
                candidates=[],
                selected_scenario_ids=[],
                use_case_ids=[],
                created_scenario_id=None,
                created_use_case_ids=[],
                confidence=0.0,
                missing_required_fields=[],
                gaps=[],
                next_steps=[],
            ),
        )

        report = build_analysis_report(result, self.library)

        candidates = report["review"]["top_scenario_candidates"]
        self.assertEqual(len(report["scenarios"]["matches"]), 1)
        self.assertEqual(len(candidates), 2)
        self.assertGreaterEqual(candidates[0]["score"], candidates[1]["score"])
        self.assertTrue(all(item["score"] < 0.45 for item in candidates))
        review_ids = {item["id"] for item in candidates}
        field_rows = [
            item
            for item in report["field_comparison"]
            if item["source_type"] == "场景库候选"
        ]
        self.assertEqual({item["sc_id"] for item in field_rows}, review_ids)
        self.assertEqual({item["candidate_rank"] for item in field_rows}, {1, 2})

    def test_report_includes_specified_scenario_evaluation(self) -> None:
        result = self._result()
        requirement = self.library.get_requirement("IR-XXXX-001")
        ir = IRRequirementInput.model_validate(
            requirement.model_dump(exclude={"id", "created_at", "updated_at"})
        )
        evaluation = self.library.evaluate_scenario_fit(ir, "SCN-XXXX-001")
        result.tool_calls.append(
            ToolCallRecord(
                name="evaluate_scenario_fit",
                arguments={"scenario_id": "SCN-XXXX-001"},
                result={"ok": True, "evaluation": evaluation},
            )
        )

        report = build_analysis_report(result, self.library)

        self.assertEqual(
            report["evaluations"]["scenario_fit"][0]["scenario_id"],
            "SCN-XXXX-001",
        )
        self.assertTrue(
            any(row["source_type"] == "指定 SC 评估" for row in report["field_comparison"])
        )

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
