from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ir_agent.documents import read_document
from ir_agent.library import ScenarioLibrary
from ir_agent.specs import SpecCatalog
from ir_agent.tools import ToolRegistry


class SpecAndDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        library_path = Path(self.temp_dir.name) / "scenario_library.json"
        shutil.copyfile(Path("data/scenario_library.json"), library_path)
        self.library = ScenarioLibrary(library_path)
        self.spec = SpecCatalog.from_file("config/ir_sc_uc_spec.json")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_spec_drafts_scenario_without_inventing_owner(self) -> None:
        requirement = self.library.get_requirement("IR-XXXX-001")
        result = self.spec.draft_scenario(requirement)

        self.assertIn("owner", result["missing_required_fields"])
        self.assertEqual(result["draft"]["category"], "派生场景")
        self.assertTrue(result["draft"]["influence_factors"])
        self.assertEqual(
            result["draft"]["influence_factors"][0]["dimension"],
            "hardware_environment",
        )

    def test_spec_exposes_matching_rules(self) -> None:
        rules = self.spec.matching_rules

        self.assertEqual(rules["scenario_reuse_threshold"], 0.45)
        self.assertEqual(rules["scenario_strong_threshold"], 0.70)
        self.assertIn("system", rules["actor_categories"])
        self.assertIn("normal_service", rules["lifecycle_categories"])

    def test_spec_derives_uc_for_existing_scenario(self) -> None:
        requirement = self.library.get_requirement("IR-XXXX-001")
        scenario = self.library.get_scenario("SCN-XXXX-001")
        result = self.spec.draft_use_case(requirement, scenario)

        self.assertEqual(result["missing_required_fields"], [])
        self.assertEqual(result["scenario_id"], "SCN-XXXX-001")
        self.assertTrue(result["draft"]["main_success_scenario"])

    def test_draft_tools_are_read_only_and_return_parent_scenario_alternatives(self) -> None:
        registry = ToolRegistry(self.library, spec=self.spec)
        requirement = self.library.get_requirement("IR-XXXX-001")
        ir = requirement.model_dump(exclude={"id", "created_at", "updated_at"})
        before = len(self.library.list_use_cases())
        result = registry.execute(
            "draft_use_cases_from_ir",
            {
                "ir": ir,
                "candidate_scenario_ids": ["SCN-XXXX-001", "SCN-XXXX-002"],
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["drafts"]), 2)
        self.assertEqual(len(self.library.list_use_cases()), before)

    def test_document_reader_handles_text_and_json(self) -> None:
        root = Path(self.temp_dir.name)
        text_path = root / "requirement.md"
        text_path.write_text("# IR\n某系统需求", encoding="utf-8")
        json_path = root / "requirement.json"
        json_path.write_text(json.dumps({"title": "需求", "why": "改进"}), encoding="utf-8")

        self.assertIn("某系统需求", read_document(text_path))
        self.assertIn('"title": "需求"', read_document(json_path))


if __name__ == "__main__":
    unittest.main()
