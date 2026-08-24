from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ir_agent.domain import (
    CreateScenarioRequest,
    CreateUseCaseRequest,
    IRRequirementInput,
    InfluenceFactor,
)
from ir_agent.library import ScenarioLibrary


class ScenarioLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.library_path = Path(self.temp_dir.name) / "scenario_library.json"
        shutil.copyfile(Path("data/scenario_library.json"), self.library_path)
        self.library = ScenarioLibrary(self.library_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_search_returns_relevant_candidates(self) -> None:
        matches = self.library.search("企业知识库多轮检索问答", top_k=3)

        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0].scenario.id, "scn_enterprise_knowledge_qa")
        self.assertGreater(matches[0].score, 0.5)
        self.assertIn("知", matches[0].matched_terms)

    def test_scenario_cannot_adopt_a_uc_owned_by_another_scenario(self) -> None:
        initial_count = len(self.library.list_scenarios())
        created = self.library.create(
            CreateScenarioRequest(
                name="医疗知识库多跳检索",
                description="面向医疗知识库的多跳证据检索场景，用于回答需要跨文档核对的问题。",
                category="技术场景",
                actor="医疗知识用户",
                influence_factors=[
                    InfluenceFactor(
                        name="知识来源",
                        candidate_values=["指南", "病历"],
                        selected_values=["医疗知识库"],
                    )
                ],
                business_goal="验证医疗知识检索场景",
                actions=["检索证据", "人工复核"],
                constraints=["保留证据链"],
                lifecycle="正常服务",
                ir_intent="召回医疗知识和证据链，并支持人工复核。",
                tags=["医疗", "多跳", "证据"],
                status="draft",
                owner="test",
            )
        )

        self.assertTrue(created.id.startswith("SCN-DRAFT-"))
        self.assertEqual(created.use_case_ids, [])

        with self.assertRaisesRegex(ValueError, "already belongs to another scenario"):
            self.library.link_use_cases(created.id, ["uc_knowledge_retrieval_qa"])
        persisted = json.loads(self.library_path.read_text(encoding="utf-8"))
        self.assertEqual(len(persisted["scenarios"]), initial_count + 1)

    def test_match_ir_uses_scenario_and_use_case_dimensions(self) -> None:
        ir = self.library.get_requirement("IR-XXXX-001")

        result = self.library.match_ir(
            IRRequirementInput.model_validate(
                ir.model_dump(exclude={"id", "created_at", "updated_at"})
            ),
            top_k=3,
        )

        self.assertEqual(result.missing_ir_fields, [])
        self.assertIn(result.decision, {"reuse_scenario_and_uc", "reuse_scenario_create_uc"})
        self.assertIn(result.scenario_matches[0].scenario.id, {"SCN-XXXX-001", "SCN-XXXX-002"})
        self.assertIn("Actor", result.scenario_matches[0].matched_dimensions)

    def test_incomplete_ir_requires_clarification(self) -> None:
        result = self.library.match_ir(
            IRRequirementInput(title="异常检测", description="检测某异常")
        )

        self.assertEqual(result.decision, "needs_clarification")
        self.assertIn("who", result.missing_ir_fields)

    def test_create_use_case_links_existing_scenario(self) -> None:
        created = self.library.create_use_case(
            CreateUseCaseRequest(
                name="异常阈值调整",
                description="运维人员调整异常检测阈值并验证配置生效。",
                actor="运维人员",
                preconditions=["异常检测功能已启用"],
                trigger_event="需要调整异常检测灵敏度",
                success_guarantee="新阈值生效并保留审计记录",
                minimum_guarantee="配置失败时保留原阈值",
                main_success_scenario=["读取当前阈值", "提交新阈值", "校验并生效"],
                scenario_id="SCN-XXXX-001",
            )
        )

        self.assertTrue(created.id.startswith("UC-DRAFT-"))
        self.assertIn(created.id, self.library.get_scenario("SCN-XXXX-001").use_case_ids)
        self.assertNotIn(created.id, self.library.get_scenario("SCN-XXXX-002").use_case_ids)
        with self.assertRaisesRegex(ValueError, "already belongs to another scenario"):
            self.library.link_use_cases("SCN-XXXX-002", [created.id])

    def test_directory_mode_keeps_uc_library_under_scenario_root(self) -> None:
        root = Path(self.temp_dir.name) / "scene_library"
        root.mkdir()
        scenario_path = root / "scenarios.json"
        shutil.copyfile(Path("data/scenario_library.json"), scenario_path)

        library = ScenarioLibrary(root)
        self.assertEqual(library.path, scenario_path)
        self.assertEqual(library.use_case_path, root / "uc" / "use_cases.json")
        self.assertGreater(len(library.list_use_cases()), 0)

        created = library.create_use_case(
            CreateUseCaseRequest(
                name="目录分库测试 UC",
                description="验证场景库目录模式下 UC 文件能够独立保存。",
                actor="测试系统",
                preconditions=["场景目录已加载"],
                trigger_event="执行分库测试",
                success_guarantee="UC 被保存到独立文件并保持场景关联",
                minimum_guarantee="写入失败时不返回成功结果",
                main_success_scenario=["读取场景库", "写入 UC 库"],
                scenario_id="SCN-XXXX-001",
            )
        )

        scenario_payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        uc_payload = json.loads(library.use_case_path.read_text(encoding="utf-8"))
        self.assertEqual(scenario_payload["use_cases"], [])
        self.assertIn(created.id, {item["id"] for item in uc_payload["use_cases"]})


if __name__ == "__main__":
    unittest.main()
