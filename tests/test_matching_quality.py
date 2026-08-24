from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ir_agent.domain import IRRequirementInput
from ir_agent.library import ScenarioLibrary


class OfflineMatchingQualityTests(unittest.TestCase):
    """Small, repeatable quality gate for the deterministic matcher."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "scenario_library.json"
        shutil.copyfile(Path("data/scenario_library.json"), path)
        self.library = ScenarioLibrary(path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_known_ir_is_flagged_as_ambiguous_between_similar_scenarios(self) -> None:
        requirement = self.library.get_requirement("IR-XXXX-001")
        ir = IRRequirementInput.model_validate(
            requirement.model_dump(exclude={"id", "created_at", "updated_at"})
        )

        result = self.library.match_ir(ir, top_k=5)

        self.assertEqual(result.decision, "needs_clarification")
        self.assertTrue(result.ambiguous)
        self.assertLess(result.score_margin, 0.08)
        self.assertEqual(
            {item.scenario.id for item in result.scenario_matches[:2]},
            {"SCN-XXXX-001", "SCN-XXXX-002"},
        )

    def test_known_knowledge_ir_reuses_scenario_and_existing_uc(self) -> None:
        result = self.library.match_ir(
            IRRequirementInput(
                title="企业知识库多轮检索问答",
                description="系统在正常服务期间从企业知识库检索证据并生成可追溯回答。",
                who="企业知识用户",
                when="正常服务期间",
                where="企业知识库",
                what="检索证据并生成可追溯回答",
                how=["理解问题", "召回证据", "生成回答"],
                why="提升回答可信度",
                how_much=["回答必须可追溯"],
                constraints=["保留证据链"],
            ),
            top_k=5,
        )

        self.assertEqual(result.decision, "reuse_scenario_and_uc")
        self.assertEqual(result.scenario_matches[0].scenario.id, "scn_enterprise_knowledge_qa")
        self.assertGreaterEqual(result.confidence, 0.70)

    def test_unrelated_complete_ir_recommends_a_new_scenario(self) -> None:
        result = self.library.match_ir(
            IRRequirementInput(
                title="卫星姿态控制异常恢复",
                description="飞控系统在轨运行期间发现姿态漂移并执行推进器重配置。",
                who="飞控系统",
                when="在轨运行期间",
                where="姿态传感器与推进器",
                what="检测姿态漂移并重配置推进器",
                how=["读取传感器", "计算偏差", "切换推进器"],
                why="保持卫星姿态稳定",
                how_much=["恢复时间不超过十秒"],
                constraints=["不得影响载荷任务"],
            ),
            top_k=5,
        )

        self.assertEqual(result.decision, "create_scenario_and_uc")
        self.assertLess(result.confidence, 0.45)


if __name__ == "__main__":
    unittest.main()
