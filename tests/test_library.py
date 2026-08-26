from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ir_agent.domain import (
    CreateScenarioRequest,
    CreateUseCaseRequest,
    InfluenceFactor,
    IRRequirementInput,
    MoveUseCaseRequest,
    TransitionRecordRequest,
    UpdateScenarioRequest,
    UpdateUseCaseRequest,
)
from ir_agent.library import ScenarioLibrary, tokenize


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


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

    def test_tokenizer_keeps_short_chinese_phrases_as_evidence(self) -> None:
        tokens = tokenize("企业知识库多轮检索问答")

        self.assertIn("知", tokens)
        self.assertIn("知识", tokens)
        self.assertIn("知识库", tokens)

    def test_configured_synonym_improves_alias_recall(self) -> None:
        self.library.configure_matching(
            {
                "synonyms": {"异常检测": ["故障监测"]},
                "lexical_weight": 0.75,
                "tfidf_weight": 0.25,
            }
        )

        matches = self.library.search("故障监测", top_k=5)

        self.assertTrue(matches)
        self.assertIn(matches[0].scenario.id, {"SCN-XXXX-001", "SCN-XXXX-002"})

    def test_optional_embedding_is_fused_without_breaking_lexical_search(self) -> None:
        provider = FakeEmbeddingProvider()
        self.library.configure_matching(
            {"lexical_weight": 0.60, "tfidf_weight": 0.20, "embedding_weight": 0.20}
        )
        self.library.configure_embedding(provider)

        matches = self.library.search("企业知识库多轮检索问答", top_k=3)

        self.assertTrue(provider.calls)
        self.assertEqual(provider.calls[0][0], "企业知识库多轮检索问答")
        self.assertEqual(matches[0].scenario.id, "scn_enterprise_knowledge_qa")

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
        self.assertIn(
            result.decision,
            {"reuse_scenario_and_uc", "reuse_scenario_create_uc", "needs_clarification"},
        )
        self.assertIn(result.scenario_matches[0].scenario.id, {"SCN-XXXX-001", "SCN-XXXX-002"})
        self.assertIn("Actor", result.scenario_matches[0].matched_dimensions)
        self.assertIn("Actor", result.scenario_matches[0].matched_fields)
        self.assertTrue(result.scenario_matches[0].matched_fields["Actor"])
        self.assertGreaterEqual(result.scenario_matches[0].base_score, 0.0)
        self.assertEqual(result.scenario_matches[0].consistency_bonus, 0.0)
        self.assertGreaterEqual(
            result.scenario_matches[0].fit_score,
            result.scenario_matches[0].score,
        )
        self.assertTrue(result.scenario_matches[0].dimension_scores)
        self.assertTrue(result.confidence_label)
        self.assertTrue(
            all(
                item.use_case.id in result.scenario_matches[0].scenario.use_case_ids
                for item in result.use_case_matches
            )
        )
        self.assertTrue(
            all(
                item.parent_scenario_id == result.scenario_matches[0].scenario.id
                for item in result.use_case_matches
            )
        )
        if result.ambiguous:
            self.assertLess(result.score_margin, 0.08)

    def test_match_separates_fit_score_from_evidence_completeness(self) -> None:
        result = self.library.match_ir(
            IRRequirementInput(
                title="某指令异常检测改进",
                description="提升异常检测能力。",
                who="本系统",
                what="改进某指令异常检测机制",
            ),
            top_k=3,
        )

        self.assertTrue(result.scenario_matches)
        self.assertAlmostEqual(result.evidence_completeness, 0.60)
        self.assertEqual(result.supplied_dimensions, ["目标/行为", "Actor"])
        top = result.scenario_matches[0]
        self.assertAlmostEqual(top.evidence_completeness, 0.60)
        self.assertGreater(top.fit_score, top.score)
        self.assertEqual(top.consistency_bonus, 0.0)

    def test_ir_to_uc_matching_compares_behavior_contract_fields(self) -> None:
        result = self.library.match_ir(self.library.get_requirement("IR-TEST-001"), top_k=3)

        self.assertTrue(result.use_case_matches)
        top = result.use_case_matches[0]
        self.assertEqual(
            set(top.dimension_scores),
            {"目标/行为", "Actor", "触发/前置", "处理步骤", "保证/DFX", "约束"},
        )
        self.assertGreater(top.dimension_scores["触发/前置"].score, 0.0)
        self.assertIn("处理步骤", top.matched_dimensions)
        self.assertAlmostEqual(top.evidence_completeness, 1.0)

    def test_evaluate_scenario_fit_returns_explainable_dimension_scores(self) -> None:
        ir = self.library.get_requirement("IR-XXXX-001")
        normalized_ir = IRRequirementInput.model_validate(
            ir.model_dump(exclude={"id", "created_at", "updated_at"})
        )

        evaluation = self.library.evaluate_scenario_fit(
            normalized_ir,
            "SCN-XXXX-001",
        )

        self.assertEqual(evaluation["scenario_id"], "SCN-XXXX-001")
        self.assertIn("目标/行为", evaluation["dimension_scores"])
        self.assertIn("matching_rules", evaluation)
        self.assertIn("low_score_reasons", evaluation)
        self.assertIsInstance(evaluation["score"], float)

    def test_explicit_actor_conflict_requires_clarification(self) -> None:
        result = self.library.match_ir(
            IRRequirementInput(
                title="企业知识库多轮检索问答",
                description="系统在正常服务期间从企业知识库检索证据并生成可追溯回答。",
                who="本系统",
                when="系统正常运行时",
                where="企业知识库",
                what="检索知识库证据并生成回答",
                how=["理解问题", "召回证据", "生成回答"],
                why="提升回答可信度",
                how_much=["回答必须可追溯"],
            ),
            top_k=3,
        )

        self.assertEqual(result.decision, "needs_clarification")
        self.assertIn("Actor 明确冲突", result.scenario_matches[0].conflicts)
        self.assertTrue(any("硬冲突" in item for item in result.rationale))

    def test_uncovered_critical_dimension_blocks_auto_reuse(self) -> None:
        result = self.library.match_ir(
            IRRequirementInput(
                title="某指令异常检测和隔离",
                description="控制器在正常运行时检测某部件异常并隔离节点。",
                who="控制器",
                when="系统正常运行时",
                where="某部件",
                what="检测某指令异常并隔离节点",
                how=["检测异常", "隔离节点"],
                why="提升可靠性",
                how_much=["检测后告警"],
            ),
            top_k=3,
        )

        self.assertEqual(result.decision, "needs_clarification")
        self.assertTrue(any("关键维度" in item for item in result.rationale))
        self.assertIn("Actor未覆盖", result.rationale[-1])

    def test_incomplete_ir_requires_clarification(self) -> None:
        result = self.library.match_ir(
            IRRequirementInput(title="异常检测", description="检测某异常")
        )

        self.assertEqual(result.decision, "needs_clarification")
        self.assertIn("who", result.missing_ir_fields)
        self.assertIn("what", result.missing_ir_fields)

    def test_only_who_and_what_are_required_for_ir_matching(self) -> None:
        result = self.library.match_ir(
            IRRequirementInput(
                title="某指令异常检测机制改进",
                description="系统正常运行时，某部件异常导致关键进程反复复位。",
                who="本系统",
                what="改进某指令异常检测机制",
            ),
            top_k=3,
        )

        self.assertEqual(result.missing_ir_fields, [])
        self.assertTrue(result.scenario_matches)
        self.assertFalse(any("缺少必填字段" in item for item in result.rationale))
        self.assertTrue(any("可选 5W2H 字段" in item for item in result.rationale))

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

    def test_quality_report_finds_broken_uc_references(self) -> None:
        payload = json.loads(self.library_path.read_text(encoding="utf-8"))
        payload["scenarios"][0]["use_case_ids"].append("UC-MISSING")
        orphan = dict(payload["use_cases"][0])
        orphan["id"] = "UC-ORPHAN"
        payload["use_cases"].append(orphan)
        self.library_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        report = self.library.quality_report()

        self.assertFalse(report["ok"])
        issue_kinds = {item["kind"] for item in report["issues"]}
        self.assertIn("missing_use_case_reference", issue_kinds)
        self.assertIn("orphan_use_case", issue_kinds)

    def test_update_transition_and_move_preserve_revisions_and_parent_rule(self) -> None:
        scenario = self.library.create(
            CreateScenarioRequest(
                name="生命周期测试场景",
                description="用于验证场景修改、状态流转和 UC 迁移。",
                category="Scenario",
                actor="测试系统",
                influence_factors=[
                    InfluenceFactor(
                        name="测试硬件",
                        kind="environment",
                        dimension="hardware_environment",
                        candidate_values=["测试节点"],
                        selected_values=["测试节点"],
                    )
                ],
                owner="test",
                business_goal="验证生命周期能力",
                actions=["执行测试"],
                constraints=["仅用于测试"],
                lifecycle="正常服务",
            )
        )
        updated_scenario = self.library.update_scenario(
            UpdateScenarioRequest(
                scenario_id=scenario.id,
                description="用于验证场景修改、状态流转和 UC 迁移的扩展描述。",
            )
        )
        self.assertEqual(updated_scenario.revision, 2)

        for workflow_status in ("Inwork", "Review", "Publish"):
            updated_scenario = self.library.transition_record(
                TransitionRecordRequest(
                    record_type="scenario",
                    record_id=scenario.id,
                    workflow_status=workflow_status,
                )
            )
        self.assertEqual(updated_scenario.workflow_status, "Publish")
        self.assertEqual(updated_scenario.status, "published")

        use_case = self.library.create_use_case(
            CreateUseCaseRequest(
                name="生命周期测试用例",
                description="验证一个完整的 UC 可以被修改并迁移到另一个父场景。",
                actor="测试系统",
                preconditions=["测试场景已发布"],
                trigger_event="执行迁移测试",
                success_guarantee="UC 被准确迁移并保持完整行为链",
                minimum_guarantee="迁移失败时原父场景关系保持不变",
                main_success_scenario=["读取 UC", "切换父场景", "校验唯一归属"],
                scenario_id=scenario.id,
            )
        )
        updated_use_case = self.library.update_use_case(
            UpdateUseCaseRequest(
                use_case_id=use_case.id,
                trigger_event="执行修改后的迁移测试",
            )
        )
        self.assertEqual(updated_use_case.revision, 2)

        moved = self.library.move_use_case(
            MoveUseCaseRequest(
                use_case_id=use_case.id,
                target_scenario_id="SCN-XXXX-001",
            )
        )
        self.assertEqual(moved.revision, 3)
        self.assertNotIn(use_case.id, self.library.get_scenario(scenario.id).use_case_ids)
        self.assertIn(use_case.id, self.library.get_scenario("SCN-XXXX-001").use_case_ids)


if __name__ == "__main__":
    unittest.main()
