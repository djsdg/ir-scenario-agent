from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ir_agent.library import ScenarioLibrary
from ir_agent.specs import SpecCatalog
from ir_agent.tools import ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "scenario_library.json"
        shutil.copyfile(Path("data/scenario_library.json"), path)
        self.registry = ToolRegistry(ScenarioLibrary(path))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_definitions_are_strict_function_tools(self) -> None:
        definitions = self.registry.definitions()
        definitions_by_name = {definition["name"]: definition for definition in definitions}

        self.assertEqual(
            {definition["name"] for definition in definitions},
            {
                "match_ir_requirement",
                "draft_scenario_from_ir",
                "draft_use_cases_from_ir",
                "save_ir_requirement",
                "get_ir_requirement",
                "match_scenario",
                "search_scenarios",
                "get_scenario",
                "match_use_case",
                "search_use_cases",
                "get_use_case",
                "list_use_cases",
                "create_scenario",
                "create_use_case",
                "link_scenario_use_cases",
            },
        )
        for definition in definitions:
            self.assertEqual(definition["type"], "function")
            self.assertTrue(definition["strict"])
            self.assertFalse(definition["parameters"]["additionalProperties"])
            self._assert_strict_schema(definition["parameters"])

        create_use_case_properties = definitions_by_name["create_use_case"]["parameters"][
            "properties"
        ]
        self.assertIn("scenario_id", create_use_case_properties)
        self.assertNotIn("scenario_ids", create_use_case_properties)
        match_use_case_properties = definitions_by_name["match_use_case"]["parameters"][
            "properties"
        ]
        self.assertIn("scenario_id", match_use_case_properties)
        self.assertNotIn("scenario_ids", match_use_case_properties)

    def _assert_strict_schema(self, schema) -> None:
        if schema.get("type") == "object":
            self.assertIs(schema.get("additionalProperties"), False)
            properties = schema.get("properties", {})
            self.assertEqual(set(schema.get("required", [])), set(properties))
            for value in properties.values():
                self._assert_strict_schema(value)
        if schema.get("type") == "array":
            self._assert_strict_schema(schema["items"])
        for value in schema.get("anyOf", []):
            self._assert_strict_schema(value)

    def test_tool_errors_are_returned_as_model_visible_data(self) -> None:
        result = self.registry.execute("get_scenario", {"scenario_id": "does-not-exist"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "tool_execution_failed")

    def test_match_ir_tool_reports_missing_fields(self) -> None:
        result = self.registry.execute(
            "match_ir_requirement",
            {
                "ir": {
                    "code": None,
                    "title": "异常检测",
                    "description": "检测某部件异常",
                    "source": None,
                    "who": None,
                    "when": None,
                    "where": None,
                    "what": None,
                    "how": [],
                    "why": None,
                    "how_much": [],
                    "constraints": [],
                    "performance": None,
                    "reliability": None,
                    "serviceability": None,
                    "maintainability": None,
                    "sales": None,
                    "delivery_time": None,
                    "tags": [],
                },
                "top_k": 3,
                "min_score": 0.0,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["match"]["decision"], "needs_clarification")
        self.assertIn("who", result["match"]["missing_ir_fields"])

    def test_standalone_scenario_match_recommends_reuse(self) -> None:
        result = self.registry.execute(
            "match_scenario",
            {
                "query": "企业知识库多轮检索问答",
                "top_k": 5,
                "min_score": 0.0,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["decision"], "reuse_existing")
        self.assertEqual(result["matches"][0]["scenario"]["id"], "scn_enterprise_knowledge_qa")
        self.assertGreaterEqual(result["confidence"], result["reuse_threshold"])

    def test_standalone_match_uses_spec_reuse_threshold(self) -> None:
        payload = SpecCatalog.default().payload
        payload["matching"]["scenario_reuse_threshold"] = 0.99
        registry = ToolRegistry(
            self.registry.library,
            spec=SpecCatalog(payload),
        )

        result = registry.execute(
            "match_scenario",
            {
                "query": "企业知识库多轮检索问答",
                "top_k": 5,
                "min_score": 0.0,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["reuse_threshold"], 0.99)
        self.assertEqual(result["decision"], "create_new")

    def test_standalone_use_case_match_can_be_scoped_to_parent_scenario(self) -> None:
        result = self.registry.execute(
            "match_use_case",
            {
                "query": "用户提交问题，召回证据并生成可追溯回答",
                "scenario_id": "scn_enterprise_knowledge_qa",
                "top_k": 5,
                "min_score": 0.0,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["decision"], "reuse_existing")
        self.assertEqual(result["matches"][0]["use_case"]["id"], "uc_knowledge_retrieval_qa")
        self.assertEqual(result["scenario_id"], "scn_enterprise_knowledge_qa")

    def test_use_case_match_rejects_unknown_scenario_scope(self) -> None:
        result = self.registry.execute(
            "match_use_case",
            {
                "query": "查询并返回结果",
                "scenario_id": "does-not-exist",
                "top_k": 5,
                "min_score": 0.0,
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "tool_execution_failed")

    def test_scenario_required_fields_are_enforced(self) -> None:
        result = self.registry.execute(
            "create_scenario",
            {
                "name": "缺少字段的场景",
                "description": "这个场景故意没有类别、Actor、影响因素和责任人。",
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "invalid_arguments")
        self.assertIn("category", result["message"])
        self.assertIn("influence_factors", result["message"])

    def test_empty_use_case_shell_is_rejected(self) -> None:
        result = self.registry.execute(
            "create_use_case",
            {
                "name": "空壳 UC",
                "description": "只有名称和描述，没有完整行为链。",
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "invalid_arguments")
        self.assertIn("trigger_event", result["message"])


if __name__ == "__main__":
    unittest.main()
