from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ir_agent.agent import IRScenarioAgent
from ir_agent.api import _build_parser as _build_api_parser
from ir_agent.config import Settings
from ir_agent.library import ScenarioLibrary
from ir_agent.mcp import MCPConfig
from ir_agent.memory import MemoryStore
from ir_agent.plugins import PluginContext, PluginManager
from ir_agent.skills import SkillCatalog
from ir_agent.tools import ToolRegistry
from ir_agent.tui import _build_parser


class FinalTextTransport:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return {
            "id": "resp_final",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "完成"}],
                }
            ],
            "output_text": "完成",
        }


class ApprovalTransport:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "id": "resp_approval",
                "output": [
                    {
                        "type": "mcp_approval_request",
                        "id": "approval_1",
                        "server_label": "demo",
                        "name": "search",
                        "arguments": "{\"query\": \"IR\"}",
                    }
                ],
            }
        return {
            "id": "resp_done",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "已完成"}]}],
            "output_text": "已完成",
        }


class ExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        library_path = self.root / "scenario_library.json"
        shutil.copyfile(Path("data/scenario_library.json"), library_path)
        self.library = ScenarioLibrary(library_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_skill_catalog_and_auto_context(self) -> None:
        skills = SkillCatalog(Path("skills"))
        matches = skills.search("IR 场景 use case 匹配")

        self.assertEqual(matches[0].skill.name, "ir-scenario-analysis")
        self.assertIn("active_skills", skills.prompt_context("IR 场景匹配"))

        transport = FinalTextTransport()
        agent = IRScenarioAgent(
            transport,
            self.library,
            settings=Settings(api_key=None, model="fake"),
            skills=skills,
        )
        agent.run("帮我做 IR 场景匹配")
        tool_names = {item["name"] for item in transport.last_request["tools"]}
        self.assertIn("search_skills", tool_names)
        self.assertIn("ir-scenario-analysis", transport.last_request["instructions"])

    def test_memory_is_user_scoped_and_rejects_secrets(self) -> None:
        memory = MemoryStore(self.root / "memory.sqlite3")
        memory.save("alice", "用户偏好先给出场景匹配结果，再说明缺口", tags=["preference"])
        memory.save("bob", "用户偏好英文回答")

        alice_results = memory.search("alice", "场景匹配")
        bob_results = memory.search("alice", "英文")
        self.assertEqual(len(alice_results), 1)
        self.assertEqual(bob_results, [])
        with self.assertRaises(ValueError):
            memory.save("alice", "api_key=sk-123456789012345")

    def test_mcp_config_expands_env_and_builds_responses_tool(self) -> None:
        config_path = self.root / "mcp.json"
        config_path.write_text(
            json.dumps(
                {
                    "servers": [
                        {
                            "server_label": "demo",
                            "server_url": "https://example.com/mcp",
                            "authorization": "${TEST_MCP_TOKEN}",
                            "allowed_tools": ["search"],
                            "require_approval": "always",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        previous = os.environ.get("TEST_MCP_TOKEN")
        os.environ["TEST_MCP_TOKEN"] = "secret-value"
        try:
            config = MCPConfig.from_file(config_path)
        finally:
            if previous is None:
                os.environ.pop("TEST_MCP_TOKEN", None)
            else:
                os.environ["TEST_MCP_TOKEN"] = previous

        tool = config.responses_tools()[0]
        self.assertEqual(tool["type"], "mcp")
        self.assertEqual(tool["server_label"], "demo")
        self.assertEqual(tool["authorization"], "secret-value")
        self.assertEqual(tool["require_approval"], "always")

    def test_plugin_is_discovered_and_registers_tool(self) -> None:
        registry = ToolRegistry(self.library)
        settings = Settings(api_key=None, model="fake")
        report = PluginManager(Path("plugins")).load_into(
            registry,
            PluginContext(
                settings=settings,
                library=self.library,
                skills=SkillCatalog(Path("skills")),
                memory=None,
                user_id="alice",
            ),
        )

        self.assertEqual(report.loaded, ["example"])
        self.assertEqual(registry.execute("example_plugin_info", {}), {
            "ok": True,
            "plugin": "example",
            "message": "这是一个可被动态发现的本地插件工具。",
            "user_id": "alice",
        })

    def test_agent_handles_mcp_approval_callback(self) -> None:
        transport = ApprovalTransport()
        agent = IRScenarioAgent(
            transport,
            self.library,
            settings=Settings(api_key=None, model="fake"),
            mcp_config=MCPConfig(
                servers=[
                    {
                        "server_label": "demo",
                        "server_url": "https://example.com/mcp",
                        "require_approval": "always",
                    }
                ]
            ),
            mcp_approval_callback=lambda request: request["name"] == "search",
        )

        result = agent.run("调用 MCP 搜索")

        self.assertEqual(result.output_text, "已完成")
        self.assertTrue(
            any(item.get("type") == "mcp_approval_response" for item in transport.calls[1]["input"])
        )

    def test_tui_parser_supports_provider_and_document_options(self) -> None:
        args = _build_parser().parse_args(
            [
                "--api-mode",
                "chat_completions",
                "--uc-library",
                "library\\uc\\use_cases.json",
                "--input-file",
                "request.md",
                "--auto-approve-writes",
            ]
        )

        self.assertEqual(args.api_mode, "chat_completions")
        self.assertEqual(args.uc_library, "library\\uc\\use_cases.json")
        self.assertEqual(args.input_file, "request.md")
        self.assertTrue(args.auto_approve_writes)

    def test_api_parser_supports_storage_and_token_options(self) -> None:
        args = _build_api_parser().parse_args(
            [
                "--library",
                "library.sqlite3",
                "--api-token",
                "secret",
                "--auto-approve-writes",
            ]
        )

        self.assertEqual(args.library, "library.sqlite3")
        self.assertEqual(args.api_token, "secret")
        self.assertTrue(args.auto_approve_writes)


if __name__ == "__main__":
    unittest.main()
