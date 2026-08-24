from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ir_agent.agent import (
    AgentSession,
    IRScenarioAgent,
    OpenAIChatCompletionsTransport,
    RetryingResponsesTransport,
    SessionStore,
    _chat_messages_from_input,
)
from ir_agent.audit import AuditLogger
from ir_agent.config import Settings
from ir_agent.library import ScenarioLibrary


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "id": "resp_search",
                "output": [
                    {
                        "type": "function_call",
                        "name": "search_scenarios",
                        "arguments": json.dumps(
                            {"query": "企业知识库多轮检索问答", "top_k": 3, "min_score": 0.0},
                            ensure_ascii=False,
                        ),
                        "call_id": "call_search",
                    }
                ],
            }
        return {
            "id": "resp_final",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "已找到企业知识库多轮检索问答场景。"}],
                }
            ],
            "output_text": "已找到企业知识库多轮检索问答场景。",
        }


class StructuredFinalTransport:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        payload = {
            "status": "matched",
            "decision": "reuse_scenario_and_uc",
            "ir_id": None,
            "request_summary": "企业知识库多轮问答",
            "candidates": [
                {
                    "scenario_id": "scn_enterprise_knowledge_qa",
                    "score": 0.91,
                    "matched_terms": ["知识库", "问答"],
                    "matched_dimensions": ["目标/行为", "Actor"],
                    "gaps": [],
                    "reason": "覆盖多轮知识库问答",
                }
            ],
            "selected_scenario_ids": ["scn_enterprise_knowledge_qa"],
            "use_case_ids": ["uc_knowledge_retrieval_qa"],
            "created_scenario_id": None,
            "created_use_case_ids": [],
            "confidence": 0.91,
            "missing_required_fields": [],
            "gaps": [],
            "next_steps": ["确认评测指标"],
        }
        return {
            "id": "resp_structured",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(payload, ensure_ascii=False)}],
                }
            ],
            "output_text": json.dumps(payload, ensure_ascii=False),
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }


class CompactingTransport:
    def __init__(self) -> None:
        self.compact_calls = 0

    def compact(self, **kwargs):
        self.compact_calls += 1
        return {"output": [{"type": "compaction", "id": "cmp_1"}]}

    def create(self, **kwargs):
        return {
            "id": "resp_compacted",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "完成"}]}],
            "output_text": "完成",
        }


class FlakyTransport:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise TimeoutError("temporary")
        return {"id": "resp_ok"}


class FakeChatCompletions:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            id="chat_1",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content="需要先查询场景库",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="search_scenarios",
                                    arguments='{"query":"异常检测"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )


class FakeChatClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class WriteToolTransport:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "id": "resp_write",
                "output": [
                    {
                        "type": "function_call",
                        "name": "create_scenario",
                        "arguments": json.dumps(
                            {
                                "name": "审批测试场景",
                                "description": "这是一个需要人工批准的测试场景。",
                                "category": "测试场景",
                                "actor": "测试系统",
                                "influence_factors": [
                                    {
                                        "name": "测试因素",
                                        "kind": "environment",
                                        "dimension": "hardware_environment",
                                        "candidate_values": ["A"],
                                        "selected_values": ["A"],
                                    }
                                ],
                                "business_goal": "验证写工具审批",
                                "actions": ["执行测试"],
                                "constraints": ["仅用于测试"],
                                "lifecycle": "正常服务",
                                "ir_intent": "验证写工具审批",
                                "tags": ["test"],
                                "status": "draft",
                                "workflow_status": "Draft",
                                "owner": "test",
                                "affected_components": [],
                                "source_ir_ids": [],
                                "security_level": None,
                                "esn_id": None,
                                "topology_diagram": None,
                            },
                            ensure_ascii=False,
                        ),
                        "call_id": "call_write",
                    }
                ],
            }
        return {
            "id": "resp_write_done",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "完成"}]}],
            "output_text": "完成",
        }


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "scenario_library.json"
        shutil.copyfile(Path("data/scenario_library.json"), path)
        self.library = ScenarioLibrary(path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_agent_executes_tool_then_returns_final_text(self) -> None:
        transport = FakeTransport()
        settings = Settings(api_key=None, model="fake-model", max_tool_rounds=4)
        agent = IRScenarioAgent(transport, self.library, settings=settings)
        session = AgentSession(id="test")

        result = agent.run("帮我找一个企业知识库多轮问答场景", session=session)

        self.assertEqual(result.output_text, "已找到企业知识库多轮检索问答场景。")
        self.assertEqual(result.response_id, "resp_final")
        self.assertEqual(result.turns, 2)
        self.assertEqual([call.name for call in result.tool_calls], ["search_scenarios"])
        self.assertTrue(any(item.get("type") == "function_call_output" for item in transport.calls[1]["input"]))
        self.assertEqual(session.input_items[0]["role"], "user")

    def test_session_store_round_trip(self) -> None:
        store = SessionStore(Path(self.temp_dir.name) / "sessions")
        session = AgentSession(id="round-trip")
        session.bind_context({"library_path": "library.json", "spec_path": "spec.json"})
        session.add_user_message("hello")
        store.save(session)

        loaded = store.load("round-trip")
        self.assertEqual(loaded.id, "round-trip")
        self.assertEqual(loaded.input_items, [{"role": "user", "content": "hello"}])
        self.assertEqual(loaded.context["library_path"], "library.json")

    def test_session_context_switch_clears_old_library_history(self) -> None:
        session = AgentSession(id="context-switch")
        session.bind_context({"library_path": "old-library", "spec_path": "spec.json"})
        session.add_user_message("旧场景库中的需求")

        reset = session.bind_context(
            {"library_path": "new-library", "spec_path": "spec.json"}
        )

        self.assertTrue(reset)
        self.assertEqual(session.input_items, [])
        self.assertEqual(session.context["library_path"], "new-library")

    def test_agent_parses_structured_resolution_and_usage(self) -> None:
        transport = StructuredFinalTransport()
        agent = IRScenarioAgent(
            transport,
            self.library,
            settings=Settings(api_key=None, model="fake-model"),
        )

        result = agent.run("找一个企业知识库多轮问答场景")

        self.assertIsNotNone(result.resolution)
        self.assertEqual(result.resolution.status, "matched")
        self.assertEqual(result.resolution.selected_scenario_ids, ["scn_enterprise_knowledge_qa"])
        self.assertEqual(result.usage["total_tokens"], 30)
        self.assertEqual(transport.last_request["text"]["format"]["type"], "json_schema")
        self.assertNotIn("$defs", transport.last_request["text"]["format"]["schema"])

    def test_context_compaction_is_used_before_request(self) -> None:
        transport = CompactingTransport()
        settings = Settings(
            api_key=None,
            model="fake-model",
            max_session_items=4,
            max_context_chars=1_000,
        )
        agent = IRScenarioAgent(transport, self.library, settings=settings)
        session = AgentSession(id="compact")
        session.input_items = [{"role": "user", "content": "x" * 400}] * 5

        result = agent.run("继续", session=session)

        self.assertEqual(result.compactions, 1)
        self.assertEqual(transport.compact_calls, 1)

    def test_retrying_transport_retries_transient_errors(self) -> None:
        flaky = FlakyTransport()
        transport = RetryingResponsesTransport(flaky, max_retries=2, backoff=0, sleep=lambda _: None)

        result = transport.create(model="fake")

        self.assertEqual(result["id"], "resp_ok")
        self.assertEqual(flaky.calls, 3)
        self.assertEqual(transport.last_retry_count, 2)

    def test_chat_completions_adapter_normalizes_tools_and_reasoning(self) -> None:
        transport = object.__new__(OpenAIChatCompletionsTransport)
        transport._client = FakeChatClient()

        response = transport.create(
            model="deepseek-v4-pro",
            instructions="输出 JSON",
            input=[{"role": "user", "content": "匹配异常检测"}],
            tools=[
                {
                    "type": "function",
                    "name": "search_scenarios",
                    "description": "搜索场景",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            tool_choice="auto",
            text={"format": {"type": "json_schema", "schema": {"type": "object"}}},
        )

        self.assertEqual(response["output"][0]["type"], "function_call")
        self.assertEqual(response["output"][0]["name"], "search_scenarios")
        self.assertEqual(response["output"][0]["_reasoning_content"], "需要先查询场景库")
        sent = transport._client.chat.completions.last_kwargs
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertEqual(sent["tools"][0]["function"]["name"], "search_scenarios")

        messages = _chat_messages_from_input(
            [
                {"role": "user", "content": "匹配异常检测"},
                *response["output"],
                {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
            ],
            instructions="系统指令",
        )
        self.assertEqual(messages[-2]["reasoning_content"], "需要先查询场景库")
        self.assertEqual(messages[-1]["role"], "tool")

    def test_audit_logger_redacts_secret_like_values(self) -> None:
        path = Path(self.temp_dir.name) / "audit.jsonl"
        logger = AuditLogger(path)

        event_id = logger.record(
            "tool_call",
            user_id="alice",
            session_id="s1",
            payload={"arguments": {"text": "api_key=sk-secret-value"}},
        )

        self.assertTrue(event_id.startswith("audit_"))
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("sk-secret-value", content)
        self.assertIn("[REDACTED]", content)

    def test_write_tool_requires_application_approval(self) -> None:
        transport = WriteToolTransport()
        initial_count = len(self.library.list_scenarios())
        agent = IRScenarioAgent(
            transport,
            self.library,
            settings=Settings(api_key=None, model="fake-model"),
            tool_approval_callback=lambda request: False,
        )

        result = agent.run("新建一个测试场景")

        self.assertEqual(result.tool_calls[0].result["error"], "approval_denied")
        self.assertEqual(len(self.library.list_scenarios()), initial_count)


if __name__ == "__main__":
    unittest.main()
