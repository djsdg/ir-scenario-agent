from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ir_agent.agent import AgentSession
from ir_agent.config import Settings
from ir_agent.domain import AgentResult
from ir_agent import tui


@unittest.skipIf(tui._TEXTUAL_IMPORT_ERROR is not None, "Textual is not installed")
class TUITests(unittest.TestCase):
    def test_input_output_panels_and_result_file(self) -> None:
        class FakeAgent:
            memory = None
            library = SimpleNamespace(
                path=Path("scenarios.json"),
                use_case_path=Path("uc/use_cases.json"),
            )
            mcp_config = SimpleNamespace(servers=[])
            transport = SimpleNamespace(supports_mcp=False)

            def run(self, text: str, *, session: AgentSession) -> AgentResult:
                return AgentResult(output_text=f"完成：{text}")

        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                runtime = SimpleNamespace(
                    agent=FakeAgent(),
                    settings=Settings(
                        api_key=None,
                        model="fake",
                        outputs_dir=Path(temp_dir),
                    ),
                    session=AgentSession(id="tui-test"),
                    session_store=None,
                    plugin_report=SimpleNamespace(loaded=[], errors=[]),
                )
                app = tui.IRScenarioTUI(
                    runtime,
                    initial_message="启动测试",
                    initial_source="test.md",
                )
                async with app.run_test(size=(140, 45)) as pilot:
                    for _ in range(20):
                        await pilot.pause(0.05)
                        if not app._busy:
                            break
                    self.assertFalse(app._busy)
                    self.assertIsNotNone(app.query_one("#input-panel"))
                    self.assertIsNotNone(app.query_one("#output-panel"))
                    self.assertIsNotNone(app.query_one("#candidate-table", tui.DataTable))
                    self.assertEqual(len(list(Path(temp_dir).rglob("*.json"))), 1)

        asyncio.run(exercise())

    def test_path_inputs_load_document_and_switch_library(self) -> None:
        class FakeAgent:
            memory = None
            mcp_config = SimpleNamespace(servers=[])
            transport = SimpleNamespace(supports_mcp=False)

            def __init__(self) -> None:
                self.library = SimpleNamespace(
                    path=Path("scenarios.json"),
                    use_case_path=Path("uc/use_cases.json"),
                )
                self.last_input: str | None = None

            def run(self, text: str, *, session: AgentSession) -> AgentResult:
                self.last_input = text
                return AgentResult(output_text=f"完成：{text}")

        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                ir_path = root / "ir.md"
                ir_path.write_text("# IR\n\n检测节点异常并隔离。", encoding="utf-8")
                library_path = root / "scenario_library.json"
                library_path.write_text(
                    json.dumps(
                        {
                            "version": 2,
                            "requirements": [],
                            "scenarios": [],
                            "use_cases": [],
                        }
                    ),
                    encoding="utf-8",
                )
                fake_agent = FakeAgent()
                runtime = SimpleNamespace(
                    agent=fake_agent,
                    settings=Settings(
                        api_key=None,
                        model="fake",
                        outputs_dir=root / "outputs",
                    ),
                    session=AgentSession(id="path-test"),
                    session_store=None,
                    plugin_report=SimpleNamespace(loaded=[], errors=[]),
                )
                app = tui.IRScenarioTUI(runtime)
                async with app.run_test(size=(140, 55)) as pilot:
                    await pilot.pause(0.1)
                    app.query_one("#ir-path", tui.Input).value = str(ir_path)
                    app.query_one("#library-path", tui.Input).value = str(library_path)
                    app._submit_from_paths()
                    for _ in range(20):
                        await pilot.pause(0.05)
                        if not app._busy:
                            break

                    self.assertFalse(app._busy)
                    self.assertEqual(fake_agent.last_input, ir_path.read_text(encoding="utf-8"))
                    self.assertEqual(fake_agent.library.path.resolve(), library_path.resolve())
                    self.assertEqual(
                        app.query_one("#prompt", tui.TextArea).text,
                        ir_path.read_text(encoding="utf-8").strip(),
                    )
                    self.assertEqual(len(list((root / "outputs").rglob("*.json"))), 1)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
