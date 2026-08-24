from __future__ import annotations

import asyncio
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
                    self.assertEqual(len(list(Path(temp_dir).rglob("*.json"))), 1)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
