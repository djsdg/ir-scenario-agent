from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import (
    AgentRunError,
    AgentSession,
    IRScenarioAgent,
    OpenAIChatCompletionsTransport,
    OpenAIResponsesTransport,
    RetryingResponsesTransport,
    SessionStore,
)
from .audit import AuditLogger
from .config import Settings
from .documents import read_document
from .library import ScenarioLibrary
from .mcp import MCPConfig
from .memory import MemoryStore
from .plugins import PluginContext, PluginLoadReport, PluginManager
from .skills import SkillCatalog
from .specs import SpecCatalog, SpecError
from .tools import ToolRegistry


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR scenario-library Textual TUI")
    parser.add_argument("--message", help="启动后自动发送一条消息")
    parser.add_argument(
        "--ir-path",
        "--input-file",
        dest="input_file",
        help="启动后读取并发送 IR/SC/UC 文档（.txt/.md/.json/.docx/.pdf）",
    )
    parser.add_argument("--session-id", default="default", help="本地会话 id")
    parser.add_argument("--library", help="场景库 JSON 文件或目录路径")
    parser.add_argument("--uc-library", help="独立 UC 库 JSON 路径")
    parser.add_argument("--spec", help="IR→SC→UC 业务 Spec JSON 路径")
    parser.add_argument("--model", help="覆盖 IR_AGENT_MODEL / OPENAI_MODEL")
    parser.add_argument(
        "--api-mode",
        choices=["responses", "chat_completions"],
        help="API 协议：Responses 或 OpenAI 兼容 Chat Completions",
    )
    parser.add_argument("--user-id", help="长期记忆的用户隔离 id")
    parser.add_argument("--skills-dir", help="Skill 目录")
    parser.add_argument("--plugins-dir", help="插件目录")
    parser.add_argument("--mcp-config", help="MCP 配置 JSON 路径")
    parser.add_argument("--audit-path", help="JSONL 审计日志路径")
    parser.add_argument("--output-dir", help="TUI 结果 JSON 输出目录")
    parser.add_argument("--no-memory", action="store_true", help="关闭 SQLite 长期记忆")
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="关闭严格 JSON 最终输出",
    )
    parser.add_argument(
        "--auto-approve-writes",
        action="store_true",
        help="自动批准场景库和记忆写入工具",
    )
    parser.add_argument("--no-session-save", action="store_true", help="不加载/保存本地会话")
    return parser


@dataclass(slots=True)
class _Runtime:
    agent: IRScenarioAgent
    settings: Settings
    session: AgentSession
    session_store: SessionStore | None
    plugin_report: PluginLoadReport


class _ApprovalGate:
    """Bridge a synchronous agent callback to a Textual approval screen."""

    def __init__(self, app: Any, *, title: str, auto_approve: bool = False):
        self.app = app
        self.title = title
        self.auto_approve = auto_approve
        self._lock = threading.Lock()
        self._pending: tuple[threading.Event, dict[str, object], bool | None] | None = None

    def __call__(self, request: dict[str, object]) -> bool:
        if self.auto_approve:
            return True

        event = threading.Event()
        pending = (event, request, None)
        with self._lock:
            if self._pending is not None:
                return False
            self._pending = pending

        try:
            self.app.call_from_thread(self.app.open_approval, self.title, request, self)
        except Exception:
            self._clear_pending(event)
            return False

        if not event.wait(timeout=300):
            self._clear_pending(event)
            return False
        with self._lock:
            current = self._pending
            if current is not None and current[0] is event:
                result = current[2]
                self._pending = None
                return bool(result)
        return False

    def resolve(self, event: threading.Event, approved: bool) -> None:
        with self._lock:
            current = self._pending
            if current is None or current[0] is not event:
                return
            self._pending = (current[0], current[1], approved)
            event.set()

    def _clear_pending(self, event: threading.Event) -> None:
        with self._lock:
            if self._pending is not None and self._pending[0] is event:
                self._pending = None


try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        DataTable,
        Label,
        RichLog,
        Static,
        TabbedContent,
        TabPane,
        TextArea,
    )
except ImportError as exc:  # Optional dependency: keep the CLI usable without Textual.
    _TEXTUAL_IMPORT_ERROR: ImportError | None = exc
else:
    _TEXTUAL_IMPORT_ERROR = None


if _TEXTUAL_IMPORT_ERROR is None:

    def _format_approval_request(request: dict[str, object]) -> str:
        """Render an approval request as a readable write preview."""

        tool_name = str(request.get("tool_name") or request.get("name") or "未知工具")
        lines = [f"操作：{tool_name}"]
        if request.get("session_id"):
            lines.append(f"会话：{request['session_id']}")
        if request.get("user_id"):
            lines.append(f"用户：{request['user_id']}")
        arguments = request.get("arguments")
        lines.append("写入内容预览：")
        if isinstance(arguments, dict) and arguments:
            for key, value in arguments.items():
                rendered = (
                    json.dumps(value, ensure_ascii=False, indent=2, default=str)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
                lines.append(f"- {key}: {rendered}")
        else:
            lines.append("- 无参数")
        return "\n".join(lines)

    class _ApprovalScreen(ModalScreen):
        BINDINGS = [("escape", "deny", "拒绝")]

        DEFAULT_CSS = """
        _ApprovalScreen {
            align: center middle;
        }
        #approval-dialog {
            width: 80%;
            max-width: 100;
            height: auto;
            max-height: 80%;
            border: round $accent;
            background: $surface;
            padding: 1 2;
        }
        #approval-details {
            height: auto;
            max-height: 16;
            margin: 1 0;
            overflow-y: auto;
        }
        #approval-buttons {
            height: 3;
            align: right middle;
        }
        #approval-buttons Button {
            margin-left: 1;
        }
        """

        def __init__(self, title: str, request: dict[str, object], event: threading.Event):
            super().__init__()
            self.title_text = title
            self.request = request
            self.event = event

        def compose(self) -> ComposeResult:
            details = _format_approval_request(self.request)
            with Container(id="approval-dialog"):
                yield Label(f"{self.title_text} 请求授权", id="approval-title")
                yield Static(details, id="approval-details")
                with Horizontal(id="approval-buttons"):
                    yield Button("允许", id="approve", variant="success")
                    yield Button("拒绝", id="deny", variant="error")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "approve")

        def action_deny(self) -> None:
            self.dismiss(False)


    class IRScenarioTUI(App):
        """Textual front end for the same provider-neutral IR agent runtime."""

        TITLE = "IR / SC / UC Agent"
        SUB_TITLE = "需求 → 场景匹配/新增 → Use Case"
        BINDINGS = [
            ("ctrl+enter", "send", "发送"),
            ("ctrl+l", "clear_chat", "清空对话"),
            ("ctrl+q", "quit", "退出"),
        ]

        CSS = """
        Screen {
            background: $background;
        }
        #body {
            height: 1fr;
        }
        #workbench {
            width: 4fr;
            height: 1fr;
        }
        #input-panel, #output-panel {
            height: 1fr;
            padding: 0 1;
        }
        #input-panel {
            width: 2fr;
        }
        #output-panel {
            width: 3fr;
        }
        .panel-title {
            height: 3;
            content-align: left middle;
            text-style: bold;
        }
        #conversation {
            height: 1fr;
            border: round $primary;
            padding: 0 1;
            scrollbar-size: 1 1;
        }
        #result-tabs {
            height: 1fr;
            border: round $primary;
        }
        #candidate-table {
            height: auto;
            min-height: 5;
            max-height: 12;
            margin: 0 1;
            scrollbar-size: 1 1;
        }
        .comparison-title {
            height: 1;
            margin: 0 1;
            text-style: bold;
        }
        #use-case-table {
            height: auto;
            min-height: 3;
            max-height: 8;
            margin: 0 1;
            scrollbar-size: 1 1;
        }
        #candidates, #tools {
            height: 1fr;
            padding: 0 1;
            scrollbar-size: 1 1;
        }
        #prompt {
            height: 1fr;
            border: round $accent;
        }
        #path-inputs {
            height: auto;
            margin-bottom: 1;
        }
        .path-label {
            height: 1;
            content-align: left middle;
            text-style: bold;
        }
        #path-inputs Input {
            height: 3;
            margin-bottom: 1;
            border: round $secondary;
        }
        #path-actions {
            height: 3;
            align: right middle;
        }
        #path-actions Button {
            margin-left: 1;
        }
        #input-meta {
            height: 4;
            margin-top: 1;
            border: round $secondary;
            padding: 1;
        }
        #input-actions {
            height: 3;
            align: right middle;
            margin-top: 1;
        }
        #input-actions Button {
            margin-left: 1;
        }
        #side {
            width: 1fr;
            min-width: 34;
            height: 1fr;
            padding: 0 1;
        }
        #status, #config, #paths, #result-summary {
            border: round $secondary;
            padding: 1;
        }
        #status {
            height: 5;
        }
        #config {
            height: auto;
            max-height: 12;
            margin-top: 1;
        }
        #paths {
            height: auto;
            max-height: 12;
            margin-top: 1;
        }
        #result-summary {
            height: auto;
            max-height: 9;
            margin-top: 1;
        }
        #result-actions {
            height: 3;
            align: right middle;
            margin-top: 1;
        }
        #result-actions Button {
            margin-left: 1;
        }
        """

        def __init__(
            self,
            runtime: _Runtime | None = None,
            *,
            initial_message: str | None = None,
            initial_source: str | None = None,
        ):
            super().__init__()
            self.runtime = runtime
            self.agent: IRScenarioAgent | None = None
            self.settings: Settings | None = None
            self.session: AgentSession | None = None
            self.session_store: SessionStore | None = None
            self.initial_message = initial_message
            self.initial_source = initial_source
            self._active_input: str | None = None
            self._active_input_source: str | None = None
            self._last_output_path: Path | None = None
            self._busy = False
            self._path_loading = False

            if runtime is not None:
                self.set_runtime(runtime)

        def set_runtime(self, runtime: _Runtime) -> None:
            self.runtime = runtime
            self.agent = runtime.agent
            self.settings = runtime.settings
            self.session = runtime.session
            self.session_store = runtime.session_store

        @property
        def conversation(self) -> RichLog:
            return self.query_one("#conversation", RichLog)

        @property
        def candidates_log(self) -> RichLog:
            return self.query_one("#candidates", RichLog)

        @property
        def candidate_table(self) -> DataTable:
            return self.query_one("#candidate-table", DataTable)

        @property
        def use_case_table(self) -> DataTable:
            return self.query_one("#use-case-table", DataTable)

        @property
        def tools_log(self) -> RichLog:
            return self.query_one("#tools", RichLog)

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="body"):
                with Horizontal(id="workbench"):
                    with Vertical(id="input-panel"):
                        yield Static("输入 IR / SC / UC", classes="panel-title")
                        with Vertical(id="path-inputs"):
                            yield Label("IR 文档路径", classes="path-label")
                            yield Input(
                                placeholder="可选：.txt / .md / .json / .docx / .pdf",
                                id="ir-path",
                            )
                            yield Label("场景库路径", classes="path-label")
                            yield Input(
                                placeholder="JSON 文件，或场景库目录（自动使用 uc/use_cases.json）",
                                id="library-path",
                            )
                            with Horizontal(id="path-actions"):
                                yield Button(
                                    "读取 IR 并发送",
                                    id="send-paths",
                                    variant="success",
                                )
                                yield Button("清空路径", id="clear-paths")
                        yield TextArea(
                            placeholder="粘贴 IR/SC/UC，或描述你的需求。Ctrl+Enter 发送。",
                            id="prompt",
                        )
                        yield Static("尚未提交输入。发送后原文会保留在这里。", id="input-meta")
                        with Horizontal(id="input-actions"):
                            yield Button("发送", id="send", variant="primary")
                            yield Button("清空输入", id="clear-input")
                            yield Button("退出", id="quit", variant="error")
                    with Vertical(id="output-panel"):
                        yield Static("Agent 输出", classes="panel-title")
                        with TabbedContent(initial="conversation-tab", id="result-tabs"):
                            with TabPane("对话", id="conversation-tab"):
                                yield RichLog(id="conversation", wrap=True, markup=False)
                            with TabPane("候选对比", id="candidates-tab"):
                                yield Static("SC 候选", classes="comparison-title")
                                yield DataTable(id="candidate-table", cursor_type="row")
                                yield Static("UC 候选（按父 SC 过滤）", classes="comparison-title")
                                yield DataTable(id="use-case-table", cursor_type="row")
                                yield RichLog(id="candidates", wrap=True, markup=False)
                            with TabPane("工具日志", id="tools-tab"):
                                yield RichLog(id="tools", wrap=True, markup=False)
                with Vertical(id="side"):
                    yield Static("状态：启动中", id="status")
                    yield Static("配置", id="config")
                    yield Static("路径", id="paths")
                    yield Static("暂无结果", id="result-summary")
                    with Horizontal(id="result-actions"):
                        yield Button("打开输出目录", id="open-output", disabled=True)
                        yield Button("打开结果文件", id="open-result", disabled=True)
            yield Footer()

        def on_mount(self) -> None:
            self._reset_candidate_table()
            if (
                self.runtime is None
                or self.agent is None
                or self.settings is None
                or self.session is None
            ):
                self._set_status("运行时未初始化")
                return
            mcp_enabled = bool(
                self.agent.mcp_config.servers and self.agent.transport.supports_mcp
            )
            uc_library_path = self.agent.library.use_case_path
            config_lines = [
                f"模型：{self.settings.model}",
                f"协议：{self.settings.api_mode}",
                f"会话：{self.session.id}",
                f"记忆：{'开启' if self.agent.memory is not None else '关闭'}",
                f"远程 MCP：{'开启' if mcp_enabled else '关闭'}",
                f"插件：{len(self.runtime.plugin_report.loaded)} 个",
            ]
            matching_rules = getattr(getattr(self.agent, "spec", None), "matching_rules", {})
            if isinstance(matching_rules, dict):
                config_lines.extend(
                    [
                        f"SC 复用线：{self._rule_number(matching_rules, 'scenario_reuse_threshold', 0.45):.2f}",
                        f"SC 强匹配线：{self._rule_number(matching_rules, 'scenario_strong_threshold', 0.70):.2f}",
                        f"UC 复用线：{self._rule_number(matching_rules, 'use_case_reuse_threshold', 0.45):.2f}",
                        f"歧义分差：{self._rule_number(matching_rules, 'ambiguity_margin', 0.08):.2f}",
                    ]
                )
            self.query_one("#config", Static).update("\n".join(config_lines))
            path_lines = [
                f"场景库：{self.agent.library.path.resolve()}",
                f"UC 库：{uc_library_path.resolve() if uc_library_path else '与场景库同文件'}",
                f"Spec：{self.settings.spec_path.resolve()}",
                f"输出：{self.settings.outputs_dir.resolve()}",
                f"审计：{self.settings.audit_path.resolve()}",
            ]
            self.query_one("#paths", Static).update("\n".join(path_lines))
            library_input = self.query_one("#library-path", Input)
            library_input.value = str(self.settings.library_path.resolve())
            if self.initial_source:
                initial_path = Path(self.initial_source).expanduser()
                if initial_path.is_file():
                    self.query_one("#ir-path", Input).value = str(initial_path.resolve())
            initial_session_context = {
                "library_path": str(self.agent.library.path.resolve()),
                "uc_library_path": (
                    str(self.agent.library.use_case_path.resolve())
                    if self.agent.library.use_case_path is not None
                    else ""
                ),
                "spec_path": str(self.settings.spec_path.resolve()),
            }
            if self.session.bind_context(initial_session_context):
                self.tools_log.write("当前会话与已加载场景库不一致，旧上下文已清空。")
            self.conversation.write(
                "系统：TUI 已启动。可直接粘贴文本，或填写 IR 文档路径和场景库路径后读取。"
            )
            for error in self.runtime.plugin_report.errors:
                self.tools_log.write(f"插件未加载：{error}")
            self._set_status("就绪")
            prompt = self.query_one("#prompt", TextArea)
            if self.initial_message:
                prompt.text = self.initial_message
                self._set_input_meta(
                    f"输入来源：{self.initial_source or '启动参数'}\n"
                    f"长度：{len(self.initial_message)} 字符"
                )
                self._submit(self.initial_message, source="启动参数")
            prompt.focus()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            button_id = event.button.id
            if button_id == "send":
                self._submit()
            elif button_id == "send-paths":
                self._submit_from_paths()
            elif button_id == "clear-paths":
                self.query_one("#ir-path", Input).value = ""
                self.query_one("#library-path", Input).value = ""
                self._set_input_meta("路径输入已清空；当前运行仍使用已加载的场景库。")
            elif button_id == "clear-input":
                self.query_one("#prompt", TextArea).clear()
                self._set_input_meta("输入区已清空。")
            elif button_id == "open-output":
                self._open_result_path(
                    self._last_output_path.parent
                    if self._last_output_path is not None
                    else None
                )
            elif button_id == "open-result":
                self._open_result_path(self._last_output_path)
            elif button_id == "quit":
                self.exit()

        def action_send(self) -> None:
            self._submit()

        def action_clear_chat(self) -> None:
            self.conversation.clear()
            self.candidates_log.clear()
            self._reset_candidate_table()
            self.tools_log.clear()
            self.conversation.write("系统：对话显示已清空，会话上下文仍保留。")
            self.query_one("#result-summary", Static).update("暂无结果")
            self._last_output_path = None
            self.query_one("#open-output", Button).disabled = True
            self.query_one("#open-result", Button).disabled = True

        def open_approval(
            self,
            title: str,
            request: dict[str, object],
            gate: _ApprovalGate,
        ) -> None:
            event = threading.Event()
            # The callback owns the event it waits on. Find it from the gate's
            # pending request so the screen result can resolve that exact wait.
            with gate._lock:
                if gate._pending is None:
                    return
                event = gate._pending[0]
            self.push_screen(
                _ApprovalScreen(title, request, event),
                lambda approved: gate.resolve(event, bool(approved)),
            )

        @work(thread=True, exclusive=True)
        def _run_agent(self, user_text: str) -> None:
            if self.agent is None or self.session is None:
                self.call_from_thread(self._finish_error, "Agent 运行时未初始化")
                return
            try:
                result = self.agent.run(user_text, session=self.session)
            except AgentRunError as exc:
                self.call_from_thread(self._finish_error, f"Agent 执行未完成：{exc}")
            except Exception as exc:  # UI boundary: keep the event loop alive.
                self.call_from_thread(self._finish_error, f"请求失败：{exc}")
            else:
                self.call_from_thread(self._finish_result, result)

        def _submit_from_paths(self) -> None:
            if self._busy or self._path_loading:
                self.notify("上一条请求还在处理中，请稍候。", severity="warning")
                return

            ir_value = self.query_one("#ir-path", Input).value.strip()
            library_value = self.query_one("#library-path", Input).value.strip()
            if not ir_value:
                self.notify("请填写 IR 文档路径。", severity="warning")
                return
            if not library_value:
                self.notify("请填写场景库路径。", severity="warning")
                return

            self._path_loading = True
            self._set_submit_buttons(True)
            self._set_status("读取 IR 和场景库…")
            self._load_paths(ir_value, library_value)

        @work(thread=True, exclusive=True)
        def _load_paths(self, ir_value: str, library_value: str) -> None:
            try:
                ir_path = self._existing_path(ir_value, expect_file=True)
                library_path = self._existing_path(library_value)
                ir_text = read_document(ir_path).strip()
                if not ir_text:
                    raise ValueError("IR 文档为空")
                library = ScenarioLibrary(library_path)
                library.document()
            except Exception as exc:  # User-provided document/library boundary.
                self.call_from_thread(self._finish_path_load_error, f"路径加载失败：{exc}")
                return

            self.call_from_thread(
                self._finish_path_load,
                ir_text,
                ir_path,
                library_path,
                library,
            )

        def _finish_path_load_error(self, message: str) -> None:
            self._path_loading = False
            self._set_submit_buttons(False)
            self.tools_log.write(message)
            self.conversation.write(message)
            self._set_status("路径输入有误")

        def _finish_path_load(
            self,
            ir_text: str,
            ir_path: Path,
            library_path: Path,
            library: ScenarioLibrary,
        ) -> None:
            try:
                self._reload_library(library_path, library=library)
            except Exception as exc:  # UI boundary: keep the event loop alive.
                self._finish_path_load_error(f"场景库切换失败：{exc}")
                return

            self._path_loading = False
            self.query_one("#prompt", TextArea).text = ir_text
            self._submit(
                ir_text,
                source="IR 文件 + 场景库路径",
                input_source=f"IR 文件：{ir_path}\n场景库：{library_path}",
            )

        @staticmethod
        def _existing_path(raw_value: str, *, expect_file: bool = False) -> Path:
            cleaned = raw_value.strip().strip('"').strip("'")
            path = Path(cleaned).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"路径不存在：{path}")
            if expect_file and not path.is_file():
                raise ValueError(f"IR 路径不是文件：{path}")
            if not expect_file and not (path.is_file() or path.is_dir()):
                raise ValueError(f"场景库路径不是文件或目录：{path}")
            return path

        def _reload_library(
            self,
            path: Path,
            *,
            library: ScenarioLibrary | None = None,
        ) -> None:
            if self.agent is None or self.settings is None or self.runtime is None:
                raise RuntimeError("Agent 运行时未初始化")

            library = library or ScenarioLibrary(path)
            # Validate both the scenario file and, for directory-based
            # libraries, the derived UC file before switching the agent.
            library.document()
            if isinstance(self.agent, IRScenarioAgent):
                self.agent.library = library
                self.agent.tools = ToolRegistry(
                    library,
                    skills=self.agent.skills,
                    memory=self.agent.memory,
                    spec=self.agent.spec,
                    user_id=self.agent.user_id,
                )
                settings = replace(
                    self.settings,
                    library_path=path,
                    uc_library_path=library.use_case_path,
                )
                self.settings = settings
                self.runtime.settings = settings
                self.agent.settings = settings
                self.runtime.plugin_report = PluginManager(settings.plugins_dir).load_into(
                    self.agent.tools,
                    PluginContext(
                        settings=settings,
                        library=library,
                        skills=self.agent.skills,
                        memory=self.agent.memory,
                        spec=self.agent.spec,
                        user_id=self.agent.user_id,
                    ),
                )
            else:
                # Keep lightweight test/dummy agents usable without requiring
                # them to expose the complete production runtime surface.
                self.agent.library = library
                self.settings = replace(
                    self.settings,
                    library_path=path,
                    uc_library_path=library.use_case_path,
                )
                self.runtime.settings = self.settings

            session_context = {
                "library_path": str(library.path.resolve()),
                "uc_library_path": (
                    str(library.use_case_path.resolve())
                    if library.use_case_path is not None
                    else ""
                ),
                "spec_path": str(self.settings.spec_path.resolve()),
            }
            if self.session is not None and self.session.bind_context(session_context):
                self.tools_log.write("场景库已切换，旧会话上下文已清空。")
            self.runtime.session = self.session

            uc_library_path = library.use_case_path
            self.query_one("#paths", Static).update(
                "\n".join(
                    [
                        f"场景库：{library.path.resolve()}",
                        f"UC 库：{uc_library_path.resolve() if uc_library_path else '与场景库同文件'}",
                        f"Spec：{self.settings.spec_path.resolve()}",
                        f"输出：{self.settings.outputs_dir.resolve()}",
                        f"审计：{self.settings.audit_path.resolve()}",
                    ]
                )
            )
            self.query_one("#library-path", Input).value = str(path)
            self.tools_log.write(f"已加载场景库：{library.path.resolve()}")
            if uc_library_path is not None:
                self.tools_log.write(f"已加载 UC 库：{uc_library_path.resolve()}")
            for error in self.runtime.plugin_report.errors:
                self.tools_log.write(f"插件未加载：{error}")

        def _submit(
            self,
            message: str | None = None,
            *,
            source: str = "You",
            input_source: str | None = None,
        ) -> None:
            if self._busy or self._path_loading:
                self.notify("上一条请求还在处理中，请稍候。", severity="warning")
                return
            prompt = self.query_one("#prompt", TextArea)
            user_text = (message if message is not None else prompt.text).strip()
            if not user_text:
                self.notify("请输入内容。", severity="warning")
                return
            input_source = input_source or (
                self.initial_source
                if message is not None and self.initial_source
                else "TUI 输入框"
            )
            self._set_input_meta(
                f"最近提交：{source}\n"
                f"来源：{input_source}\n"
                f"长度：{len(user_text)} 字符（原文保留在输入区）"
            )
            self._active_input = user_text
            self._active_input_source = input_source
            self.conversation.write(f"输入已提交：{source}（{len(user_text)} 字符）")
            self._busy = True
            self._set_submit_buttons(True)
            self._set_status("处理中…")
            self._run_agent(user_text)

        def _finish_result(self, result: Any) -> None:
            self._busy = False
            self._set_submit_buttons(False)
            if self.session_store is not None:
                try:
                    self.session_store.save(self.session)
                except Exception as exc:
                    self.tools_log.write(f"会话保存失败：{exc}")
            output_path = self._save_result(result)
            self._last_output_path = output_path
            self.query_one("#open-output", Button).disabled = output_path is None
            self.query_one("#open-result", Button).disabled = output_path is None
            self._write_result(result, output_path)
            self._set_status("就绪")

        def _finish_error(self, message: str) -> None:
            self._busy = False
            self._set_submit_buttons(False)
            self.conversation.write(f"错误：{message}")
            self._set_status("发生错误，可继续输入")

        def _write_result(self, result: Any, output_path: Path | None) -> None:
            resolution = result.resolution
            self.candidates_log.clear()
            self._reset_candidate_table()
            if resolution is None:
                self.query_one("#result-summary", Static).update(
                    "本轮返回非结构化文本\n"
                    + (f"输出：{output_path}" if output_path else "输出文件保存失败")
                )
                self.candidates_log.write("本轮没有结构化 SC/UC 候选。")
                self.conversation.write(f"Agent：\n{result.output_text}")
            else:
                lines = [
                    f"状态：{resolution.status}",
                    f"决策：{resolution.decision}",
                    f"需求理解：{resolution.request_summary}",
                ]
                if resolution.candidates:
                    candidates = "、".join(
                        f"{item.scenario_id}({item.score:.2f})"
                        for item in resolution.candidates
                    )
                    lines.append(f"候选场景：{candidates}")
                if resolution.selected_scenario_ids:
                    lines.append(
                        f"选中场景：{', '.join(resolution.selected_scenario_ids)}"
                    )
                if resolution.use_case_ids:
                    lines.append(f"Use case：{', '.join(resolution.use_case_ids)}")
                if resolution.created_scenario_id:
                    lines.append(f"新建场景：{resolution.created_scenario_id}")
                if resolution.created_use_case_ids:
                    lines.append(
                        f"新建 Use case：{', '.join(resolution.created_use_case_ids)}"
                    )
                lines.append(f"匹配分数（非概率）：{resolution.confidence:.2f}")
                if resolution.missing_required_fields:
                    lines.append(
                        "缺少必填字段：" + ", ".join(resolution.missing_required_fields)
                    )
                if resolution.gaps:
                    lines.append("缺口：" + "；".join(resolution.gaps))
                if resolution.next_steps:
                    lines.append("下一步：" + "；".join(resolution.next_steps))
                self.conversation.write("Agent：\n" + "\n".join(lines))

                summary_lines = [
                    f"状态：{resolution.status}",
                    f"决策：{resolution.decision}",
                    f"匹配分数（非概率）：{resolution.confidence:.2f}",
                ]
                ambiguous = bool(getattr(resolution, "ambiguous", False))
                score_margin = float(getattr(resolution, "score_margin", 0.0) or 0.0)
                for call in result.tool_calls:
                    if call.name != "match_ir_requirement":
                        continue
                    match_payload = call.result.get("match")
                    if isinstance(match_payload, dict):
                        ambiguous = bool(match_payload.get("ambiguous", ambiguous))
                        score_margin = float(match_payload.get("score_margin", score_margin) or 0.0)
                if ambiguous:
                    summary_lines.append(f"候选分差：{score_margin:.2f}（需确认）")
                if resolution.selected_scenario_ids:
                    summary_lines.append(
                        "场景：" + ", ".join(resolution.selected_scenario_ids)
                    )
                if resolution.use_case_ids:
                    summary_lines.append("UC：" + ", ".join(resolution.use_case_ids))
                if output_path:
                    summary_lines.append(f"输出：{output_path}")
                self.query_one("#result-summary", Static).update("\n".join(summary_lines))

                if not resolution.candidates:
                    self.candidates_log.write("模型没有返回结构化候选场景。")
                candidate_details = self._match_candidate_details(result)
                for index, candidate in enumerate(resolution.candidates, start=1):
                    details = candidate_details.get(candidate.scenario_id, {})
                    scenario_name = self._scenario_name(candidate.scenario_id)
                    conflicts = details.get("conflicts", [])
                    self.candidate_table.add_row(
                        str(index),
                        candidate.scenario_id,
                        self._clip(scenario_name, 34),
                        f"{candidate.score:.2f}",
                        self._clip("、".join(candidate.matched_dimensions) or "-"),
                        self._clip("；".join(candidate.gaps) or "-"),
                        self._clip("；".join(conflicts) or "-"),
                    )
                    self.candidates_log.write(
                        f"{index}. {candidate.scenario_id} | 分数 {candidate.score:.2f}"
                    )
                    if candidate.matched_dimensions:
                        self.candidates_log.write(
                            "   命中：" + "、".join(candidate.matched_dimensions)
                        )
                    if candidate.gaps:
                        self.candidates_log.write("   缺口：" + "；".join(candidate.gaps))
                    if conflicts:
                        self.candidates_log.write("   冲突：" + "；".join(conflicts))
                    self.candidates_log.write(f"   原因：{candidate.reason}")
                self._write_use_case_candidates(result)

            if output_path is not None:
                self.conversation.write(f"输出文件：{output_path}")
                self.tools_log.write(f"输出文件：{output_path}")

            if not result.tool_calls:
                self.tools_log.write("本轮未调用工具。")
                return
            self.tools_log.write(f"本轮工具调用：{len(result.tool_calls)}")
            for call in result.tool_calls:
                ok = bool(call.result.get("ok"))
                approval = ""
                if call.approved is not None:
                    approval = f", approved={call.approved}"
                self.tools_log.write(
                    f"{'✓' if ok else '✗'} {call.name} ({call.duration_ms or 0:.0f}ms{approval})"
                )

        def _reset_candidate_table(self) -> None:
            table = self.candidate_table
            table.clear(columns=True)
            table.add_columns("序号", "SC", "场景名称", "分数", "命中维度", "缺口", "冲突")
            use_case_table = self.use_case_table
            use_case_table.clear(columns=True)
            use_case_table.add_columns("序号", "UC", "用例名称", "分数", "父 SC", "命中词")

        def _write_use_case_candidates(self, result: Any) -> None:
            raw_matches: list[dict[str, Any]] = []
            for call in result.tool_calls:
                payload: object = {}
                if call.name == "match_ir_requirement":
                    match_payload = call.result.get("match")
                    if isinstance(match_payload, dict):
                        payload = match_payload.get("use_case_matches", [])
                elif call.name == "match_use_case":
                    payload = call.result.get("matches", [])
                if isinstance(payload, list):
                    raw_matches.extend(
                        item for item in payload if isinstance(item, dict)
                    )

            seen: set[str] = set()
            rows: list[tuple[str, str, str, str, str]] = []
            for item in raw_matches:
                use_case = item.get("use_case")
                if not isinstance(use_case, dict) or not use_case.get("id"):
                    continue
                use_case_id = str(use_case["id"])
                if use_case_id in seen:
                    continue
                seen.add(use_case_id)
                parent_id = str(use_case.get("scenario_id") or "-")
                matched_terms = item.get("matched_terms", [])
                rows.append(
                    (
                        use_case_id,
                        str(use_case.get("name") or use_case_id),
                        f"{float(item.get('score', 0.0) or 0.0):.2f}",
                        parent_id,
                        self._clip("、".join(str(value) for value in matched_terms) or "-"),
                    )
                )

            if not rows:
                self.candidates_log.write("本轮没有结构化 UC 候选。")
                return
            for index, (use_case_id, name, score, parent_id, matched_terms) in enumerate(
                rows, start=1
            ):
                self.use_case_table.add_row(
                    str(index),
                    use_case_id,
                    self._clip(name, 34),
                    score,
                    parent_id,
                    matched_terms,
                )

        def _match_candidate_details(self, result: Any) -> dict[str, dict[str, Any]]:
            details: dict[str, dict[str, Any]] = {}
            for call in result.tool_calls:
                if call.name != "match_ir_requirement":
                    continue
                payload = call.result.get("match")
                if not isinstance(payload, dict):
                    continue
                for item in payload.get("scenario_matches", []):
                    if not isinstance(item, dict):
                        continue
                    scenario = item.get("scenario")
                    if not isinstance(scenario, dict) or not scenario.get("id"):
                        continue
                    details[str(scenario["id"])] = {
                        "conflicts": [
                            str(value) for value in item.get("conflicts", [])
                        ],
                    }
            return details

        def _scenario_name(self, scenario_id: str) -> str:
            if self.agent is None:
                return scenario_id
            try:
                return self.agent.library.get_scenario(scenario_id).name
            except Exception:
                return scenario_id

        @staticmethod
        def _rule_number(rules: dict[str, object], key: str, default: float) -> float:
            try:
                value = float(rules.get(key, default))
            except (TypeError, ValueError):
                return default
            return value if 0.0 <= value <= 1.0 else default

        @staticmethod
        def _clip(value: str, limit: int = 42) -> str:
            return value if len(value) <= limit else value[: limit - 1] + "…"

        def _save_result(self, result: Any) -> Path | None:
            if self.settings is None or self.session is None:
                return None
            try:
                output_root = self.settings.outputs_dir / self.session.id
                output_root.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).astimezone().strftime(
                    "%Y%m%d_%H%M%S"
                )
                output_path = output_root / f"{timestamp}_{uuid4().hex[:8]}.json"
                payload = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "session_id": self.session.id,
                    "input": self._active_input,
                    "input_source": self._active_input_source,
                    "library_path": (
                        str(self.agent.library.path.resolve())
                        if self.agent is not None
                        else None
                    ),
                    "uc_library_path": (
                        str(self.agent.library.use_case_path.resolve())
                        if self.agent is not None
                        and self.agent.library.use_case_path is not None
                        else None
                    ),
                    "spec_path": str(self.settings.spec_path.resolve()),
                    "result": result.model_dump(mode="json"),
                }
                output_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return output_path.resolve()
            except Exception as exc:
                self.tools_log.write(f"结果文件保存失败：{exc}")
                return None

        def _set_input_meta(self, text: str) -> None:
            self.query_one("#input-meta", Static).update(text)

        def _open_result_path(self, path: Path | None) -> None:
            if path is None:
                self.notify("当前还没有可打开的结果。", severity="warning")
                return
            try:
                resolved = path.resolve()
                if not resolved.exists():
                    raise FileNotFoundError(resolved)
                startfile = getattr(os, "startfile", None)
                if callable(startfile):
                    startfile(str(resolved))
                else:
                    webbrowser.open(resolved.as_uri())
            except Exception as exc:
                self.tools_log.write(f"打开路径失败：{exc}")
                self.notify(f"打开路径失败：{exc}", severity="error")

        def _set_submit_buttons(self, disabled: bool) -> None:
            self.query_one("#send", Button).disabled = disabled
            self.query_one("#send-paths", Button).disabled = disabled

        def _set_status(self, text: str) -> None:
            self.query_one("#status", Static).update(f"状态：{text}")


def _build_runtime(args: argparse.Namespace, app: Any) -> _Runtime:
    settings = Settings.from_env(
        library_path=args.library,
        uc_library_path=args.uc_library,
        spec_path=args.spec,
        api_mode=args.api_mode,
        outputs_dir=args.output_dir,
        skills_dir=args.skills_dir,
        plugins_dir=args.plugins_dir,
        mcp_config_path=args.mcp_config,
        audit_path=args.audit_path,
        user_id=args.user_id,
    )
    if args.model:
        settings = replace(settings, model=args.model)
    if args.no_structured_output:
        settings = replace(settings, structured_output=False)

    library = ScenarioLibrary(
        settings.library_path,
        use_case_path=settings.uc_library_path,
    )
    try:
        spec = SpecCatalog.from_file(settings.spec_path)
    except SpecError:
        raise
    memory = None if args.no_memory else MemoryStore(settings.memory_path)
    mcp_config = MCPConfig.from_file(settings.mcp_config_path)
    transport_class = (
        OpenAIChatCompletionsTransport
        if settings.api_mode == "chat_completions"
        else OpenAIResponsesTransport
    )
    transport = RetryingResponsesTransport(
        transport_class(settings),
        max_retries=settings.max_retries,
        backoff=settings.retry_backoff,
    )
    tool_gate = _ApprovalGate(
        app,
        title="写入工具",
        auto_approve=args.auto_approve_writes,
    )
    mcp_gate = _ApprovalGate(app, title="MCP 工具")
    agent = IRScenarioAgent(
        transport,
        library,
        settings=settings,
        skills=SkillCatalog(settings.skills_dir),
        memory=memory,
        spec=spec,
        user_id=settings.user_id,
        mcp_config=mcp_config,
        mcp_approval_callback=mcp_gate,
        tool_approval_callback=tool_gate,
        audit_logger=AuditLogger(settings.audit_path),
    )
    plugin_report = PluginManager(settings.plugins_dir).load_into(
        agent.tools,
        PluginContext(
            settings=settings,
            library=library,
            skills=agent.skills,
            memory=memory,
            spec=spec,
            user_id=settings.user_id,
        ),
    )
    session_store = None if args.no_session_save else SessionStore(settings.sessions_dir)
    session = (
        AgentSession(id=args.session_id)
        if session_store is None
        else session_store.load(args.session_id)
    )
    return _Runtime(
        agent=agent,
        settings=settings,
        session=session,
        session_store=session_store,
        plugin_report=plugin_report,
    )


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = _build_parser().parse_args(argv)
    if _TEXTUAL_IMPORT_ERROR is not None:
        print(
            "TUI 需要 Textual 依赖。请运行：pip install -e \".[tui]\"",
            file=sys.stderr,
        )
        return 2
    if args.message and args.input_file:
        print("--message 和 --ir-path/--input-file 不能同时使用。", file=sys.stderr)
        return 2

    initial_message = args.message
    initial_source = "启动参数" if args.message else None
    if args.input_file:
        try:
            initial_message = read_document(Path(args.input_file))
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"输入文件读取失败：{exc}", file=sys.stderr)
            return 2
        initial_source = str(Path(args.input_file).resolve())

    try:
        settings = Settings.from_env(api_mode=args.api_mode)
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    if not settings.api_key:
        print(
            "未检测到 API key。请在 .env 中填入 IR_AGENT_API_KEY 或 OPENAI_API_KEY。",
            file=sys.stderr,
        )
        return 2

    try:
        app = IRScenarioTUI(
            initial_message=initial_message,
            initial_source=initial_source,
        )
        runtime = _build_runtime(args, app=app)
        app.set_runtime(runtime)
    except SpecError as exc:
        print(f"业务 Spec 读取失败：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"TUI 启动失败：{exc}", file=sys.stderr)
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
