from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
from .library import ScenarioLibrary, open_scenario_library
from .mcp import MCPConfig
from .memory import MemoryStore
from .plugins import PluginContext, PluginLoadReport, PluginManager
from .retrieval import OpenAIEmbeddingProvider
from .reporting import build_analysis_report, render_human_review_text, save_run_report
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

    class _CopyableTextArea(TextArea):
        """Read-only output area with native mouse selection and Ctrl+C support."""

        def __init__(self, text: str = "", **kwargs: Any):
            kwargs.setdefault("read_only", True)
            kwargs.setdefault("soft_wrap", True)
            kwargs.setdefault("show_line_numbers", False)
            super().__init__(text=text, **kwargs)

        def write(self, value: object) -> None:
            rendered = str(value)
            if not rendered:
                return
            self.text = f"{self.text}\n{rendered}" if self.text else rendered
            self.scroll_end(animate=False)

        def update(self, content: object = "") -> _CopyableTextArea:
            self.text = str(content)
            return self

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
                yield _CopyableTextArea(details, id="approval-details")
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
        #copy-hint {
            height: 1;
            color: $text-muted;
            padding: 0 1;
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
        #candidate-actions {
            height: 3;
            align: right middle;
            margin: 0 1;
        }
        #candidate-actions Button {
            margin-left: 1;
        }
        #selection-summary {
            height: auto;
            min-height: 2;
            max-height: 5;
            margin: 0 1;
            border: round $secondary;
            padding: 0 1;
        }
        #candidates, #tools {
            height: 1fr;
            padding: 0 1;
            scrollbar-size: 1 1;
        }
        #human-review {
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
            self._current_scenario_id: str | None = None
            self._current_use_case_id: str | None = None
            self._selected_scenario_ids: list[str] = []
            self._selected_use_case_ids: list[str] = []

            if runtime is not None:
                self.set_runtime(runtime)

        def set_runtime(self, runtime: _Runtime) -> None:
            self.runtime = runtime
            self.agent = runtime.agent
            self.settings = runtime.settings
            self.session = runtime.session
            self.session_store = runtime.session_store

        @property
        def conversation(self) -> _CopyableTextArea:
            return self.query_one("#conversation", _CopyableTextArea)

        @property
        def candidates_log(self) -> _CopyableTextArea:
            return self.query_one("#candidates", _CopyableTextArea)

        @property
        def candidate_table(self) -> DataTable:
            return self.query_one("#candidate-table", DataTable)

        @property
        def use_case_table(self) -> DataTable:
            return self.query_one("#use-case-table", DataTable)

        @property
        def tools_log(self) -> _CopyableTextArea:
            return self.query_one("#tools", _CopyableTextArea)

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
                        yield _CopyableTextArea(
                            "尚未提交输入。发送后原文会保留在这里。",
                            id="input-meta",
                        )
                        with Horizontal(id="input-actions"):
                            yield Button("发送", id="send", variant="primary")
                            yield Button("清空输入", id="clear-input")
                            yield Button("退出", id="quit", variant="error")
                    with Vertical(id="output-panel"):
                        yield Static("Agent 输出", classes="panel-title")
                        yield Static("输出区支持鼠标选中；Ctrl+C 复制，Ctrl+A 全选", id="copy-hint")
                        with TabbedContent(initial="conversation-tab", id="result-tabs"):
                            with TabPane("对话", id="conversation-tab"):
                                yield _CopyableTextArea(id="conversation")
                            with TabPane("候选对比", id="candidates-tab"):
                                yield Static("SC 候选", classes="comparison-title")
                                yield DataTable(id="candidate-table", cursor_type="row")
                                yield Static("UC 候选（按父 SC 过滤）", classes="comparison-title")
                                yield DataTable(id="use-case-table", cursor_type="row")
                                yield _CopyableTextArea(
                                    "尚未选择候选。点击表格行后加入选择。",
                                    id="selection-summary",
                                )
                                with Horizontal(id="candidate-actions"):
                                    yield Button("加入当前选择", id="add-selection", variant="primary")
                                    yield Button("评估当前 SC", id="evaluate-scenario")
                                    yield Button("填充确认/编辑提示", id="prepare-selection")
                                    yield Button("确认选择并发送", id="confirm-selection", variant="success")
                                    yield Button("检查库质量", id="validate-library")
                                    yield Button("清空选择", id="clear-selection")
                                yield _CopyableTextArea(id="candidates")
                            with TabPane("人工复核", id="review-tab"):
                                yield _CopyableTextArea(
                                    "本轮结果后显示候选总览和字段复核提示。完整可编辑内容请打开复核 CSV。",
                                    id="human-review",
                                )
                            with TabPane("工具日志", id="tools-tab"):
                                yield _CopyableTextArea(id="tools")
                with Vertical(id="side"):
                    yield Static("状态：启动中", id="status")
                    yield _CopyableTextArea("配置", id="config")
                    yield _CopyableTextArea("路径", id="paths")
                    yield _CopyableTextArea("暂无结果", id="result-summary")
                    with Horizontal(id="result-actions"):
                        yield Button("打开输出目录", id="open-output", disabled=True)
                        yield Button("打开结果文件", id="open-result", disabled=True)
                        yield Button("打开复核表", id="open-review", disabled=True)
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
                f"Embedding：{self.settings.embedding_model or '关闭（关键词 + TF-IDF）'}",
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
                dimension_weights = matching_rules.get("ir_dimension_weights")
                if isinstance(dimension_weights, dict):
                    config_lines.append(
                        "SC维度权重："
                        + "；".join(
                            f"{key} {self._rule_number(dimension_weights, key, 0.0):.2f}"
                            for key in ("目标/行为", "Actor", "上下文", "影响因素", "约束")
                        )
                    )
            self.query_one("#config", _CopyableTextArea).update("\n".join(config_lines))
            path_lines = [
                f"场景库：{self.agent.library.path.resolve()}",
                f"UC 库：{uc_library_path.resolve() if uc_library_path else '与场景库同文件'}",
                f"Spec：{self.settings.spec_path.resolve()}",
                f"输出：{self.settings.outputs_dir.resolve()}",
                f"审计：{self.settings.audit_path.resolve()}",
            ]
            self.query_one("#paths", _CopyableTextArea).update("\n".join(path_lines))
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
            elif button_id == "add-selection":
                self._add_current_selection()
            elif button_id == "evaluate-scenario":
                self._evaluate_current_scenario()
            elif button_id == "prepare-selection":
                self._prepare_selection_prompt(auto_submit=False)
            elif button_id == "confirm-selection":
                self._prepare_selection_prompt(auto_submit=True)
            elif button_id == "validate-library":
                self._validate_library_direct()
            elif button_id == "clear-selection":
                self._clear_selection()
            elif button_id == "open-output":
                self._open_result_path(
                    self._last_output_path.parent
                    if self._last_output_path is not None
                    else None
                )
            elif button_id == "open-result":
                self._open_result_path(self._last_output_path)
            elif button_id == "open-review":
                self._open_review_csv()
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
            self.query_one("#result-summary", _CopyableTextArea).update("暂无结果")
            self._last_output_path = None
            self.query_one("#open-output", Button).disabled = True
            self.query_one("#open-result", Button).disabled = True
            self.query_one("#open-review", Button).disabled = True
            self.query_one("#human-review", _CopyableTextArea).update(
                "本轮结果后显示候选总览和字段复核提示。完整可编辑内容请打开复核 CSV。"
            )
            self._clear_selection()

        def on_data_table_row_selected(self, event: Any) -> None:
            self._remember_table_row(event)

        def on_data_table_row_highlighted(self, event: Any) -> None:
            self._remember_table_row(event)

        def _remember_table_row(self, event: Any) -> None:
            row_key = getattr(event, "row_key", None)
            value = getattr(row_key, "value", row_key)
            if value is None:
                return
            identifier = str(value)
            table_id = getattr(getattr(event, "data_table", None), "id", None)
            if table_id == "candidate-table":
                self._current_scenario_id = identifier
            elif table_id == "use-case-table":
                self._current_use_case_id = identifier
            else:
                return
            self._refresh_selection_summary()

        def _add_current_selection(self) -> None:
            if self._current_scenario_id:
                if self._current_scenario_id not in self._selected_scenario_ids:
                    self._selected_scenario_ids.append(self._current_scenario_id)
                self.notify(f"已加入 SC：{self._current_scenario_id}", severity="information")
            elif self._current_use_case_id:
                if self._current_use_case_id not in self._selected_use_case_ids:
                    self._selected_use_case_ids.append(self._current_use_case_id)
                self.notify(f"已加入 UC：{self._current_use_case_id}", severity="information")
            else:
                self.notify("请先点击 SC 或 UC 表格中的一行。", severity="warning")
                return
            self._refresh_selection_summary()

        def _clear_selection(self) -> None:
            self._current_scenario_id = None
            self._current_use_case_id = None
            self._selected_scenario_ids.clear()
            self._selected_use_case_ids.clear()
            if self.is_mounted:
                self._refresh_selection_summary()

        def _refresh_selection_summary(self) -> None:
            if not self.is_mounted:
                return
            selected_sc = "、".join(self._selected_scenario_ids) or "无"
            selected_uc = "、".join(self._selected_use_case_ids) or "无"
            current = self._current_scenario_id or self._current_use_case_id or "无"
            self.query_one("#selection-summary", _CopyableTextArea).update(
                f"当前行：{current}\n已选 SC：{selected_sc}\n已选 UC：{selected_uc}"
            )

        def _prepare_selection_prompt(self, *, auto_submit: bool) -> None:
            if not self._selected_scenario_ids and not self._selected_use_case_ids:
                self.notify("请先加入至少一个 SC 或 UC。", severity="warning")
                return
            scenario_text = "、".join(self._selected_scenario_ids) or "无"
            use_case_text = "、".join(self._selected_use_case_ids) or "无"
            prompt = (
                "我已在 TUI 中人工选择以下候选，请以这些编号为准继续处理。\n"
                f"选定 SC：{scenario_text}\n"
                f"选定 UC：{use_case_text}\n"
                "请先读取并核对当前记录；如需补齐或修改，请先生成草稿/修改建议。"
                "只有我明确确认写入时才调用写入工具，并保留审批流程。"
            )
            self.query_one("#prompt", TextArea).text = prompt
            self._set_input_meta("已填充人工选择提示，可继续编辑后发送。")
            if auto_submit:
                self._submit(prompt, source="TUI 人工确认")

        def _evaluate_current_scenario(self) -> None:
            """Ask the agent to run the deterministic fit check for the highlighted SC."""
            if self._busy or self._path_loading:
                self.notify("上一条请求还在处理中，请稍候。", severity="warning")
                return
            if not self._current_scenario_id:
                self.notify("请先点击 SC 候选表格中的一行。", severity="warning")
                return
            prompt = (
                "请对当前选中的场景做只读符合度评估。\n"
                f"指定 SC 编号：{self._current_scenario_id}\n"
                "请从当前 IR/需求上下文中提取完整字段，必须调用 evaluate_scenario_fit，"
                "展示总分、每个维度分数、证据、低分原因、缺口和冲突；不要执行任何写入。"
            )
            self.query_one("#prompt", TextArea).text = prompt
            self._set_input_meta("已填充指定 SC 评估提示；评估是只读操作，不会更新场景库。")
            self._submit(prompt, source="TUI 指定 SC 符合度评估")

        def _validate_library_direct(self) -> None:
            if self._busy or self._path_loading:
                self.notify("上一条请求还在处理中，请稍候。", severity="warning")
                return
            if self.agent is None:
                self.notify("Agent 运行时未初始化。", severity="error")
                return
            self._busy = True
            self._set_submit_buttons(True)
            self._set_status("检查场景库…")
            self._run_library_validation()

        @work(thread=True, exclusive=True)
        def _run_library_validation(self) -> None:
            if self.agent is None:
                self.call_from_thread(self._finish_error, "Agent 运行时未初始化")
                return
            try:
                report = self.agent.tools.execute("validate_library", {})
            except Exception as exc:
                self.call_from_thread(self._finish_error, f"场景库检查失败：{exc}")
            else:
                self.call_from_thread(self._finish_library_validation, report)

        def _finish_library_validation(self, report: dict[str, Any]) -> None:
            self._busy = False
            self._set_submit_buttons(False)
            self._set_status("就绪")
            counts = report.get("counts", {}) if isinstance(report, dict) else {}
            issues = report.get("issues", []) if isinstance(report, dict) else []
            warnings = report.get("warnings", []) if isinstance(report, dict) else []
            ok = bool(report.get("ok")) if isinstance(report, dict) else False
            summary = [
                f"库质量：{'通过' if ok else '存在问题'}",
                f"IR：{counts.get('requirements', 0)}  SC：{counts.get('scenarios', 0)}  UC：{counts.get('use_cases', 0)}",
                f"问题：{counts.get('issues', len(issues))}  警告：{counts.get('warnings', len(warnings))}",
            ]
            self.query_one("#result-summary", _CopyableTextArea).update("\n".join(summary))
            self.conversation.write("库质量审计：\n" + "\n".join(summary))
            self.tools_log.write(
                f"✓ validate_library（问题 {len(issues)}，警告 {len(warnings)}）"
            )
            for item in [*issues, *warnings]:
                if isinstance(item, dict):
                    self.candidates_log.write(
                        f"{item.get('kind', 'unknown')} | {item.get('record_id', '-')} | {item.get('message', '')}"
                    )

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
                library = open_scenario_library(library_path)
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

            library = library or open_scenario_library(path)
            if self.settings.embedding_model:
                library.configure_embedding(
                    OpenAIEmbeddingProvider(
                        api_key=self.settings.api_key,
                        model=self.settings.embedding_model,
                        base_url=self.settings.base_url,
                        organization=self.settings.organization,
                        timeout=self.settings.request_timeout,
                    )
                )
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
            self.query_one("#paths", _CopyableTextArea).update(
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
            self.query_one("#open-review", Button).disabled = output_path is None
            self._write_result(result, output_path)
            self._set_status("就绪")

        def _finish_error(self, message: str) -> None:
            self._busy = False
            self._set_submit_buttons(False)
            self.conversation.write(f"错误：{message}")
            self._set_status("发生错误，可继续输入")

        def _write_result(self, result: Any, output_path: Path | None) -> None:
            resolution = result.resolution
            report = build_analysis_report(
                result,
                self.agent.library if self.agent is not None else None,
            )
            review_text = render_human_review_text(report)
            if output_path is not None:
                review_text += (
                    f"\n\n结果目录：{output_path.parent}"
                    f"\n可编辑明细：{output_path.parent / 'evaluation' / 'human_review_template.csv'}"
                    f"\n横向对比：{output_path.parent / 'evaluation' / 'human_review_matrix.csv'}"
                )
            self.query_one("#human-review", _CopyableTextArea).update(review_text)
            self.candidates_log.clear()
            self._reset_candidate_table()
            if resolution is None:
                self.query_one("#result-summary", _CopyableTextArea).update(
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
                matching = report.get("matching") or {}
                if matching:
                    lines.append(
                        f"匹配评价：{matching.get('confidence_label', '未评价')} "
                        f"（决策分 {float(matching.get('confidence', 0.0) or 0.0):.2f}；"
                        f"IR证据完整度 {float(matching.get('evidence_completeness', 0.0) or 0.0):.2f}）"
                    )
                    confidence_reasons = matching.get("confidence_reasons") or []
                    if confidence_reasons:
                        lines.append("置信度原因：" + "；".join(str(item) for item in confidence_reasons))
                    if matching.get("ambiguous"):
                        lines.append(
                            f"候选分差：{float(matching.get('score_margin', 0.0) or 0.0):.2f}（需人工确认）"
                        )
                scenario_evidence = report.get("scenarios", {}).get("matches", [])
                candidate_summary_rows = scenario_evidence or resolution.candidates
                if candidate_summary_rows:
                    candidates = "、".join(
                        (
                            f"{item.get('id')}({float(item.get('score', 0.0) or 0.0):.2f})"
                            if isinstance(item, dict)
                            else f"{item.scenario_id}({item.score:.2f})"
                        )
                        for item in candidate_summary_rows
                    )
                    lines.append(f"候选场景：{candidates}")
                if scenario_evidence:
                    lines.append("SC 匹配依据：")
                    for item in scenario_evidence:
                        lines.append(
                            f"- {item['id']}：{self._format_match_fields(item.get('matched_fields'))}"
                        )
                if resolution.selected_scenario_ids:
                    lines.append(
                        f"选中场景：{', '.join(resolution.selected_scenario_ids)}"
                    )
                if resolution.use_case_ids:
                    lines.append(f"Use case：{', '.join(resolution.use_case_ids)}")
                use_case_evidence = report.get("use_cases", {}).get("matches", [])
                if use_case_evidence:
                    lines.append("UC 匹配依据：")
                    for item in use_case_evidence:
                        parent = item.get("parent_scenario_id") or "父 SC 未解析"
                        lines.append(
                            f"- {item['id']}（父 SC：{parent}）："
                            f"{self._format_match_fields(item.get('matched_fields'))}"
                        )
                evaluations = report.get("evaluations", {}).get("scenario_fit", [])
                if evaluations:
                    lines.append("指定 SC 符合度评估：")
                    for evaluation in evaluations:
                        scenario = evaluation.get("scenario") or {}
                        lines.append(
                            f"- {evaluation.get('scenario_id', scenario.get('id', '-'))} "
                            f"{scenario.get('name', '')}："
                            f"{float(evaluation.get('score', 0.0) or 0.0):.2f} "
                            f"（{evaluation.get('evaluation', '未评价')}）"
                        )
                        reasons = evaluation.get("low_score_reasons") or []
                        if reasons:
                            lines.append("  低分原因：" + "；".join(str(item) for item in reasons))
                for relationship in report.get("use_cases", {}).get("by_scenario", []):
                    lines.append(
                        f"SC→UC：{relationship['scenario_id']} "
                        f"{relationship['scenario_name']} -> "
                        f"{', '.join(relationship.get('use_case_ids', [])) or '暂无 UC'}"
                    )
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
                if report.get("writes"):
                    lines.append(
                        "库更新："
                        + "、".join(
                            f"{item.get('action')} {item.get('id')}"
                            for item in report["writes"]
                        )
                    )
                self.conversation.write("Agent：\n" + "\n".join(lines))

                summary_lines = [
                    f"状态：{resolution.status}",
                    f"决策：{resolution.decision}",
                    f"匹配分数（非概率）：{resolution.confidence:.2f}",
                ]
                if matching:
                    summary_lines.append(
                        f"匹配评价：{matching.get('confidence_label', '未评价')}"
                    )
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
                for relationship in report.get("use_cases", {}).get("by_scenario", []):
                    summary_lines.append(
                        f"{relationship['scenario_id']} 子 UC："
                        f"{', '.join(relationship.get('use_case_ids', [])) or '暂无'}"
                    )
                if output_path:
                    summary_lines.append(f"输出：{output_path}")
                    summary_lines.append(f"CSV评估目录：{output_path.parent / 'evaluation'}")
                self.query_one("#result-summary", _CopyableTextArea).update("\n".join(summary_lines))

                candidate_details = self._match_candidate_details(result)
                resolution_candidates = {
                    candidate.scenario_id: candidate for candidate in resolution.candidates
                }
                candidate_rows: list[dict[str, Any]] = [
                    dict(item)
                    for item in scenario_evidence
                    if isinstance(item, dict) and item.get("id")
                ]
                if not candidate_rows:
                    candidate_rows = [
                        {
                            "id": candidate.scenario_id,
                            "score": candidate.score,
                            "matched_dimensions": candidate.matched_dimensions,
                            "gaps": candidate.gaps,
                            "reason": candidate.reason,
                        }
                        for candidate in resolution.candidates
                    ]
                if not candidate_rows:
                    self.candidates_log.write("本轮没有结构化候选场景。")
                for index, candidate_row in enumerate(candidate_rows, start=1):
                    scenario_id = str(candidate_row["id"])
                    fallback_candidate = resolution_candidates.get(scenario_id)
                    details = dict(candidate_details.get(scenario_id, {}))
                    details.update(candidate_row)
                    scenario_name = str(
                        details.get("name") or self._scenario_name(scenario_id)
                    )
                    score = float(details.get("score", 0.0) or 0.0)
                    matched_dimensions = details.get("matched_dimensions") or (
                        fallback_candidate.matched_dimensions if fallback_candidate else []
                    )
                    gaps = details.get("gaps") or (
                        fallback_candidate.gaps if fallback_candidate else []
                    )
                    conflicts = details.get("conflicts", [])
                    self.candidate_table.add_row(
                        str(index),
                        scenario_id,
                        self._clip(scenario_name, 34),
                        f"{score:.2f}",
                        f"{float(details.get('fit_score', 0.0) or 0.0):.2f}",
                        f"{float(details.get('evidence_completeness', 0.0) or 0.0):.2f}",
                        self._clip(str(details.get("evaluation") or "未评价"), 16),
                        self._clip("、".join(str(item) for item in matched_dimensions) or "-"),
                        self._clip("；".join(details.get("low_score_reasons", [])) or "-", 42),
                        self._clip("；".join(str(item) for item in gaps) or "-"),
                        self._clip("；".join(conflicts) or "-"),
                        key=scenario_id,
                    )
                    self.candidates_log.write(
                        f"{index}. {scenario_id} | 分数 {score:.2f}"
                    )
                    if details.get("evaluation"):
                        self.candidates_log.write(f"   评价：{details['evaluation']}")
                    fit_score = float(details.get("fit_score", 0.0) or 0.0)
                    evidence_completeness = float(
                        details.get("evidence_completeness", 0.0) or 0.0
                    )
                    self.candidates_log.write(
                        f"   评分：决策分 {score:.2f}；可用证据匹配度 {fit_score:.2f}；"
                        f"证据完整度 {evidence_completeness:.2f}"
                    )
                    for dimension_line in self._format_dimension_scores(
                        details.get("dimension_scores")
                    ):
                        self.candidates_log.write("   " + dimension_line)
                    if details.get("low_score_reasons"):
                        self.candidates_log.write(
                            "   低分原因：" + "；".join(details["low_score_reasons"])
                        )
                    if matched_dimensions:
                        self.candidates_log.write(
                            "   命中：" + "、".join(str(item) for item in matched_dimensions)
                        )
                    matched_fields = details.get("matched_fields", {})
                    if matched_fields:
                        self.candidates_log.write(
                            "   命中字段：" + self._format_match_fields(matched_fields)
                        )
                    if details.get("matched_terms"):
                        self.candidates_log.write(
                            "   命中词：" + "、".join(details["matched_terms"])
                        )
                    if gaps:
                        self.candidates_log.write(
                            "   缺口：" + "；".join(str(item) for item in gaps)
                        )
                    if conflicts:
                        self.candidates_log.write("   冲突：" + "；".join(conflicts))
                    reason = str(details.get("reason") or "工具返回的匹配候选")
                    self.candidates_log.write(f"   原因：{reason}")
                self._write_use_case_candidates(result)
                for evaluation in report.get("evaluations", {}).get("scenario_fit", []):
                    scenario = evaluation.get("scenario") or {}
                    scenario_id = evaluation.get("scenario_id") or scenario.get("id", "-")
                    self.candidates_log.write(
                        f"指定 SC 评估 {scenario_id} | "
                        f"分数 {float(evaluation.get('score', 0.0) or 0.0):.2f} | "
                        f"评价 {evaluation.get('evaluation', '未评价')}"
                    )
                    for dimension_line in self._format_dimension_scores(
                        evaluation.get("dimension_scores")
                    ):
                        self.candidates_log.write("   " + dimension_line)
                    reasons = evaluation.get("low_score_reasons") or []
                    if reasons:
                        self.candidates_log.write(
                            "   原因：" + "；".join(str(item) for item in reasons)
                        )

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
            table.add_columns(
                "序号",
                "SC",
                "场景名称",
                "分数",
                "可用证据",
                "证据完整",
                "评价",
                "命中维度",
                "低分原因",
                "缺口",
                "冲突",
            )
            use_case_table = self.use_case_table
            use_case_table.clear(columns=True)
            use_case_table.add_columns(
                "序号",
                "UC",
                "用例名称",
                "分数",
                "可用证据",
                "证据完整",
                "评价",
                "父 SC",
                "低分原因",
                "命中词",
            )

        def _write_use_case_candidates(self, result: Any) -> None:
            report = build_analysis_report(
                result,
                self.agent.library if self.agent is not None else None,
            )
            raw_matches = report.get("use_cases", {}).get("matches", [])
            seen: set[str] = set()
            rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
            for item in raw_matches:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                use_case_id = str(item["id"])
                if use_case_id in seen:
                    continue
                seen.add(use_case_id)
                parent_id = str(item.get("parent_scenario_id") or "-")
                matched_terms = item.get("matched_terms", [])
                low_score_reasons = item.get("low_score_reasons", [])
                rows.append(
                    (
                        use_case_id,
                        str(item.get("name") or use_case_id),
                        f"{float(item.get('score', 0.0) or 0.0):.2f}",
                        f"{float(item.get('fit_score', 0.0) or 0.0):.2f}",
                        f"{float(item.get('evidence_completeness', 0.0) or 0.0):.2f}",
                        str(item.get("evaluation") or "未评价"),
                        parent_id,
                        self._clip(
                            "；".join(str(value) for value in low_score_reasons) or "-",
                            42,
                        ),
                        self._clip("、".join(str(value) for value in matched_terms) or "-"),
                    )
                )

            if not rows:
                self.candidates_log.write("本轮没有结构化 UC 候选。")
                return
            for index, (
                use_case_id,
                name,
                score,
                fit_score,
                evidence_completeness,
                evaluation,
                parent_id,
                low_score_reasons,
                matched_terms,
            ) in enumerate(
                rows, start=1
            ):
                item = next(
                    candidate
                    for candidate in raw_matches
                    if isinstance(candidate, dict) and str(candidate.get("id")) == use_case_id
                )
                self.use_case_table.add_row(
                    str(index),
                    use_case_id,
                    self._clip(name, 34),
                    score,
                    fit_score,
                    evidence_completeness,
                    self._clip(evaluation, 16),
                    parent_id,
                    low_score_reasons,
                    matched_terms,
                    key=use_case_id,
                )
                self.candidates_log.write(
                    f"UC {use_case_id} | 分数 {score} | 评价 {evaluation} | 父 SC {parent_id}"
                )
                fit_score = float(item.get("fit_score", 0.0) or 0.0)
                evidence_completeness = float(
                    item.get("evidence_completeness", 0.0) or 0.0
                )
                self.candidates_log.write(
                    f"   评分：决策分 {score}；可用证据匹配度 {fit_score:.2f}；"
                    f"证据完整度 {evidence_completeness:.2f}"
                )
                for dimension_line in self._format_dimension_scores(
                    item.get("dimension_scores")
                ):
                    self.candidates_log.write("   " + dimension_line)
                if low_score_reasons != "-":
                    self.candidates_log.write(f"   低分原因：{low_score_reasons}")

        def _match_candidate_details(self, result: Any) -> dict[str, dict[str, Any]]:
            details: dict[str, dict[str, Any]] = {}
            for call in result.tool_calls:
                if call.name not in {"match_ir_requirement", "match_scenario"}:
                    continue
                payload = call.result.get("match") if call.name == "match_ir_requirement" else call.result
                if not isinstance(payload, dict):
                    continue
                raw_matches = (
                    payload.get("scenario_matches", [])
                    if call.name == "match_ir_requirement"
                    else payload.get("matches", [])
                )
                for item in raw_matches:
                    if not isinstance(item, dict):
                        continue
                    scenario = item.get("scenario")
                    if not isinstance(scenario, dict) or not scenario.get("id"):
                        continue
                    details[str(scenario["id"])] = {
                        "conflicts": [
                            str(value) for value in item.get("conflicts", [])
                        ],
                        "matched_fields": {
                            str(key): [str(value) for value in values]
                            for key, values in (item.get("matched_fields") or {}).items()
                        },
                        "matched_terms": [
                            str(value) for value in item.get("matched_terms", [])
                        ],
                        "base_score": float(item.get("base_score", 0.0) or 0.0),
                        "fit_score": float(item.get("fit_score", 0.0) or 0.0),
                        "evidence_completeness": float(
                            item.get("evidence_completeness", 0.0) or 0.0
                        ),
                        "evaluation": str(item.get("evaluation") or "未评价"),
                        "dimension_scores": item.get("dimension_scores") or {},
                        "low_score_reasons": [
                            str(value) for value in item.get("low_score_reasons", [])
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

        @staticmethod
        def _format_match_fields(fields: object) -> str:
            if not isinstance(fields, dict) or not fields:
                return "无字段级命中证据"
            parts = []
            for key, values in fields.items():
                terms = values if isinstance(values, list) else [values]
                parts.append(f"{key}[{'、'.join(str(value) for value in terms)}]")
            return "；".join(parts)

        @staticmethod
        def _format_dimension_scores(scores: object) -> list[str]:
            if not isinstance(scores, dict) or not scores:
                return ["维度评分：暂无"]
            lines: list[str] = []
            for label, detail in scores.items():
                if not isinstance(detail, dict):
                    continue
                evidence = detail.get("evidence") or []
                evidence_values = [str(value) for value in evidence]
                evidence_text = "、".join(evidence_values[:12]) or "无"
                if len(evidence_values) > 12:
                    evidence_text += f"等{len(evidence_values)}项"
                lines.append(
                    f"维度 {label}：{float(detail.get('score', 0.0) or 0.0):.2f} × "
                    f"权重 {float(detail.get('weight', 0.0) or 0.0):.2f} = "
                    f"{float(detail.get('weighted_score', 0.0) or 0.0):.2f}；"
                    f"{detail.get('level', '未评价')}；证据：{evidence_text}；"
                    f"{detail.get('reason', '')}"
                )
            return lines or ["维度评分：暂无"]

        def _save_result(self, result: Any) -> Path | None:
            if self.settings is None or self.session is None:
                return None
            try:
                return save_run_report(
                    result,
                    self.settings.outputs_dir,
                    session_id=self.session.id,
                    input_text=self._active_input,
                    input_source=self._active_input_source,
                    library=self.agent.library if self.agent is not None else None,
                    spec_path=self.settings.spec_path,
                )
            except Exception as exc:
                self.tools_log.write(f"结果文件保存失败：{exc}")
                return None

        def _set_input_meta(self, text: str) -> None:
            self.query_one("#input-meta", _CopyableTextArea).update(text)

        def _open_review_csv(self) -> None:
            if self._last_output_path is None:
                self.notify("当前还没有可打开的复核表。", severity="warning")
                return
            self._open_result_path(
                self._last_output_path.parent / "evaluation" / "human_review_template.csv"
            )

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
            self.query_one("#evaluate-scenario", Button).disabled = disabled

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

    library = open_scenario_library(
        settings.library_path,
        use_case_path=settings.uc_library_path,
    )
    if settings.embedding_model:
        library.configure_embedding(
            OpenAIEmbeddingProvider(
                api_key=settings.api_key,
                model=settings.embedding_model,
                base_url=settings.base_url,
                organization=settings.organization,
                timeout=settings.request_timeout,
            )
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
