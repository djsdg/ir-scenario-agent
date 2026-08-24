from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .agent import (
    AgentRunError,
    AgentSession,
    OpenAIChatCompletionsTransport,
    IRScenarioAgent,
    OpenAIResponsesTransport,
    RetryingResponsesTransport,
    SessionStore,
)
from .audit import AuditLogger
from .config import Settings
from .documents import read_document
from .library import open_scenario_library
from .mcp import MCPConfig
from .memory import MemoryStore
from .plugins import PluginContext, PluginManager
from .retrieval import OpenAIEmbeddingProvider
from .skills import SkillCatalog
from .specs import SpecCatalog, SpecError
from .sqlite_library import migrate_json_to_sqlite
from .tools import ToolRegistry


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR scenario-library agent")
    parser.add_argument("--message", help="Run one request and exit")
    parser.add_argument(
        "--input-file",
        help="Read an IR/SC/UC document (.txt/.md/.json/.docx/.pdf) and run once",
    )
    parser.add_argument("--session-id", default="default", help="Local session id for conversation memory")
    parser.add_argument("--library", help="Path to the scenario library JSON file or directory")
    parser.add_argument("--uc-library", help="Path to the separate use-case library JSON file")
    parser.add_argument("--spec", help="Path to the IR→SC→UC business specification JSON")
    parser.add_argument("--model", help="Override OPENAI_MODEL for this run")
    parser.add_argument(
        "--api-mode",
        choices=["responses", "chat_completions"],
        help="API protocol: Responses API (default) or OpenAI-compatible Chat Completions",
    )
    parser.add_argument("--user-id", help="User scope for long-term memory")
    parser.add_argument("--skills-dir", help="Directory containing SKILL.md files")
    parser.add_argument("--plugins-dir", help="Directory containing plugin.json files")
    parser.add_argument("--mcp-config", help="MCP server configuration JSON")
    parser.add_argument("--audit-path", help="Append-only JSONL audit log path")
    parser.add_argument("--no-memory", action="store_true", help="Disable SQLite long-term memory")
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Disable strict JSON final output (useful with older/incompatible models)",
    )
    parser.add_argument(
        "--auto-approve-writes",
        action="store_true",
        help="Automatically approve scenario-library and memory write tools",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print the machine-readable final JSON instead of the human summary",
    )
    parser.add_argument(
        "--no-session-save",
        action="store_true",
        help="Do not load or save the local JSON session",
    )
    parser.add_argument("--show-tools", action="store_true", help="Print tool calls after each turn")
    parser.add_argument(
        "--validate-library",
        action="store_true",
        help="Run the local read-only library quality audit without calling a model",
    )
    parser.add_argument(
        "--migrate-to-sqlite",
        metavar="PATH",
        help="Copy the selected JSON/directory library to a SQLite database and exit",
    )
    return parser


def _print_result(result, *, show_tools: bool, json_output: bool) -> None:
    if result.resolution is not None and not json_output:
        resolution = result.resolution
        print(f"状态：{resolution.status}")
        print(f"决策：{resolution.decision}")
        if resolution.ir_id:
            print(f"IR：{resolution.ir_id}")
        print(f"需求理解：{resolution.request_summary}")
        if resolution.candidates:
            candidates = "、".join(
                f"{item.scenario_id}({item.score:.2f})" for item in resolution.candidates
            )
            print(f"候选场景：{candidates}")
        if resolution.selected_scenario_ids:
            print(f"选中场景：{', '.join(resolution.selected_scenario_ids)}")
        if resolution.use_case_ids:
            print(f"Use case：{', '.join(resolution.use_case_ids)}")
        if resolution.created_scenario_id:
            print(f"新建场景：{resolution.created_scenario_id}")
        if resolution.created_use_case_ids:
            print(f"新建 Use case：{', '.join(resolution.created_use_case_ids)}")
        print(f"置信度：{resolution.confidence:.2f}")
        if resolution.missing_required_fields:
            print(f"缺少必填字段：{', '.join(resolution.missing_required_fields)}")
        if resolution.gaps:
            print(f"缺口：{'；'.join(resolution.gaps)}")
        if resolution.next_steps:
            print(f"下一步：{'；'.join(resolution.next_steps)}")
    else:
        print(result.output_text)
    if show_tools and result.tool_calls:
        print("\n[tool calls]")
        for call in result.tool_calls:
            print(f"- {call.name}: {call.result.get('ok', False)}")


def _approve_mcp_request(request: dict[str, object]) -> bool:
    if not sys.stdin.isatty():
        return False
    print(
        "\nMCP 工具需要授权："
        f" server={request.get('server_label')}"
        f" tool={request.get('name')}"
        f" args={request.get('arguments')}"
    )
    answer = input("允许这次 MCP 调用吗？[y/N] ").strip().casefold()
    return answer in {"y", "yes"}


def _approve_write_request(request: dict[str, object], *, auto_approve: bool) -> bool:
    if auto_approve:
        return True
    if not sys.stdin.isatty():
        return False
    print(
        "\nAgent 请求执行写入工具："
        f" tool={request.get('tool_name')}"
        f" args={request.get('arguments')}"
    )
    answer = input("允许这次写入吗？[y/N] ").strip().casefold()
    return answer in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = _build_parser().parse_args(argv)
    settings = Settings.from_env(
        library_path=args.library,
        uc_library_path=args.uc_library,
        spec_path=args.spec,
        api_mode=args.api_mode,
        skills_dir=args.skills_dir,
        plugins_dir=args.plugins_dir,
        mcp_config_path=args.mcp_config,
        audit_path=args.audit_path,
        user_id=args.user_id,
    )

    if args.migrate_to_sqlite:
        try:
            migrated = migrate_json_to_sqlite(
                settings.library_path,
                Path(args.migrate_to_sqlite),
            )
        except (OSError, ValueError, FileExistsError) as exc:
            print(f"SQLite 迁移失败：{exc}", file=sys.stderr)
            return 2
        print(f"SQLite 场景库已创建：{migrated.path.resolve()}")
        return 0

    if args.validate_library:
        try:
            library = open_scenario_library(
                settings.library_path,
                use_case_path=settings.uc_library_path,
            )
            spec = SpecCatalog.from_file(settings.spec_path)
            report = ToolRegistry(library, spec=spec).execute("validate_library", {})
        except (OSError, ValueError, SpecError) as exc:
            print(f"场景库检查失败：{exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    if args.model:
        settings = replace(settings, model=args.model)
    if args.no_structured_output:
        settings = replace(settings, structured_output=False)
    if not settings.api_key:
        print(
            "未检测到 API key。请在 .env 中填入 IR_AGENT_API_KEY 或 OPENAI_API_KEY，"
            "或在当前 shell 中设置环境变量。",
            file=sys.stderr,
        )
        return 2

    library = open_scenario_library(
        settings.library_path,
        use_case_path=settings.uc_library_path,
    )
    if settings.embedding_model:
        try:
            library.configure_embedding(
                OpenAIEmbeddingProvider(
                    api_key=settings.api_key,
                    model=settings.embedding_model,
                    base_url=settings.base_url,
                    organization=settings.organization,
                    timeout=settings.request_timeout,
                )
            )
        except RuntimeError as exc:
            print(f"Embedding 配置失败：{exc}", file=sys.stderr)
            return 2
    try:
        spec = SpecCatalog.from_file(settings.spec_path)
    except SpecError as exc:
        print(f"业务 Spec 读取失败：{exc}", file=sys.stderr)
        return 2
    skills = SkillCatalog(settings.skills_dir)
    memory = None if args.no_memory else MemoryStore(settings.memory_path)
    try:
        mcp_config = MCPConfig.from_file(settings.mcp_config_path)
    except Exception as exc:
        print(f"MCP 配置读取失败：{exc}", file=sys.stderr)
        return 2
    try:
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
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    agent = IRScenarioAgent(
        transport,
        library,
        settings=settings,
        skills=skills,
        memory=memory,
        spec=spec,
        user_id=settings.user_id,
        mcp_config=mcp_config,
        mcp_approval_callback=_approve_mcp_request,
        tool_approval_callback=lambda request: _approve_write_request(
            request, auto_approve=args.auto_approve_writes
        ),
        audit_logger=AuditLogger(settings.audit_path),
    )
    plugin_report = PluginManager(settings.plugins_dir).load_into(
        agent.tools,
        PluginContext(
            settings=settings,
            library=library,
            skills=skills,
            memory=memory,
            spec=spec,
            user_id=settings.user_id,
        ),
    )
    for error in plugin_report.errors:
        print(f"插件未加载：{error}", file=sys.stderr)

    session_store = None if args.no_session_save else SessionStore(settings.sessions_dir)
    session = AgentSession(id=args.session_id) if session_store is None else session_store.load(args.session_id)

    def run_one(message: str) -> None:
        nonlocal session
        try:
            result = agent.run(message, session=session)
        except AgentRunError as exc:
            print(f"Agent 执行未完成：{exc}", file=sys.stderr)
            return
        except Exception as exc:  # CLI boundary: make API errors readable.
            print(f"请求失败：{exc}", file=sys.stderr)
            return
        _print_result(result, show_tools=args.show_tools, json_output=args.json_output)
        if session_store is not None:
            session_store.save(session)

    if args.message and args.input_file:
        print("--message 和 --input-file 不能同时使用。", file=sys.stderr)
        return 2
    if args.input_file:
        try:
            file_message = read_document(Path(args.input_file))
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"输入文件读取失败：{exc}", file=sys.stderr)
            return 2
        run_one(file_message)
        return 0
    if args.message:
        run_one(args.message)
        return 0

    print("IR Scenario Agent 已启动。输入 exit / quit 退出。")
    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.casefold() in {"exit", "quit", "\u9000\u51fa"}:
            break
        run_one(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
