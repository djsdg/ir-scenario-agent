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
from .reporting import (
    apply_human_review,
    build_analysis_report,
    save_reviewed_report,
    save_run_report,
)
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
        "--output-dir",
        help="Save each run into a report directory containing scenarios/ and use_cases/",
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
    parser.add_argument(
        "--apply-review",
        metavar="CSV",
        help="Merge a human-edited human_review_template.csv into a report JSON and exit",
    )
    parser.add_argument(
        "--review-report",
        metavar="PATH",
        help="Report JSON used with --apply-review (normally evaluation/report.json's parent report.json)",
    )
    parser.add_argument(
        "--review-output",
        metavar="PATH",
        help="Output JSON for --apply-review; defaults to reviewed_report.json next to the input report",
    )
    return parser


def _format_match_fields(fields: object) -> str:
    if not isinstance(fields, dict) or not fields:
        return "无字段级命中证据"
    parts = []
    for key, values in fields.items():
        terms = values if isinstance(values, list) else [values]
        parts.append(f"{key}[{'、'.join(str(value) for value in terms)}]")
    return "；".join(parts)


def _format_dimension_scores(scores: object) -> list[str]:
    if not isinstance(scores, dict) or not scores:
        return ["  维度评分：暂无"]
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
            f"  {label}：{float(detail.get('score', 0.0) or 0.0):.2f} × "
            f"权重 {float(detail.get('weight', 0.0) or 0.0):.2f} = "
            f"{float(detail.get('weighted_score', 0.0) or 0.0):.2f}；"
            f"{detail.get('level', '未评价')}；证据：{evidence_text}；"
            f"{detail.get('reason', '')}"
        )
    return lines or ["  维度评分：暂无"]


def _print_result(
    result,
    *,
    show_tools: bool,
    json_output: bool,
    library=None,
) -> None:
    if result.resolution is not None and not json_output:
        resolution = result.resolution
        report = build_analysis_report(result, library)
        print(f"状态：{resolution.status}")
        print(f"决策：{resolution.decision}")
        if resolution.ir_id:
            print(f"IR：{resolution.ir_id}")
        print(f"需求理解：{resolution.request_summary}")
        matching = report.get("matching") or {}
        if matching:
            print(
                f"匹配评价：{matching.get('confidence_label', '未评价')} "
                f"（决策分 {float(matching.get('confidence', 0.0) or 0.0):.2f}；"
                f"IR证据完整度 {float(matching.get('evidence_completeness', 0.0) or 0.0):.2f}）"
            )
            for reason in matching.get("confidence_reasons") or []:
                print(f"置信度原因：{reason}")
            if matching.get("ambiguous"):
                print(
                    f"候选分差：{float(matching.get('score_margin', 0.0) or 0.0):.2f}（需人工确认）"
                )
        if resolution.candidates:
            candidates = "、".join(
                f"{item.scenario_id}({item.score:.2f})" for item in resolution.candidates
            )
            print(f"候选场景：{candidates}")
        for item in report.get("scenarios", {}).get("matches", []):
            print(
                f"SC匹配依据：{item['id']} {item['name']} -> "
                f"{_format_match_fields(item.get('matched_fields'))}"
            )
            print(
                f"  评分：决策分 {float(item.get('score', 0.0) or 0.0):.2f}；"
                f"可用证据匹配度 {float(item.get('fit_score', 0.0) or 0.0):.2f}；"
                f"证据完整度 {float(item.get('evidence_completeness', 0.0) or 0.0):.2f}；"
                f"评价：{item.get('evaluation', '未评价')}"
            )
            print("\n".join(_format_dimension_scores(item.get("dimension_scores"))))
            if item.get("low_score_reasons"):
                print(f"  低分原因：{'；'.join(item['low_score_reasons'])}")
            if item.get("gaps"):
                print(f"  未覆盖：{'；'.join(item['gaps'])}")
            if item.get("conflicts"):
                print(f"  冲突：{'；'.join(item['conflicts'])}")
        if resolution.selected_scenario_ids:
            print(f"选中场景：{', '.join(resolution.selected_scenario_ids)}")
        if resolution.use_case_ids:
            print(f"Use case：{', '.join(resolution.use_case_ids)}")
        for item in report.get("use_cases", {}).get("matches", []):
            print(
                f"UC匹配依据：{item['id']} {item['name']} "
                f"（父 SC：{item.get('parent_scenario_id') or '未解析'}） -> "
                f"{_format_match_fields(item.get('matched_fields'))}"
            )
            print(
                f"  评分：决策分 {float(item.get('score', 0.0) or 0.0):.2f}；"
                f"可用证据匹配度 {float(item.get('fit_score', 0.0) or 0.0):.2f}；"
                f"证据完整度 {float(item.get('evidence_completeness', 0.0) or 0.0):.2f}；"
                f"评价：{item.get('evaluation', '未评价')}"
            )
            print("\n".join(_format_dimension_scores(item.get("dimension_scores"))))
            if item.get("low_score_reasons"):
                print(f"  低分原因：{'；'.join(item['low_score_reasons'])}")
        for evaluation in report.get("evaluations", {}).get("scenario_fit", []):
            scenario = evaluation.get("scenario") or {}
            print(
                f"指定 SC 符合度：{evaluation.get('scenario_id', scenario.get('id', '-'))} "
                f"{scenario.get('name', '')} -> "
                f"{float(evaluation.get('score', 0.0) or 0.0):.2f} "
                f"（{evaluation.get('evaluation', '未评价')}）"
            )
            print("\n".join(_format_dimension_scores(evaluation.get("dimension_scores"))))
            if evaluation.get("low_score_reasons"):
                print(f"  低分原因：{'；'.join(evaluation['low_score_reasons'])}")
        for relationship in report.get("use_cases", {}).get("by_scenario", []):
            print(
                f"SC→UC：{relationship['scenario_id']} "
                f"{relationship['scenario_name']} -> "
                f"{', '.join(relationship.get('use_case_ids', [])) or '暂无 UC'}"
            )
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
        if report.get("writes"):
            print(
                "库已更新："
                + "、".join(
                    f"{item.get('action')} {item.get('id')}"
                    for item in report["writes"]
                )
            )
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
        outputs_dir=args.output_dir,
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

    if args.apply_review:
        if not args.review_report:
            print("--apply-review 必须同时提供 --review-report。", file=sys.stderr)
            return 2
        try:
            report_path = Path(args.review_report).expanduser()
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            report = payload.get("report") if isinstance(payload, dict) else None
            if not isinstance(report, dict):
                report = payload
            if not isinstance(report, dict):
                raise ValueError("输入 JSON 不是分析报告。")
            reviewed = apply_human_review(report, args.apply_review)
            output_path = (
                Path(args.review_output).expanduser()
                if args.review_output
                else report_path.with_name("reviewed_report.json")
            )
            saved = save_reviewed_report(reviewed, output_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            print(f"人工复核回填失败：{exc}", file=sys.stderr)
            return 2
        print(f"人工复核已回填：{saved}")
        print(f"Markdown：{saved.with_suffix('.md')}")
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
        _print_result(
            result,
            show_tools=args.show_tools,
            json_output=args.json_output,
            library=library,
        )
        try:
            output_path = save_run_report(
                result,
                settings.outputs_dir,
                session_id=session.id,
                input_text=message,
                input_source="CLI",
                library=library,
                spec_path=settings.spec_path,
            )
            if args.json_output:
                print(f"结果目录：{output_path.parent}", file=sys.stderr)
                print(f"CSV评估目录：{output_path.parent / 'evaluation'}", file=sys.stderr)
            else:
                print(f"结果目录：{output_path.parent}")
                print(f"CSV评估目录：{output_path.parent / 'evaluation'}")
        except OSError as exc:
            print(f"结果报告保存失败：{exc}", file=sys.stderr)
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
