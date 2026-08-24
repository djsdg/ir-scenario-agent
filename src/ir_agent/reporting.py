from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .domain import AgentResult
from .library import ScenarioLibrary


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if item is not None and str(item)]


def _merge_strings(left: list[str], right: Any) -> list[str]:
    return list(dict.fromkeys([*left, *_string_list(right)]))


def _merge_fields(left: dict[str, list[str]], right: Any) -> dict[str, list[str]]:
    merged = {str(key): _string_list(value) for key, value in left.items()}
    if not isinstance(right, dict):
        return merged
    for key, value in right.items():
        name = str(key)
        merged[name] = _merge_strings(merged.get(name, []), value)
    return merged


def _normalize_scenario_match(item: Any, *, source: str) -> dict[str, Any] | None:
    scenario = _value(item, "scenario", {})
    scenario_id = _value(scenario, "id")
    if not scenario_id:
        return None
    return {
        "id": str(scenario_id),
        "name": str(_value(scenario, "name") or scenario_id),
        "score": float(_value(item, "score", 0.0) or 0.0),
        "matched_terms": _string_list(_value(item, "matched_terms")),
        "matched_fields": {
            str(key): _string_list(value)
            for key, value in (_value(item, "matched_fields", {}) or {}).items()
        },
        "matched_dimensions": _string_list(_value(item, "matched_dimensions")),
        "gaps": _string_list(_value(item, "gaps")),
        "conflicts": _string_list(_value(item, "conflicts")),
        "reason": str(_value(item, "reason") or ""),
        "source": source,
        "record": scenario,
    }


def _normalize_use_case_match(
    item: Any,
    *,
    source: str,
    parent_by_use_case: dict[str, str],
) -> dict[str, Any] | None:
    use_case = _value(item, "use_case", {})
    use_case_id = _value(use_case, "id")
    if not use_case_id:
        return None
    use_case_id = str(use_case_id)
    parent_id = _value(item, "parent_scenario_id") or parent_by_use_case.get(use_case_id)
    return {
        "id": use_case_id,
        "name": str(_value(use_case, "name") or use_case_id),
        "score": float(_value(item, "score", 0.0) or 0.0),
        "parent_scenario_id": str(parent_id) if parent_id else None,
        "matched_terms": _string_list(_value(item, "matched_terms")),
        "matched_fields": {
            str(key): _string_list(value)
            for key, value in (_value(item, "matched_fields", {}) or {}).items()
        },
        "source": source,
        "record": use_case,
    }


def _merge_match_rows(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    for existing in rows:
        if existing.get("id") != row.get("id"):
            continue
        existing["matched_terms"] = _merge_strings(
            existing.get("matched_terms", []), row.get("matched_terms")
        )
        existing["matched_fields"] = _merge_fields(
            existing.get("matched_fields", {}), row.get("matched_fields")
        )
        for field in ("matched_dimensions", "gaps", "conflicts"):
            if field in row:
                existing[field] = _merge_strings(existing.get(field, []), row.get(field))
        if row.get("score", 0.0) > existing.get("score", 0.0):
            existing["score"] = row["score"]
            existing["record"] = row.get("record", existing.get("record", {}))
        existing["source"] = _merge_strings(
            _string_list(existing.get("source")), row.get("source")
        )
        return
    rows.append(row)


def _parent_map(library: ScenarioLibrary | None) -> dict[str, str]:
    if library is None:
        return {}
    parents: dict[str, str] = {}
    try:
        for scenario in library.list_scenarios():
            for use_case_id in scenario.use_case_ids:
                parents.setdefault(use_case_id, scenario.id)
    except Exception:
        return {}
    return parents


def _match_rows(result: AgentResult, library: ScenarioLibrary | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_rows: list[dict[str, Any]] = []
    use_case_rows: list[dict[str, Any]] = []
    parent_by_use_case = _parent_map(library)
    for call in result.tool_calls:
        payload = call.result if isinstance(call.result, dict) else {}
        if call.name == "match_ir_requirement":
            match = payload.get("match")
            if not isinstance(match, dict):
                continue
            raw_scenarios = match.get("scenario_matches", [])
            raw_use_cases = match.get("use_case_matches", [])
            source = "match_ir_requirement"
        elif call.name in {"match_scenario", "search_scenarios"}:
            raw_scenarios = payload.get("matches", [])
            raw_use_cases = []
            source = call.name
        elif call.name in {"match_use_case", "search_use_cases"}:
            raw_scenarios = []
            raw_use_cases = payload.get("matches", [])
            source = call.name
        else:
            continue
        for item in raw_scenarios if isinstance(raw_scenarios, list) else []:
            normalized = _normalize_scenario_match(item, source=source)
            if normalized:
                _merge_match_rows(scenario_rows, normalized)
        for item in raw_use_cases if isinstance(raw_use_cases, list) else []:
            normalized = _normalize_use_case_match(
                item,
                source=source,
                parent_by_use_case=parent_by_use_case,
            )
            if normalized:
                _merge_match_rows(use_case_rows, normalized)
    scenario_rows.sort(key=lambda item: (-float(item.get("score", 0.0)), item["name"]))
    use_case_rows.sort(key=lambda item: (-float(item.get("score", 0.0)), item["name"]))
    return scenario_rows, use_case_rows


def _write_rows(result: AgentResult) -> dict[str, list[dict[str, Any]]]:
    writes = {
        "scenarios": {"created": [], "updated": []},
        "use_cases": {"created": [], "updated": []},
        "operations": [],
    }

    def append(kind: str, action: str, record: Any, call: Any) -> None:
        if not isinstance(record, dict) or not record.get("id"):
            return
        item = {
            "id": str(record["id"]),
            "action": action,
            "tool": call.name,
            "approved": call.approved,
            "record": record,
        }
        writes[kind][action].append(item)
        writes["operations"].append(item)

    for call in result.tool_calls:
        payload = call.result if isinstance(call.result, dict) else {}
        if not payload.get("ok", True):
            continue
        if call.name == "create_scenario":
            append("scenarios", "created", payload.get("scenario"), call)
        elif call.name == "create_use_case":
            append("use_cases", "created", payload.get("use_case"), call)
        elif call.name == "update_scenario":
            append("scenarios", "updated", payload.get("scenario"), call)
        elif call.name == "update_use_case":
            append("use_cases", "updated", payload.get("use_case"), call)
        elif call.name == "link_scenario_use_cases":
            append("scenarios", "updated", payload.get("scenario"), call)
        elif call.name == "move_use_case":
            append("use_cases", "updated", payload.get("use_case"), call)
        elif call.name == "transition_record":
            record_type = str(call.arguments.get("record_type") or "")
            kind = "scenarios" if record_type == "scenario" else "use_cases"
            append(kind, "updated", payload.get("record"), call)
    return writes


def _records_by_ids(library: ScenarioLibrary | None, ids: list[str], kind: str) -> list[dict[str, Any]]:
    if library is None:
        return []
    records: list[dict[str, Any]] = []
    getter = library.get_scenario if kind == "scenario" else library.get_use_case
    for record_id in dict.fromkeys(ids):
        try:
            records.append(getter(record_id).model_dump(mode="json"))
        except (AttributeError, KeyError, ValueError):
            continue
    return records


def build_analysis_report(result: AgentResult, library: ScenarioLibrary | None = None) -> dict[str, Any]:
    """Build a stable, explainable SC/UC report from tool facts and the final resolution."""

    scenario_matches, use_case_matches = _match_rows(result, library)
    resolution = result.resolution.model_dump(mode="json") if result.resolution else None
    writes = _write_rows(result)
    selected_scenario_ids = list(resolution.get("selected_scenario_ids", [])) if resolution else []
    selected_use_case_ids = list(resolution.get("use_case_ids", [])) if resolution else []
    selected_scenario_ids.extend(item["id"] for item in writes["scenarios"]["created"])
    selected_scenario_ids.extend(item["id"] for item in writes["scenarios"]["updated"])
    selected_use_case_ids.extend(item["id"] for item in writes["use_cases"]["created"])
    selected_use_case_ids.extend(item["id"] for item in writes["use_cases"]["updated"])

    parent_ids = [
        str(item["parent_scenario_id"])
        for item in use_case_matches
        if item.get("parent_scenario_id")
    ]
    selected_scenarios = _records_by_ids(
        library,
        [*selected_scenario_ids, *parent_ids],
        "scenario",
    )
    selected_use_cases = _records_by_ids(library, selected_use_case_ids, "use_case")
    by_scenario: list[dict[str, Any]] = []
    if library is not None:
        for scenario_id in dict.fromkeys([*selected_scenario_ids, *parent_ids]):
            try:
                scenario = library.get_scenario(scenario_id)
            except (AttributeError, KeyError, ValueError):
                continue
            children = _records_by_ids(library, scenario.use_case_ids, "use_case")
            by_scenario.append(
                {
                    "scenario_id": scenario.id,
                    "scenario_name": scenario.name,
                    "use_case_ids": list(scenario.use_case_ids),
                    "use_cases": children,
                }
            )

    library_payload = {
        "scenario_path": str(library.path.resolve()) if library is not None else None,
        "use_case_path": (
            str(library.use_case_path.resolve())
            if library is not None and library.use_case_path is not None
            else None
        ),
        "updated_by_this_run": bool(writes["operations"]),
    }
    return {
        "resolution": resolution,
        "scenarios": {
            "matches": scenario_matches,
            "selected": selected_scenarios,
            "created": writes["scenarios"]["created"],
            "updated": writes["scenarios"]["updated"],
        },
        "use_cases": {
            "matches": use_case_matches,
            "selected": selected_use_cases,
            "created": writes["use_cases"]["created"],
            "updated": writes["use_cases"]["updated"],
            "by_scenario": by_scenario,
        },
        "writes": writes["operations"],
        "library": library_payload,
    }


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:120] or "record"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _evidence_text(fields: Any) -> str:
    if not isinstance(fields, dict) or not fields:
        return "无字段级命中证据"
    return "；".join(
        f"{key}：{'、'.join(_string_list(value)) or '-'}" for key, value in fields.items()
    )


def render_markdown_report(report: dict[str, Any]) -> str:
    resolution = report.get("resolution") or {}
    lines = [
        "# IR / SC / UC 分析报告",
        "",
        f"- 状态：{resolution.get('status', '未结构化')}",
        f"- 决策：{resolution.get('decision', '-')}",
        f"- 需求理解：{resolution.get('request_summary', '-')}",
        "",
        "## 场景 SC",
        "",
    ]
    scenario_matches = report.get("scenarios", {}).get("matches", [])
    if scenario_matches:
        for item in scenario_matches:
            lines.append(
                f"- **{item['id']} {item['name']}**（分数 {float(item.get('score', 0.0)):.2f}）"
            )
            lines.append(f"  - 命中部分：{_evidence_text(item.get('matched_fields'))}")
            if item.get("gaps"):
                lines.append(f"  - 未覆盖：{'；'.join(item['gaps'])}")
            if item.get("conflicts"):
                lines.append(f"  - 冲突：{'；'.join(item['conflicts'])}")
    else:
        lines.append("未找到结构化 SC 候选；如确认无匹配，可根据草稿补齐后新建 SC。")

    lines.extend(["", "## 用例 UC", ""])
    use_case_matches = report.get("use_cases", {}).get("matches", [])
    if use_case_matches:
        for item in use_case_matches:
            parent = item.get("parent_scenario_id") or "父 SC 未解析"
            lines.append(
                f"- **{item['id']} {item['name']}**（父 SC：{parent}，分数 {float(item.get('score', 0.0)):.2f}）"
            )
            lines.append(f"  - 命中部分：{_evidence_text(item.get('matched_fields'))}")
    else:
        lines.append("未找到结构化 UC 候选；如 SC 可复用但行为链不同，可在该 SC 下新增 UC。")

    lines.extend(["", "## SC → UC 关系", ""])
    relationships = report.get("use_cases", {}).get("by_scenario", [])
    if relationships:
        for item in relationships:
            ids = "、".join(item.get("use_case_ids", [])) or "暂无 UC"
            lines.append(f"- {item['scenario_id']} {item['scenario_name']}：{ids}")
    else:
        lines.append("本轮没有可读取的父子关系。")

    lines.extend(["", "## 库写入", ""])
    writes = report.get("writes", [])
    if writes:
        for item in writes:
            lines.append(
                f"- {item.get('action')} {item.get('id')}（工具：{item.get('tool')}，已审批：{item.get('approved')}）"
            )
        lines.append("原场景库已更新；结果目录中的 SC/UC 文件是本轮快照。")
    else:
        lines.append("本轮没有执行库写入。")
    return "\n".join(lines) + "\n"


def save_run_report(
    result: AgentResult,
    output_dir: str | Path,
    *,
    session_id: str,
    input_text: str | None = None,
    input_source: str | None = None,
    library: ScenarioLibrary | None = None,
    spec_path: str | Path | None = None,
) -> Path:
    """Persist one run as result + report + separate SC/UC artifact folders."""

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    run_root = Path(output_dir) / session_id / f"{timestamp}_{uuid4().hex[:8]}"
    report = build_analysis_report(result, library)
    envelope = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "input": input_text,
        "input_source": input_source,
        "library_path": report["library"].get("scenario_path"),
        "uc_library_path": report["library"].get("use_case_path"),
        "spec_path": str(Path(spec_path).resolve()) if spec_path else None,
        "result": result.model_dump(mode="json"),
        "report": report,
    }
    _write_json(run_root / "result.json", envelope)
    _write_json(run_root / "report.json", report)
    (run_root / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "report.md").write_text(
        render_markdown_report(report), encoding="utf-8"
    )
    _write_json(run_root / "scenarios" / "matches.json", report["scenarios"]["matches"])
    _write_json(run_root / "scenarios" / "selected.json", report["scenarios"]["selected"])
    _write_json(run_root / "scenarios" / "created.json", report["scenarios"]["created"])
    _write_json(run_root / "scenarios" / "updated.json", report["scenarios"]["updated"])
    _write_json(run_root / "use_cases" / "matches.json", report["use_cases"]["matches"])
    _write_json(run_root / "use_cases" / "selected.json", report["use_cases"]["selected"])
    _write_json(run_root / "use_cases" / "created.json", report["use_cases"]["created"])
    _write_json(run_root / "use_cases" / "updated.json", report["use_cases"]["updated"])
    for relationship in report["use_cases"]["by_scenario"]:
        _write_json(
            run_root
            / "use_cases"
            / "by_scenario"
            / f"{_safe_filename(relationship['scenario_id'])}.json",
            relationship,
        )
    manifest = {
        "result": str((run_root / "result.json").resolve()),
        "report_json": str((run_root / "report.json").resolve()),
        "report_markdown": str((run_root / "report.md").resolve()),
        "scenarios_dir": str((run_root / "scenarios").resolve()),
        "use_cases_dir": str((run_root / "use_cases").resolve()),
    }
    _write_json(run_root / "manifest.json", manifest)
    return (run_root / "result.json").resolve()
