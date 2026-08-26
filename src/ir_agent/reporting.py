from __future__ import annotations

import csv
from copy import deepcopy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .domain import AgentResult, IRRequirementInput
from .library import ScenarioLibrary


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _record_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, dict) else {}
    return {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
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


def _dimension_scores(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): dict(item)
        for key, item in value.items()
        if isinstance(item, dict)
    }


def _normalize_scenario_match(item: Any, *, source: str) -> dict[str, Any] | None:
    scenario = _value(item, "scenario", {})
    scenario_id = _value(scenario, "id")
    if not scenario_id:
        return None
    scenario_record = _record_dict(scenario)
    score = float(_value(item, "score", 0.0) or 0.0)
    raw_fit_score = _value(item, "fit_score", None)
    raw_completeness = _value(item, "evidence_completeness", None)
    return {
        "id": str(scenario_id),
        "name": str(_value(scenario, "name") or scenario_id),
        "score": score,
        "fit_score": score if raw_fit_score is None else float(raw_fit_score or 0.0),
        "evidence_completeness": (
            1.0 if raw_completeness is None else float(raw_completeness or 0.0)
        ),
        "matched_terms": _string_list(_value(item, "matched_terms")),
        "matched_fields": {
            str(key): _string_list(value)
            for key, value in (_value(item, "matched_fields", {}) or {}).items()
        },
        "matched_dimensions": _string_list(_value(item, "matched_dimensions")),
        "gaps": _string_list(_value(item, "gaps")),
        "conflicts": _string_list(_value(item, "conflicts")),
        "base_score": float(_value(item, "base_score", 0.0) or 0.0),
        "consistency_bonus": float(_value(item, "consistency_bonus", 0.0) or 0.0),
        "evaluation": str(_value(item, "evaluation") or "未评价"),
        "dimension_scores": _dimension_scores(_value(item, "dimension_scores", {})),
        "low_score_reasons": _string_list(_value(item, "low_score_reasons")),
        "reason": str(_value(item, "reason") or ""),
        "source": source,
        "record": scenario_record,
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
    score = float(_value(item, "score", 0.0) or 0.0)
    raw_fit_score = _value(item, "fit_score", None)
    raw_completeness = _value(item, "evidence_completeness", None)
    return {
        "id": use_case_id,
        "name": str(_value(use_case, "name") or use_case_id),
        "score": score,
        "fit_score": score if raw_fit_score is None else float(raw_fit_score or 0.0),
        "evidence_completeness": (
            1.0 if raw_completeness is None else float(raw_completeness or 0.0)
        ),
        "parent_scenario_id": str(parent_id) if parent_id else None,
        "matched_terms": _string_list(_value(item, "matched_terms")),
        "matched_fields": {
            str(key): _string_list(value)
            for key, value in (_value(item, "matched_fields", {}) or {}).items()
        },
        "matched_dimensions": _string_list(_value(item, "matched_dimensions")),
        "gaps": _string_list(_value(item, "gaps")),
        "conflicts": _string_list(_value(item, "conflicts")),
        "base_score": float(_value(item, "base_score", 0.0) or 0.0),
        "consistency_bonus": float(_value(item, "consistency_bonus", 0.0) or 0.0),
        "evaluation": str(_value(item, "evaluation") or "未评价"),
        "dimension_scores": _dimension_scores(_value(item, "dimension_scores", {})),
        "low_score_reasons": _string_list(_value(item, "low_score_reasons")),
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
        existing["low_score_reasons"] = _merge_strings(
            existing.get("low_score_reasons", []), row.get("low_score_reasons")
        )
        if row.get("score", 0.0) > existing.get("score", 0.0):
            existing["score"] = row["score"]
            existing["record"] = row.get("record", existing.get("record", {}))
            for field in (
                "fit_score",
                "evidence_completeness",
                "base_score",
                "consistency_bonus",
                "evaluation",
                "dimension_scores",
            ):
                if field in row:
                    existing[field] = row[field]
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


def _review_top_scenario_rows(
    scenario_rows: list[dict[str, Any]],
    *,
    ir: dict[str, Any],
    library: ScenarioLibrary | None,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Return the best SC candidates for human review, even below reuse threshold.

    The agent normally asks the matcher for multiple candidates. If a model or
    caller requested fewer than two, a report-only deterministic re-ranking
    fills the review list from the active library. It never changes the agent
    decision and never writes the library.
    """

    if limit < 1:
        return []
    review_rows = [dict(item) for item in scenario_rows]
    if len(review_rows) < limit and library is not None and ir:
        try:
            normalized_ir = IRRequirementInput.model_validate(ir)
            supplemental = library.match_ir(normalized_ir, top_k=limit, min_score=0.0)
        except (TypeError, ValueError, KeyError):
            supplemental = None
        if supplemental is not None:
            known_ids = {str(item.get("id") or "") for item in review_rows}
            for match in supplemental.scenario_matches:
                normalized = _normalize_scenario_match(
                    match,
                    source="report_review_top2",
                )
                if normalized is None or normalized["id"] in known_ids:
                    continue
                review_rows.append(normalized)
                known_ids.add(normalized["id"])

    review_rows.sort(key=lambda item: (-float(item.get("score", 0.0)), item["name"]))
    return review_rows[:limit]


def _matching_summary(result: AgentResult) -> dict[str, Any]:
    for call in result.tool_calls:
        payload = call.result if isinstance(call.result, dict) else {}
        if call.name == "match_ir_requirement" and isinstance(payload.get("match"), dict):
            match = payload["match"]
            return {
                "tool": call.name,
                "confidence": float(match.get("confidence", 0.0) or 0.0),
                "evidence_completeness": float(
                    match.get("evidence_completeness", 0.0) or 0.0
                ),
                "supplied_dimensions": _string_list(match.get("supplied_dimensions")),
                "confidence_label": str(match.get("confidence_label") or "未评价"),
                "confidence_reasons": _string_list(match.get("confidence_reasons")),
                "score_margin": float(match.get("score_margin", 0.0) or 0.0),
                "ambiguous": bool(match.get("ambiguous", False)),
                "decision": str(match.get("decision") or ""),
            }
        if call.name in {"match_scenario", "match_use_case"}:
            return {
                "tool": call.name,
                "confidence": float(payload.get("confidence", 0.0) or 0.0),
                "evidence_completeness": 1.0,
                "supplied_dimensions": [],
                "confidence_label": "可复用候选"
                if payload.get("decision") == "reuse_existing"
                else "低分/建议新增",
                "confidence_reasons": _string_list(payload.get("rationale")),
                "score_margin": 0.0,
                "ambiguous": False,
                "decision": str(payload.get("decision") or ""),
            }
        if call.name == "evaluate_scenario_fit" and isinstance(payload.get("evaluation"), dict):
            evaluation = payload["evaluation"]
            return {
                "tool": call.name,
                "confidence": float(evaluation.get("score", 0.0) or 0.0),
                "evidence_completeness": float(
                    evaluation.get("evidence_completeness", 1.0) or 0.0
                ),
                "supplied_dimensions": [],
                "confidence_label": str(evaluation.get("confidence_label") or evaluation.get("evaluation") or "未评价"),
                "confidence_reasons": _string_list(
                    evaluation.get("confidence_reasons")
                    or evaluation.get("low_score_reasons")
                ),
                "score_margin": 0.0,
                "ambiguous": False,
                "decision": "evaluate_scenario_fit",
            }
    return {
        "tool": None,
        "confidence": 0.0,
        "evidence_completeness": 0.0,
        "supplied_dimensions": [],
        "confidence_label": "未调用匹配工具",
        "confidence_reasons": ["本轮没有匹配工具事实。"],
        "score_margin": 0.0,
        "ambiguous": False,
        "decision": "needs_clarification",
    }


def _scenario_fit_evaluations(result: AgentResult) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for call in result.tool_calls:
        if call.name != "evaluate_scenario_fit":
            continue
        payload = call.result if isinstance(call.result, dict) else {}
        evaluation = payload.get("evaluation")
        if isinstance(evaluation, dict):
            evaluations.append(dict(evaluation))
    return evaluations


def _ir_payload(result: AgentResult, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    for evaluation in evaluations:
        ir = evaluation.get("ir")
        if isinstance(ir, dict):
            return ir
    for call in result.tool_calls:
        if call.name != "match_ir_requirement":
            continue
        payload = call.result if isinstance(call.result, dict) else {}
        match = payload.get("match")
        if isinstance(match, dict) and isinstance(match.get("ir"), dict):
            return dict(match["ir"])
    return {}


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


_SC_FIELD_COMPARISON = (
    ("description", "description", "description", "目标/行为", "SC description 必填"),
    ("category", None, "category", "目标/行为", "SC category 必须符合 Spec"),
    ("business_goal", "what", "business_goal", "目标/行为", "SC business_goal 必填"),
    ("actor", "who", "actor", "Actor", "SC actor 必填"),
    ("actions", "how", "actions", "目标/行为", "SC actions 必填"),
    ("influence_factors", "where", "influence_factors", "影响因素", "至少一个影响因素且有 selected_values"),
    ("lifecycle", "when", "lifecycle", "上下文", "SC lifecycle 必填"),
    ("constraints", "constraints", "constraints", "约束", "SC constraints 必填"),
    ("owner", "owner", "owner", "Actor", "SC owner 必填"),
)

_REVIEW_CONSISTENCY_VALUES = ("一致", "部分一致", "不一致", "无法判断", "待人工确认")
_REVIEW_STATUS_VALUES = ("待复核", "已确认", "需新增", "需修改")
_REVIEW_DECISION_VALUES = ("复用", "部分复用", "不复用", "新增场景", "修改后复用", "待定")


def _review_candidate_summary_rows(
    candidates: list[dict[str, Any]],
    selected_scenario_ids: list[str],
) -> list[dict[str, Any]]:
    """Render a compact, human-first summary for the top SC review candidates."""

    selected = {str(item) for item in selected_scenario_ids}
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(candidates, start=1):
        candidate_id = str(item.get("id") or "")
        matched_dimensions = _string_list(item.get("matched_dimensions"))
        if not matched_dimensions:
            matched_dimensions = [
                str(key) for key in (item.get("matched_fields") or {}) if str(key)
            ]
        rows.append(
            {
                "rank": rank,
                "id": candidate_id,
                "name": str(item.get("name") or candidate_id),
                "score": float(item.get("score", 0.0) or 0.0),
                "fit_score": float(item.get("fit_score", 0.0) or 0.0),
                "evidence_completeness": float(
                    item.get("evidence_completeness", 0.0) or 0.0
                ),
                "evaluation": str(item.get("evaluation") or "未评价"),
                "matched_dimensions": matched_dimensions,
                "strengths": "、".join(matched_dimensions) or "暂无明确命中维度",
                "gaps": _string_list(item.get("gaps")),
                "conflicts": _string_list(item.get("conflicts")),
                "low_score_reasons": _string_list(item.get("low_score_reasons")),
                "selection_state": "已选中" if candidate_id in selected else "候选待确认",
                "review_status": "待复核",
                "human_decision": "",
                "human_notes": "",
            }
        )
    return rows


def _field_comparison_rows(
    ir: dict[str, Any],
    scenario_rows: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources: list[tuple[dict[str, Any], str, dict[str, Any], int | None]] = []
    for rank, item in enumerate(scenario_rows[:2], start=1):
        sources.append((item.get("record") or {}, "场景库候选", item, rank))
    for evaluation in evaluations:
        scenario = evaluation.get("scenario")
        if isinstance(scenario, dict):
            match_like = {
                "id": evaluation.get("scenario_id"),
                "name": scenario.get("name"),
                "score": evaluation.get("score", 0.0),
                "evaluation": evaluation.get("evaluation", "未评价"),
                "dimension_scores": evaluation.get("dimension_scores", {}),
                "matched_fields": {},
                "low_score_reasons": evaluation.get("low_score_reasons", []),
                "gaps": evaluation.get("gaps", []),
                "conflicts": evaluation.get("conflicts", []),
                "source": "evaluate_scenario_fit",
            }
            sources.append((scenario, "指定 SC 评估", match_like, None))

    ir_code = str(ir.get("code") or ir.get("id") or "")
    for scenario, source_type, match, candidate_rank in sources:
        dimension_scores = match.get("dimension_scores") or {}
        matched_fields = match.get("matched_fields") or {}
        candidate_id = str(match.get("id") or scenario.get("id") or "")
        candidate_name = str(scenario.get("name") or "")
        candidate_score = float(match.get("score", 0.0) or 0.0)
        candidate_label = (
            f"#{candidate_rank} {candidate_id} {candidate_name}（{candidate_score:.2f}）"
            if candidate_rank is not None
            else f"指定 SC {candidate_id} {candidate_name}（{candidate_score:.2f}）"
        )
        for field_name, ir_field, sc_field, dimension, spec_rule in _SC_FIELD_COMPARISON:
            ai_value = scenario.get(sc_field)
            if ir_field is None:
                basis = "当前场景记录与 active_business_spec 校验"
            elif _has_value(ir.get(ir_field)):
                basis = f"IR {ir_field} → SC {sc_field}"
            else:
                basis = (
                    f"IR {ir_field} 未提供；依据标题/描述、Who、What、约束、DFX 等"
                    f"已提供内容推断 SC {sc_field}"
                )
            evidence = _string_list(matched_fields.get(dimension))
            detail = dimension_scores.get(dimension) or {}
            score = float(detail.get("score", 0.0) or 0.0)
            if not ai_value:
                ai_consistency_hint = "缺失"
            elif detail.get("level") == "not_provided":
                ai_consistency_hint = "推断/待人工"
            elif score >= 0.70:
                ai_consistency_hint = "初步一致"
            elif score >= 0.45:
                ai_consistency_hint = "部分一致"
            elif dimension in dimension_scores:
                ai_consistency_hint = "不一致/需核对"
            else:
                ai_consistency_hint = "待人工"
            rows.append(
                {
                    "ir_code": ir_code,
                    "candidate_rank": candidate_rank if candidate_rank is not None else "指定",
                    "candidate_score": f"{candidate_score:.2f}",
                    "candidate_evaluation": str(match.get("evaluation") or "未评价"),
                    "candidate_label": candidate_label,
                    "sc_id": candidate_id,
                    "sc_name": candidate_name,
                    "source_type": source_type,
                    "field_name": field_name,
                    "ai_value": _display_value(ai_value),
                    "analysis_basis": basis,
                    "skill": "IR→SC 场景匹配",
                    "spec_rule": spec_rule,
                    "method": "字段覆盖 + 同义词扩展 + 加权评分 + 冲突检查",
                    "human_value": "",
                    "consistency": "待人工确认",
                    "review_status": "待复核",
                    "human_decision": "",
                    "human_notes": "",
                    "ai_consistency_hint": ai_consistency_hint,
                    "consistency_reason": (
                        f"人工分析字段值为空；AI预判：{ai_consistency_hint}。"
                        f"{str(detail.get('reason') or '') or '请人工填写分析值后复核。'}"
                    ),
                    "dimension_score": f"{score:.2f}",
                    "evidence": "、".join(evidence),
                    "low_score_reason": "；".join(_string_list(match.get("low_score_reasons"))),
                }
            )
    return rows


def _human_review_matrix_rows(field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a spreadsheet-friendly side-by-side view for the two SC candidates."""

    candidate_rows = [
        row
        for row in field_rows
        if str(row.get("source_type") or "") == "场景库候选"
        and str(row.get("candidate_rank") or "") in {"1", "2"}
    ]
    by_field: dict[str, dict[str, dict[str, Any]]] = {}
    field_order: list[str] = []
    for row in candidate_rows:
        field_name = str(row.get("field_name") or "")
        if not field_name:
            continue
        if field_name not in by_field:
            by_field[field_name] = {}
            field_order.append(field_name)
        by_field[field_name][str(row.get("candidate_rank"))] = row

    rows: list[dict[str, Any]] = []
    for field_name in field_order:
        candidates = by_field[field_name]
        first = candidates.get("1") or candidates.get("2") or {}
        matrix_row: dict[str, Any] = {
            "ir_code": first.get("ir_code", ""),
            "field_name": field_name,
            "analysis_basis": first.get("analysis_basis", ""),
            "skill": first.get("skill", ""),
            "spec_rule": first.get("spec_rule", ""),
            "method": first.get("method", ""),
        }
        for rank in (1, 2):
            candidate = candidates.get(str(rank), {})
            prefix = f"candidate_{rank}_"
            for column, source_key in (
                ("id", "sc_id"),
                ("label", "candidate_label"),
                ("score", "candidate_score"),
                ("evaluation", "candidate_evaluation"),
                ("ai_value", "ai_value"),
                ("dimension_score", "dimension_score"),
                ("evidence", "evidence"),
                ("low_score_reason", "low_score_reason"),
                ("ai_consistency_hint", "ai_consistency_hint"),
                ("human_value", "human_value"),
                ("consistency", "consistency"),
                ("review_status", "review_status"),
                ("human_decision", "human_decision"),
                ("human_notes", "human_notes"),
            ):
                matrix_row[prefix + column] = candidate.get(source_key, "")
        rows.append(matrix_row)
    return rows


def _match_summary_rows(
    scenario_matches: list[dict[str, Any]],
    use_case_matches: list[dict[str, Any]],
    matching: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_type, matches in (("SC", scenario_matches), ("UC", use_case_matches)):
        for rank, item in enumerate(matches, start=1):
            dimension_scores = item.get("dimension_scores") or {}
            row: dict[str, Any] = {
                "record_type": record_type,
                "record_id": item.get("id", ""),
                "record_name": item.get("name", ""),
                "parent_scenario_id": item.get("parent_scenario_id", ""),
                "rank": rank,
                "total_score": f"{float(item.get('score', 0.0) or 0.0):.4f}",
                "fit_score": f"{float(item.get('fit_score', 0.0) or 0.0):.4f}",
                "evidence_completeness": f"{float(item.get('evidence_completeness', 0.0) or 0.0):.4f}",
                "base_score": f"{float(item.get('base_score', 0.0) or 0.0):.4f}",
                "evaluation": item.get("evaluation", "未评价"),
                "confidence_label": matching.get("confidence_label", "未评价"),
                "low_score_reasons": "；".join(_string_list(item.get("low_score_reasons"))),
                "gaps": "；".join(_string_list(item.get("gaps"))),
                "conflicts": "；".join(_string_list(item.get("conflicts"))),
                "source": "、".join(_string_list(item.get("source"))),
            }
            for dimension, detail in dimension_scores.items():
                row[f"{dimension}_score"] = f"{float(detail.get('score', 0.0) or 0.0):.4f}"
                row[f"{dimension}_weighted"] = f"{float(detail.get('weighted_score', 0.0) or 0.0):.4f}"
                row[f"{dimension}_level"] = str(detail.get("level") or "")
                row[f"{dimension}_evidence"] = "、".join(
                    _string_list(detail.get("evidence"))
                )
                row[f"{dimension}_reason"] = str(detail.get("reason") or "")
            rows.append(row)
    return rows


def _scenario_fit_csv_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        scenario = evaluation.get("scenario") or {}
        row: dict[str, Any] = {
            "scenario_id": evaluation.get("scenario_id") or scenario.get("id", ""),
            "scenario_name": scenario.get("name", ""),
            "score": f"{float(evaluation.get('score', 0.0) or 0.0):.4f}",
            "fit_score": f"{float(evaluation.get('fit_score', 0.0) or 0.0):.4f}",
            "evidence_completeness": f"{float(evaluation.get('evidence_completeness', 0.0) or 0.0):.4f}",
            "evaluation": evaluation.get("evaluation", "未评价"),
            "confidence_label": evaluation.get("confidence_label", "未评价"),
            "low_score_reasons": "；".join(_string_list(evaluation.get("low_score_reasons"))),
            "gaps": "；".join(_string_list(evaluation.get("gaps"))),
            "conflicts": "；".join(_string_list(evaluation.get("conflicts"))),
        }
        for dimension, detail in (evaluation.get("dimension_scores") or {}).items():
            if not isinstance(detail, dict):
                continue
            row[f"{dimension}_score"] = f"{float(detail.get('score', 0.0) or 0.0):.4f}"
            row[f"{dimension}_weighted"] = f"{float(detail.get('weighted_score', 0.0) or 0.0):.4f}"
            row[f"{dimension}_level"] = str(detail.get("level") or "")
            row[f"{dimension}_evidence"] = "、".join(
                _string_list(detail.get("evidence"))
            )
            row[f"{dimension}_reason"] = str(detail.get("reason") or "")
        rows.append(row)
    return rows


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
    evaluations = _scenario_fit_evaluations(result)
    matching = _matching_summary(result)
    ir = _ir_payload(result, evaluations)
    review_scenario_matches = _review_top_scenario_rows(
        scenario_matches,
        ir=ir,
        library=library,
    )
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
        "matching": matching,
        "review": {
            "candidate_limit": 2,
            "top_scenario_candidates": review_scenario_matches,
            "candidate_summary": _review_candidate_summary_rows(
                review_scenario_matches,
                selected_scenario_ids,
            ),
            "editable_columns": [
                "human_value",
                "consistency",
                "review_status",
                "human_decision",
                "human_notes",
            ],
            "allowed_consistency": list(_REVIEW_CONSISTENCY_VALUES),
            "allowed_review_status": list(_REVIEW_STATUS_VALUES),
            "allowed_human_decision": list(_REVIEW_DECISION_VALUES),
        },
        "field_comparison": _field_comparison_rows(ir, review_scenario_matches, evaluations),
        "evaluations": {"scenario_fit": evaluations},
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


def _dimension_lines(item: dict[str, Any], *, indent: str = "  ") -> list[str]:
    lines: list[str] = []
    dimension_scores = item.get("dimension_scores") or {}
    for label, detail in dimension_scores.items():
        if not isinstance(detail, dict):
            continue
        evidence = "、".join(_string_list(detail.get("evidence"))) or "无"
        lines.append(
            f"{indent}- {label}：{float(detail.get('score', 0.0) or 0.0):.2f} "
            f"× {float(detail.get('weight', 0.0) or 0.0):.2f} = "
            f"{float(detail.get('weighted_score', 0.0) or 0.0):.2f} "
            f"（{detail.get('level', '未评价')}；证据：{evidence}）"
        )
        if detail.get("reason"):
            lines.append(f"{indent}  说明：{detail['reason']}")
    return lines


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def render_human_review_text(report: dict[str, Any]) -> str:
    """Render a compact review workspace for the TUI and other text clients."""

    review = report.get("review") or {}
    candidates = review.get("candidate_summary") or []
    field_rows = report.get("field_comparison") or []
    lines = [
        "人工复核工作区",
        "先看候选总览，再核对字段；完整可编辑内容请打开 evaluation/human_review_template.csv。",
        "可填写列：human_value / consistency / review_status / human_decision / human_notes",
        "一致性：一致、部分一致、不一致、无法判断；状态：待复核、已确认、需新增、需修改。",
        "",
        "候选总览",
    ]
    if not candidates:
        lines.append("- 本轮没有可供人工复核的 SC 候选。")
    for item in candidates:
        lines.append(
            f"[{item.get('rank', '-')}] {item.get('id', '-')} {item.get('name', '')}"
            f" | 决策分 {float(item.get('score', 0.0) or 0.0):.2f}"
            f" | 可用证据 {float(item.get('fit_score', 0.0) or 0.0):.2f}"
            f" | 证据完整 {float(item.get('evidence_completeness', 0.0) or 0.0):.2f}"
        )
        lines.append(
            f"  AI评价：{item.get('evaluation', '未评价')} | 状态：{item.get('selection_state', '候选待确认')}"
        )
        lines.append(f"  主要命中：{item.get('strengths') or '暂无'}")
        lines.append(
            f"  缺口：{'；'.join(_string_list(item.get('gaps'))) or '无'} | "
            f"冲突：{'；'.join(_string_list(item.get('conflicts'))) or '无'}"
        )

    lines.extend(["", "字段明细（人工填写在 CSV 中完成）"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in field_rows:
        grouped.setdefault(str(row.get("candidate_label") or "候选"), []).append(row)
    if not grouped:
        lines.append("- 本轮没有可生成的字段复核行。")
    for label, rows in grouped.items():
        lines.append(f"- {label}")
        for row in rows:
            lines.append(
                f"  {row.get('field_name', '-')}: AI值={row.get('ai_value') or '空'} | "
                f"AI预判={row.get('ai_consistency_hint') or '待人工'} | "
                f"人工值={row.get('human_value') or '待填写'} | "
                f"一致性={row.get('consistency') or '待人工确认'} | "
                f"状态={row.get('review_status') or '待复核'}"
            )
    return "\n".join(lines)


def _render_markdown_report_legacy(report: dict[str, Any]) -> str:
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


def render_markdown_report(report: dict[str, Any]) -> str:
    resolution = report.get("resolution") or {}
    matching = report.get("matching") or {}
    lines = [
        "# IR / SC / UC 分析报告",
        "",
        f"- 状态：{resolution.get('status', '未结构化')}",
        f"- 决策：{resolution.get('decision', '-')}",
        f"- 需求理解：{resolution.get('request_summary', '-')}",
        f"- 匹配工具：{matching.get('tool') or '未调用'}",
        f"- 决策分：{float(matching.get('confidence', 0.0) or 0.0):.2f}",
        f"- IR 证据完整度：{float(matching.get('evidence_completeness', 0.0) or 0.0):.2f}"
        + (
            f"（可用：{'、'.join(_string_list(matching.get('supplied_dimensions')))})"
            if _string_list(matching.get('supplied_dimensions'))
            else ""
        ),
        f"- 置信度评价：{matching.get('confidence_label', '未评价')}",
        "",
        "## 置信度原因",
        "",
    ]
    confidence_reasons = _string_list(matching.get("confidence_reasons"))
    lines.extend(f"- {reason}" for reason in confidence_reasons or ["无额外说明。"])
    if matching.get("ambiguous"):
        lines.append(
            f"- 候选分差：{float(matching.get('score_margin', 0.0) or 0.0):.2f}，存在歧义。"
        )

    lines.extend(["", "## 场景 SC", ""])
    scenario_matches = report.get("scenarios", {}).get("matches", [])
    if scenario_matches:
        for item in scenario_matches:
            lines.append(
                f"- **{item['id']} {item['name']}**（分数 {float(item.get('score', 0.0)):.2f}）"
            )
            lines.append(
                f"  - 评分：决策分 {float(item.get('score', 0.0) or 0.0):.2f}；"
                f"可用证据匹配度 {float(item.get('fit_score', 0.0) or 0.0):.2f}；"
                f"证据完整度 {float(item.get('evidence_completeness', 0.0) or 0.0):.2f}"
            )
            lines.append(f"  - 评价：{item.get('evaluation', '未评价')}")
            lines.extend(_dimension_lines(item))
            lines.append(f"  - 命中部分：{_evidence_text(item.get('matched_fields'))}")
            if item.get("low_score_reasons"):
                lines.append(f"  - 低分原因：{'；'.join(item['low_score_reasons'])}")
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
            lines.append(
                f"  - 评分：决策分 {float(item.get('score', 0.0) or 0.0):.2f}；"
                f"可用证据匹配度 {float(item.get('fit_score', 0.0) or 0.0):.2f}；"
                f"证据完整度 {float(item.get('evidence_completeness', 0.0) or 0.0):.2f}"
            )
            lines.append(f"  - 评价：{item.get('evaluation', '未评价')}")
            lines.extend(_dimension_lines(item))
            lines.append(f"  - 命中部分：{_evidence_text(item.get('matched_fields'))}")
            if item.get("low_score_reasons"):
                lines.append(f"  - 低分原因：{'；'.join(item['low_score_reasons'])}")
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

    lines.extend(["", "## 指定 SC 符合度评估", ""])
    evaluations = report.get("evaluations", {}).get("scenario_fit", [])
    if evaluations:
        for evaluation in evaluations:
            scenario = evaluation.get("scenario") or {}
            lines.append(
                f"- **{evaluation.get('scenario_id', scenario.get('id', '-'))} "
                f"{scenario.get('name', '')}**：{float(evaluation.get('score', 0.0) or 0.0):.2f} "
                f"（{evaluation.get('evaluation', '未评价')}）"
            )
            lines.extend(_dimension_lines(evaluation))
            reasons = _string_list(evaluation.get("low_score_reasons"))
            if reasons:
                lines.append(f"  - 原因：{'；'.join(reasons)}")
    else:
        lines.append("本轮没有调用指定 SC 符合度评估。")

    lines.extend(["", "## 人工复核字段表", ""])
    review = report.get("review", {})
    review_candidates = review.get("top_scenario_candidates", []) if isinstance(review, dict) else []
    if review_candidates:
        lines.append(
            "默认展示评分最高的两个 SC 候选；即使未达到复用阈值，也用于人工对比。"
        )
        lines.append(
            "建议先看候选总览，再看分候选字段明细。人工填写请修改 CSV 中的 "
            "human_value、consistency、review_status、human_decision、human_notes；"
            "回填只更新复核报告，不会自动写入场景库。"
        )
        lines.append("")
        lines.extend(
            [
                "### 候选总览",
                "",
                "| 优先级 | SC | 决策分 | 可用证据匹配度 | 证据完整度 | AI评价 | 主要命中 | 缺口 | 冲突 | 选择状态 | 复核状态 |",
                "|---:|---|---:|---:|---:|---|---|---|---|---|---|",
            ]
        )
        candidate_summary = review.get("candidate_summary", []) if isinstance(review, dict) else []
        for item in candidate_summary:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        item.get("rank", ""),
                        f"{item.get('id', '')} {item.get('name', '')}",
                        f"{float(item.get('score', 0.0) or 0.0):.2f}",
                        f"{float(item.get('fit_score', 0.0) or 0.0):.2f}",
                        f"{float(item.get('evidence_completeness', 0.0) or 0.0):.2f}",
                        item.get("evaluation", "未评价"),
                        item.get("strengths", ""),
                        "；".join(_string_list(item.get("gaps"))) or "无",
                        "；".join(_string_list(item.get("conflicts"))) or "无",
                        item.get("selection_state", "候选待确认"),
                        item.get("review_status", "待复核"),
                    )
                )
                + " |"
            )
        lines.append("")
    field_rows = report.get("field_comparison", [])
    if field_rows:
        lines.append("### 分候选字段明细")
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for row in field_rows:
            grouped_rows.setdefault(str(row.get("candidate_label") or "候选"), []).append(row)
        for label, rows in grouped_rows.items():
            lines.extend(
                [
                    "",
                    f"#### {_markdown_cell(label)}",
                    "",
                    "| 字段 | AI/候选字段值 | 维度分 | 分析依据 | Spec规则 | AI预判 | 人工分析字段值 | 一致性 | 状态 | 人工结论 | 备注 |",
                    "|---|---|---:|---|---|---|---|---|---|---|---|",
                ]
            )
            for row in rows:
                lines.append(
                    "| "
                    + " | ".join(
                        _markdown_cell(row.get(key))
                        for key in (
                            "field_name",
                            "ai_value",
                            "dimension_score",
                            "analysis_basis",
                            "spec_rule",
                            "ai_consistency_hint",
                            "human_value",
                            "consistency",
                            "review_status",
                            "human_decision",
                            "human_notes",
                        )
                    )
                    + " |"
                )
    else:
        lines.append("本轮没有可生成的字段复核行。")

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


def _write_csv(path: Path, rows: list[dict[str, Any]], base_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if str(key) not in base_fields
        }
    )
    fieldnames = [*base_fields, *extra_fields]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _review_choice(
    value: Any,
    *,
    allowed: tuple[str, ...],
    field_name: str,
    row_number: int,
) -> str:
    choice = str(value or "").strip()
    if not choice:
        return ""
    if choice not in allowed:
        raise ValueError(
            f"复核 CSV 第 {row_number} 行的 {field_name}={choice!r} 无效；"
            f"可选值：{'、'.join(allowed)}"
        )
    return choice


def apply_human_review(
    report: dict[str, Any],
    review_csv: str | Path,
) -> dict[str, Any]:
    """Merge human-edited review columns into a report without touching the library."""

    target = deepcopy(report)
    field_rows = target.get("field_comparison")
    if not isinstance(field_rows, list) or not field_rows:
        raise ValueError("报告中没有可回填的 field_comparison 字段表。")

    csv_path = Path(review_csv).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {str(item) for item in (reader.fieldnames or [])}
        required = {"sc_id", "field_name", "human_value", "consistency"}
        missing = sorted(required - headers)
        if missing:
            raise ValueError(f"复核 CSV 缺少列：{', '.join(missing)}")
        input_rows = list(reader)

    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    fallback_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for report_row in field_rows:
        if not isinstance(report_row, dict):
            continue
        key = (
            str(report_row.get("sc_id") or ""),
            str(report_row.get("field_name") or ""),
            str(report_row.get("source_type") or ""),
            str(report_row.get("candidate_rank") or ""),
        )
        index[key] = report_row
        fallback_key = (key[0], key[1], key[3])
        fallback_index.setdefault(fallback_key, []).append(report_row)

    touched_candidates: dict[str, dict[str, str]] = {}
    updated_rows = 0
    for row_number, input_row in enumerate(input_rows, start=2):
        if not any(str(value or "").strip() for value in input_row.values()):
            continue
        sc_id = str(input_row.get("sc_id") or "").strip()
        field_name = str(input_row.get("field_name") or "").strip()
        source_type = str(input_row.get("source_type") or "").strip()
        candidate_rank = str(input_row.get("candidate_rank") or "").strip()
        exact_key = (sc_id, field_name, source_type, candidate_rank)
        report_row = index.get(exact_key)
        if report_row is None:
            options = fallback_index.get((sc_id, field_name, candidate_rank), [])
            if len(options) == 1:
                report_row = options[0]
        if report_row is None:
            raise ValueError(
                f"复核 CSV 第 {row_number} 行找不到对应字段："
                f"sc_id={sc_id!r}, field_name={field_name!r}, candidate_rank={candidate_rank!r}"
            )

        consistency = _review_choice(
            input_row.get("consistency"),
            allowed=_REVIEW_CONSISTENCY_VALUES,
            field_name="consistency",
            row_number=row_number,
        )
        review_status = _review_choice(
            input_row.get("review_status"),
            allowed=_REVIEW_STATUS_VALUES,
            field_name="review_status",
            row_number=row_number,
        )
        human_decision = _review_choice(
            input_row.get("human_decision"),
            allowed=_REVIEW_DECISION_VALUES,
            field_name="human_decision",
            row_number=row_number,
        )
        row_changed = False
        if "human_value" in input_row:
            human_value = str(input_row.get("human_value") or "").strip()
            if human_value != str(report_row.get("human_value") or ""):
                report_row["human_value"] = human_value
                row_changed = True
        if consistency and consistency != str(report_row.get("consistency") or ""):
            report_row["consistency"] = consistency
            report_row["consistency_reason"] = (
                f"人工复核结论：{consistency}。"
                + (
                    f"人工备注：{str(input_row.get('human_notes') or '').strip()}"
                    if str(input_row.get("human_notes") or "").strip()
                    else ""
                )
            )
            row_changed = True
        if review_status and review_status != str(report_row.get("review_status") or ""):
            report_row["review_status"] = review_status
            row_changed = True
        if human_decision and human_decision != str(report_row.get("human_decision") or ""):
            report_row["human_decision"] = human_decision
            row_changed = True
        if "human_notes" in input_row:
            human_notes = str(input_row.get("human_notes") or "").strip()
            if human_notes != str(report_row.get("human_notes") or ""):
                report_row["human_notes"] = human_notes
                row_changed = True
        if row_changed:
            touched_candidates[sc_id] = {
                "review_status": report_row.get("review_status", "待复核"),
                "human_decision": report_row.get("human_decision", ""),
                "human_notes": report_row.get("human_notes", ""),
            }
            updated_rows += 1

    review = target.setdefault("review", {})
    if not isinstance(review, dict):
        review = {}
        target["review"] = review
    for candidate in review.get("candidate_summary", []) or []:
        if not isinstance(candidate, dict):
            continue
        values = touched_candidates.get(str(candidate.get("id") or ""))
        if values:
            candidate.update(values)
    review["reviewed_from_csv"] = str(csv_path.resolve())
    review["reviewed_rows"] = updated_rows
    review["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    return target


def save_reviewed_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Save a human-reviewed report and its refreshed Markdown rendering."""

    target = Path(output_path).expanduser().resolve()
    _write_json(target, report)
    target.with_suffix(".md").write_text(render_markdown_report(report), encoding="utf-8")
    return target


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
    _write_json(
        run_root / "evaluation" / "review_candidates.json",
        report.get("review", {}).get("top_scenario_candidates", []),
    )
    summary_rows = _match_summary_rows(
        report["scenarios"]["matches"],
        report["use_cases"]["matches"],
        report.get("matching", {}),
    )
    _write_csv(
        run_root / "evaluation" / "match_summary.csv",
        summary_rows,
        [
            "record_type",
            "record_id",
            "record_name",
            "parent_scenario_id",
            "rank",
            "total_score",
            "fit_score",
            "evidence_completeness",
            "base_score",
            "evaluation",
            "confidence_label",
            "low_score_reasons",
            "gaps",
            "conflicts",
            "source",
        ],
    )
    _write_csv(
        run_root / "evaluation" / "field_comparison.csv",
        report.get("field_comparison", []),
        [
            "ir_code",
            "candidate_rank",
            "candidate_score",
            "candidate_evaluation",
            "candidate_label",
            "sc_id",
            "sc_name",
            "source_type",
            "field_name",
            "ai_value",
            "analysis_basis",
            "skill",
            "spec_rule",
            "method",
            "human_value",
            "consistency",
            "ai_consistency_hint",
            "consistency_reason",
            "dimension_score",
            "evidence",
            "low_score_reason",
        ],
    )
    _write_csv(
        run_root / "evaluation" / "human_review_template.csv",
        report.get("field_comparison", []),
        [
            "ir_code",
            "candidate_rank",
            "candidate_score",
            "candidate_evaluation",
            "candidate_label",
            "sc_id",
            "sc_name",
            "source_type",
            "field_name",
            "ai_value",
            "analysis_basis",
            "skill",
            "spec_rule",
            "method",
            "human_value",
            "consistency",
            "review_status",
            "human_decision",
            "human_notes",
            "ai_consistency_hint",
            "consistency_reason",
        ],
    )
    _write_csv(
        run_root / "evaluation" / "human_review_matrix.csv",
        _human_review_matrix_rows(report.get("field_comparison", [])),
        [
            "ir_code",
            "field_name",
            "analysis_basis",
            "skill",
            "spec_rule",
            "method",
            "candidate_1_id",
            "candidate_1_label",
            "candidate_1_score",
            "candidate_1_evaluation",
            "candidate_1_ai_value",
            "candidate_1_dimension_score",
            "candidate_1_evidence",
            "candidate_1_low_score_reason",
            "candidate_1_ai_consistency_hint",
            "candidate_1_human_value",
            "candidate_1_consistency",
            "candidate_1_review_status",
            "candidate_1_human_decision",
            "candidate_1_human_notes",
            "candidate_2_id",
            "candidate_2_label",
            "candidate_2_score",
            "candidate_2_evaluation",
            "candidate_2_ai_value",
            "candidate_2_dimension_score",
            "candidate_2_evidence",
            "candidate_2_low_score_reason",
            "candidate_2_ai_consistency_hint",
            "candidate_2_human_value",
            "candidate_2_consistency",
            "candidate_2_review_status",
            "candidate_2_human_decision",
            "candidate_2_human_notes",
        ],
    )
    _write_csv(
        run_root / "evaluation" / "scenario_fit.csv",
        _scenario_fit_csv_rows(report.get("evaluations", {}).get("scenario_fit", [])),
        [
            "scenario_id",
            "scenario_name",
            "score",
            "fit_score",
            "evidence_completeness",
            "evaluation",
            "confidence_label",
            "low_score_reasons",
            "gaps",
            "conflicts",
        ],
    )
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
        "evaluation_dir": str((run_root / "evaluation").resolve()),
        "review_candidates": str((run_root / "evaluation" / "review_candidates.json").resolve()),
        "match_summary_csv": str((run_root / "evaluation" / "match_summary.csv").resolve()),
        "field_comparison_csv": str((run_root / "evaluation" / "field_comparison.csv").resolve()),
        "human_review_template_csv": str(
            (run_root / "evaluation" / "human_review_template.csv").resolve()
        ),
        "human_review_matrix_csv": str(
            (run_root / "evaluation" / "human_review_matrix.csv").resolve()
        ),
        "scenario_fit_csv": str((run_root / "evaluation" / "scenario_fit.csv").resolve()),
    }
    _write_json(run_root / "manifest.json", manifest)
    return (run_root / "result.json").resolve()
