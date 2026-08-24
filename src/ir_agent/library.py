from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from threading import RLock
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .domain import (
    CreateScenarioRequest,
    CreateUseCaseRequest,
    IRMatchResult,
    IRRequirementInput,
    InformationRequirement,
    Scenario,
    ScenarioMatch,
    UseCase,
    UseCaseMatch,
    utc_now,
)


class LibraryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=2, ge=2)
    requirements: list[InformationRequirement] = Field(default_factory=list)
    use_cases: list[UseCase] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)


_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "the",
    "to",
    "with",
    "以及",
    "一个",
    "可以",
    "需要",
    "某",
    "当前",
    "相关",
    "进行",
    "其中",
    "一种",
    "明确",
    "可能",
    "能够",
    "支持",
    "用于",
    "实现",
    "根据",
    "对于",
    "系统",
}

_CJK_RUN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")

_ACTOR_CATEGORIES: dict[str, tuple[str, ...]] = {
    "system": ("本系统", "系统自身", "软件", "进程"),
    "user": ("用户", "客户", "操作员", "人员"),
    "maintenance": ("运维", "管理员", "维护人员", "管理人员"),
    "external_system": ("外部系统", "周边系统", "第三方系统"),
    "device": ("设备", "部件", "节点", "硬件"),
}

_LIFECYCLE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "normal_service": ("正常服务", "正常运行", "正常工作", "运行时"),
    "maintenance": ("维护", "检修", "维修"),
    "deployment": ("灌装", "部署", "安装", "配置"),
    "upgrade": ("升级", "更新"),
    "retirement": ("退网", "退役", "下线"),
    "fault_recovery": ("故障恢复", "故障", "异常恢复"),
}

_COMPONENT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "type_a": ("类型a", "typea"),
    "type_b": ("类型b", "typeb"),
    "single_node": ("单节点",),
    "multi_node": ("多节点", "集群"),
    "unrestricted": ("不限", "不限制"),
}

_SCENARIO_REUSE_THRESHOLD = 0.45
_SCENARIO_STRONG_THRESHOLD = 0.70
_USE_CASE_REUSE_THRESHOLD = 0.45
_AMBIGUITY_MARGIN_THRESHOLD = 0.08
_CRITICAL_DIMENSIONS_FOR_REUSE = ("Actor", "上下文", "影响因素")


def _configured_threshold(
    rules: dict[str, object],
    key: str,
    default: float,
) -> float:
    try:
        value = float(rules.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if 0.0 <= value <= 1.0 else default


def _configured_categories(
    rules: dict[str, object],
    key: str,
    defaults: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    raw = rules.get(key)
    if not isinstance(raw, dict):
        return defaults
    configured: dict[str, tuple[str, ...]] = dict(defaults)
    changed = False
    for category, terms in raw.items():
        if not isinstance(terms, (list, tuple)):
            continue
        cleaned = tuple(
            str(term).strip()
            for term in terms
            if term is not None and str(term).strip()
        )
        if cleaned:
            category_name = str(category)
            existing = configured.get(category_name, ())
            configured[category_name] = tuple(dict.fromkeys([*existing, *cleaned]))
            changed = True
    return configured if changed else defaults


def _configured_strings(
    rules: dict[str, object],
    key: str,
    defaults: tuple[str, ...],
) -> tuple[str, ...]:
    raw = rules.get(key)
    if not isinstance(raw, (list, tuple)):
        return defaults
    configured = tuple(
        str(item).strip()
        for item in raw
        if item is not None and str(item).strip()
    )
    return configured or defaults


def tokenize(text: str) -> list[str]:
    """Tokenize words, Chinese characters, and short Chinese phrases.

    Keeping single Chinese characters preserves recall for legacy data, while
    adding bi/tri-grams makes an exact phrase stronger evidence than a few
    unrelated shared characters.
    """

    normalized = text.casefold()
    tokens = _TOKEN_PATTERN.findall(normalized)
    for run in _CJK_RUN_PATTERN.findall(normalized):
        for size in (2, 3):
            if len(run) < size:
                continue
            tokens.extend(run[index : index + size] for index in range(len(run) - size + 1))
    return [token for token in tokens if token not in _STOPWORDS]


def _compact(text: str) -> str:
    return re.sub(r"[\s_\-/]+", "", text.casefold())


def _category_hits(text: str, categories: dict[str, tuple[str, ...]]) -> set[str]:
    normalized = _compact(text)
    return {
        category
        for category, terms in categories.items()
        if any(_compact(term) in normalized for term in terms)
    }


def _exclusive_conflict(
    query: str,
    document: str,
    categories: dict[str, tuple[str, ...]],
) -> bool:
    query_categories = _category_hits(query, categories)
    document_categories = _category_hits(document, categories)
    return bool(
        query_categories
        and document_categories
        and query_categories.isdisjoint(document_categories)
    )


def _scenario_conflicts(
    ir: IRRequirementInput,
    scenario: Scenario,
    *,
    actor_categories: dict[str, tuple[str, ...]] | None = None,
    lifecycle_categories: dict[str, tuple[str, ...]] | None = None,
    component_categories: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    actor_categories = actor_categories or _ACTOR_CATEGORIES
    lifecycle_categories = lifecycle_categories or _LIFECYCLE_CATEGORIES
    component_categories = component_categories or _COMPONENT_CATEGORIES
    conflicts: list[str] = []
    if _exclusive_conflict(ir.who or "", scenario.actor, actor_categories):
        conflicts.append("Actor 明确冲突")
    if _exclusive_conflict(ir.when or "", scenario.lifecycle or "", lifecycle_categories):
        conflicts.append("生命周期明确冲突")

    scenario_values = " ".join(
        [factor.name for factor in scenario.influence_factors]
        + [
            value
            for factor in scenario.influence_factors
            for value in factor.selected_values
        ]
        + scenario.affected_components
        + scenario.constraints
    )
    requirement_values = " ".join(
        [ir.where or "", ir.description, *ir.constraints, *ir.tags]
    )
    if _exclusive_conflict(requirement_values, scenario_values, component_categories):
        conflicts.append("影响部件或范围明确冲突")
    return conflicts


def _coverage(query: str, document: str) -> tuple[float, set[str]]:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0, set()
    document_terms = set(tokenize(document))
    matched = query_terms & document_terms
    total_weight = sum(_token_weight(token) for token in query_terms)
    matched_weight = sum(_token_weight(token) for token in matched)
    return (matched_weight / total_weight if total_weight else 0.0), matched


def _token_weight(token: str) -> float:
    if re.fullmatch(r"[\u4e00-\u9fff]+", token):
        if len(token) == 1:
            return 0.5
        return float(min(len(token), 3))
    return 1.0


class ScenarioLibrary:
    """File-backed IR/Scenario/UC library with deterministic matching boundaries.

    The legacy format stores IR, scenarios, and use cases in one JSON file.
    A directory path or an explicit ``use_case_path`` enables split storage:
    ``scenarios.json`` plus ``uc/use_cases.json`` (or the configured UC path).
    """

    def __init__(self, path: str | Path, *, use_case_path: str | Path | None = None):
        raw_path = Path(path)
        is_directory = (raw_path.exists() and raw_path.is_dir()) or (
            not raw_path.exists() and not raw_path.suffix
        )
        if is_directory:
            self.root = raw_path
            self.path = raw_path / "scenarios.json"
            self.use_case_path = (
                Path(use_case_path)
                if use_case_path
                else raw_path / "uc" / "use_cases.json"
            )
        else:
            self.root = raw_path.parent
            self.path = raw_path
            self.use_case_path = Path(use_case_path) if use_case_path else None
        if self.use_case_path is not None and self.use_case_path.resolve() == self.path.resolve():
            self.use_case_path = None
        self._lock = RLock()
        self._matching_rules: dict[str, object] = {}
        self._ensure_exists()

    def configure_matching(self, rules: dict[str, object] | None) -> None:
        """Set business-domain matching rules without changing library data."""

        with self._lock:
            self._matching_rules = deepcopy(rules) if isinstance(rules, dict) else {}

    def matching_rules(self) -> dict[str, object]:
        """Return a defensive copy of the active matching rules."""

        with self._lock:
            return deepcopy(self._matching_rules)

    def _ensure_exists(self) -> None:
        if self.path.exists():
            if self.use_case_path is not None and not self.use_case_path.exists():
                try:
                    self._atomic_write(self._read())
                except ValueError:
                    # Preserve the previous lazy validation behavior for a
                    # malformed existing file; the normal read path reports it.
                    pass
            return
        self._atomic_write(LibraryDocument())

    def _read(self) -> LibraryDocument:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            document = LibraryDocument.model_validate(_migrate_payload(payload))
            if self.use_case_path is not None and self.use_case_path.exists():
                document = document.model_copy(update={"use_cases": self._read_use_cases()})
            return document
        except FileNotFoundError:
            self._atomic_write(LibraryDocument())
            return LibraryDocument()
        except json.JSONDecodeError as exc:
            raise ValueError(f"Scenario library is not valid JSON: {self.path}") from exc

    def _atomic_write(self, document: LibraryDocument) -> None:
        if self.use_case_path is None:
            self._atomic_write_payload(self.path, document.model_dump(mode="json"))
            return

        scenario_document = document.model_copy(update={"use_cases": []})
        self._atomic_write_payload(self.path, scenario_document.model_dump(mode="json"))
        self._atomic_write_payload(
            self.use_case_path,
            {
                "version": document.version,
                "use_cases": [item.model_dump(mode="json") for item in document.use_cases],
            },
        )

    def _atomic_write_payload(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _read_use_cases(self) -> list[UseCase]:
        if self.use_case_path is None:
            return []
        try:
            payload = json.loads(self.use_case_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            raise ValueError(f"Use-case library is not valid JSON: {self.use_case_path}") from exc

        if isinstance(payload, list):
            payload = {"version": 2, "use_cases": payload}
        if not isinstance(payload, dict):
            raise ValueError(
                "Use-case library must be a JSON object or list: "
                f"{self.use_case_path}"
            )
        payload.setdefault("version", 2)
        payload.setdefault("use_cases", [])
        return LibraryDocument.model_validate(payload).use_cases

    def document(self) -> LibraryDocument:
        with self._lock:
            return self._read()

    def list_requirements(self) -> list[InformationRequirement]:
        return self.document().requirements

    def get_requirement(self, requirement_id: str) -> InformationRequirement:
        for requirement in self.list_requirements():
            if requirement.id == requirement_id or requirement.code == requirement_id:
                return requirement
        raise KeyError(f"Unknown IR: {requirement_id}")

    def save_requirement(self, request: IRRequirementInput) -> InformationRequirement:
        with self._lock:
            document = self._read()
            requirement_id = request.code or f"IR-DRAFT-{uuid4().hex[:8].upper()}"
            now = utc_now()
            for index, existing in enumerate(document.requirements):
                matches_id = existing.id == requirement_id
                matches_code = request.code is not None and existing.code == request.code
                if not matches_id and not matches_code:
                    continue
                updated = InformationRequirement(
                    **request.model_dump(),
                    id=existing.id,
                    created_at=existing.created_at,
                    updated_at=now,
                )
                document.requirements[index] = updated
                self._atomic_write(document)
                return updated

            created = InformationRequirement(
                **request.model_dump(),
                id=requirement_id,
                created_at=now,
                updated_at=now,
            )
            document.requirements.append(created)
            self._atomic_write(document)
            return created

    def list_use_cases(self) -> list[UseCase]:
        return self.document().use_cases

    def get_use_case(self, use_case_id: str) -> UseCase:
        for use_case in self.list_use_cases():
            if use_case.id == use_case_id:
                return use_case
        raise KeyError(f"Unknown use case: {use_case_id}")

    def list_scenarios(self) -> list[Scenario]:
        return self.document().scenarios

    def get_scenario(self, scenario_id: str) -> Scenario:
        for scenario in self.list_scenarios():
            if scenario.id == scenario_id:
                return scenario
        raise KeyError(f"Unknown scenario: {scenario_id}")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[ScenarioMatch]:
        if not query.strip():
            return []
        _validate_search_limits(top_k, min_score)
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        matches: list[ScenarioMatch] = []
        for scenario in self.list_scenarios():
            searchable_text = " ".join(
                [
                    scenario.name,
                    scenario.description,
                    scenario.ir_intent,
                    scenario.business_goal or "",
                    *scenario.actions,
                    *scenario.tags,
                ]
            )
            coverage, matched = _coverage(query, searchable_text)
            if not matched:
                continue

            name_coverage, _ = _coverage(query, scenario.name)
            tag_coverage, _ = _coverage(query, " ".join(scenario.tags))
            score = min(1.0, 0.65 * coverage + 0.25 * name_coverage + 0.10 * tag_coverage)
            if score >= min_score:
                matches.append(
                    ScenarioMatch(
                        scenario=scenario,
                        score=round(score, 4),
                        matched_terms=sorted(matched),
                    )
                )

        matches.sort(key=lambda item: (-item.score, item.scenario.name))
        return matches[:top_k]

    def search_use_cases(
        self,
        query: str,
        *,
        scenario_id: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[UseCaseMatch]:
        if not query.strip():
            return []
        _validate_search_limits(top_k, min_score)
        allowed_use_case_ids: set[str] | None = None
        if scenario_id:
            scenarios = {scenario.id: scenario for scenario in self.list_scenarios()}
            if scenario_id not in scenarios:
                raise ValueError(f"Unknown scenario id: {scenario_id}")
            allowed_use_case_ids = set(scenarios[scenario_id].use_case_ids)
        query_terms = set(tokenize(query))
        matches: list[UseCaseMatch] = []
        for use_case in self.list_use_cases():
            if allowed_use_case_ids is not None and use_case.id not in allowed_use_case_ids:
                continue
            document = " ".join(
                [
                    use_case.name,
                    use_case.description,
                    use_case.actor,
                    use_case.trigger_event,
                    use_case.success_guarantee,
                    use_case.minimum_guarantee,
                    *use_case.preconditions,
                    *use_case.main_success_scenario,
                    *use_case.extension_scenarios,
                    *use_case.constraints,
                    *use_case.dfx,
                    *use_case.tags,
                ]
            )
            coverage, matched = _coverage(query, document)
            if not matched:
                continue
            name_coverage, _ = _coverage(query, use_case.name)
            score = min(1.0, 0.8 * coverage + 0.2 * name_coverage)
            if score >= min_score:
                matches.append(
                    UseCaseMatch(
                        use_case=use_case,
                        score=round(score, 4),
                        matched_terms=sorted(matched),
                    )
                )
        matches.sort(key=lambda item: (-item.score, item.use_case.name))
        return matches[:top_k]

    def match_ir(
        self,
        ir: IRRequirementInput,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> IRMatchResult:
        _validate_search_limits(top_k, min_score)
        rules = self.matching_rules()
        scenario_reuse_threshold = _configured_threshold(
            rules, "scenario_reuse_threshold", _SCENARIO_REUSE_THRESHOLD
        )
        scenario_strong_threshold = _configured_threshold(
            rules, "scenario_strong_threshold", _SCENARIO_STRONG_THRESHOLD
        )
        use_case_reuse_threshold = _configured_threshold(
            rules, "use_case_reuse_threshold", _USE_CASE_REUSE_THRESHOLD
        )
        ambiguity_margin_threshold = _configured_threshold(
            rules, "ambiguity_margin", _AMBIGUITY_MARGIN_THRESHOLD
        )
        actor_categories = _configured_categories(
            rules, "actor_categories", _ACTOR_CATEGORIES
        )
        lifecycle_categories = _configured_categories(
            rules, "lifecycle_categories", _LIFECYCLE_CATEGORIES
        )
        component_categories = _configured_categories(
            rules, "component_categories", _COMPONENT_CATEGORIES
        )
        critical_dimensions = _configured_strings(
            rules,
            "critical_dimensions_for_reuse",
            _CRITICAL_DIMENSIONS_FOR_REUSE,
        )
        matches: list[ScenarioMatch] = []
        for scenario in self.list_scenarios():
            intent_query = " ".join(
                [ir.title, ir.description, ir.what or "", ir.why or "", *ir.how]
            )
            intent_document = " ".join(
                [
                    scenario.name,
                    scenario.description,
                    scenario.business_goal or "",
                    scenario.ir_intent,
                    *scenario.actions,
                    *scenario.tags,
                ]
            )
            actor_score, actor_terms = _coverage(ir.who or "", scenario.actor)
            context_score, context_terms = _coverage(
                " ".join([ir.when or "", ir.where or ""]),
                " ".join(
                    [
                        scenario.lifecycle or "",
                        scenario.description,
                        *scenario.affected_components,
                    ]
                ),
            )
            impact_values = [
                value
                for factor in scenario.influence_factors
                for value in [factor.name, *factor.candidate_values, *factor.selected_values]
            ]
            impact_score, impact_terms = _coverage(
                " ".join([ir.where or "", ir.description, *ir.constraints, *ir.how_much]),
                " ".join([*impact_values, *scenario.affected_components]),
            )
            constraint_score, constraint_terms = _coverage(
                " ".join([*ir.constraints, *ir.how_much]),
                " ".join(scenario.constraints),
            )
            intent_score, intent_terms = _coverage(intent_query, intent_document)
            score = min(
                1.0,
                0.55 * intent_score
                + 0.15 * actor_score
                + 0.10 * context_score
                + 0.10 * impact_score
                + 0.10 * constraint_score,
            )

            dimensions: list[str] = []
            gaps: list[str] = []
            for name, dimension_score, supplied in (
                ("目标/行为", intent_score, bool(intent_query.strip())),
                ("Actor", actor_score, bool(ir.who)),
                ("上下文", context_score, bool(ir.when or ir.where)),
                ("影响因素", impact_score, bool(ir.where or ir.constraints or ir.how_much)),
                ("约束", constraint_score, bool(ir.constraints or ir.how_much)),
            ):
                if dimension_score > 0:
                    dimensions.append(name)
                elif supplied:
                    gaps.append(f"{name}未覆盖")

            # A cross-field agreement is stronger evidence than raw character
            # coverage alone, especially for long Chinese IR descriptions.
            score = min(1.0, score + 0.04 * len(dimensions))
            if score < min_score:
                continue

            matched_terms = intent_terms | actor_terms | context_terms | impact_terms | constraint_terms
            conflicts = _scenario_conflicts(
                ir,
                scenario,
                actor_categories=actor_categories,
                lifecycle_categories=lifecycle_categories,
                component_categories=component_categories,
            )
            matches.append(
                ScenarioMatch(
                    scenario=scenario,
                    score=round(score, 4),
                    matched_terms=sorted(matched_terms),
                    matched_dimensions=dimensions,
                    gaps=gaps,
                    conflicts=conflicts,
                )
            )

        matches.sort(key=lambda item: (-item.score, item.scenario.name))
        ranked_matches = matches
        top_score = ranked_matches[0].score if ranked_matches else 0.0
        score_margin = (
            round(max(0.0, ranked_matches[0].score - ranked_matches[1].score), 4)
            if len(ranked_matches) >= 2
            else (1.0 if ranked_matches else 0.0)
        )
        ambiguous = bool(
            len(ranked_matches) >= 2
            and top_score >= scenario_reuse_threshold
            and score_margin < ambiguity_margin_threshold
        )
        matches = matches[:top_k]
        global_use_case_matches = self.search_use_cases(
            ir.search_text(), top_k=top_k, min_score=0.0
        )
        use_case_matches = global_use_case_matches
        missing_fields = ir.missing_fields()
        rationale: list[str] = []
        top_match = matches[0] if matches else None
        linked_matches: list[UseCaseMatch] = []
        critical_gaps: list[str] = []
        if top_match is not None:
            critical_gaps = [
                gap
                for gap in top_match.gaps
                if any(gap.startswith(f"{dimension}未") for dimension in critical_dimensions)
            ]
            # UC is a child of the selected SC. Search inside that parent's
            # children for the decision instead of filtering a global top-k
            # result, which could hide the relevant child UC.
            linked_matches = self.search_use_cases(
                ir.search_text(),
                scenario_id=top_match.scenario.id,
                top_k=top_k,
                min_score=0.0,
            )
            use_case_matches = linked_matches

        if missing_fields:
            decision = "needs_clarification"
            rationale.append(f"IR 缺少 5W2H 字段：{', '.join(missing_fields)}")
        elif not matches or top_score < scenario_reuse_threshold:
            decision = "create_scenario_and_uc"
            rationale.append("没有达到可复用阈值的场景，需要新建场景并派生 UC 草稿。")
        elif top_match is not None and top_match.conflicts:
            decision = "needs_clarification"
            rationale.append("候选场景存在硬冲突，不能自动复用：" + "、".join(top_match.conflicts))
        elif critical_gaps:
            decision = "needs_clarification"
            rationale.append(
                "候选场景未覆盖自动复用所需的关键维度：" + "、".join(critical_gaps)
            )
        elif ambiguous:
            decision = "needs_clarification"
            rationale.append(
                f"最高候选与次高候选分差仅 {score_margin:.2f}，需要人工确认场景边界。"
            )
        else:
            if (
                top_score >= scenario_strong_threshold
                and linked_matches
                and linked_matches[0].score >= use_case_reuse_threshold
            ):
                decision = "reuse_scenario_and_uc"
                rationale.append("场景关键维度一致，且已有 UC 已覆盖主要触发和处理链路。")
            else:
                decision = "reuse_scenario_create_uc"
                rationale.append("场景上下文可以复用，但没有足够匹配的 UC 覆盖完整行为链路。")

        return IRMatchResult(
            ir=ir,
            missing_ir_fields=missing_fields,
            scenario_matches=matches,
            use_case_matches=use_case_matches,
            decision=decision,
            confidence=round(top_score, 4),
            score_margin=score_margin,
            ambiguous=ambiguous,
            rationale=rationale,
        )

    def create(self, request: CreateScenarioRequest) -> Scenario:
        with self._lock:
            document = self._read()
            existing_names = {scenario.name.casefold() for scenario in document.scenarios}
            if request.name.casefold() in existing_names:
                raise ValueError(f"A scenario named {request.name!r} already exists")
            _use_case_parent_scenarios(document)

            now = utc_now()
            scenario = Scenario(
                id=f"SCN-DRAFT-{uuid4().hex[:8].upper()}",
                **request.model_dump(),
                created_at=now,
                updated_at=now,
            )
            document.scenarios.append(scenario)
            self._atomic_write(document)
            return scenario

    def create_use_case(self, request: CreateUseCaseRequest) -> UseCase:
        with self._lock:
            document = self._read()
            existing_names = {use_case.name.casefold() for use_case in document.use_cases}
            if request.name.casefold() in existing_names:
                raise ValueError(f"A use case named {request.name!r} already exists")
            scenario_map = {item.id: index for index, item in enumerate(document.scenarios)}
            if request.scenario_id not in scenario_map:
                raise ValueError(f"Unknown scenario id: {request.scenario_id}")
            _use_case_parent_scenarios(document)

            now = utc_now()
            payload = request.model_dump(exclude={"scenario_id"})
            use_case = UseCase(
                id=f"UC-DRAFT-{uuid4().hex[:8].upper()}",
                **payload,
                created_at=now,
                updated_at=now,
            )
            document.use_cases.append(use_case)
            index = scenario_map[request.scenario_id]
            scenario = document.scenarios[index]
            document.scenarios[index] = scenario.model_copy(
                update={
                    "use_case_ids": list(dict.fromkeys([*scenario.use_case_ids, use_case.id])),
                    "updated_at": now,
                }
            )
            self._atomic_write(document)
            return use_case

    def link_use_cases(self, scenario_id: str, use_case_ids: list[str]) -> Scenario:
        with self._lock:
            document = self._read()
            _ensure_known_ids(use_case_ids, {item.id for item in document.use_cases}, "use case")
            parent_scenarios = _use_case_parent_scenarios(document)
            for index, scenario in enumerate(document.scenarios):
                if scenario.id != scenario_id:
                    continue
                conflicts = {
                    use_case_id: parent_scenarios[use_case_id]
                    for use_case_id in use_case_ids
                    if use_case_id in parent_scenarios
                    and parent_scenarios[use_case_id] != scenario_id
                }
                if conflicts:
                    details = ", ".join(
                        f"{use_case_id}→{parent_scenario_id}"
                        for use_case_id, parent_scenario_id in sorted(conflicts.items())
                    )
                    raise ValueError(f"Use case already belongs to another scenario: {details}")
                updated = scenario.model_copy(
                    update={
                        "use_case_ids": list(
                            dict.fromkeys([*scenario.use_case_ids, *use_case_ids])
                        ),
                        "updated_at": utc_now(),
                    }
                )
                document.scenarios[index] = updated
                self._atomic_write(document)
                return updated
            raise KeyError(f"Unknown scenario: {scenario_id}")


def _validate_search_limits(top_k: int, min_score: float) -> None:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")


def _ensure_known_ids(values: list[str], known: set[str], label: str) -> None:
    missing = [value for value in values if value not in known]
    if missing:
        raise ValueError(f"Unknown {label} ids: {', '.join(missing)}")


def _use_case_parent_scenarios(document: LibraryDocument) -> dict[str, str]:
    """Return the single SC parent for each UC and reject shared children."""

    parents: dict[str, str] = {}
    for scenario in document.scenarios:
        for use_case_id in scenario.use_case_ids:
            parent_scenario_id = parents.get(use_case_id)
            if parent_scenario_id and parent_scenario_id != scenario.id:
                raise ValueError(
                    "UC hierarchy violation: "
                    f"{use_case_id} belongs to both {parent_scenario_id} and {scenario.id}"
                )
            parents[use_case_id] = scenario.id
    known_use_case_ids = {use_case.id for use_case in document.use_cases}
    unknown_use_case_ids = sorted(set(parents) - known_use_case_ids)
    if unknown_use_case_ids:
        raise ValueError(
            "UC hierarchy violation: unknown use cases linked by scenarios: "
            f"{', '.join(unknown_use_case_ids)}"
        )
    orphan_use_case_ids = sorted(known_use_case_ids - set(parents))
    if orphan_use_case_ids:
        raise ValueError(
            "UC hierarchy violation: use cases must have exactly one parent scenario: "
            f"{', '.join(orphan_use_case_ids)}"
        )
    return parents


def _migrate_payload(payload: object) -> object:
    """Read old v1 prototype libraries without discarding user data."""

    if not isinstance(payload, dict) or int(payload.get("version", 1)) >= 2:
        return payload
    migrated = dict(payload)
    migrated["version"] = 2
    migrated.setdefault("requirements", [])
    for use_case in migrated.get("use_cases", []):
        if not isinstance(use_case, dict):
            continue
        description = str(use_case.get("description") or use_case.get("name") or "待补充")
        use_case.setdefault("actor", "待补充")
        use_case.setdefault("preconditions", ["待补充"])
        use_case.setdefault("trigger_event", "待补充")
        use_case.setdefault("success_guarantee", "待补充")
        use_case.setdefault("minimum_guarantee", "待补充")
        use_case.setdefault("main_success_scenario", [description])
        use_case.setdefault("extension_scenarios", [])
        use_case.setdefault("constraints", [])
        use_case.setdefault("dfx", [])
        use_case.setdefault("catalog", None)
        use_case.setdefault("status", "draft")
        use_case.setdefault("workflow_status", _workflow_status_for(use_case.get("status")))
        use_case.setdefault("security_level", None)
        use_case.setdefault("source_ir_ids", [])
    for scenario in migrated.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        metadata = scenario.setdefault("metadata", {})
        scenario.setdefault("category", metadata.get("category", "待补充"))
        scenario.setdefault("actor", metadata.get("actor", "待补充"))
        scenario.setdefault(
            "influence_factors",
            [{"name": "待补充", "candidate_values": [], "selected_values": ["待补充"]}],
        )
        scenario["owner"] = scenario.get("owner") or "待补充"
        scenario.setdefault("business_goal", None)
        scenario.setdefault("actions", [])
        scenario.setdefault("constraints", [])
        scenario.setdefault("dfx", [])
        scenario.setdefault("affected_components", [])
        scenario.setdefault("lifecycle", None)
        scenario.setdefault("workflow_status", _workflow_status_for(scenario.get("status")))
        scenario.setdefault("source_ir_ids", [])
        scenario.setdefault("security_level", None)
        scenario.setdefault("esn_id", None)
        scenario.setdefault("topology_diagram", None)
    return migrated


def _workflow_status_for(status: object) -> str:
    return {
        "draft": "Draft",
        "working": "Inwork",
        "published": "Publish",
        "active": "Publish",
        "archived": "Obsolete",
    }.get(str(status), "Draft")
