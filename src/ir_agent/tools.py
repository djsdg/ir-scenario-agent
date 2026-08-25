from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .domain import (
    CreateScenarioRequest,
    CreateUseCaseRequest,
    IRRequirementInput,
    LinkUseCasesRequest,
    MoveUseCaseRequest,
    TransitionRecordRequest,
    UpdateScenarioRequest,
    UpdateUseCaseRequest,
)
from .library import ScenarioLibrary
from .memory import MemoryStore
from .skills import SkillCatalog
from .specs import SpecCatalog


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchScenariosArgs(StrictArgs):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MatchScenarioArgs(StrictArgs):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MatchUseCaseArgs(StrictArgs):
    query: str = Field(min_length=1)
    scenario_id: str | None = Field(default=None, min_length=1, max_length=120)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MatchIRArgs(StrictArgs):
    ir: IRRequirementInput
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluateScenarioFitArgs(StrictArgs):
    ir: IRRequirementInput
    scenario_id: str = Field(min_length=1, max_length=120)


class DraftScenarioArgs(StrictArgs):
    ir: IRRequirementInput


class DraftUseCasesArgs(StrictArgs):
    ir: IRRequirementInput
    candidate_scenario_ids: list[str] = Field(min_length=1, max_length=50)


class GetIRArgs(StrictArgs):
    ir_id: str = Field(min_length=1, max_length=120)


class GetScenarioArgs(StrictArgs):
    scenario_id: str = Field(min_length=1, max_length=120)


class GetUseCaseArgs(StrictArgs):
    use_case_id: str = Field(min_length=1, max_length=120)


class SearchSkillsArgs(StrictArgs):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class LoadSkillArgs(StrictArgs):
    name: str = Field(min_length=1, max_length=120)


class SearchMemoryArgs(StrictArgs):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class SaveMemoryArgs(StrictArgs):
    content: str = Field(min_length=1, max_length=2_000)
    kind: str = Field(default="fact", min_length=1, max_length=50)
    tags: list[str] = Field(default_factory=list, max_length=20)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    input_model: type[BaseModel] | None
    handler: Callable[[BaseModel | None], dict[str, Any]]
    requires_approval: bool = False

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


def _nullable_string(*, max_length: int = 4_000) -> dict[str, Any]:
    return {"anyOf": [{"type": "string", "maxLength": max_length}, {"type": "null"}]}


def _string_array(*, min_items: int = 0, max_items: int = 100) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": min_items,
        "maxItems": max_items,
        "items": {"type": "string"},
    }


def _nullable_array(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _ir_schema() -> dict[str, Any]:
    properties = {
        "code": _nullable_string(max_length=120),
        "title": {"type": "string", "minLength": 1, "maxLength": 300},
        "description": {"type": "string", "minLength": 1, "maxLength": 12_000},
        "source": _nullable_string(max_length=1_000),
        "owner": _nullable_string(max_length=300),
        "who": _nullable_string(max_length=1_000),
        "when": _nullable_string(max_length=2_000),
        "where": _nullable_string(max_length=2_000),
        "what": _nullable_string(max_length=4_000),
        "how": _string_array(max_items=50),
        "why": _nullable_string(max_length=4_000),
        "how_much": _string_array(max_items=50),
        "constraints": _string_array(),
        "performance": _nullable_string(max_length=2_000),
        "reliability": _nullable_string(max_length=2_000),
        "serviceability": _nullable_string(max_length=2_000),
        "maintainability": _nullable_string(max_length=2_000),
        "sales": _nullable_string(max_length=2_000),
        "delivery_time": _nullable_string(max_length=1_000),
        "tags": _string_array(max_items=50),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _match_ir_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ir": _ir_schema(),
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            "min_score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["ir", "top_k", "min_score"],
    }


def _evaluate_scenario_fit_parameters() -> dict[str, Any]:
    properties = {
        "ir": _ir_schema(),
        "scenario_id": {"type": "string", "minLength": 1, "maxLength": 120},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _save_ir_parameters() -> dict[str, Any]:
    return _ir_schema()


def _search_parameters(description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "description": description},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            "min_score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["query", "top_k", "min_score"],
    }


def _match_use_case_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "description": "UC 的触发、处理步骤、成功保证和异常分支等行为描述。",
            },
            "scenario_id": {
                **_nullable_string(max_length=120),
                "description": "可选的父场景 ID；传 null 表示在整个 UC 库中匹配。",
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            "min_score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["query", "scenario_id", "top_k", "min_score"],
    }


def _single_record_parameters(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: {"type": "string", "description": description}},
        "required": [name],
    }


def _influence_factor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 200},
            "kind": {"type": "string", "enum": ["environment", "activity"]},
            "dimension": {"type": "string", "minLength": 1, "maxLength": 120},
            "candidate_values": _string_array(max_items=50),
            "selected_values": _string_array(min_items=1, max_items=50),
        },
        "required": ["name", "kind", "dimension", "candidate_values", "selected_values"],
    }


def _status_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": ["draft", "working", "published", "active", "archived"],
    }


def _workflow_status_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": ["Draft", "Inwork", "Review", "Publish", "Obsolete"],
    }


def _create_scenario_parameters() -> dict[str, Any]:
    properties = {
        "name": {"type": "string", "minLength": 3, "maxLength": 300},
        "description": {"type": "string", "minLength": 10, "maxLength": 8_000},
        "category": {"type": "string", "minLength": 1, "maxLength": 200},
        "actor": {"type": "string", "minLength": 1, "maxLength": 1_000},
        "influence_factors": {
            "type": "array",
            "minItems": 1,
            "maxItems": 100,
            "items": _influence_factor_schema(),
        },
        "owner": {"type": "string", "minLength": 1, "maxLength": 300},
        "business_goal": {"type": "string", "minLength": 1, "maxLength": 4_000},
        "actions": _string_array(min_items=1),
        "constraints": _string_array(min_items=1),
        "dfx": _string_array(),
        "affected_components": _string_array(),
        "lifecycle": {"type": "string", "minLength": 1, "maxLength": 500},
        "tags": _string_array(max_items=50),
        "source_ir_ids": _string_array(max_items=50),
        "status": _status_schema(),
        "workflow_status": _workflow_status_schema(),
        "security_level": _nullable_string(max_length=100),
        "esn_id": _nullable_string(max_length=120),
        "topology_diagram": _nullable_string(max_length=2_000),
        "ir_intent": {"type": "string", "maxLength": 4_000},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _create_use_case_parameters() -> dict[str, Any]:
    properties = {
        "name": {"type": "string", "minLength": 3, "maxLength": 300},
        "description": {"type": "string", "minLength": 10, "maxLength": 8_000},
        "actor": {"type": "string", "minLength": 1, "maxLength": 1_000},
        "preconditions": _string_array(min_items=1),
        "trigger_event": {"type": "string", "minLength": 1, "maxLength": 4_000},
        "success_guarantee": {"type": "string", "minLength": 1, "maxLength": 4_000},
        "minimum_guarantee": {"type": "string", "minLength": 1, "maxLength": 4_000},
        "main_success_scenario": _string_array(min_items=1),
        "extension_scenarios": _string_array(),
        "constraints": _string_array(),
        "dfx": _string_array(),
        "catalog": _nullable_string(max_length=1_000),
        "status": _status_schema(),
        "workflow_status": _workflow_status_schema(),
        "security_level": _nullable_string(max_length=100),
        "tags": _string_array(max_items=50),
        "source_ir_ids": _string_array(max_items=50),
        "scenario_id": {"type": "string", "minLength": 1, "maxLength": 120},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _link_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scenario_id": {"type": "string", "maxLength": 120},
            "use_case_ids": _string_array(min_items=1, max_items=50),
        },
        "required": ["scenario_id", "use_case_ids"],
    }


def _update_scenario_parameters() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "scenario_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "name": _nullable_string(max_length=300),
        "description": _nullable_string(max_length=8_000),
        "category": _nullable_string(max_length=200),
        "actor": _nullable_string(max_length=1_000),
        "influence_factors": _nullable_array(
            {
                "type": "array",
                "maxItems": 100,
                "items": _influence_factor_schema(),
            }
        ),
        "owner": _nullable_string(max_length=300),
        "business_goal": _nullable_string(max_length=4_000),
        "actions": _nullable_array(_string_array(max_items=100)),
        "constraints": _nullable_array(_string_array(max_items=100)),
        "dfx": _nullable_array(_string_array(max_items=100)),
        "affected_components": _nullable_array(_string_array(max_items=100)),
        "lifecycle": _nullable_string(max_length=500),
        "tags": _nullable_array(_string_array(max_items=50)),
        "source_ir_ids": _nullable_array(_string_array(max_items=50)),
        "security_level": _nullable_string(max_length=100),
        "esn_id": _nullable_string(max_length=120),
        "topology_diagram": _nullable_string(max_length=2_000),
        "ir_intent": _nullable_string(max_length=4_000),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _update_use_case_parameters() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "use_case_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "name": _nullable_string(max_length=300),
        "description": _nullable_string(max_length=8_000),
        "actor": _nullable_string(max_length=1_000),
        "preconditions": _nullable_array(_string_array(max_items=100)),
        "trigger_event": _nullable_string(max_length=4_000),
        "success_guarantee": _nullable_string(max_length=4_000),
        "minimum_guarantee": _nullable_string(max_length=4_000),
        "main_success_scenario": _nullable_array(_string_array(max_items=100)),
        "extension_scenarios": _nullable_array(_string_array(max_items=100)),
        "constraints": _nullable_array(_string_array(max_items=100)),
        "dfx": _nullable_array(_string_array(max_items=100)),
        "catalog": _nullable_string(max_length=1_000),
        "tags": _nullable_array(_string_array(max_items=50)),
        "source_ir_ids": _nullable_array(_string_array(max_items=50)),
        "security_level": _nullable_string(max_length=100),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _transition_parameters() -> dict[str, Any]:
    properties = {
        "record_type": {"type": "string", "enum": ["scenario", "use_case"]},
        "record_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "workflow_status": {
            "type": "string",
            "enum": ["Draft", "Inwork", "Review", "Publish", "Obsolete"],
        },
        "comment": _nullable_string(max_length=2_000),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _move_use_case_parameters() -> dict[str, Any]:
    properties = {
        "use_case_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "target_scenario_id": {"type": "string", "minLength": 1, "maxLength": 120},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _list_parameters() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": {}, "required": []}


def _match_recommendation(matches: list[Any], *, reuse_threshold: float = 0.45) -> dict[str, Any]:
    confidence = matches[0].score if matches else 0.0
    decision = "reuse_existing" if confidence >= reuse_threshold else "create_new"
    if decision == "reuse_existing":
        rationale = ["最高候选达到独立匹配的复用阈值，建议先读取候选详情并确认是否复用。"]
    else:
        rationale = ["没有候选达到独立匹配的复用阈值，建议按当前 Spec 生成草稿并人工确认后新建。"]
    return {
        "matches": [item.model_dump(mode="json") for item in matches],
        "decision": decision,
        "confidence": round(confidence, 4),
        "reuse_threshold": reuse_threshold,
        "rationale": rationale,
    }


def _search_skills_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query", "top_k"],
    }


def _search_memory_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["query", "limit"],
    }


def _save_memory_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "kind": {"type": "string", "minLength": 1, "maxLength": 50},
            "tags": _string_array(max_items=20),
        },
        "required": ["content", "kind", "tags"],
    }


class ToolRegistry:
    """Agent-facing tools backed by one ScenarioLibrary instance."""

    def __init__(
        self,
        library: ScenarioLibrary,
        *,
        skills: SkillCatalog | None = None,
        memory: MemoryStore | None = None,
        spec: SpecCatalog | None = None,
        user_id: str = "default",
    ):
        self.library = library
        self.skills = skills
        self.memory = memory
        self.spec = spec or SpecCatalog.default()
        self.library.configure_matching(self.spec.matching_rules)
        self.user_id = user_id
        self._specs = self._build_specs()
        self._specs.update(self._optional_specs())

    def _reuse_threshold(self, kind: str = "scenario") -> float:
        key = (
            "use_case_reuse_threshold"
            if kind == "use_case"
            else "scenario_reuse_threshold"
        )
        try:
            value = float(self.library.matching_rules().get(key, 0.45))
        except (TypeError, ValueError):
            return 0.45
        return value if 0.0 <= value <= 1.0 else 0.45

    def _build_specs(self) -> dict[str, ToolSpec]:
        return {
            "match_ir_requirement": ToolSpec(
                name="match_ir_requirement",
                description=(
                    "Normalize and match one IR against scenarios and use cases using 5W2H, actor, "
                    "lifecycle, influence factors, constraints, triggers, and guarantees. Always call this first."
                ),
                parameters=_match_ir_parameters(),
                input_model=MatchIRArgs,
                handler=lambda args: {
                    "match": self.library.match_ir(
                        args.ir, top_k=args.top_k, min_score=args.min_score
                    ).model_dump(mode="json")
                },
            ),
            "evaluate_scenario_fit": ToolSpec(
                name="evaluate_scenario_fit",
                description=(
                    "Evaluate one explicitly selected SC against a normalized IR. "
                    "Read-only; returns total score, every dimension score, low-score reasons, "
                    "gaps, conflicts, and child UC coverage for testing or human review."
                ),
                parameters=_evaluate_scenario_fit_parameters(),
                input_model=EvaluateScenarioFitArgs,
                handler=lambda args: {
                    "evaluation": self.library.evaluate_scenario_fit(
                        args.ir, args.scenario_id
                    )
                },
            ),
            "draft_scenario_from_ir": ToolSpec(
                name="draft_scenario_from_ir",
                description=(
                    "Use the active business Spec to map one normalized IR into an SC draft. "
                    "This is read-only and returns missing required fields; it never writes the library."
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"ir": _ir_schema()},
                    "required": ["ir"],
                },
                input_model=DraftScenarioArgs,
                handler=lambda args: self.spec.draft_scenario(args.ir),
            ),
            "draft_use_cases_from_ir": ToolSpec(
                name="draft_use_cases_from_ir",
                description=(
                    "Use the active business Spec to derive one candidate UC draft per potential parent "
                    "scenario. This is read-only and never writes the library."
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ir": _ir_schema(),
                        "candidate_scenario_ids": _string_array(min_items=1, max_items=50),
                    },
                    "required": ["ir", "candidate_scenario_ids"],
                },
                input_model=DraftUseCasesArgs,
                handler=lambda args: {
                    "drafts": [
                        self.spec.draft_use_case(args.ir, self.library.get_scenario(scenario_id))
                        for scenario_id in args.candidate_scenario_ids
                    ]
                },
            ),
            "save_ir_requirement": ToolSpec(
                name="save_ir_requirement",
                description="Persist or update the normalized IR for traceability after user approval.",
                parameters=_save_ir_parameters(),
                input_model=IRRequirementInput,
                requires_approval=True,
                handler=lambda args: {
                    "saved": True,
                    "ir": self.library.save_requirement(args).model_dump(mode="json"),
                },
            ),
            "get_ir_requirement": ToolSpec(
                name="get_ir_requirement",
                description="Read one saved IR by id or code.",
                parameters=_single_record_parameters("ir_id", "IR id or code."),
                input_model=GetIRArgs,
                handler=lambda args: {
                    "ir": self.library.get_requirement(args.ir_id).model_dump(mode="json")
                },
            ),
            "match_scenario": ToolSpec(
                name="match_scenario",
                description=(
                    "Independently match a standalone SC description against the scenario library. "
                    "Read-only; returns candidates, a reuse/create recommendation, confidence, and rationale."
                ),
                parameters=_search_parameters(
                    "Standalone scenario description including actor, action, context, and influence factors."
                ),
                input_model=MatchScenarioArgs,
                handler=lambda args: {
                    "query": args.query,
                    **_match_recommendation(
                        self.library.search(args.query, top_k=args.top_k, min_score=args.min_score),
                        reuse_threshold=self._reuse_threshold(),
                    ),
                },
            ),
            "search_scenarios": ToolSpec(
                name="search_scenarios",
                description="Perform a lightweight free-text scenario search when no full IR is available.",
                parameters=_search_parameters("Normalized requirement text."),
                input_model=SearchScenariosArgs,
                handler=lambda args: {
                    "query": args.query,
                    "matches": [
                        item.model_dump(mode="json")
                        for item in self.library.search(
                            args.query, top_k=args.top_k, min_score=args.min_score
                        )
                    ],
                },
            ),
            "get_scenario": ToolSpec(
                name="get_scenario",
                description="Read full scenario facts after a candidate id is known.",
                parameters=_single_record_parameters("scenario_id", "Scenario id."),
                input_model=GetScenarioArgs,
                handler=lambda args: {
                    "scenario": self.library.get_scenario(args.scenario_id).model_dump(mode="json")
                },
            ),
            "search_use_cases": ToolSpec(
                name="search_use_cases",
                description="Search UC behavior chains: trigger, preconditions, success path, extensions, and guarantees.",
                parameters=_search_parameters("IR behavior and expected handling chain."),
                input_model=SearchScenariosArgs,
                handler=lambda args: {
                    "query": args.query,
                    "matches": [
                        item.model_dump(mode="json")
                        for item in self.library.search_use_cases(
                            args.query, top_k=args.top_k, min_score=args.min_score
                        )
                    ],
                },
            ),
            "match_use_case": ToolSpec(
                name="match_use_case",
                description=(
                    "Independently match a standalone UC behavior chain against the UC library. "
                    "Optionally restrict matching to one parent scenario. "
                    "Read-only; returns candidates, a reuse/create recommendation, confidence, and rationale."
                ),
                parameters=_match_use_case_parameters(),
                input_model=MatchUseCaseArgs,
                handler=lambda args: {
                    "query": args.query,
                    "scenario_id": args.scenario_id,
                    **_match_recommendation(
                        self.library.search_use_cases(
                            args.query,
                            scenario_id=args.scenario_id,
                            top_k=args.top_k,
                            min_score=args.min_score,
                        ),
                        reuse_threshold=self._reuse_threshold("use_case"),
                    ),
                },
            ),
            "get_use_case": ToolSpec(
                name="get_use_case",
                description="Read one UC including its trigger, success path, extensions, guarantees, and constraints.",
                parameters=_single_record_parameters("use_case_id", "Use case id."),
                input_model=GetUseCaseArgs,
                handler=lambda args: {
                    "use_case": self.library.get_use_case(args.use_case_id).model_dump(mode="json")
                },
            ),
            "update_scenario": ToolSpec(
                name="update_scenario",
                description=(
                    "Update the content fields of one existing SC after human confirmation. "
                    "Workflow status changes must use transition_record; obsolete SCs cannot be edited."
                ),
                parameters=_update_scenario_parameters(),
                input_model=UpdateScenarioRequest,
                requires_approval=True,
                handler=lambda args: {
                    "updated": True,
                    "scenario": self._update_scenario(args),
                },
            ),
            "update_use_case": ToolSpec(
                name="update_use_case",
                description=(
                    "Update the content fields of one existing UC after human confirmation. "
                    "Use move_use_case for changing its parent SC."
                ),
                parameters=_update_use_case_parameters(),
                input_model=UpdateUseCaseRequest,
                requires_approval=True,
                handler=lambda args: {
                    "updated": True,
                    "use_case": self._update_use_case(args),
                },
            ),
            "transition_record": ToolSpec(
                name="transition_record",
                description=(
                    "Move one SC or UC through Draft/Inwork/Review/Publish/Obsolete with "
                    "validated workflow transitions. Obsolete is terminal."
                ),
                parameters=_transition_parameters(),
                input_model=TransitionRecordRequest,
                requires_approval=True,
                handler=lambda args: {
                    "updated": True,
                    "record": self._transition_record(args),
                },
            ),
            "move_use_case": ToolSpec(
                name="move_use_case",
                description=(
                    "Move one UC to another parent SC while preserving the exactly-one-parent rule. "
                    "This changes both scenario links and the UC revision."
                ),
                parameters=_move_use_case_parameters(),
                input_model=MoveUseCaseRequest,
                requires_approval=True,
                handler=lambda args: {
                    "updated": True,
                    "use_case": self._move_use_case(args),
                    "target_scenario_id": args.target_scenario_id,
                },
            ),
            "validate_library": ToolSpec(
                name="validate_library",
                description=(
                    "Read-only quality audit for the active IR/SC/UC library: checks Spec required fields, "
                    "SC-to-UC parent references, duplicate IDs, orphan UC records, and unresolved IR traces."
                ),
                parameters=_list_parameters(),
                input_model=None,
                handler=lambda _args: self._validate_library(),
            ),
            "list_use_cases": ToolSpec(
                name="list_use_cases",
                description="List use cases currently available in the library.",
                parameters=_list_parameters(),
                input_model=None,
                handler=lambda _args: {
                    "use_cases": [
                        item.model_dump(mode="json") for item in self.library.list_use_cases()
                    ]
                },
            ),
            "create_scenario": ToolSpec(
                name="create_scenario",
                description=(
                    "Create a scenario only after matching and approval. description, category, actor, "
                    "influence_factors with selected values, and owner are mandatory."
                ),
                parameters=_create_scenario_parameters(),
                input_model=CreateScenarioRequest,
                requires_approval=True,
                handler=lambda args: {
                    "created": True,
                    "scenario": self._create_scenario(args),
                },
            ),
            "create_use_case": ToolSpec(
                name="create_use_case",
                description=(
                    "Create one UC child under exactly one parent scenario after matching and approval. "
                    "A complete trigger-to-guarantee behavior chain is mandatory."
                ),
                parameters=_create_use_case_parameters(),
                input_model=CreateUseCaseRequest,
                requires_approval=True,
                handler=lambda args: {
                    "created": True,
                    "use_case": self._create_use_case(args),
                    "linked_scenario_id": args.scenario_id,
                },
            ),
            "link_scenario_use_cases": ToolSpec(
                name="link_scenario_use_cases",
                description=(
                    "Add one or more existing, unowned UC children to a scenario. "
                    "Fails if a UC already belongs to another scenario."
                ),
                parameters=_link_parameters(),
                input_model=LinkUseCasesRequest,
                requires_approval=True,
                handler=lambda args: {
                    "updated": True,
                    "scenario": self.library.link_use_cases(
                        args.scenario_id, args.use_case_ids
                    ).model_dump(mode="json"),
                },
            ),
        }

    def _create_scenario(self, args: CreateScenarioRequest) -> dict[str, Any]:
        gaps = self.spec.validate_scenario_payload(args.model_dump())
        if gaps:
            raise ValueError(f"场景不符合当前 Spec，待补字段：{', '.join(gaps)}")
        return self.library.create(args).model_dump(mode="json")

    def _create_use_case(self, args: CreateUseCaseRequest) -> dict[str, Any]:
        gaps = self.spec.validate_use_case_payload(args.model_dump())
        if gaps:
            raise ValueError(f"UC 不符合当前 Spec，待补字段：{', '.join(gaps)}")
        return self.library.create_use_case(args).model_dump(mode="json")

    def _update_scenario(self, args: UpdateScenarioRequest) -> dict[str, Any]:
        current = self.library.get_scenario(args.scenario_id)
        payload = current.model_dump(mode="json")
        payload.update(args.model_dump(exclude={"scenario_id"}, exclude_unset=True))
        gaps = self.spec.validate_scenario_payload(payload)
        if gaps:
            raise ValueError(f"SC 不符合当前 Spec，待补字段：{', '.join(gaps)}")
        return self.library.update_scenario(args).model_dump(mode="json")

    def _update_use_case(self, args: UpdateUseCaseRequest) -> dict[str, Any]:
        current = self.library.get_use_case(args.use_case_id)
        payload = current.model_dump(mode="json")
        payload.update(args.model_dump(exclude={"use_case_id"}, exclude_unset=True))
        gaps = self.spec.validate_use_case_payload(payload)
        if gaps:
            raise ValueError(f"UC 不符合当前 Spec，待补字段：{', '.join(gaps)}")
        return self.library.update_use_case(args).model_dump(mode="json")

    def _transition_record(self, args: TransitionRecordRequest) -> dict[str, Any]:
        return self.library.transition_record(args).model_dump(mode="json")

    def _move_use_case(self, args: MoveUseCaseRequest) -> dict[str, Any]:
        return self.library.move_use_case(args).model_dump(mode="json")

    def _validate_library(self) -> dict[str, Any]:
        report = self.library.quality_report()
        issues = list(report.get("issues", []))
        counts = dict(report.get("counts", {}))
        for scenario in self.library.list_scenarios():
            gaps = self.spec.validate_scenario_payload(scenario.model_dump(mode="json"))
            if gaps:
                issues.append(
                    {
                        "kind": "scenario_spec_fields",
                        "record_id": scenario.id,
                        "message": "SC 不符合当前 Spec：" + "、".join(gaps),
                    }
                )
        for use_case in self.library.list_use_cases():
            gaps = self.spec.validate_use_case_payload(use_case.model_dump(mode="json"))
            if gaps:
                issues.append(
                    {
                        "kind": "use_case_spec_fields",
                        "record_id": use_case.id,
                        "message": "UC 不符合当前 Spec：" + "、".join(gaps),
                    }
                )
        counts["issues"] = len(issues)
        return {
            "ok": not issues,
            "counts": counts,
            "issues": issues,
            "warnings": list(report.get("warnings", [])),
        }

    def _optional_specs(self) -> dict[str, ToolSpec]:
        specs: dict[str, ToolSpec] = {}
        if self.skills is not None:
            specs.update(
                {
                    "list_skills": ToolSpec(
                        name="list_skills",
                        description="List available project skills and their intended use.",
                        parameters=_list_parameters(),
                        input_model=None,
                        handler=lambda _args: {
                            "skills": [item.summary() for item in self.skills.list()]
                        },
                    ),
                    "search_skills": ToolSpec(
                        name="search_skills",
                        description="Find project skills relevant to the current request.",
                        parameters=_search_skills_parameters(),
                        input_model=SearchSkillsArgs,
                        handler=lambda args: {
                            "matches": [
                                {
                                    "score": item.score,
                                    "matched_terms": list(item.matched_terms),
                                    "skill": item.skill.summary(),
                                }
                                for item in self.skills.search(args.query, top_k=args.top_k)
                            ]
                        },
                    ),
                    "load_skill": ToolSpec(
                        name="load_skill",
                        description="Load the full trusted instructions of one project skill.",
                        parameters={
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                        input_model=LoadSkillArgs,
                        handler=lambda args: {
                            "skill": self.skills.get(args.name).summary(),
                            "instructions": self.skills.get(args.name).instructions[:8_000],
                        },
                    ),
                }
            )
        if self.memory is not None:
            specs.update(
                {
                    "search_memory": ToolSpec(
                        name="search_memory",
                        description="Search user-scoped project facts, preferences, and prior decisions.",
                        parameters=_search_memory_parameters(),
                        input_model=SearchMemoryArgs,
                        handler=lambda args: {
                            "memories": [
                                item.as_dict()
                                for item in self.memory.search(
                                    self.user_id, args.query, limit=args.limit
                                )
                            ]
                        },
                    ),
                    "save_memory": ToolSpec(
                        name="save_memory",
                        description="Save a stable non-secret preference, decision, or project fact.",
                        parameters=_save_memory_parameters(),
                        input_model=SaveMemoryArgs,
                        requires_approval=True,
                        handler=lambda args: {
                            "saved": True,
                            "memory": self.memory.save(
                                self.user_id,
                                args.content,
                                kind=args.kind,
                                tags=args.tags,
                            ).as_dict(),
                        },
                    ),
                }
            )
        return specs

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.as_openai_tool() for spec in self._specs.values()]

    def requires_approval(self, name: str) -> bool:
        spec = self._specs.get(name)
        return bool(spec and spec.requires_approval)

    def execute(self, name: str, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        spec = self._specs.get(name)
        if spec is None:
            return {"ok": False, "error": "unknown_tool", "message": f"Unknown tool: {name}"}
        try:
            arguments = spec.input_model.model_validate(raw_arguments) if spec.input_model else None
            return {"ok": True, **spec.handler(arguments)}
        except ValidationError as exc:
            return {"ok": False, "error": "invalid_arguments", "message": str(exc)}
        except (KeyError, ValueError) as exc:
            return {"ok": False, "error": "tool_execution_failed", "message": str(exc)}
        except Exception as exc:
            return {
                "ok": False,
                "error": "tool_execution_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }
