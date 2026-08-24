from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RecordStatus = Literal["draft", "working", "published", "active", "archived"]
WorkflowStatus = Literal["Draft", "Inwork", "Review", "Publish", "Obsolete"]
MatchDecision = Literal[
    "reuse_scenario_and_uc",
    "reuse_scenario_create_uc",
    "create_scenario_and_uc",
    "needs_clarification",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IRRequirementInput(BaseModel):
    """Normalized 5W2H/DFX content extracted from one IR document."""

    model_config = ConfigDict(extra="forbid")

    code: str | None = Field(default=None, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=12_000)
    source: str | None = Field(default=None, max_length=1_000)
    owner: str | None = Field(default=None, max_length=300)
    who: str | None = Field(default=None, max_length=1_000)
    when: str | None = Field(default=None, max_length=2_000)
    where: str | None = Field(default=None, max_length=2_000)
    what: str | None = Field(default=None, max_length=4_000)
    how: list[str] = Field(default_factory=list, max_length=50)
    why: str | None = Field(default=None, max_length=4_000)
    how_much: list[str] = Field(default_factory=list, max_length=50)
    constraints: list[str] = Field(default_factory=list, max_length=100)
    performance: str | None = Field(default=None, max_length=2_000)
    reliability: str | None = Field(default=None, max_length=2_000)
    serviceability: str | None = Field(default=None, max_length=2_000)
    maintainability: str | None = Field(default=None, max_length=2_000)
    sales: str | None = Field(default=None, max_length=2_000)
    delivery_time: str | None = Field(default=None, max_length=1_000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    def missing_fields(self) -> list[str]:
        required = {
            "who": self.who,
            "when": self.when,
            "where": self.where,
            "what": self.what,
            "how": self.how,
            "why": self.why,
            "how_much": self.how_much,
        }
        return [name for name, value in required.items() if not value]

    def search_text(self) -> str:
        values: list[str] = [self.title, self.description]
        for value in (self.who, self.when, self.where, self.what, self.why):
            if value:
                values.append(value)
        values.extend(self.how)
        values.extend(self.how_much)
        values.extend(self.constraints)
        values.extend(self.tags)
        return " ".join(values)


class InformationRequirement(IRRequirementInput):
    id: str = Field(min_length=1, max_length=120)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class InfluenceFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    kind: Literal["environment", "activity"] = "environment"
    dimension: str = Field(default="hardware_environment", min_length=1, max_length=120)
    candidate_values: list[str] = Field(default_factory=list, max_length=50)
    selected_values: list[str] = Field(min_length=1, max_length=50)


class UseCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=8_000)
    actor: str = Field(min_length=1, max_length=1_000)
    preconditions: list[str] = Field(default_factory=list, max_length=100)
    trigger_event: str = Field(min_length=1, max_length=4_000)
    success_guarantee: str = Field(min_length=1, max_length=4_000)
    minimum_guarantee: str = Field(min_length=1, max_length=4_000)
    main_success_scenario: list[str] = Field(min_length=1, max_length=100)
    extension_scenarios: list[str] = Field(default_factory=list, max_length=100)
    constraints: list[str] = Field(default_factory=list, max_length=100)
    dfx: list[str] = Field(default_factory=list, max_length=100)
    catalog: str | None = Field(default=None, max_length=1_000)
    status: RecordStatus = "draft"
    workflow_status: WorkflowStatus = "Draft"
    security_level: str | None = Field(default=None, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=50)
    source_ir_ids: list[str] = Field(default_factory=list, max_length=50)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=8_000)
    category: str = Field(min_length=1, max_length=200)
    actor: str = Field(min_length=1, max_length=1_000)
    influence_factors: list[InfluenceFactor] = Field(min_length=1, max_length=100)
    owner: str = Field(min_length=1, max_length=300)
    business_goal: str | None = Field(default=None, max_length=4_000)
    actions: list[str] = Field(default_factory=list, max_length=100)
    constraints: list[str] = Field(default_factory=list, max_length=100)
    dfx: list[str] = Field(default_factory=list, max_length=100)
    affected_components: list[str] = Field(default_factory=list, max_length=100)
    lifecycle: str | None = Field(default=None, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=50)
    use_case_ids: list[str] = Field(default_factory=list, max_length=100)
    source_ir_ids: list[str] = Field(default_factory=list, max_length=50)
    status: RecordStatus = "draft"
    workflow_status: WorkflowStatus = "Draft"
    security_level: str | None = Field(default=None, max_length=100)
    esn_id: str | None = Field(default=None, max_length=120)
    topology_diagram: str | None = Field(default=None, max_length=2_000)
    ir_intent: str = Field(default="", max_length=4_000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, str] = Field(default_factory=dict)


class ScenarioMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: Scenario
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    matched_dimensions: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class UseCaseMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_case: UseCase
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)


class IRMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ir: IRRequirementInput
    missing_ir_fields: list[str]
    scenario_matches: list[ScenarioMatch]
    use_case_matches: list[UseCaseMatch]
    decision: MatchDecision
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: list[str]


class CreateScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=300)
    description: str = Field(min_length=10, max_length=8_000)
    category: str = Field(min_length=1, max_length=200)
    actor: str = Field(min_length=1, max_length=1_000)
    influence_factors: list[InfluenceFactor] = Field(min_length=1, max_length=100)
    owner: str = Field(min_length=1, max_length=300)
    business_goal: str = Field(min_length=1, max_length=4_000)
    actions: list[str] = Field(min_length=1, max_length=100)
    constraints: list[str] = Field(min_length=1, max_length=100)
    dfx: list[str] = Field(default_factory=list, max_length=100)
    affected_components: list[str] = Field(default_factory=list, max_length=100)
    lifecycle: str = Field(min_length=1, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=50)
    source_ir_ids: list[str] = Field(default_factory=list, max_length=50)
    status: RecordStatus = "draft"
    workflow_status: WorkflowStatus = "Draft"
    security_level: str | None = Field(default=None, max_length=100)
    esn_id: str | None = Field(default=None, max_length=120)
    topology_diagram: str | None = Field(default=None, max_length=2_000)
    ir_intent: str = Field(default="", max_length=4_000)
    metadata: dict[str, str] = Field(default_factory=dict)


class CreateUseCaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=300)
    description: str = Field(min_length=10, max_length=8_000)
    actor: str = Field(min_length=1, max_length=1_000)
    preconditions: list[str] = Field(min_length=1, max_length=100)
    trigger_event: str = Field(min_length=1, max_length=4_000)
    success_guarantee: str = Field(min_length=1, max_length=4_000)
    minimum_guarantee: str = Field(min_length=1, max_length=4_000)
    main_success_scenario: list[str] = Field(min_length=1, max_length=100)
    extension_scenarios: list[str] = Field(default_factory=list, max_length=100)
    constraints: list[str] = Field(default_factory=list, max_length=100)
    dfx: list[str] = Field(default_factory=list, max_length=100)
    catalog: str | None = Field(default=None, max_length=1_000)
    status: RecordStatus = "draft"
    workflow_status: WorkflowStatus = "Draft"
    security_level: str | None = Field(default=None, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=50)
    source_ir_ids: list[str] = Field(default_factory=list, max_length=50)
    scenario_id: str = Field(min_length=1, max_length=120)


class LinkUseCasesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1, max_length=120)
    use_case_ids: list[str] = Field(min_length=1, max_length=50)


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    result: dict[str, object] = Field(default_factory=dict)
    approved: bool | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    audit_event_id: str | None = None


class ResolutionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1, max_length=120)
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str]
    matched_dimensions: list[str]
    gaps: list[str]
    reason: str


class ScenarioResolution(BaseModel):
    """Strict final contract returned by the agent for downstream systems."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["matched", "created", "needs_clarification", "no_match"]
    decision: MatchDecision
    ir_id: str | None
    request_summary: str
    candidates: list[ResolutionCandidate]
    selected_scenario_ids: list[str]
    use_case_ids: list[str]
    created_scenario_id: str | None
    created_use_case_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    missing_required_fields: list[str]
    gaps: list[str]
    next_steps: list[str]


class AgentResult(BaseModel):
    output_text: str
    response_id: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    turns: int = 0
    resolution: ScenarioResolution | None = None
    usage: dict[str, object] | None = None
    request_id: str | None = None
    compactions: int = 0
    retries: int = 0
    audit_event_ids: list[str] = Field(default_factory=list)
