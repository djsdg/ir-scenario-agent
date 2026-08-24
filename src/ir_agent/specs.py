from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .domain import IRRequirementInput


class SpecError(ValueError):
    """Raised when a business specification file cannot be loaded."""


class SpecCatalog:
    """Loadable business rules for the IR → SC → UC analysis pipeline.

    The catalog is deliberately data-driven.  It does not try to replace the
    language model's semantic interpretation; it defines the field contract,
    allowed workflow values, influence-factor dimensions, and deterministic
    draft/validation behavior around that interpretation.
    """

    def __init__(self, payload: dict[str, Any]):
        if not isinstance(payload, dict):
            raise SpecError("spec root must be a JSON object")
        self._payload = deepcopy(payload)
        self.version = int(payload.get("version", 1))
        self.name = str(payload.get("name", "IR→SC→UC specification"))

    @classmethod
    def from_file(cls, path: str | Path) -> "SpecCatalog":
        spec_path = Path(path)
        try:
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SpecError(f"spec file does not exist: {spec_path}") from exc
        except json.JSONDecodeError as exc:
            raise SpecError(f"spec file is not valid JSON: {spec_path}") from exc
        return cls(payload)

    @classmethod
    def default(cls) -> "SpecCatalog":
        project_spec = Path(__file__).resolve().parents[2] / "config" / "ir_sc_uc_spec.json"
        if project_spec.exists():
            try:
                return cls.from_file(project_spec)
            except SpecError:
                pass
        return cls(_fallback_spec())

    @property
    def payload(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    @property
    def categories(self) -> list[str]:
        return [str(item) for item in self._payload.get("categories", [])]

    @property
    def workflow_statuses(self) -> list[str]:
        return [str(item) for item in self._payload.get("workflow_statuses", [])]

    @property
    def quality_outputs(self) -> list[str]:
        return [str(item) for item in self._payload.get("quality_outputs", [])]

    @property
    def default_category(self) -> str:
        return str(self._payload.get("default_category", self.categories[-1] if self.categories else "派生场景"))

    @property
    def default_workflow_status(self) -> str:
        return str(self._payload.get("default_workflow_status", "Draft"))

    @property
    def default_owner(self) -> str | None:
        owner = self._payload.get("default_owner")
        return str(owner) if owner else None

    @property
    def matching_rules(self) -> dict[str, Any]:
        """Return deterministic matching knobs for the active business domain."""

        rules = self._payload.get("matching", {})
        return deepcopy(rules) if isinstance(rules, dict) else {}

    def scenario_required_fields(self) -> list[str]:
        return [
            str(item)
            for item in self._payload.get(
                "hard_required_scenario_fields",
                [
                    "description",
                    "category",
                    "business_goal",
                    "actor",
                    "actions",
                    "influence_factors",
                    "lifecycle",
                    "constraints",
                    "owner",
                ],
            )
        ]

    def use_case_required_fields(self) -> list[str]:
        return [
            str(item)
            for item in self._payload.get(
                "hard_required_uc_fields",
                [
                    "description",
                    "actor",
                    "preconditions",
                    "trigger_event",
                    "success_guarantee",
                    "minimum_guarantee",
                    "main_success_scenario",
                ],
            )
        ]

    def influence_dimensions(self) -> list[dict[str, Any]]:
        dimensions: list[dict[str, Any]] = []
        for kind, values in self._payload.get("influence_factor_dimensions", {}).items():
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict) or not value.get("id"):
                    continue
                dimensions.append(
                    {
                        "kind": str(kind),
                        "id": str(value["id"]),
                        "name": str(value.get("name", value["id"])),
                        "examples": [str(item) for item in value.get("examples", [])],
                    }
                )
        return dimensions

    def prompt_context(self) -> str:
        """Return the business rules that should be visible to the model."""

        public_spec = {
            "name": self.name,
            "version": self.version,
            "pipeline": self._payload.get("pipeline", []),
            "categories": self.categories,
            "workflow_statuses": self.workflow_statuses,
            "required_scenario_fields": self.scenario_required_fields(),
            "required_uc_fields": self.use_case_required_fields(),
            "relationship_constraints": self._payload.get("relationship_constraints", {}),
            "matching": self.matching_rules,
            "influence_factor_dimensions": self._payload.get("influence_factor_dimensions", {}),
            "ir_to_scenario_mapping": self._payload.get("ir_to_scenario_mapping", []),
            "identification_views": self._payload.get("identification_views", []),
            "quality_outputs": self._payload.get("quality_outputs", []),
            "forbidden_patterns": self._payload.get("forbidden_patterns", []),
        }
        return (
            "\n\n[active_business_spec]\n"
            "下面是本项目当前生效的 IR→SC→UC 业务规范。它优先于泛化模板；"
            "Spec 用于映射、校验和草稿，不代表可以跳过人工确认。\n"
            + json.dumps(public_spec, ensure_ascii=False, indent=2)
        )

    def validate_scenario_payload(self, payload: dict[str, Any]) -> list[str]:
        missing = self._missing_fields(payload, self.scenario_required_fields())
        if payload.get("description") and len(str(payload["description"])) < 10:
            missing.append("description(min_length=10)")

        category = payload.get("category")
        if category and self.categories and category not in self.categories:
            missing.append("category:not_in_spec")

        workflow_status = payload.get("workflow_status")
        if workflow_status and self.workflow_statuses and workflow_status not in self.workflow_statuses:
            missing.append("workflow_status:not_in_spec")

        factors = payload.get("influence_factors")
        if isinstance(factors, list):
            for index, factor in enumerate(factors):
                if not isinstance(factor, dict):
                    missing.append(f"influence_factors[{index}]")
                    continue
                if not _present(factor.get("name")):
                    missing.append(f"influence_factors[{index}].name")
                if not _present(factor.get("selected_values")):
                    missing.append(f"influence_factors[{index}].selected_values")
                if not _present(factor.get("kind")):
                    missing.append(f"influence_factors[{index}].kind")
                if not _present(factor.get("dimension")):
                    missing.append(f"influence_factors[{index}].dimension")
        return _unique(missing)

    def validate_use_case_payload(self, payload: dict[str, Any]) -> list[str]:
        missing = self._missing_fields(payload, self.use_case_required_fields())
        if payload.get("description") and len(str(payload["description"])) < 10:
            missing.append("description(min_length=10)")
        return _unique(missing)

    def infer_influence_factors(self, ir: IRRequirementInput) -> list[dict[str, Any]]:
        """Map IR text to candidate dimensions without inventing a value."""

        where_text = ir.where or ""
        all_text = " ".join(
            [
                ir.title,
                ir.description,
                where_text,
                ir.what or "",
                ir.why or "",
                *ir.how,
                *ir.constraints,
            ]
        ).casefold()
        factors: list[dict[str, Any]] = []
        for dimension in self.influence_dimensions():
            examples = dimension["examples"]
            hits = [example for example in examples if example.casefold() in all_text]
            is_environment = dimension["kind"] == "environment"
            if not hits:
                continue
            selected_values = [where_text.strip()] if is_environment and where_text.strip() else hits[:3]
            factors.append(
                {
                    "name": dimension["name"],
                    "kind": dimension["kind"],
                    "dimension": dimension["id"],
                    "candidate_values": examples,
                    "selected_values": _unique(selected_values),
                }
            )

        # A Where value is itself evidence of an environmental factor even
        # when it uses a product-specific term absent from the example list.
        if where_text.strip() and not any(item["kind"] == "environment" for item in factors):
            hardware = next(
                (
                    item
                    for item in self.influence_dimensions()
                    if item["id"] == "hardware_environment"
                ),
                {"name": "硬件环境", "kind": "environment", "id": "hardware_environment", "examples": []},
            )
            factors.insert(
                0,
                {
                    "name": hardware["name"],
                    "kind": hardware["kind"],
                    "dimension": hardware["id"],
                    "candidate_values": hardware["examples"],
                    "selected_values": [where_text.strip()],
                },
            )
        return factors

    def draft_scenario(self, ir: IRRequirementInput) -> dict[str, Any]:
        constraints = _unique([*ir.constraints, *ir.how_much])
        draft = {
            "name": ir.title.strip(),
            "description": ir.description.strip(),
            "category": self.default_category,
            "business_goal": ir.what.strip() if ir.what else ir.title.strip(),
            "actor": ir.who.strip() if ir.who else None,
            "actions": [item.strip() for item in ir.how if item.strip()],
            "influence_factors": self.infer_influence_factors(ir),
            "lifecycle": ir.when.strip() if ir.when else None,
            "constraints": constraints,
            "dfx": _quality_values(ir),
            "owner": ir.owner.strip() if ir.owner else self.default_owner,
            "affected_components": [ir.where.strip()] if ir.where and ir.where.strip() else [],
            "tags": list(dict.fromkeys(ir.tags)),
            "source_ir_ids": [ir.code] if ir.code else [],
            "status": "draft",
            "workflow_status": self.default_workflow_status,
            "security_level": None,
            "esn_id": None,
            "topology_diagram": None,
            "ir_intent": (ir.why or ir.description).strip(),
            "metadata": {"draft_source": "business_spec", "spec_version": str(self.version)},
        }
        return {
            "draft": draft,
            "missing_required_fields": self.validate_scenario_payload(draft),
            "mapping": self._payload.get("ir_to_scenario_mapping", []),
            "identification_views": self._payload.get("identification_views", []),
            "quality_outputs": self.quality_outputs,
        }

    def draft_use_case(self, ir: IRRequirementInput, scenario: Any) -> dict[str, Any]:
        scenario_id = _read(scenario, "id")
        scenario_name = _read(scenario, "name") or "目标场景"
        actor = ir.who or _read(scenario, "actor")
        constraints = _unique(ir.constraints)
        quality = _quality_values(ir)
        minimum_guarantee = _minimum_guarantee(ir)
        name = ir.what.strip() if ir.what else ir.title.strip()
        if name.casefold() == str(scenario_name).casefold():
            name = f"{name}具体处理"
        draft = {
            "name": name,
            "description": ir.description.strip(),
            "actor": actor,
            "preconditions": [ir.when.strip()] if ir.when and ir.when.strip() else [],
            "trigger_event": (ir.what or ir.description).strip(),
            "success_guarantee": _join(quality or ir.how_much),
            "minimum_guarantee": minimum_guarantee,
            "main_success_scenario": [item.strip() for item in ir.how if item.strip()],
            "extension_scenarios": _extension_candidates(ir),
            "constraints": constraints,
            "dfx": quality,
            "catalog": f"{_read(scenario, 'lifecycle') or '需求分析'}/派生用例",
            "status": "draft",
            "workflow_status": self.default_workflow_status,
            "security_level": None,
            "tags": list(dict.fromkeys(ir.tags)),
            "source_ir_ids": [ir.code] if ir.code else [],
            "scenario_id": scenario_id,
        }
        missing = self.validate_use_case_payload(draft)
        scenario_actor = _read(scenario, "actor")
        if actor and scenario_actor and actor.strip().casefold() != str(scenario_actor).strip().casefold():
            missing.append("actor:scenario_mismatch")
        return {
            "draft": draft,
            "missing_required_fields": _unique(missing),
            "scenario_id": scenario_id,
        }

    @staticmethod
    def _missing_fields(payload: dict[str, Any], fields: list[str]) -> list[str]:
        return [field for field in fields if not _present(payload.get(field))]


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _join(values: list[str]) -> str | None:
    cleaned = _unique(values)
    return "；".join(cleaned) if cleaned else None


def _quality_values(ir: IRRequirementInput) -> list[str]:
    values: list[str] = []
    for label, value in (
        ("性能", ir.performance),
        ("可靠性", ir.reliability),
        ("可服务性", ir.serviceability),
        ("可维护性", ir.maintainability),
        ("可销售性", ir.sales),
    ):
        if value and value.strip():
            values.append(f"{label}：{value.strip()}")
    return values


def _minimum_guarantee(ir: IRRequirementInput) -> str | None:
    candidates = [*ir.constraints, *ir.how_much]
    for value in candidates:
        normalized = value.casefold()
        if any(marker in normalized for marker in ("不能", "不保证", "保留", "至少", "不执行")):
            return value.strip()
    return None


def _extension_candidates(ir: IRRequirementInput) -> list[str]:
    candidates: list[str] = []
    for value in ir.constraints:
        normalized = value.casefold()
        if any(marker in normalized for marker in ("误判", "排除", "反复", "无法", "不限制")):
            candidates.append(value.strip())
    return _unique(candidates)


def _fallback_spec() -> dict[str, Any]:
    return {
        "version": 1,
        "name": "IR→SC→UC需求分析规范",
        "pipeline": ["IR", "SC", "UC", "FUNCTION_IMPACT", "SR"],
        "categories": ["Scenario Directory", "Scenario", "派生场景"],
        "workflow_statuses": ["Draft", "Inwork", "Review", "Publish", "Obsolete"],
        "default_workflow_status": "Draft",
        "default_category": "派生场景",
        "matching": {
            "scenario_reuse_threshold": 0.45,
            "scenario_strong_threshold": 0.70,
            "use_case_reuse_threshold": 0.45,
            "ambiguity_margin": 0.08,
            "lexical_weight": 0.75,
            "tfidf_weight": 0.25,
            "embedding_weight": 0.20,
            "synonyms": {
                "异常恢复": ["故障恢复", "恢复节点", "复位修复"],
                "隔离": ["下电隔离", "节点隔离", "故障隔离"],
                "反复复位": ["反复 core", "重复复位", "进程反复复位"],
                "检索问答": ["知识库问答", "多轮问答", "问题回答"],
                "证据链": ["可追溯回答", "来源追溯", "证据追溯"],
            },
            "critical_dimensions_for_reuse": ["Actor", "上下文", "影响因素"],
            "actor_categories": {
                "system": ["本系统", "系统自身", "软件", "进程"],
                "user": ["用户", "客户", "操作员", "人员"],
                "maintenance": ["运维", "管理员", "维护人员", "管理人员"],
                "external_system": ["外部系统", "周边系统", "第三方系统"],
                "device": ["设备", "部件", "节点", "硬件"],
            },
            "lifecycle_categories": {
                "normal_service": ["正常服务", "正常运行", "正常工作", "运行时"],
                "maintenance": ["维护", "检修", "维修"],
                "deployment": ["灌装", "部署", "安装", "配置"],
                "upgrade": ["升级", "更新"],
                "retirement": ["退网", "退役", "下线"],
                "fault_recovery": ["故障恢复", "故障", "异常恢复"],
            },
            "component_categories": {
                "type_a": ["类型a", "typea"],
                "type_b": ["类型b", "typeb"],
                "single_node": ["单节点"],
                "multi_node": ["多节点", "集群"],
                "unrestricted": ["不限", "不限制"],
            },
        },
        "hard_required_scenario_fields": [
            "description", "category", "business_goal", "actor", "actions",
            "influence_factors", "lifecycle", "constraints", "owner",
        ],
        "hard_required_uc_fields": [
            "description", "actor", "preconditions", "trigger_event",
            "success_guarantee", "minimum_guarantee", "main_success_scenario",
        ],
        "influence_factor_dimensions": {
            "environment": [
                {"id": "hardware_environment", "name": "硬件环境", "examples": ["部件", "节点", "硬盘", "控制器"]},
                {"id": "network_topology", "name": "组网场景", "examples": ["网络", "组网"]},
                {"id": "protocol_connection", "name": "协议连接", "examples": ["协议", "接口"]},
            ],
            "activity": [
                {"id": "storage_architecture", "name": "存储架构", "examples": ["架构"]},
                {"id": "business_scenario", "name": "业务场景", "examples": ["文件服务", "块服务", "对象服务"]},
                {"id": "operations", "name": "运维场景", "examples": ["配置", "升级", "扩容", "维护", "修复", "恢复", "隔离", "告警"]},
            ],
        },
    }
