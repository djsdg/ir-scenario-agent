from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    api_key: str | None
    model: str = "gpt-5.5"
    api_mode: str = "responses"
    embedding_model: str | None = None
    base_url: str | None = None
    organization: str | None = None
    library_path: Path = Path("data/scenario_library.json")
    uc_library_path: Path | None = None
    spec_path: Path = Path("config/ir_sc_uc_spec.json")
    sessions_dir: Path = Path("data/sessions")
    outputs_dir: Path = Path("data/outputs")
    memory_path: Path = Path("data/memory.sqlite3")
    skills_dir: Path = Path("skills")
    plugins_dir: Path = Path("plugins")
    mcp_config_path: Path = Path("config/mcp.json")
    user_id: str = "default"
    max_tool_rounds: int = 8
    request_timeout: float = 120.0
    max_retries: int = 2
    retry_backoff: float = 0.5
    max_session_items: int = 100
    max_context_chars: int = 120_000
    structured_output: bool = True
    require_tool_approval: bool = True
    audit_path: Path = Path("data/audit.jsonl")

    @classmethod
    def from_env(
        cls,
        *,
        library_path: str | Path | None = None,
        uc_library_path: str | Path | None = None,
        spec_path: str | Path | None = None,
        api_mode: str | None = None,
        sessions_dir: str | Path | None = None,
        outputs_dir: str | Path | None = None,
        memory_path: str | Path | None = None,
        skills_dir: str | Path | None = None,
        plugins_dir: str | Path | None = None,
        mcp_config_path: str | Path | None = None,
        audit_path: str | Path | None = None,
        user_id: str | None = None,
    ) -> "Settings":
        resolved_library_path = Path(
            library_path or os.getenv("IR_AGENT_LIBRARY_PATH", "data/scenario_library.json")
        )
        raw_uc_library_path = uc_library_path or os.getenv("IR_AGENT_UC_LIBRARY_PATH")
        resolved_uc_library_path = Path(raw_uc_library_path) if raw_uc_library_path else None
        resolved_spec_path = Path(
            spec_path or os.getenv("IR_AGENT_SPEC_PATH", "config/ir_sc_uc_spec.json")
        )
        resolved_sessions_dir = Path(
            sessions_dir or os.getenv("IR_AGENT_SESSION_DIR", "data/sessions")
        )
        resolved_outputs_dir = Path(
            outputs_dir or os.getenv("IR_AGENT_OUTPUT_DIR", "data/outputs")
        )
        resolved_memory_path = Path(
            memory_path or os.getenv("IR_AGENT_MEMORY_PATH", "data/memory.sqlite3")
        )
        resolved_skills_dir = Path(skills_dir or os.getenv("IR_AGENT_SKILLS_DIR", "skills"))
        resolved_plugins_dir = Path(plugins_dir or os.getenv("IR_AGENT_PLUGINS_DIR", "plugins"))
        resolved_mcp_config_path = Path(
            mcp_config_path or os.getenv("IR_AGENT_MCP_CONFIG", "config/mcp.json")
        )
        resolved_audit_path = Path(
            audit_path or os.getenv("IR_AGENT_AUDIT_PATH", "data/audit.jsonl")
        )
        resolved_api_mode = api_mode or os.getenv("IR_AGENT_API_MODE") or "responses"
        if resolved_api_mode not in {"responses", "chat_completions"}:
            raise ValueError(
                "IR_AGENT_API_MODE must be 'responses' or 'chat_completions'"
            )
        return cls(
            api_key=os.getenv("IR_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY"),
            model=os.getenv("IR_AGENT_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.5"),
            api_mode=resolved_api_mode,
            embedding_model=os.getenv("IR_AGENT_EMBEDDING_MODEL") or None,
            base_url=os.getenv("IR_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
            organization=os.getenv("OPENAI_ORG_ID") or None,
            library_path=resolved_library_path,
            uc_library_path=resolved_uc_library_path,
            spec_path=resolved_spec_path,
            sessions_dir=resolved_sessions_dir,
            outputs_dir=resolved_outputs_dir,
            memory_path=resolved_memory_path,
            skills_dir=resolved_skills_dir,
            plugins_dir=resolved_plugins_dir,
            mcp_config_path=resolved_mcp_config_path,
            user_id=user_id or os.getenv("IR_AGENT_USER_ID", "default"),
            max_tool_rounds=_read_int("IR_AGENT_MAX_TOOL_ROUNDS", 8),
            request_timeout=_read_float("IR_AGENT_REQUEST_TIMEOUT", 120.0),
            max_retries=_read_int("IR_AGENT_MAX_RETRIES", 2),
            retry_backoff=_read_float("IR_AGENT_RETRY_BACKOFF", 0.5),
            max_session_items=_read_int("IR_AGENT_MAX_SESSION_ITEMS", 100),
            max_context_chars=_read_int("IR_AGENT_MAX_CONTEXT_CHARS", 120_000),
            structured_output=_read_bool("IR_AGENT_STRUCTURED_OUTPUT", True),
            require_tool_approval=_read_bool("IR_AGENT_REQUIRE_TOOL_APPROVAL", True),
            audit_path=resolved_audit_path,
        )
