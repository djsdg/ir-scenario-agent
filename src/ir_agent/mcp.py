from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MCPServerConfig(BaseModel):
    """Remote MCP server configuration for the Responses API."""

    model_config = ConfigDict(extra="ignore")

    server_label: str = Field(min_length=1, max_length=100)
    server_url: str | None = None
    connector_id: str | None = None
    server_description: str | None = None
    allowed_tools: list[str] | dict[str, Any] | None = None
    authorization: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    require_approval: str | dict[str, Any] = "never"

    @model_validator(mode="after")
    def validate_endpoint(self) -> "MCPServerConfig":
        if bool(self.server_url) == bool(self.connector_id):
            raise ValueError("exactly one of server_url or connector_id is required")
        if isinstance(self.require_approval, str):
            if self.require_approval not in {"always", "never"}:
                raise ValueError("require_approval must be 'always' or 'never'")
        elif not isinstance(self.require_approval, dict):
            raise ValueError("require_approval must be 'always', 'never', or a filter object")
        return self

    def as_responses_tool(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "mcp",
            "server_label": self.server_label,
            "require_approval": self.require_approval,
        }
        if self.server_url:
            payload["server_url"] = self.server_url
        if self.connector_id:
            payload["connector_id"] = self.connector_id
        if self.server_description:
            payload["server_description"] = self.server_description
        if self.allowed_tools:
            payload["allowed_tools"] = self.allowed_tools
        if self.authorization:
            payload["authorization"] = self.authorization
        if self.headers:
            payload["headers"] = self.headers
        return payload


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    servers: list[MCPServerConfig] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "MCPConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.model_validate(_expand_env(payload))

    def responses_tools(self) -> list[dict[str, Any]]:
        return [server.as_responses_tool() for server in self.servers]


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value
