"""IR scenario-library agent."""

from .audit import AuditLogger
from .agent import (
    AgentResult,
    AgentRunError,
    AgentSession,
    IRScenarioAgent,
    OpenAIChatCompletionsTransport,
    OpenAIResponsesTransport,
    RetryingResponsesTransport,
    SessionStore,
)
from .config import Settings
from .documents import read_document
from .domain import (
    IRMatchResult,
    IRRequirementInput,
    InformationRequirement,
    InfluenceFactor,
    Scenario,
    ScenarioMatch,
    ScenarioResolution,
    UseCase,
    UseCaseMatch,
)
from .library import ScenarioLibrary
from .mcp import MCPConfig, MCPServerConfig
from .memory import MemoryStore
from .plugins import PluginContext, PluginManager
from .skills import SkillCatalog
from .specs import SpecCatalog, SpecError

__all__ = [
    "AgentResult",
    "AgentRunError",
    "AgentSession",
    "AuditLogger",
    "IRScenarioAgent",
    "OpenAIChatCompletionsTransport",
    "IRMatchResult",
    "IRRequirementInput",
    "InformationRequirement",
    "InfluenceFactor",
    "OpenAIResponsesTransport",
    "RetryingResponsesTransport",
    "SessionStore",
    "MCPConfig",
    "MCPServerConfig",
    "MemoryStore",
    "PluginContext",
    "PluginManager",
    "Scenario",
    "ScenarioLibrary",
    "ScenarioMatch",
    "ScenarioResolution",
    "Settings",
    "SkillCatalog",
    "SpecCatalog",
    "SpecError",
    "UseCase",
    "UseCaseMatch",
    "read_document",
]
