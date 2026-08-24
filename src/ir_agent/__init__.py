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
from .library import ScenarioLibrary, open_scenario_library
from .mcp import MCPConfig, MCPServerConfig
from .memory import MemoryStore
from .plugins import PluginContext, PluginManager
from .reporting import build_analysis_report, save_run_report
from .skills import SkillCatalog
from .specs import SpecCatalog, SpecError
from .sqlite_library import SQLiteScenarioLibrary

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
    "build_analysis_report",
    "save_run_report",
    "Scenario",
    "ScenarioLibrary",
    "SQLiteScenarioLibrary",
    "ScenarioMatch",
    "ScenarioResolution",
    "Settings",
    "SkillCatalog",
    "SpecCatalog",
    "SpecError",
    "UseCase",
    "UseCaseMatch",
    "open_scenario_library",
    "read_document",
]
