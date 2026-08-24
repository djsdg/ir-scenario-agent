from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import Settings
from .library import ScenarioLibrary
from .memory import MemoryStore
from .skills import SkillCatalog
from .specs import SpecCatalog
from .tools import ToolRegistry, ToolSpec


@dataclass(frozen=True, slots=True)
class PluginManifest:
    name: str
    version: str
    description: str
    entrypoint: str
    enabled: bool = True


@dataclass(slots=True)
class PluginContext:
    settings: Settings
    library: ScenarioLibrary
    skills: SkillCatalog
    memory: MemoryStore | None
    user_id: str
    spec: SpecCatalog = field(default_factory=SpecCatalog.default)


@dataclass(slots=True)
class PluginLoadReport:
    loaded: list[str]
    errors: list[str]


class PluginManager:
    """Discover trusted local plugins and register their function tools."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def manifests(self) -> list[tuple[PluginManifest, Path]]:
        if not self.root.exists():
            return []
        discovered: list[tuple[PluginManifest, Path]] = []
        for manifest_path in sorted(self.root.rglob("plugin.json")):
            discovered.append(self._read_manifest(manifest_path))
        return discovered

    def load_into(self, registry: ToolRegistry, context: PluginContext) -> PluginLoadReport:
        loaded: list[str] = []
        errors: list[str] = []
        if not self.root.exists():
            return PluginLoadReport(loaded=loaded, errors=errors)
        for manifest_path in sorted(self.root.rglob("plugin.json")):
            plugin_name = manifest_path.parent.name
            try:
                manifest, plugin_dir = self._read_manifest(manifest_path)
                plugin_name = manifest.name
                if not manifest.enabled:
                    continue
                module, factory_name = self._load_module(manifest, plugin_dir)
                factory = getattr(module, factory_name)
                specs = factory(context)
                if isinstance(specs, ToolSpec):
                    specs = [specs]
                for spec in specs:
                    if not isinstance(spec, ToolSpec):
                        raise TypeError("plugin factory must return ToolSpec objects")
                    registry.register(spec)
                loaded.append(manifest.name)
            except Exception as exc:  # A broken optional plugin should not kill the core agent.
                errors.append(f"{plugin_name}: {exc}")
        return PluginLoadReport(loaded=loaded, errors=errors)

    @staticmethod
    def _read_manifest(manifest_path: Path) -> tuple[PluginManifest, Path]:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = PluginManifest(
            name=str(payload["name"]),
            version=str(payload.get("version", "0.0.0")),
            description=str(payload.get("description", "")),
            entrypoint=str(payload["entrypoint"]),
            enabled=bool(payload.get("enabled", True)),
        )
        return manifest, manifest_path.parent

    @staticmethod
    def _load_module(manifest: PluginManifest, plugin_dir: Path) -> tuple[ModuleType, str]:
        module_part, separator, factory_name = manifest.entrypoint.partition(":")
        if not separator or not module_part or not factory_name:
            raise ValueError("entrypoint must look like 'plugin.py:create_plugin'")
        module_path = (plugin_dir / module_part).resolve()
        if plugin_dir.resolve() not in module_path.parents:
            raise ValueError("entrypoint must stay inside the plugin directory")
        if not module_path.exists():
            raise FileNotFoundError(module_path)

        module_name = f"ir_agent_plugin_{manifest.name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, factory_name
