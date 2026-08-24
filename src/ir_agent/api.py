from __future__ import annotations

import argparse
import os
from typing import Any

from .agent import (
    AgentSession,
    IRScenarioAgent,
    OpenAIChatCompletionsTransport,
    OpenAIResponsesTransport,
    RetryingResponsesTransport,
)
from .audit import AuditLogger
from .config import Settings
from .domain import IRRequirementInput
from .library import open_scenario_library
from .mcp import MCPConfig
from .memory import MemoryStore
from .retrieval import OpenAIEmbeddingProvider
from .skills import SkillCatalog
from .specs import SpecCatalog
from .tools import ToolRegistry


def create_app(
    library,
    *,
    spec=None,
    agent: IRScenarioAgent | None = None,
    api_token: str | None = None,
):
    """Create an optional FastAPI service around the provider-neutral runtime."""

    try:
        from fastapi import FastAPI, Header, HTTPException, Query
    except ImportError as exc:
        raise RuntimeError(
            "The web API requires optional dependencies. Run: pip install -e '.[web]'"
        ) from exc

    app = FastAPI(title="IR / SC / UC Agent API", version="0.1.0")

    def authorize(api_key: str | None, *, write: bool = False) -> None:
        if write and not api_token:
            raise HTTPException(
                status_code=503,
                detail="Write/agent endpoints require IR_AGENT_API_TOKEN on the server.",
            )
        if api_token and api_key != api_token:
            raise HTTPException(status_code=401, detail="Invalid API token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        document = library.document()
        return {
            "ok": True,
            "storage": str(library.path.resolve()),
            "requirements": len(document.requirements),
            "scenarios": len(document.scenarios),
            "use_cases": len(document.use_cases),
            "agent_enabled": agent is not None,
        }

    @app.post("/match")
    def match(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            raw_ir = payload.get("ir", payload)
            ir = IRRequirementInput.model_validate(raw_ir)
            top_k = int(payload.get("top_k", 5))
            min_score = float(payload.get("min_score", 0.0))
            return library.match_ir(ir, top_k=top_k, min_score=min_score).model_dump(mode="json")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/scenarios")
    def scenarios(
        q: str = Query(default="", min_length=0),
        top_k: int = Query(default=20, ge=1, le=50),
    ) -> dict[str, Any]:
        if q.strip():
            items = library.search(q, top_k=min(top_k, 20), min_score=0.0)
            return {"matches": [item.model_dump(mode="json") for item in items]}
        return {
            "scenarios": [item.model_dump(mode="json") for item in library.list_scenarios()[:top_k]]
        }

    @app.get("/scenarios/{scenario_id}")
    def scenario(scenario_id: str) -> dict[str, Any]:
        try:
            return {"scenario": library.get_scenario(scenario_id).model_dump(mode="json")}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/use-cases")
    def use_cases(
        q: str = Query(default="", min_length=0),
        scenario_id: str | None = Query(default=None),
        top_k: int = Query(default=20, ge=1, le=50),
    ) -> dict[str, Any]:
        try:
            if q.strip():
                items = library.search_use_cases(
                    q,
                    scenario_id=scenario_id,
                    top_k=min(top_k, 20),
                    min_score=0.0,
                )
                return {"matches": [item.model_dump(mode="json") for item in items]}
            items = library.list_use_cases()
            if scenario_id:
                allowed = set(library.get_scenario(scenario_id).use_case_ids)
                items = [item for item in items if item.id in allowed]
            return {"use_cases": [item.model_dump(mode="json") for item in items[:top_k]]}
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/library/validate")
    def validate_library() -> dict[str, Any]:
        return ToolRegistry(library, spec=spec).execute("validate_library", {})

    @app.post("/agent/run")
    def run_agent(
        payload: dict[str, Any],
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_api_key, write=True)
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent API is not configured")
        message = str(payload.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=422, detail="message is required")
        session_id = str(payload.get("session_id") or "api-default")
        try:
            result = agent.run(message, session=AgentSession(id=session_id))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR / SC / UC Agent REST API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--library")
    parser.add_argument("--uc-library")
    parser.add_argument("--spec")
    parser.add_argument("--api-mode", choices=["responses", "chat_completions"])
    parser.add_argument("--api-token", help="Protect /agent/run with this token")
    parser.add_argument("--auto-approve-writes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        import uvicorn
    except ImportError:
        print("REST API 需要可选依赖，请运行：pip install -e '.[web]'")
        return 2

    args = _build_parser().parse_args(argv)
    settings = Settings.from_env(
        library_path=args.library,
        uc_library_path=args.uc_library,
        spec_path=args.spec,
        api_mode=args.api_mode,
    )
    library = open_scenario_library(
        settings.library_path,
        use_case_path=settings.uc_library_path,
    )
    if settings.embedding_model:
        library.configure_embedding(
            OpenAIEmbeddingProvider(
                api_key=settings.api_key,
                model=settings.embedding_model,
                base_url=settings.base_url,
                organization=settings.organization,
                timeout=settings.request_timeout,
            )
        )
    spec = SpecCatalog.from_file(settings.spec_path)
    agent = None
    if settings.api_key:
        transport_class = (
            OpenAIChatCompletionsTransport
            if settings.api_mode == "chat_completions"
            else OpenAIResponsesTransport
        )
        transport = RetryingResponsesTransport(
            transport_class(settings),
            max_retries=settings.max_retries,
            backoff=settings.retry_backoff,
        )
        agent = IRScenarioAgent(
            transport,
            library,
            settings=settings,
            skills=SkillCatalog(settings.skills_dir),
            memory=MemoryStore(settings.memory_path),
            spec=spec,
            mcp_config=MCPConfig.from_file(settings.mcp_config_path),
            tool_approval_callback=(lambda _request: args.auto_approve_writes),
            audit_logger=AuditLogger(settings.audit_path),
        )
    api_token = args.api_token or os.getenv("IR_AGENT_API_TOKEN")
    app = create_app(library, spec=spec, agent=agent, api_token=api_token)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
