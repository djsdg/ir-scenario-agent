from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


class AuditLogger:
    """Append-only JSONL audit log for tool calls and agent lifecycle events."""

    _SECRET_PATTERN = re.compile(
        r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|(?:api[_ -]?key|password|token|authorization)\s*[:=]\s*[^\s,;]+)"
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def record(
        self,
        event: str,
        *,
        user_id: str,
        session_id: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        event_id = f"audit_{uuid4().hex}"
        item = {
            "event_id": event_id,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "payload": self._redact(payload or {}),
        }
        line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return event_id

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, tuple):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            return self._SECRET_PATTERN.sub("[REDACTED]", value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)
