from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .library import tokenize


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    user_id: str
    content: str
    kind: str
    tags: tuple[str, ...]
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "kind": self.kind,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MemoryStore:
    """SQLite long-term memory with user scoping and small lexical retrieval."""

    _SECRET_PATTERN = re.compile(
        r"(?:sk-[A-Za-z0-9]{12,}|api[_ -]?key\s*[:=]|password\s*[:=]|token\s*[:=])",
        re.IGNORECASE,
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save(
        self,
        user_id: str,
        content: str,
        *,
        kind: str = "fact",
        tags: list[str] | None = None,
    ) -> MemoryRecord:
        user_id = user_id.strip()
        content = content.strip()
        kind = kind.strip() or "fact"
        normalized_tags = list(dict.fromkeys(item.strip() for item in (tags or []) if item.strip()))
        if not user_id:
            raise ValueError("user_id must not be empty")
        if not content:
            raise ValueError("content must not be empty")
        if self._SECRET_PATTERN.search(content):
            raise ValueError("memory refuses to store values that look like secrets")

        now = _now()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT * FROM memories WHERE user_id = ? AND content = ? LIMIT 1",
                (user_id, content),
            ).fetchone()
            if existing:
                connection.execute(
                    "UPDATE memories SET kind = ?, tags_json = ?, updated_at = ? WHERE id = ?",
                    (kind, json.dumps(normalized_tags, ensure_ascii=False), now, existing["id"]),
                )
                record_id = existing["id"]
                created_at = existing["created_at"]
            else:
                record_id = f"mem_{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO memories(id, user_id, content, kind, tags_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        user_id,
                        content,
                        kind,
                        json.dumps(normalized_tags, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                created_at = now
        return MemoryRecord(record_id, user_id, content, kind, tuple(normalized_tags), created_at, now)

    def search(self, user_id: str, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        query_terms = set(tokenize(query))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC LIMIT 500",
                (user_id,),
            ).fetchall()

        ranked: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            record = _record_from_row(row)
            if not query_terms:
                score = 0.0
            else:
                document_terms = set(tokenize(" ".join([record.content, record.kind, *record.tags])))
                matched = query_terms & document_terms
                if not matched:
                    continue
                score = len(matched) / len(query_terms)
            ranked.append((score, record))
        # SQLite returned rows are already newest-first; Python's sort is stable,
        # so equal lexical scores retain recency order.
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in ranked[:limit]]

    def recent(self, user_id: str, *, limit: int = 10) -> list[MemoryRecord]:
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [_record_from_row(row) for row in rows]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        kind=row["kind"],
        tags=tuple(json.loads(row["tags_json"])),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
