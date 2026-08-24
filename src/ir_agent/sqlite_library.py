from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock

from .library import LibraryDocument, ScenarioLibrary


class ConcurrentLibraryWriteError(RuntimeError):
    """Raised when another process changed the SQLite library first."""


class SQLiteScenarioLibrary(ScenarioLibrary):
    """Transactional SQLite-backed library with the same public API as JSON storage."""

    def __init__(self, path: str | Path):
        database_path = Path(path)
        if database_path.suffix.casefold() not in {".sqlite", ".sqlite3", ".db"}:
            raise ValueError("SQLite library path must end with .sqlite, .sqlite3, or .db")
        self.root = database_path.parent
        self.path = database_path
        self.use_case_path = None
        self._lock = RLock()
        self._matching_rules: dict[str, object] = {}
        self._embedding_provider = None
        self._ensure_database()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _ensure_database(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS library_state "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL, "
                "revision INTEGER NOT NULL DEFAULT 0, "
                "payload TEXT NOT NULL)"
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(library_state)").fetchall()
            }
            if "revision" not in columns:
                connection.execute(
                    "ALTER TABLE library_state ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
                )
            row = connection.execute(
                "SELECT id FROM library_state WHERE id = 1"
            ).fetchone()
            if row is None:
                document = LibraryDocument()
                connection.execute(
                    "INSERT INTO library_state (id, version, revision, payload) VALUES (1, ?, ?, ?)",
                    (
                        document.version,
                        document.revision,
                        json.dumps(document.model_dump(mode="json"), ensure_ascii=False),
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def _read(self) -> LibraryDocument:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT payload FROM library_state WHERE id = 1"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return LibraryDocument()
        payload = json.loads(str(row[0]))
        return LibraryDocument.model_validate(payload)

    def _atomic_write(self, document: LibraryDocument) -> None:
        payload = json.dumps(document.model_dump(mode="json"), ensure_ascii=False)
        expected_revision = document.revision
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision FROM library_state WHERE id = 1"
            ).fetchone()
            current_revision = int(row[0]) if row is not None else 0
            if current_revision != expected_revision:
                raise ConcurrentLibraryWriteError(
                    "SQLite library changed while this operation was in progress; retry the operation."
                )
            next_revision = expected_revision + 1
            document = document.model_copy(update={"revision": next_revision})
            payload = json.dumps(document.model_dump(mode="json"), ensure_ascii=False)
            connection.execute(
                "UPDATE library_state SET version = ?, revision = ?, payload = ? WHERE id = 1",
                (document.version, next_revision, payload),
            )
            if connection.execute("SELECT changes()").fetchone()[0] == 0:
                connection.execute(
                    "INSERT INTO library_state (id, version, revision, payload) VALUES (1, ?, ?, ?)",
                    (document.version, next_revision, payload),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def migrate_json_to_sqlite(
    source_path: str | Path,
    target_path: str | Path,
    *,
    overwrite: bool = False,
) -> SQLiteScenarioLibrary:
    """Copy one JSON/directory library into a new SQLite library."""

    target = SQLiteScenarioLibrary(target_path)
    current = target.document()
    if not overwrite and (current.requirements or current.scenarios or current.use_cases):
        raise FileExistsError(f"Target SQLite library is not empty: {target.path}")
    source = ScenarioLibrary(source_path)
    target._atomic_write(source.document())
    return target
