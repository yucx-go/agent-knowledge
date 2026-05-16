"""Structured event index — companion to the Source/Claim store.

Why this exists
---------------
Sources hold rendered text bodies optimized for keyword retrieval; Claims
hold the agent-knowledge layer's distilled assertions. Neither is great
at answering "give me the timeline of bash tool calls in this session"
or "show every file change between 10:00 and 11:00 on May 14". For
those questions you need flat, queryable, time-ordered events.

The EventIndex stores raw UMSF events alongside their source_id, in
SQLite. It's an *additional* index, not a replacement — Source/Claim/
Vault remain the primary persistence layer. The index is populated
automatically by :func:`ingest_umsf` and can be queried directly:

    idx = vault.event_index
    bash_calls = idx.query(type="tool_use", session_id="sess1", limit=50)
    file_changes = idx.query(type="file_change", since_ts=time.time() - 3600)

Storage
-------
Single SQLite database at ``vault/.ak-events.db`` with one table. A few
indexes cover the common query patterns; this is *not* meant to be a
fully relational schema, just a focused secondary store.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from .umsf import UMSFEvent


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL DEFAULT '',
    agent       TEXT NOT NULL DEFAULT '',
    ts          REAL NOT NULL,
    type        TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT '',
    name        TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL DEFAULT '',
    args_json   TEXT NOT NULL DEFAULT '',
    result      TEXT NOT NULL DEFAULT '',
    error       TEXT NOT NULL DEFAULT '',
    path        TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_events_source  ON events(source_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_agent   ON events(agent);
"""


class EventIndex:
    """SQLite-backed flat index of UMSF events keyed by source.

    Thread-safe enough for single-process use: every call opens its own
    connection. For high-throughput multi-writer use, swap to a connection
    pool — the public API doesn't change.
    """

    DB_FILENAME = ".ak-events.db"

    def __init__(self, vault_root: Path):
        self.db_path = Path(vault_root) / self.DB_FILENAME
        self._init_lock = threading.Lock()
        self._initialized = False

    # ── Lifecycle ──

    def _ensure_init(self) -> None:
        # Lazy init so vault.event_index access doesn't immediately
        # create a SQLite file in vaults that don't use it.
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
            self._initialized = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── Mutations ──

    def add(
        self,
        source_id: str,
        session_id: str,
        events: list["UMSFEvent"],
        agent: str = "",
    ) -> int:
        """Insert events for a source. Returns rows-inserted count.

        Existing rows for ``source_id`` are NOT cleared. Callers that want
        replace-semantics should call :meth:`delete_source_events` first.
        """
        if not events:
            return 0
        self._ensure_init()
        rows = []
        for e in events:
            rows.append((
                source_id,
                session_id or "",
                agent or "",
                float(e.ts),
                str(e.type),
                str(e.role or ""),
                str(e.name or ""),
                str(e.content or ""),
                json.dumps(e.args, ensure_ascii=False) if e.args else "",
                str(e.result or ""),
                str(e.error or ""),
                str(e.path or ""),
                json.dumps(e.metadata, ensure_ascii=False) if e.metadata else "",
            ))
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO events
                  (source_id, session_id, agent, ts, type, role, name,
                   content, args_json, result, error, path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def delete_source_events(self, source_id: str) -> int:
        """Remove all events tied to a source. Returns rows-deleted count."""
        self._ensure_init()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM events WHERE source_id = ?", (source_id,)
            )
            conn.commit()
            return cur.rowcount

    # ── Queries ──

    def query(
        self,
        *,
        type: Optional[str] = None,         # noqa: A002 (shadow ok in kwargs)
        session_id: Optional[str] = None,
        agent: Optional[str] = None,
        source_id: Optional[str] = None,
        since_ts: Optional[float] = None,
        until_ts: Optional[float] = None,
        limit: int = 100,
        order: str = "asc",
    ) -> list[dict]:
        """Filter events. Returns list of dicts with all columns hydrated.

        ``args_json`` and ``metadata_json`` are decoded back to dicts for
        callers' convenience; the raw row keys (``args``, ``metadata``)
        carry the parsed forms.
        """
        self._ensure_init()
        clauses: list[str] = []
        params: list = []
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if agent is not None:
            clauses.append("agent = ?")
            params.append(agent)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(since_ts)
        if until_ts is not None:
            clauses.append("ts <= ?")
            params.append(until_ts)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_dir = "DESC" if order.lower() == "desc" else "ASC"
        sql = f"SELECT * FROM events {where} ORDER BY ts {order_dir} LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            cur = conn.execute(sql, params)
            results = []
            for row in cur:
                d = dict(row)
                d["args"] = json.loads(d["args_json"]) if d["args_json"] else {}
                d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
                results.append(d)
        return results

    def by_source(self, source_id: str) -> list[dict]:
        """All events for one source, ordered by ts ascending."""
        return self.query(source_id=source_id, limit=10_000, order="asc")

    def by_session(self, session_id: str, limit: int = 1000) -> list[dict]:
        """All events in one session, ordered by ts ascending."""
        return self.query(session_id=session_id, limit=limit, order="asc")

    # ── Stats ──

    def stats(self) -> dict:
        """Return per-type and per-agent event counts."""
        self._ensure_init()
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            per_type = dict(conn.execute(
                "SELECT type, COUNT(*) FROM events GROUP BY type"
            ).fetchall())
            per_agent = dict(conn.execute(
                "SELECT agent, COUNT(*) FROM events GROUP BY agent"
            ).fetchall())
            unique_sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM events WHERE session_id != ''"
            ).fetchone()[0]
        return {
            "total_events": total,
            "by_type": per_type,
            "by_agent": per_agent,
            "unique_sessions": unique_sessions,
        }
