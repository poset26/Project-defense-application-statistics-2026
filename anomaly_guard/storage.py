"""SQLite-backed event store for dashboard and replay."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from anomaly_guard.core.decision import ScoreResult
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord


class EventStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE,
                    timestamp TEXT,
                    method TEXT,
                    route TEXT,
                    decision TEXT,
                    request_json TEXT,
                    score_json TEXT,
                    ground_truth_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_decision ON events(decision);
                CREATE INDEX IF NOT EXISTS idx_events_route ON events(route);
                """
            )

    def upsert_event(
        self,
        record: RequestRecord,
        score: ScoreResult,
        ground_truth: GroundTruthMeta | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (request_id, timestamp, method, route, decision, request_json, score_json, ground_truth_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    decision=excluded.decision,
                    score_json=excluded.score_json,
                    ground_truth_json=excluded.ground_truth_json
                """,
                (
                    record.request_id,
                    record.timestamp,
                    record.method,
                    record.route,
                    score.decision,
                    json.dumps(record.to_dict()),
                    json.dumps(score.to_dict()),
                    json.dumps(ground_truth.to_dict()) if ground_truth else None,
                ),
            )

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM events")

    def search(
        self,
        decision: str | None = None,
        route: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if decision:
            q += " AND decision = ?"
            params.append(decision)
        if route:
            q += " AND route = ?"
            params.append(route)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE request_id = ?", (request_id,)
            ).fetchone()
        return dict(row) if row else None

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT decision, COUNT(*) as c FROM events GROUP BY decision"
            ).fetchall()
        return {r["decision"]: r["c"] for r in rows}


class MemoryEventStore:
    """In-memory event store for hosted demo sessions (no shared SQLite)."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def upsert_event(
        self,
        record: RequestRecord,
        score: ScoreResult,
        ground_truth: GroundTruthMeta | None = None,
    ) -> None:
        rid = record.request_id
        if rid not in self._rows:
            self._order.append(rid)
        self._rows[rid] = {
            "request_id": rid,
            "timestamp": record.timestamp,
            "method": record.method,
            "route": record.route,
            "decision": score.decision,
            "request_json": json.dumps(record.to_dict()),
            "score_json": json.dumps(score.to_dict()),
            "ground_truth_json": json.dumps(ground_truth.to_dict()) if ground_truth else None,
        }

    def clear(self) -> None:
        self._rows.clear()
        self._order.clear()

    def search(
        self,
        decision: str | None = None,
        route: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rid in reversed(self._order):
            row = self._rows[rid]
            if decision and row["decision"] != decision:
                continue
            if route and row["route"] != route:
                continue
            out.append(dict(row))
            if len(out) >= limit:
                break
        return out

    def get(self, request_id: str) -> dict[str, Any] | None:
        row = self._rows.get(request_id)
        return dict(row) if row else None

    def counts(self) -> dict[str, int]:
        tallies: dict[str, int] = {}
        for row in self._rows.values():
            d = row["decision"]
            tallies[d] = tallies.get(d, 0) + 1
        return tallies
