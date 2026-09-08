"""Persistence for training runs and the organisms they produce.

A run's losing organisms are pruned from disk as it goes, so these rows are the
only lasting record of a population. The library owns them rather than each host
reimplementing the schema.

Tables are prefixed rather than kept in their own schema, so a host can point
this at its existing database and keep foreign keys to ezmt_Model enforced.
Attaching a file already open as main is the case SQLite warns against.

Uses the standard library through a worker thread. The writes are a few dozen
rows per run, which does not justify an async driver dependency on a host that
may not want one.
"""

import asyncio
import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

_log = logging.getLogger("ezmt.storage")

TABLE_PREFIX = "ezmt_"
RUN_TABLE = f"{TABLE_PREFIX}TrainingRun"
MODEL_TABLE = f"{TABLE_PREFIX}Model"

# Milliseconds a write waits for another connection's lock. A host keeps its own
# pool open on the same file.
BUSY_TIMEOUT_MS = 30_000

_CREATE_RUN_TABLE = f"""
CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name     TEXT    NOT NULL,
    version      TEXT    NOT NULL,
    started_at   INTEGER NOT NULL,
    finished_at  INTEGER,
    pop_size     INTEGER,
    generations  INTEGER,
    notes        TEXT,
    UNIQUE(run_name, version)
)
"""

# generation and organism_index complete the key: a whole population shares one
# name and version. Rows written before runs were recorded keep NULL for both.
_CREATE_MODEL_TABLE = f"""
CREATE TABLE IF NOT EXISTS {MODEL_TABLE} (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id      INTEGER REFERENCES {RUN_TABLE}(id),
    organism_name        TEXT NOT NULL,
    organism_version     TEXT NOT NULL,
    generation           INTEGER,
    organism_index       INTEGER,
    is_winner            INTEGER NOT NULL DEFAULT 0,
    created_at           INTEGER NOT NULL,
    score                REAL,
    fitness              REAL,
    parameters           TEXT,
    dna_summary          TEXT,
    data_quality_metrics TEXT,
    model_performance    TEXT,
    UNIQUE(organism_name, organism_version, generation, organism_index)
)
"""


def _as_json(value: Any) -> Optional[str]:
    """Serialise a value for a JSON column, or None when there is nothing."""
    return json.dumps(value, default=str) if value else None


class TrainingStore:
    """Reads and writes training runs in a SQLite database.

    Args:
        database_path: File to store runs in. The host may point this at its own
            database so foreign keys to ezmt_Model stay enforced.
    """

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with the busy timeout the host's pool requires."""
        connection = sqlite3.connect(self.database_path, timeout=BUSY_TIMEOUT_MS / 1000)
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        return connection

    def _write(self, statements: List[tuple]) -> Optional[int]:
        """Run (sql, params) pairs in one transaction.

        Args:
            statements: Statements to execute in order.

        Returns:
            The last inserted row id.
        """
        connection = self._connect()
        try:
            with connection:
                cursor = connection.cursor()
                for sql, params in statements:
                    cursor.execute(sql, params)
                return cursor.lastrowid
        finally:
            connection.close()

    def _read(self, sql: str, params: tuple = ()) -> List[tuple]:
        """Run one query and return every row."""
        connection = self._connect()
        try:
            return connection.execute(sql, params).fetchall()
        finally:
            connection.close()

    async def create_tables(self) -> None:
        """Create both tables if they are absent."""
        await asyncio.to_thread(
            self._write, [(_CREATE_RUN_TABLE, ()), (_CREATE_MODEL_TABLE, ())]
        )

    async def save_run(self, summary: dict, notes: Optional[str] = None) -> dict:
        """Record a finished run and every organism in it.

        The winner's row is written like any other; a host with metrics of its
        own attaches them afterwards with update_model_metrics.

        Args:
            summary: The dict ModelTuner.run returns.
            notes: Free text describing what the run was testing.

        Returns:
            {"training_run_id": int, "winner_model_id": int}.
        """
        await self.create_tables()
        run_id = await asyncio.to_thread(self._insert_run, summary, notes)
        await asyncio.to_thread(self._insert_organisms, run_id, summary)
        winner_id = await self.get_winner_id(
            summary["run_name"], summary["version"]
        )
        _log.info(
            "Recorded run %s/%s: %s organisms",
            summary["run_name"], summary["version"],
            sum(len(g["organisms"]) for g in summary["generations_detail"]),
        )
        return {"training_run_id": run_id, "winner_model_id": winner_id}

    def _insert_run(self, summary: dict, notes: Optional[str]) -> int:
        """Insert the run row and return its id."""
        return self._write([(
            f"INSERT INTO {RUN_TABLE} (run_name, version, started_at,"
            f" finished_at, pop_size, generations, notes)"
            f" VALUES (?, ?, ?, ?, ?, ?, ?)",
            (summary["run_name"], summary["version"], summary["started_at"],
             summary.get("finished_at"), summary.get("pop_size"),
             summary.get("generations"), notes),
        )])

    def _insert_organisms(self, run_id: int, summary: dict) -> None:
        """Insert one row per organism of every generation."""
        best = summary["best"]
        created_at = int(time.time())
        statements = []
        for generation in summary["generations_detail"]:
            for organism in generation["organisms"]:
                is_winner = (organism["generation"], organism["organism_index"]) == (
                    best["generation"], best["organism_index"])
                statements.append((
                    f"INSERT INTO {MODEL_TABLE} (training_run_id, organism_name,"
                    f" organism_version, generation, organism_index, is_winner,"
                    f" created_at, score, fitness, parameters, dna_summary)"
                    f" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, summary["run_name"], summary["version"],
                     organism["generation"], organism["organism_index"],
                     int(is_winner), created_at, organism["score"],
                     organism["fitness"], _as_json(organism["parameters"]),
                     organism["dna"]),
                ))
        self._write(statements)

    async def update_model_metrics(
        self,
        model_id: int,
        data_quality_metrics: Optional[dict] = None,
        model_performance: Optional[dict] = None,
    ) -> None:
        """Attach a host's metrics to one organism's row.

        Kept separate from save_run because only the winning organism is loaded
        back with its knowledge intact.

        Args:
            model_id: Row to update.
            data_quality_metrics: Properties of the input data.
            model_performance: How well the trained model performed.
        """
        await asyncio.to_thread(self._write, [(
            f"UPDATE {MODEL_TABLE} SET data_quality_metrics = ?,"
            f" model_performance = ? WHERE id = ?",
            (_as_json(data_quality_metrics), _as_json(model_performance), model_id),
        )])

    async def get_winner_id(self, run_name: str, version: str) -> Optional[int]:
        """Id of the winning organism of one run.

        Args:
            run_name: Organism name the run used.
            version: Exact version string, or "latest" for the most recent run.

        Returns:
            The row id, or None if there is no winner recorded.
        """
        if version == "latest":
            # id breaks ties when two runs land in the same second.
            sql = (f"SELECT id FROM {MODEL_TABLE} WHERE organism_name = ?"
                   f" AND is_winner = 1 ORDER BY created_at DESC, id DESC LIMIT 1")
            params: tuple = (run_name,)
        else:
            sql = (f"SELECT id FROM {MODEL_TABLE} WHERE organism_name = ?"
                   f" AND organism_version = ? AND is_winner = 1")
            params = (run_name, version)
        rows = await asyncio.to_thread(self._read, sql, params)
        return rows[0][0] if rows else None

    async def get_population(self, training_run_id: int) -> List[Dict[str, Any]]:
        """Every organism of every generation of one run.

        Args:
            training_run_id: The run to read.

        Returns:
            One dict per organism, in generation and population order.
        """
        rows = await asyncio.to_thread(
            self._read,
            f"SELECT generation, organism_index, score, fitness, dna_summary,"
            f" parameters, is_winner FROM {MODEL_TABLE} WHERE training_run_id = ?"
            f" ORDER BY generation, organism_index",
            (training_run_id,),
        )
        columns = ("generation", "organism_index", "score", "fitness",
                   "dna_summary", "parameters", "is_winner")
        return [dict(zip(columns, row)) for row in rows]
