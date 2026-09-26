"""SQLite/WAL durable reference implementation for SPM checkpoint semantics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sqlite3
import threading
from typing import Any

from .state_runtime import CheckpointEnvelope, InMemoryCheckpointStore, _require_digest


class SQLiteCheckpointStore:
    """File-backed store with transactional CAS and lineage-scoped idempotency."""

    def __init__(self, path: str | Path, *, timeout: float = 5.0) -> None:
        self.path = str(Path(path))
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=timeout,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                digest TEXT PRIMARY KEY,
                model_digest TEXT NOT NULL,
                schema_digest TEXT NOT NULL,
                prefix_digest TEXT NOT NULL,
                parent_digest TEXT,
                sequence INTEGER NOT NULL,
                state_payload_digest TEXT NOT NULL,
                state_payload BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS checkpoints_resolve_idx
                ON checkpoints(model_digest, schema_digest, prefix_digest);
            CREATE TABLE IF NOT EXISTS heads (
                lineage_id TEXT PRIMARY KEY,
                checkpoint_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS idempotency (
                lineage_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                checkpoint_digest TEXT NOT NULL,
                expected_head TEXT,
                PRIMARY KEY(lineage_id, idempotency_key)
            );
            """
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_checkpoint(row: sqlite3.Row | None) -> CheckpointEnvelope | None:
        if row is None:
            return None
        payload = bytes(row["state_payload"])
        rebuilt = CheckpointEnvelope.create(
            model_digest=row["model_digest"],
            schema_digest=row["schema_digest"],
            prefix_digest=row["prefix_digest"],
            parent_digest=row["parent_digest"],
            sequence=row["sequence"],
            state_payload=payload,
        )
        if rebuilt.digest != row["digest"]:
            raise ValueError("durable checkpoint integrity mismatch")
        if rebuilt.state_payload_digest != row["state_payload_digest"]:
            raise ValueError("durable checkpoint integrity mismatch")
        return rebuilt

    def get(self, digest: str) -> CheckpointEnvelope | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE digest = ?", (digest,)
            ).fetchone()
            return self._row_checkpoint(row)

    def head(self, lineage_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT checkpoint_digest FROM heads WHERE lineage_id = ?",
                (lineage_id,),
            ).fetchone()
            return None if row is None else row["checkpoint_digest"]

    def fork_lineage(self, lineage_id: str, from_digest: str) -> None:
        if not isinstance(lineage_id, str) or not lineage_id:
            raise ValueError("lineage_id must be non-empty")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                source = self._conn.execute(
                    "SELECT * FROM checkpoints WHERE digest = ?", (from_digest,)
                ).fetchone()
                if source is None:
                    raise ValueError("fork source checkpoint is unknown")
                self._row_checkpoint(source)
                if self._conn.execute(
                    "SELECT 1 FROM heads WHERE lineage_id = ?", (lineage_id,)
                ).fetchone() is not None:
                    raise ValueError("lineage already exists")
                self._conn.execute(
                    "INSERT INTO heads(lineage_id, checkpoint_digest) VALUES (?, ?)",
                    (lineage_id, from_digest),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    @staticmethod
    def _validate_envelope(checkpoint: CheckpointEnvelope) -> None:
        rebuilt = CheckpointEnvelope.create(
            model_digest=checkpoint.model_digest,
            schema_digest=checkpoint.schema_digest,
            prefix_digest=checkpoint.prefix_digest,
            parent_digest=checkpoint.parent_digest,
            sequence=checkpoint.sequence,
            state_payload=checkpoint.state_payload,
        )
        if rebuilt != checkpoint:
            raise ValueError("checkpoint envelope integrity mismatch")

    def commit(
        self,
        checkpoint: CheckpointEnvelope,
        *,
        lineage_id: str,
        expected_head: str | None,
        idempotency_key: str,
    ) -> CheckpointEnvelope:
        if not isinstance(lineage_id, str) or not lineage_id:
            raise ValueError("lineage_id must be non-empty")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        self._validate_envelope(checkpoint)

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                return self._commit_locked(
                    checkpoint, lineage_id, expected_head, idempotency_key
                )
            except Exception:
                self._conn.rollback()
                raise

    def _commit_locked(
        self,
        checkpoint: CheckpointEnvelope,
        lineage_id: str,
        expected_head: str | None,
        idempotency_key: str,
    ) -> CheckpointEnvelope:
        prior = self._conn.execute(
            """SELECT checkpoint_digest, expected_head FROM idempotency
               WHERE lineage_id = ? AND idempotency_key = ?""",
            (lineage_id, idempotency_key),
        ).fetchone()
        if prior is not None:
            if (
                prior["checkpoint_digest"] != checkpoint.digest
                or prior["expected_head"] != expected_head
            ):
                raise ValueError("idempotency key reused for a different transition")
            stored = self._conn.execute(
                "SELECT * FROM checkpoints WHERE digest = ?", (checkpoint.digest,)
            ).fetchone()
            if stored is None:
                raise ValueError("idempotent transition checkpoint was evicted")
            result = self._row_checkpoint(stored)
            assert result is not None
            self._conn.commit()
            return result

        head = self._conn.execute(
            "SELECT checkpoint_digest FROM heads WHERE lineage_id = ?", (lineage_id,)
        ).fetchone()
        current_head = None if head is None else head["checkpoint_digest"]
        if current_head != expected_head:
            raise ValueError("stale head: expected lineage head does not match")
        if checkpoint.parent_digest != expected_head:
            raise ValueError("checkpoint parent does not match expected head")

        if expected_head is None:
            if checkpoint.sequence != 0:
                raise ValueError("root sequence must be zero")
        else:
            parent_row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE digest = ?", (expected_head,)
            ).fetchone()
            if parent_row is None:
                raise ValueError("expected parent checkpoint is unknown")
            parent = self._row_checkpoint(parent_row)
            assert parent is not None
            if (
                checkpoint.model_digest != parent.model_digest
                or checkpoint.schema_digest != parent.schema_digest
            ):
                raise ValueError("checkpoint subject does not match parent subject")
            if checkpoint.sequence != parent.sequence + 1:
                raise ValueError("checkpoint sequence must advance parent by exactly one")

        existing_row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE digest = ?", (checkpoint.digest,)
        ).fetchone()
        if existing_row is not None:
            existing = self._row_checkpoint(existing_row)
            if existing != checkpoint:
                raise ValueError("checkpoint digest collision")
        else:
            self._conn.execute(
                """INSERT INTO checkpoints(
                   digest, model_digest, schema_digest, prefix_digest, parent_digest,
                   sequence, state_payload_digest, state_payload
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    checkpoint.digest, checkpoint.model_digest, checkpoint.schema_digest,
                    checkpoint.prefix_digest, checkpoint.parent_digest, checkpoint.sequence,
                    checkpoint.state_payload_digest, checkpoint.state_payload,
                ),
            )

        self._conn.execute(
            """INSERT INTO heads(lineage_id, checkpoint_digest) VALUES (?, ?)
               ON CONFLICT(lineage_id) DO UPDATE SET checkpoint_digest = excluded.checkpoint_digest""",
            (lineage_id, checkpoint.digest),
        )
        self._conn.execute(
            """INSERT INTO idempotency(
               lineage_id, idempotency_key, checkpoint_digest, expected_head
               ) VALUES (?, ?, ?, ?)""",
            (lineage_id, idempotency_key, checkpoint.digest, expected_head),
        )
        self._conn.commit()
        return checkpoint

    def resolve(
        self, model_digest: str, schema_digest: str, prefix_digest: str
    ) -> CheckpointEnvelope | None:
        model = _require_digest(model_digest, "model_digest")
        schema = _require_digest(schema_digest, "schema_digest")
        prefix = _require_digest(prefix_digest, "prefix_digest")
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM checkpoints
                   WHERE model_digest = ? AND schema_digest = ? AND prefix_digest = ?""",
                (model, schema, prefix),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("ambiguous checkpoint resolution for model/schema/prefix")
        return self._row_checkpoint(rows[0])

    def evict(self, digest: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM checkpoints WHERE digest = ?", (digest,))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            checkpoint_rows = self._conn.execute(
                "SELECT * FROM checkpoints ORDER BY digest"
            ).fetchall()
            head_rows = self._conn.execute(
                "SELECT lineage_id, checkpoint_digest FROM heads ORDER BY lineage_id"
            ).fetchall()
            idempotency_rows = self._conn.execute(
                """SELECT lineage_id, idempotency_key, checkpoint_digest, expected_head
                   FROM idempotency ORDER BY lineage_id, idempotency_key"""
            ).fetchall()

        checkpoints: list[dict[str, Any]] = []
        for row in checkpoint_rows:
            checkpoint = self._row_checkpoint(row)
            assert checkpoint is not None
            checkpoints.append({
                "model_digest": checkpoint.model_digest,
                "schema_digest": checkpoint.schema_digest,
                "prefix_digest": checkpoint.prefix_digest,
                "parent_digest": checkpoint.parent_digest,
                "sequence": checkpoint.sequence,
                "state_payload_digest": checkpoint.state_payload_digest,
                "state_payload_hex": checkpoint.state_payload.hex(),
                "digest": checkpoint.digest,
            })

        snapshot: dict[str, Any] = {
            "schema_version": 1,
            "checkpoints": checkpoints,
            "heads": {row["lineage_id"]: row["checkpoint_digest"] for row in head_rows},
            "idempotency": [
                {
                    "lineage_id": row["lineage_id"],
                    "idempotency_key": row["idempotency_key"],
                    "checkpoint_digest": row["checkpoint_digest"],
                    "expected_head": row["expected_head"],
                }
                for row in idempotency_rows
            ],
        }
        InMemoryCheckpointStore.from_snapshot(snapshot)
        return snapshot
