from __future__ import annotations

import sqlite3
import threading

import pytest

from spm_bench.sqlite_store import SQLiteCheckpointStore
from spm_bench.state_runtime import (
    CheckpointEnvelope,
    LineageRuntime,
    canonical_prefix_digest,
)


MODEL = "3" * 64
SCHEMA = "4" * 64


def updater(previous: bytes, prefix) -> bytes:
    return previous + b"|" + canonical_prefix_digest(prefix).encode("ascii")[:8]


def make_root(state=b"seed"):
    return CheckpointEnvelope.create(
        model_digest=MODEL,
        schema_digest=SCHEMA,
        prefix_digest=canonical_prefix_digest(()),
        parent_digest=None,
        sequence=0,
        state_payload=state,
    )


def test_close_reopen_preserves_exact_checkpoint_and_head(tmp_path):
    path = tmp_path / "state.db"
    store = SQLiteCheckpointStore(path)
    runtime = LineageRuntime(store, model_digest=MODEL, schema_digest=SCHEMA, updater=updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    child = runtime.advance(
        "main", expected_parent=root.digest,
        prefix=({"role": "user", "content": "persist"},), idempotency_key="p1",
    )
    store.close()

    reopened = SQLiteCheckpointStore(path)
    try:
        assert reopened.get(child.digest) == child
        assert reopened.head("main") == child.digest
        assert reopened.resolve(MODEL, SCHEMA, child.prefix_digest) == child
    finally:
        reopened.close()


def test_lineage_scoped_idempotency_survives_reopen(tmp_path):
    path = tmp_path / "state.db"
    store = SQLiteCheckpointStore(path)
    root = make_root()
    store.commit(root, lineage_id="main", expected_head=None, idempotency_key="root")
    store.fork_lineage("branch", root.digest)
    store.close()

    reopened = SQLiteCheckpointStore(path)
    try:
        main = CheckpointEnvelope.create(
            model_digest=MODEL, schema_digest=SCHEMA, prefix_digest="a" * 64,
            parent_digest=root.digest, sequence=1, state_payload=b"main",
        )
        branch = CheckpointEnvelope.create(
            model_digest=MODEL, schema_digest=SCHEMA, prefix_digest="b" * 64,
            parent_digest=root.digest, sequence=1, state_payload=b"branch",
        )
        reopened.commit(main, lineage_id="main", expected_head=root.digest, idempotency_key="same")
        reopened.commit(branch, lineage_id="branch", expected_head=root.digest, idempotency_key="same")
        assert reopened.head("main") == main.digest
        assert reopened.head("branch") == branch.digest
    finally:
        reopened.close()


def test_two_connections_racing_one_lineage_yield_one_winner(tmp_path):
    path = tmp_path / "state.db"
    setup = SQLiteCheckpointStore(path)
    root = make_root()
    setup.commit(root, lineage_id="main", expected_head=None, idempotency_key="root")
    setup.close()

    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(name: str):
        store = SQLiteCheckpointStore(path)
        try:
            prefix = ({"role": "user", "content": name},)
            child = CheckpointEnvelope.create(
                model_digest=MODEL,
                schema_digest=SCHEMA,
                prefix_digest=canonical_prefix_digest(prefix),
                parent_digest=root.digest,
                sequence=1,
                state_payload=name.encode(),
            )
            barrier.wait(timeout=3)
            try:
                store.commit(child, lineage_id="main", expected_head=root.digest, idempotency_key=name)
                result = "ok"
            except ValueError as error:
                result = str(error)
            with lock:
                outcomes.append(result)
        finally:
            store.close()

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("left", "right")]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=5)
    assert outcomes.count("ok") == 1
    assert sum("stale head" in item for item in outcomes) == 1


def test_corrupted_durable_payload_fails_closed_on_readback(tmp_path):
    path = tmp_path / "state.db"
    store = SQLiteCheckpointStore(path)
    root = make_root()
    store.commit(root, lineage_id="main", expected_head=None, idempotency_key="root")
    store.close()

    raw = sqlite3.connect(path)
    try:
        raw.execute("UPDATE checkpoints SET state_payload = ? WHERE digest = ?", (b"tampered", root.digest))
        raw.commit()
    finally:
        raw.close()

    reopened = SQLiteCheckpointStore(path)
    try:
        with pytest.raises(ValueError, match="integrity"):
            reopened.get(root.digest)
    finally:
        reopened.close()


def test_snapshot_is_json_safe_and_matches_reopened_subject(tmp_path):
    path = tmp_path / "state.db"
    store = SQLiteCheckpointStore(path)
    runtime = LineageRuntime(store, model_digest=MODEL, schema_digest=SCHEMA, updater=updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    child = runtime.advance(
        "main", expected_parent=root.digest,
        prefix=({"role": "user", "content": "snapshot"},), idempotency_key="s1",
    )
    snapshot = store.snapshot()
    store.close()

    import json
    json.dumps(snapshot)
    assert snapshot["schema_version"] == 1
    assert snapshot["heads"]["main"] == child.digest
    assert any(item["digest"] == child.digest for item in snapshot["checkpoints"])

    reopened = SQLiteCheckpointStore(path)
    try:
        assert reopened.get(child.digest) == child
    finally:
        reopened.close()


def test_resolve_rejects_malformed_subject_digests(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError, match="model_digest"):
            store.resolve("bad", SCHEMA, "a" * 64)
        with pytest.raises(ValueError, match="schema_digest"):
            store.resolve(MODEL, "bad", "a" * 64)
        with pytest.raises(ValueError, match="prefix_digest"):
            store.resolve(MODEL, SCHEMA, "bad")
    finally:
        store.close()


def test_snapshot_rejects_dangling_head_after_fault_eviction(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.db")
    runtime = LineageRuntime(store, model_digest=MODEL, schema_digest=SCHEMA, updater=updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    child = runtime.advance(
        "main", expected_parent=root.digest,
        prefix=({"role": "user", "content": "x"},), idempotency_key="x",
    )
    store.evict(child.digest)
    try:
        with pytest.raises(ValueError, match="head"):
            store.snapshot()
    finally:
        store.close()
