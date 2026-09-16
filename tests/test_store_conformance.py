from __future__ import annotations

import pytest

from spm_bench.sqlite_store import SQLiteCheckpointStore
from spm_bench.state_runtime import (
    InMemoryCheckpointStore,
    LineageRuntime,
    canonical_prefix_digest,
)

MODEL = "3" * 64
SCHEMA = "4" * 64


def updater(previous: bytes, prefix) -> bytes:
    return previous + b"|" + canonical_prefix_digest(prefix).encode("ascii")[:8]


@pytest.fixture(params=("memory", "sqlite"))
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryCheckpointStore()
        return
    durable = SQLiteCheckpointStore(tmp_path / "state.db")
    try:
        yield durable
    finally:
        durable.close()


def test_common_root_advance_retry_and_resume(store):
    runtime = LineageRuntime(store, model_digest=MODEL, schema_digest=SCHEMA, updater=updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    prefix = ({"role": "user", "content": "one"},)
    child = runtime.advance(
        "main", expected_parent=root.digest, prefix=prefix, idempotency_key="c1"
    )
    retry = runtime.advance(
        "main", expected_parent=root.digest, prefix=prefix, idempotency_key="c1"
    )
    assert retry == child
    assert runtime.resume(child.digest) == child
    assert store.head("main") == child.digest
    assert store.resolve(MODEL, SCHEMA, child.prefix_digest) == child


def test_common_stale_rejection_and_explicit_fork(store):
    runtime = LineageRuntime(store, model_digest=MODEL, schema_digest=SCHEMA, updater=updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    runtime.fork("branch", root.digest)
    main = runtime.advance(
        "main", expected_parent=root.digest,
        prefix=({"role": "user", "content": "main"},), idempotency_key="m1",
    )
    branch = runtime.advance(
        "branch", expected_parent=root.digest,
        prefix=({"role": "user", "content": "branch"},), idempotency_key="b1",
    )
    assert main.digest != branch.digest

    with pytest.raises(ValueError, match="stale head"):
        runtime.advance(
            "main", expected_parent=root.digest,
            prefix=({"role": "user", "content": "stale"},), idempotency_key="stale",
        )


def test_common_snapshot_and_replay_are_exact(store):
    runtime = LineageRuntime(store, model_digest=MODEL, schema_digest=SCHEMA, updater=updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    prefixes = [
        ({"role": "user", "content": "one"},),
        ({"role": "user", "content": "two"},),
    ]
    first = runtime.advance(
        "main", expected_parent=root.digest, prefix=prefixes[0], idempotency_key="m1"
    )
    target = runtime.advance(
        "main", expected_parent=first.digest, prefix=prefixes[1], idempotency_key="m2"
    )
    replayed = runtime.replay(
        "replay", root.digest, prefixes, idempotency_prefix="replay"
    )
    assert replayed.digest == target.digest
    snapshot = store.snapshot()
    assert snapshot["heads"]["main"] == target.digest
    assert snapshot["heads"]["replay"] == target.digest
