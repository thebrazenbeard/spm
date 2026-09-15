from __future__ import annotations

import json
import threading

import pytest

from spm_bench.state_runtime import (
    CheckpointEnvelope,
    InMemoryCheckpointStore,
    LineageRuntime,
    canonical_prefix_digest,
    payload_digest,
)


D1 = "1" * 64
D2 = "2" * 64
D3 = "3" * 64


def test_canonical_prefix_digest_is_deterministic_and_order_sensitive():
    a = ({"role": "user", "content": "one"}, {"role": "assistant", "content": "two"})
    b = tuple(reversed(a))
    assert canonical_prefix_digest(a) == canonical_prefix_digest(a)
    assert canonical_prefix_digest(a) != canonical_prefix_digest(b)
    assert len(canonical_prefix_digest(a)) == 64


def test_payload_digest_is_content_bound():
    assert payload_digest(b"state") == payload_digest(b"state")
    assert payload_digest(b"state") != payload_digest(b"State")


def make_envelope(*, state=b"x", sequence=1, parent=D1, prefix=D2):
    return CheckpointEnvelope.create(
        model_digest=D3,
        schema_digest="4" * 64,
        prefix_digest=prefix,
        parent_digest=parent,
        sequence=sequence,
        state_payload=state,
    )


def test_checkpoint_envelope_binds_all_identity_fields():
    base = make_envelope()
    variants = [
        make_envelope(state=b"y"),
        make_envelope(sequence=2),
        make_envelope(parent="5" * 64),
        make_envelope(prefix="6" * 64),
        CheckpointEnvelope.create(
            model_digest="7" * 64, schema_digest="4" * 64,
            prefix_digest=D2, parent_digest=D1, sequence=1, state_payload=b"x"
        ),
    ]
    assert all(item.digest != base.digest for item in variants)
    assert base.state_payload_digest == payload_digest(b"x")


def test_checkpoint_envelope_rejects_bad_digests_and_sequence():
    with pytest.raises(ValueError, match="model_digest"):
        CheckpointEnvelope.create(
            model_digest="bad", schema_digest="4" * 64,
            prefix_digest=D2, parent_digest=D1, sequence=1, state_payload=b"x"
        )
    with pytest.raises(ValueError, match="sequence"):
        CheckpointEnvelope.create(
            model_digest=D3, schema_digest="4" * 64,
            prefix_digest=D2, parent_digest=D1, sequence=-1, state_payload=b"x"
        )
    with pytest.raises(TypeError, match="bytes"):
        CheckpointEnvelope.create(
            model_digest=D3, schema_digest="4" * 64,
            prefix_digest=D2, parent_digest=D1, sequence=1, state_payload="x"
        )



def test_store_commit_retry_and_lookup_are_content_addressed():
    store = InMemoryCheckpointStore()
    root = CheckpointEnvelope.create(
        model_digest=D3, schema_digest="4" * 64, prefix_digest="0" * 64,
        parent_digest=None, sequence=0, state_payload=b"root"
    )
    committed = store.commit(root, lineage_id="session-a", expected_head=None, idempotency_key="root-a")
    retried = store.commit(root, lineage_id="session-a", expected_head=None, idempotency_key="root-a")
    assert committed == retried == root
    assert store.get(root.digest) == root
    assert store.get("f" * 64) is None
    with pytest.raises(Exception):
        root.sequence = 9


def test_store_rejects_same_idempotency_key_with_different_transition():
    store = InMemoryCheckpointStore()
    root = CheckpointEnvelope.create(
        model_digest=D3, schema_digest="4" * 64, prefix_digest="0" * 64,
        parent_digest=None, sequence=0, state_payload=b"root"
    )
    store.commit(root, lineage_id="session-a", expected_head=None, idempotency_key="same")
    other = CheckpointEnvelope.create(
        model_digest=D3, schema_digest="4" * 64, prefix_digest="9" * 64,
        parent_digest=None, sequence=0, state_payload=b"other"
    )
    with pytest.raises(ValueError, match="idempotency"):
        store.commit(other, lineage_id="session-a", expected_head=None, idempotency_key="same")


def test_store_cas_rejects_stale_head_and_explicit_fork_allows_sibling():
    store = InMemoryCheckpointStore()
    root = CheckpointEnvelope.create(
        model_digest=D3, schema_digest="4" * 64, prefix_digest="0" * 64,
        parent_digest=None, sequence=0, state_payload=b"root"
    )
    store.commit(root, lineage_id="main", expected_head=None, idempotency_key="r")
    child = make_envelope(parent=root.digest, prefix="6" * 64, sequence=1, state=b"child")
    store.commit(child, lineage_id="main", expected_head=root.digest, idempotency_key="c")

    stale = make_envelope(parent=root.digest, prefix="7" * 64, sequence=1, state=b"stale")
    with pytest.raises(ValueError, match="stale head"):
        store.commit(stale, lineage_id="main", expected_head=root.digest, idempotency_key="stale")

    store.fork_lineage("branch", root.digest)
    sibling = make_envelope(parent=root.digest, prefix="8" * 64, sequence=1, state=b"sibling")
    store.commit(sibling, lineage_id="branch", expected_head=root.digest, idempotency_key="s")
    assert store.head("main") == child.digest
    assert store.head("branch") == sibling.digest


def synthetic_updater(previous: bytes, prefix) -> bytes:
    return previous + b"|" + canonical_prefix_digest(prefix).encode("ascii")[:8]


def test_lineage_root_advance_resume_and_reset_are_exact():
    store = InMemoryCheckpointStore()
    runtime = LineageRuntime(store, model_digest=D3, schema_digest="4" * 64, updater=synthetic_updater)
    root = runtime.root("session-a", initial_state=b"seed", idempotency_key="root")
    assert root.model_digest == D3
    assert root.schema_digest == "4" * 64
    assert root.sequence == 0
    assert root.parent_digest is None

    prefix = ({"role": "user", "content": "hello"},)
    child = runtime.advance("session-a", expected_parent=root.digest, prefix=prefix, idempotency_key="a1")
    assert child.sequence == 1
    assert child.parent_digest == root.digest
    assert child.prefix_digest == canonical_prefix_digest(prefix)
    assert runtime.resume(child.digest) == child
    assert runtime.reset(root.digest) == root


def test_lineage_explicit_forks_diverge_from_same_ancestor():
    store = InMemoryCheckpointStore()
    runtime = LineageRuntime(store, model_digest=D3, schema_digest="4" * 64, updater=synthetic_updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    store.fork_lineage("left", root.digest)
    store.fork_lineage("right", root.digest)

    left_prefix = ({"role": "user", "content": "left"},)
    right_prefix = ({"role": "user", "content": "right"},)
    left = runtime.advance("left", expected_parent=root.digest, prefix=left_prefix, idempotency_key="left-1")
    right = runtime.advance("right", expected_parent=root.digest, prefix=right_prefix, idempotency_key="right-1")
    assert left.digest != right.digest
    assert left.parent_digest == right.parent_digest == root.digest


def test_replay_reconstructs_same_checkpoint_digest():
    store = InMemoryCheckpointStore()
    runtime = LineageRuntime(store, model_digest=D3, schema_digest="4" * 64, updater=synthetic_updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    prefixes = [
        ({"role": "user", "content": "one"},),
        ({"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}),
    ]
    first = runtime.advance("main", expected_parent=root.digest, prefix=prefixes[0], idempotency_key="m1")
    second = runtime.advance("main", expected_parent=first.digest, prefix=prefixes[1], idempotency_key="m2")

    replayed = runtime.replay("replay", root.digest, prefixes, idempotency_prefix="replay")
    assert replayed.digest == second.digest
    assert replayed.state_payload == second.state_payload


def test_store_snapshot_reload_preserves_exact_checkpoint_and_heads():
    store = InMemoryCheckpointStore()
    runtime = LineageRuntime(store, model_digest=D3, schema_digest="4" * 64, updater=synthetic_updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    prefix = ({"role": "user", "content": "persist"},)
    child = runtime.advance("main", expected_parent=root.digest, prefix=prefix, idempotency_key="p1")

    snapshot = store.snapshot()
    json.dumps(snapshot)
    restored = InMemoryCheckpointStore.from_snapshot(snapshot)
    assert restored.get(child.digest) == child
    assert restored.head("main") == child.digest
    assert restored.resolve(D3, "4" * 64, child.prefix_digest) == child
    assert restored.resolve(D3, "4" * 64, "f" * 64) is None


def test_eviction_then_replay_reconstructs_exact_checkpoint():
    store = InMemoryCheckpointStore()
    runtime = LineageRuntime(store, model_digest=D3, schema_digest="4" * 64, updater=synthetic_updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    prefixes = [({"role": "user", "content": "one"},), ({"role": "user", "content": "two"},)]
    first = runtime.advance("main", expected_parent=root.digest, prefix=prefixes[0], idempotency_key="e1")
    target = runtime.advance("main", expected_parent=first.digest, prefix=prefixes[1], idempotency_key="e2")
    store.evict(target.digest)
    with pytest.raises(ValueError, match="unknown"):
        runtime.resume(target.digest)
    reconstructed = runtime.replay("recovered", root.digest, prefixes, idempotency_prefix="recover")
    assert reconstructed.digest == target.digest


def test_two_workers_racing_one_lineage_cannot_both_advance():
    barrier = threading.Barrier(2)

    def racing_updater(previous, prefix):
        barrier.wait(timeout=2)
        return synthetic_updater(previous, prefix)

    store = InMemoryCheckpointStore()
    runtime = LineageRuntime(store, model_digest=D3, schema_digest="4" * 64, updater=racing_updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    outcomes: list[str] = []

    def worker(name):
        try:
            runtime.advance(
                "main", expected_parent=root.digest,
                prefix=({"role": "user", "content": name},), idempotency_key=name,
            )
            outcomes.append("ok")
        except ValueError as error:
            outcomes.append(str(error))

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("left", "right")]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=3)
    assert outcomes.count("ok") == 1
    assert sum("stale head" in item for item in outcomes) == 1


def test_idempotency_key_is_scoped_to_lineage_not_global_store():
    store = InMemoryCheckpointStore()
    root = make_envelope(parent=None, prefix="0" * 64, sequence=0, state=b"root")
    store.commit(root, lineage_id="main", expected_head=None, idempotency_key="root")
    store.fork_lineage("branch", root.digest)

    main_child = make_envelope(parent=root.digest, prefix="a" * 64, sequence=1, state=b"main")
    branch_child = make_envelope(parent=root.digest, prefix="b" * 64, sequence=1, state=b"branch")
    store.commit(main_child, lineage_id="main", expected_head=root.digest, idempotency_key="same")
    store.commit(branch_child, lineage_id="branch", expected_head=root.digest, idempotency_key="same")

    assert store.head("main") == main_child.digest
    assert store.head("branch") == branch_child.digest


def test_store_rejects_forged_checkpoint_envelope():
    store = InMemoryCheckpointStore()
    valid = make_envelope(parent=None, prefix="0" * 64, sequence=0, state=b"root")
    forged = CheckpointEnvelope(
        model_digest=valid.model_digest,
        schema_digest=valid.schema_digest,
        prefix_digest=valid.prefix_digest,
        parent_digest=valid.parent_digest,
        sequence=valid.sequence,
        state_payload_digest=valid.state_payload_digest,
        state_payload=valid.state_payload,
        digest="f" * 64,
    )
    with pytest.raises(ValueError, match="integrity"):
        store.commit(forged, lineage_id="main", expected_head=None, idempotency_key="root")


def test_resolve_fails_closed_when_same_prefix_has_multiple_roots():
    store = InMemoryCheckpointStore()
    prefix = canonical_prefix_digest(())
    left = make_envelope(parent=None, prefix=prefix, sequence=0, state=b"left-root")
    right = make_envelope(parent=None, prefix=prefix, sequence=0, state=b"right-root")
    store.commit(left, lineage_id="left", expected_head=None, idempotency_key="root")
    store.commit(right, lineage_id="right", expected_head=None, idempotency_key="root")

    with pytest.raises(ValueError, match="ambiguous"):
        store.resolve(D3, "4" * 64, prefix)


def test_snapshot_rejects_checkpoint_whose_parent_is_missing():
    store = InMemoryCheckpointStore()
    runtime = LineageRuntime(store, model_digest=D3, schema_digest="4" * 64, updater=synthetic_updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    child = runtime.advance(
        "main", expected_parent=root.digest,
        prefix=({"role": "user", "content": "child"},), idempotency_key="c1",
    )
    snapshot = store.snapshot()
    snapshot["checkpoints"] = [item for item in snapshot["checkpoints"] if item["digest"] == child.digest]
    snapshot["heads"] = {}
    snapshot["idempotency"] = []

    with pytest.raises(ValueError, match="parent"):
        InMemoryCheckpointStore.from_snapshot(snapshot)


def test_replay_rehydrates_an_evicted_active_head_exactly():
    store = InMemoryCheckpointStore()
    runtime = LineageRuntime(store, model_digest=D3, schema_digest="4" * 64, updater=synthetic_updater)
    root = runtime.root("main", initial_state=b"seed", idempotency_key="root")
    prefixes = [({"role": "user", "content": "one"},)]
    target = runtime.advance("main", expected_parent=root.digest, prefix=prefixes[0], idempotency_key="m1")
    store.evict(target.digest)
    assert store.head("main") == target.digest
    assert store.get(target.digest) is None

    replayed = runtime.replay("repair", root.digest, prefixes, idempotency_prefix="repair")
    assert replayed.digest == target.digest
    assert store.get(store.head("main")) == target


def test_store_rejects_nonzero_sequence_root():
    store = InMemoryCheckpointStore()
    invalid_root = make_envelope(parent=None, prefix="0" * 64, sequence=2, state=b"bad-root")
    with pytest.raises(ValueError, match="root sequence"):
        store.commit(invalid_root, lineage_id="main", expected_head=None, idempotency_key="root")


def test_store_rejects_child_sequence_gap():
    store = InMemoryCheckpointStore()
    root = make_envelope(parent=None, prefix="0" * 64, sequence=0, state=b"root")
    store.commit(root, lineage_id="main", expected_head=None, idempotency_key="root")
    gap = make_envelope(parent=root.digest, prefix="a" * 64, sequence=3, state=b"gap")
    with pytest.raises(ValueError, match="sequence"):
        store.commit(gap, lineage_id="main", expected_head=root.digest, idempotency_key="gap")


def test_store_rejects_child_subject_mismatch():
    store = InMemoryCheckpointStore()
    root = make_envelope(parent=None, prefix="0" * 64, sequence=0, state=b"root")
    store.commit(root, lineage_id="main", expected_head=None, idempotency_key="root")
    child = CheckpointEnvelope.create(
        model_digest="9" * 64,
        schema_digest=root.schema_digest,
        prefix_digest="a" * 64,
        parent_digest=root.digest,
        sequence=1,
        state_payload=b"wrong-subject",
    )
    with pytest.raises(ValueError, match="subject"):
        store.commit(child, lineage_id="main", expected_head=root.digest, idempotency_key="subject")
