# SPM-0 Lineage Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use test-driven development; each task must observe RED before production code.

**Goal:** Implement the source-bound opaque-state lineage runtime required by the adversarial SPM-0 design.

**Architecture:** Canonical message prefixes map to collision-resistant lineage digests. Immutable checkpoint envelopes bind model/schema/prefix/parent/sequence/state payload. A compare-and-swap store provides idempotent commits and rejects stale/different duplicate writes. A lineage runtime implements root, advance, fork, resume, reset, deterministic replay, and intervention-safe state substitution without interpreting payload semantics.

**Tech Stack:** Python 3.11 standard library, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-spm0-adversarial-state-runtime-design.md`

## Global constraints

- No learned SPM weights or hidden answer labels.
- State payload is opaque bytes.
- Unknown model/schema revisions fail closed.
- No mutable chat-local identity is authority for state selection.
- All committed transitions are immutable and content-bound.
- Duplicate identical idempotent requests return the same checkpoint.
- Duplicate keys with differing payloads fail closed.
- Stale-parent writes fail closed; explicit forks are separate lineages.

---

### Task 1 — canonical identities and checkpoint envelope

Create `src/spm_bench/state_runtime.py` and `tests/test_state_runtime.py`.

Tests first: canonical prefix hashing is deterministic/order-sensitive; payload digest is checked; envelope digest changes when parent/model/schema/prefix/sequence/state changes; malformed digests and negative sequence are rejected.
### Task 2 — compare-and-swap conformance store

Add `InMemoryCheckpointStore` with `get`, `commit`, and idempotency lookup.

Tests first: root/child commit succeeds; checkpoint mutation is impossible; identical retry returns same digest; same idempotency key with different transition fails; expected-parent mismatch fails; independently named fork from one parent succeeds; unknown checkpoint returns `None`.

### Task 3 — lineage operations and deterministic reconstruction

Add `LineageRuntime` and a deterministic synthetic updater protocol.

Tests first: `root()` binds model/schema; `advance()` increments sequence and binds canonical prefix; `reset()` returns root; forks from the same ancestor diverge; `resume()` retrieves exact checkpoint; missing descendant can be replayed from root/ancestor using supplied deterministic updater; replay reproduces the same digest.

### Task 4 — isolation, crash/reload, concurrency semantics

Tests first: two logical sessions with different prefixes cannot resolve one another; stale-worker advance is rejected; store serialization/reload preserves checkpoint identities; reordered requests produce distinct lineage; duplicate request is idempotent; forced eviction followed by replay reconstructs the exact checkpoint.

### Task 5 — intervention surface

Add non-mutating `intervene()` helpers for reset/freeze/substitute/scramble-test adapters. Interventions must produce derived experimental state without overwriting source checkpoints.

Tests first: substitution never mutates either source; freeze preserves prior payload while advancing experimental prefix; intervention provenance binds source checkpoint plus intervention descriptor; same intervention is deterministic.

### Final gate

Run focused tests, full suite, `compileall`, and `git diff --check`. Commit only on all-green. Runtime PASS remains separate from learned-model/behavioral qualification.
