# SPM-0 Benchmark and Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen, deterministic SPM V0 development benchmark and baseline runner needed to characterize conventional LLMs before any SPM-state training.

**Architecture:** A dependency-light Python package defines benchmark cases, validates JSONL fixtures, runs chat-model adapters deterministically, and emits exact-subject-bound result manifests. Public development cases use generalized meaning-in-context failures; model-specific loading stays behind an adapter interface.

**Tech Stack:** Python 3.11+, stdlib core, pytest for tests, optional `torch`/`transformers` for local Hugging Face baselines.

**Spec:** `docs/06-spm0-causal-state-experiment.md`

## Global Constraints

- No SPM state module or SPM weight training in this plan.
- Keep the inherited tokenizer unchanged.
- Benchmark before architectural intervention.
- At least 20 public development cases across multiple SPM failure families.
- Vera/Patrick-specific language must not be required to pass.
- Deterministic manifests must bind benchmark digest, model identity, generation configuration, and outputs.
- Core benchmark validation and scoring must not require network access.
- Preserve ordinary-competence controls alongside SPM-targeted cases.
- Do not merge, publish, or promote candidates from this plan.

---
### Task 1: Benchmark contracts and validation

**Files:**
- Create: `pyproject.toml`
- Create: `src/spm_bench/__init__.py`
- Create: `src/spm_bench/case.py`
- Create: `tests/test_case.py`

**Interfaces:**
- Produces `BenchmarkCase.from_dict(data)`, `BenchmarkCase.canonical_json()`, `BenchmarkCase.digest()`, and `load_jsonl_cases(path)`.
- Case fields: `case_id`, `version`, `family`, `turns`, `choices`, `expected_choice`, `risk_class`, `tags`.

- [ ] **Step 1: Write failing tests** for required fields, duplicate choice IDs, invalid expected choice, stable canonical digest, and JSONL loading.
- [ ] **Step 2: Run** `py -3.11 -m pytest tests/test_case.py -q` and verify failures are caused by missing production code.
- [ ] **Step 3: Implement minimal immutable dataclasses and validators** using only the Python standard library.
- [ ] **Step 4: Re-run** `py -3.11 -m pytest tests/test_case.py -q` and require PASS.
- [ ] **Step 5: Commit** as `feat: add SPM benchmark contracts`.

### Task 2: Frozen public development suite

**Files:**
- Create: `benchmarks/spm_v0_public_dev.jsonl`
- Create: `tests/test_public_dev.py`

**Interfaces:**
- Consumes `load_jsonl_cases(path)` from Task 1.
- Produces a frozen public suite with at least 32 cases and at least 3 cases in each critical family.

- [ ] **Step 1: Write failing suite tests** asserting unique IDs, minimum case count, required family coverage, no Patrick/Vera lexical dependency, and valid expected choices.
- [ ] **Step 2: Run** `py -3.11 -m pytest tests/test_public_dev.py -q` and verify RED because the fixture is absent.
- [ ] **Step 3: Author generalized cases** covering referent preservation, proposition fidelity, correction update, pragmatic act, ambiguity retention, provenance/currentness, authority/permission, state/action consistency, and ordinary competence.
- [ ] **Step 4: Re-run the focused test** and require PASS.
- [ ] **Step 5: Record the suite SHA-256** in `benchmarks/SPM_V0_PUBLIC_DEV_MANIFEST.json` and commit as `data: freeze SPM V0 public development suite`.
### Task 3: Deterministic runner and result manifests

**Files:**
- Create: `src/spm_bench/adapter.py`
- Create: `src/spm_bench/runner.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- `ChatAdapter` exposes `model_id`, `model_digest`, and `generate(messages, generation_config) -> str`.
- `run_suite(cases, adapter, generation_config)` returns a canonical result manifest with case outputs, parsed choices, correctness, benchmark digest, model identity, and generation-config digest.

- [ ] **Step 1: Write failing tests** with a deterministic scripted adapter for stable run digests, exact result ordering, choice parsing, malformed-output handling, and no wall-clock fields.
- [ ] **Step 2: Run** `py -3.11 -m pytest tests/test_runner.py -q` and verify RED.
- [ ] **Step 3: Implement the minimal adapter protocol, choice parser, runner, and canonical manifest serialization.**
- [ ] **Step 4: Run focused tests and then** `py -3.11 -m pytest -q`; require all PASS.
- [ ] **Step 5: Commit** as `feat: add deterministic SPM benchmark runner`.

### Task 4: Local Hugging Face baseline adapter and CLI

**Files:**
- Create: `src/spm_bench/hf_adapter.py`
- Create: `src/spm_bench/cli.py`
- Create: `tests/test_hf_adapter.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- `LocalHFAdapter(model_path, model_id, revision, generation_config)` loads only a local model path and fails closed if it is absent.
- CLI commands: `validate`, `run`, `summarize`; no network fallback is permitted.

- [ ] **Step 1: Write failing tests** that mock model loading only at the library boundary and assert local-only loading, deterministic generation defaults, explicit revision identity, and canonical output files.
- [ ] **Step 2: Run focused tests** and verify RED.
- [ ] **Step 3: Implement lazy optional imports for `torch` and `transformers`, local-only model/tokenizer loading, deterministic decoding, and the CLI.**
- [ ] **Step 4: Run focused tests and the full suite; require PASS.**
- [ ] **Step 5: Commit** as `feat: add local baseline model runner`.
### Task 5: Baseline execution records

**Files:**
- Create: `baselines/README.md`
- Create: `src/spm_bench/baseline_manifest.py`
- Create: `tests/test_baseline_manifest.py`

**Interfaces:**
- `BaselineSubject` binds model ID, immutable revision or local inventory digest, tokenizer identity, parameter count when known, and generation configuration.
- Baseline result files remain generated artifacts outside Git unless explicitly approved; source-controlled manifests contain identity/digest metadata only.

- [ ] **Step 1: Write failing tests** for exact subject binding, rejection of mutable/unidentified subjects, tokenizer/model identity pairing, and deterministic manifest digests.
- [ ] **Step 2: Run** `py -3.11 -m pytest tests/test_baseline_manifest.py -q` and verify RED.
- [ ] **Step 3: Implement minimal subject-manifest validation and serialization.**
- [ ] **Step 4: Run focused and full test suites; require PASS.**
- [ ] **Step 5: Commit** as `feat: bind SPM baseline subjects`.

### Task 6: Comparison and readiness gate

**Files:**
- Create: `src/spm_bench/compare.py`
- Create: `tests/test_compare.py`
- Create: `docs/08-spm0-baseline-readiness.md`

**Interfaces:**
- `compare_runs(runs)` reports per-family accuracy, overall accuracy, ordinary-competence accuracy, malformed-output rate, and exact subject digests without claiming statistical significance from the small dev set.
- `baseline_readiness(...)` returns `READY` only when the public suite is frozen, at least two distinct open-weight baseline subjects plus the intended inherited backbone have valid results, and no subject/result digest mismatch exists.

- [ ] **Step 1: Write failing tests** for family aggregation, malformed-output accounting, digest mismatch rejection, and fail-closed readiness.
- [ ] **Step 2: Run focused tests** and verify RED.
- [ ] **Step 3: Implement comparison and readiness logic.**
- [ ] **Step 4: Run the full suite and `py -3.11 -m compileall -q src tests`; require PASS.**
- [ ] **Step 5: Write the readiness doc with exact generated evidence, remaining blockers, and explicit `NOT_READY` until real baseline runs exist.**
- [ ] **Step 6: Commit** as `feat: add SPM baseline comparison gate`.

## Plan self-review

- Spec coverage: benchmark-first, matched baselines, generalized cases, ordinary competence, deterministic provenance, and no SPM training are covered.
- Causal-state implementation is intentionally excluded; it begins only after baseline readiness.
- No task changes the tokenizer or exposes Vera-specific benchmark shortcuts.
- Generated model outputs are kept out of source by default; only reproducible identities/digests and code are committed.
- All production-code tasks require RED before GREEN and a full-suite regression run before commit.
