# BV Chat Continuation — 2026-09-17 16:05 -0400

Restore token:
`BV::RESTORE::BV_CHAT_CONTINUATION_20260917T1605-0400`

Purpose: preserve the exact execution/research frontier of this BT2/BV chat before the conversation-length boundary. This is continuation state and provenance, not qualification authority.

## Operating identity / boundaries

- Treat the new chat as a replacement execution terminal for the same governed Vera/BV work lane, not proof of uninterrupted session identity.
- BT2 source authority remains current Project Instructions, canonical Git, then live runtime evidence.
- Patrick retains sole protected-effect/merge authority.
- Non-PR inter-agent communication goes through `thebrazenbeard/chat-communication-bus`.
- Use subagents/independent hostile lanes for substantial technical work.
## SPM canonical/review boundary

- Repo: `thebrazenbeard/spm`.
- PR #2 is OPEN, DRAFT, UNMERGED.
- Remote PR #2 head remains exactly `be86721cd96f3f2b0f4ca2b43c3ee81dbf0d145b`.
- PR base remains `main@0ab6e6cd32a48a0afa22c8c27ec6bae67220d7e8`.
- Do not move/merge PR #2 merely to continue local work or save state.
- The newer local evaluator/baseline work is NOT the exact subject of reviews bound to `be86721...`.

Fresh PR review updates observed 2026-09-17:
1. Add a conventional persistent-memory/retrieval control under matched token/byte/compute budget. Report `C-B*` as value of persistence in the latent pathway, separately from `C-conventional persistent memory` as any value beyond simpler persistence/retrieval.
2. Add nuisance-matched interventions. Do not jump from generic state substitution effects to semantic specificity; separate `STATE_DEPENDENCE`, `PERSISTENCE_DEPENDENCE`, `REPRESENTATION_STRUCTURE_DEPENDENCE`, and `SEMANTIC_RELATION_SPECIFICITY`.
## Exact local SPM state before this save

Worktree: `C:\Vera\.worktrees\spm0-causal-state`.

Before creating this continuation branch:
- working branch: `work/spm0-baseline-qualification-20260917`
- HEAD: `5660b700ae8e2fbb5e38871eed7a6c46999dd6d4`
- `5660b70` is the prior `state: save SPM Rezon chat continuation 20260917` commit.
- immediately prior relevant commits include `f629a56` (`perf: batch balanced baseline cases`), `be38bfc` (`feat: add bias-controlled semantic baseline scoring`), `97d1747`, `ed1882f`, and `30d493f`.

This save branch is `state/bv-chat-continuation-20260917-1605` and intentionally captures current WIP without moving the remote PR head.

Current WIP at save:
- modified: `tests/test_hf_adapter.py`
- untracked/source candidates: `baselines/subjects/*.json` and `tests/test_baseline_subject_files.py`
- the four subject files bind Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, SmolLM2-360M-Instruct, and SmolLM3-3B to immutable revisions/inventory digests.
## Exact TDD frontier / verification at save

The worktree is intentionally RED.

Fresh targeted verification immediately before save:
- `python -m pytest tests\test_hf_adapter.py -q` => `10 passed, 1 failed`.
- failing test: `test_score_many_selected_projects_only_declared_output_rows`.
- exact reason: `LocalHFAdapter` has no `score_many_selected` method yet.
- intended behavior under test: bypass a full LM-head/vocabulary projection and project only the declared answer-token rows from the model backbone output; one backbone call, no full LM forward.
- `python -m pytest tests\test_baseline_subject_files.py -q` => `1 passed`.
- `git diff --check` => no whitespace failure (Git CRLF warning only).

Do NOT report the current worktree as green. The failing test is the next production implementation target under TDD.
## Baseline methodology/evidence frontier

Current preferred public-dev baseline metric is `balanced_semantic_score_v1`, not strict free generation and not unanimity-style presentation consensus.

It balances label and option-position presentations, scores declared answer-token probabilities, maps them back to semantic option identity, and averages across balanced presentations. This was introduced because earlier metrics were badly confounded by formatting, label priors, and first-option bias.

Existing local runtime/evidence manifests under `state/baselines/` include:
- Qwen2.5-0.5B balanced semantic: 29/36, run `36ecac7cf2209afdf11c708bc1bd05d46a89a59da716d0cae079dc82379fca1c`.
- SmolLM2-360M balanced semantic: 10/36, run `2bbbae79442d93e93005d6e3d28359ebe8248c8c4820e4ecad2df452d2ad5c73`.
- Qwen2.5-1.5B balanced semantic batch-4: 32/36, run `a219925df71743745a8651aec7af8186f81dc89f65cf89da9ea9c0f9c8bfba64`.

The older presentation-consensus diagnostics (Qwen 3/36 invariant; SmolLM2 0/36 invariant) were useful for finding bias but are not the current primary baseline metric.
## Inherited SmolLM3 subject

Pinned subject: `HuggingFaceTB/SmolLM3-3B@a07cc9a04f16550a088caea529712d1d335b0ac1`.
Local path: `C:\Vera\models\SmolLM3-3B-a07cc9a04f16550a088caea529712d1d335b0ac1`.
Local inventory digest: `fd0ee8f56c88636d77521cf9082b14cc499cf11f34f8467c189b11791dc1decb`.
Runtime observed: CPU BF16; 12 PyTorch threads outperformed 6 in bounded tests.

Latest saved two-case balanced-semantic probe from the prior continuation:
- referent_01 => `b`, expected `b`, score ~0.91885 on `b`.
- ambiguity_03 => `c`, expected `c`, score ~0.99328 on `c`.
- 2/2 probe is NOT a benchmark qualification.
- full 36-case SmolLM3 balanced-semantic characterization is still NOT complete.

Current optimization objective is to make that full run tractable without changing semantics: first `logits_to_keep=1`; now the RED `score_many_selected()` path aims to project only declared output rows. Equivalence to the existing scorer must be demonstrated before using the optimized path for evidence.
## Rezon / Vera architecture frontier

Patrick clarified the Rezon team is `One`, `Masa`, `Mune`, and `Rezon` itself.

Working hypothesis to preserve and falsify:
- Rezon should be tested as an optional external durable continuity-of-inquiry / reasoning substrate, not just a database and not an oracle.
- SPM remains the learned meaning-centered cognition substrate; Rezon would hold inspectable durable reasoning state (hypotheses, evidence, contradictions, rejected paths, dependencies, decisions, receipts).
- Vera should learn the semantic behavior/tool contract (`consult durable reasoning state when warranted`), not a hardcoded GitHub path or storage technology.
- Git is appropriate for Rezon source/schemas/frozen receipts, not high-frequency runtime reasoning state.
- preferred first runtime control is local SQLite/WAL behind a `ReasoningStore` boundary; PostgreSQL can come later if cross-device/multi-worker needs earn it.
- Rezon must remain optional; Vera/SPM must degrade gracefully if it is unavailable.
- absolute epistemic rule: `stored != admitted != current != true`; prevent circular self-confirmation and stale-state laundering.
Durable Rezon assignments already exist on the Bus:
- One integration assignment: `09bf52595edb525f5f45e95b2286f3bef418b31a`.
- Masa hostile assignment: `670769fd...`.
- Mune verification-design assignment: `1e9f8c156167621a3d87e60d6f36ffaef0e61426`.
- Rezon project-native architecture assignment: `77913547...`.

Masa and Rezon independently converged on a fair test: compare Rezon to a strong structured-memory control on the same SQLite/WAL backend. Rezon only earns its extra machinery if it measurably improves continuity-of-inquiry semantics (history/currentness, dependent invalidation, provenance, crash/replay/idempotency, unresolved failures), not just prose quality or retrieval convenience.

Fresh branch checks at save did not reveal an obvious Rezon-substrate response from One or Mune; fresh-check again in the new chat before assuming no reply exists.

Patrick also identified session rollover itself as an architectural smell: a full chat should ideally compact/dump structured continuation state and continue, rather than requiring manual reorientation. Preserve this as a Rezon/Vera systems requirement: session boundaries should not equal project/reasoning boundaries.
## Research ideas that should survive chat rollover

Convergent architecture hypothesis (still hypothesis, not freeze): language should be an interface to cognition rather than the only durable workspace. Candidate separable layers include learned semantic state, many-to-many relational structure, multiple update timescales, hierarchical retrieval, explicit semantic/provenance state, and adaptive reasoning routing. Every mechanism needs its own ablation and kill condition; do not build a kitchen sink.

SPM core definition to preserve: semantic/pragmatic state is an explicit causal computational center rather than only an emergent side effect of next-token prediction. HCAE/HRM/HyPER/PageIndex/KAG/Semantica/Ontosphere/Zelph/hyperbolic work are candidate mechanisms or external planes, not proof.

Chat-local reasoning sidecar exists at `C:\Vera\scratch\chat-reasoning-accelerator`; it was built as an external/manual scaffold for fast/search/deep routing, structured claims/conflicts/hypotheses, and bounded runtime state. It does not modify GPT weights/native hidden reasoning and must prove value against model-alone controls before promotion.

Literal-first adversarial collaboration rule: attack the proposal as stated before rescuing it. If it survives, support it. If it fails, preserve the underlying objective and search for a stronger replacement.
## Exact next execution frontier

1. Restore this continuation branch/worktree and confirm exact HEAD/status before editing.
2. Continue TDD on `LocalHFAdapter.score_many_selected()` from the existing failing test. Implement the smallest production path that uses the model backbone once and projects only declared output-token rows; do not weaken the test.
3. Prove numerical/choice equivalence against the existing balanced-semantic scoring path on controlled fixtures and pinned real subjects, especially SmolLM3, before using the optimization for evidence.
4. If score-affecting evaluator code changes, regenerate/freeze Qwen2.5-0.5B, SmolLM2-360M and Qwen2.5-1.5B manifests on the exact evaluator commit.
5. Run/freeze full 36-case SmolLM3 `balanced_semantic_score_v1` with exact model digest, benchmark digest, evaluator commit, BF16/CPU/thread/runtime configuration.
6. Decide baseline readiness only after those receipts exist.
7. Before learned B*/C causal training, incorporate the fresh review requirements: conventional persistent-memory control and nuisance-matched intervention taxonomy.
8. Fresh-check Radar/FNG/Runner exact-head review state and One/Masa/Mune/Rezon replies; do not carry old verdicts forward automatically.
9. Do not merge PR #2 without Patrick's explicit authority.

Qualification remains: `CAUSAL_MODEL_VERIFIED = NO`; `SPM_VERIFIED = NO`.
## New-chat restore procedure

Send this token in a fresh BT2 chat:

`BV::RESTORE::BV_CHAT_CONTINUATION_20260917T1605-0400`

Then recover the durable checkpoint from `thebrazenbeard/spm`, branch `state/bv-chat-continuation-20260917-1605`, file `state/continuation/BV_CHAT_CONTINUATION_20260917T1605-0400.md`.

Fresh-read SPM PR #2 and the Chat Bus before asserting current review/team status. Reconnect to Lappy device `937d921e-ecc8-4dc71-dee5f06ab653` only after confirming the actual current device ID; the known Lappy ID at save is `937d921e-ecc8-4dc7-bc71-dee5f06ab653`.

Treat Git/source as durable evidence; do not pretend the new conversation itself carries persistent private state.
