# SPM / Rezon Chat Continuation — 2026-09-17 14:53 -0400

Restore token:
`SPM_REZON::RESTORE::SPM_REZON_CHAT_CONTINUATION_20260917T1453-0400`

Purpose: preserve the exact execution/research frontier of the BT2 SPM/Rezon chat before the ChatGPT conversation-length boundary. This file is continuation state/evidence, not qualification authority.

## Canonical / review boundaries

- SPM repo: `thebrazenbeard/spm`
- SPM PR: #2, DRAFT, open, unmerged.
- PR #2 remote review head remains exactly `be86721cd96f3f2b0f4ca2b43c3ee81dbf0d145b`.
- PR base remains `main@0ab6e6cd32a48a0afa22c8c27ec6bae67220d7e8`.
- Do NOT move/merge PR #2 merely to save continuation state.
- Fresh Radar/FNG hostile review was requested against `be86721...`; no completed exact-head verdict was observed in the last checks.
- Local/newer evaluator work must not inherit PASS/FAIL from that older exact review subject.
## Exact local SPM state at save

- Worktree: `C:\Vera\.worktrees\spm0-causal-state`
- Original working branch before save: `work/spm0-causal-state-20260914`
- Original local HEAD before save: `f629a56d1003f22f798e5250d8a6cbb250a17af4`
- Continuation branch created from that working state: `state/spm-rezon-chat-continuation-20260917-1453`
- Local commits newer than remote PR review head:
  - `30d493f3463de60547b0207fda77de15e4aab580` — bias-resistant constrained-choice evaluation
  - `ed1882f761e12ce8e6fe7d339d5158c36ab42245` — label-permutation baseline evaluation
  - `97d17472da30eb1366fbba289e368e8434530453` — presentation-invariant baseline choices
  - `be38bfc` — bias-controlled semantic baseline scoring
  - `f629a56d1003f22f798e5250d8a6cbb250a17af4` — batch balanced baseline cases

At the instant the continuation branch was created, two source/test files had verified but uncommitted changes:
- `src/spm_bench/hf_adapter.py`
- `tests/test_hf_adapter.py`
These add an equal-length prompt optimization using `logits_to_keep=1` and select the last returned logit while preserving the full-logit path for unequal-length batches.
## Fresh verification immediately before save

Current dirty continuation state was freshly verified:
- `python -m pytest tests\test_hf_adapter.py -q` => `10 passed`
- `python -m pytest -q` => `106 passed`
- `python -m compileall -q src tests` => PASS
- `git diff --check` => PASS (only Git CRLF conversion warning)

Earlier published runtime subject `be86721...` had 88/88 tests plus repeated in-memory and SQLite/WAL race/recovery stress, but those results remain bound to that exact older source subject.

## Baseline methodology frontier

The public-dev benchmark remains 36 cases across 9 semantic/pragmatic families. The earlier strict free-generation and bare-label constrained scores were found to be badly confounded by output-format, label, and option-position priors.

Current preferred diagnostic methodology is `balanced_semantic_score_v1`:
- method: `balanced_label_position_probability_mean`
- base selector: `forced_bracket_prefix_declared_label_softmax`
- layouts: crossed cyclic label + option-order presentations
- semantic option probability is averaged across balanced presentations
- answer key is not passed into model scoring
- ties are recorded explicitly.
## Baseline evidence currently on Lappy

Saved manifests under `state/baselines/` (runtime/evidence files, not canonical qualification):
- Qwen2.5-0.5B balanced semantic: 29/36, run `36ecac7cf2209afdf11c708bc1bd05d46a89a59da716d0cae079dc82379fca1c`, model digest `6080fc05cb5e0ccfa35e64523b11a902cc1f3e35672f85135a19eb16b722f8b8`.
- SmolLM2-360M balanced semantic: 10/36, run `2bbbae79442d93e93005d6e3d28359ebe8248c8c4820e4ecad2df452d2ad5c73`, model digest `3ac36bcfe007decda631d7d47b6503b3469a8f6d1bef090c662f0acfc449efb3`.
- Qwen2.5-1.5B balanced semantic, batch size 4: 32/36, run `a219925df71743745a8651aec7af8186f81dc89f65cf89da9ea9c0f9c8bfba64`, model digest `866e9c64bd94ae56fbaf5f2a46c0d6578b9a44baeab1ebc269e43018f3b1766d`.

Older presentation-consensus diagnostic manifests:
- Qwen2.5-0.5B: 3/36 presentation-invariant, 33 unstable, run `11e52a11e6173a37584c32ba1994095006d30a50acd35e4a30a10aaa0d9bea00`.
- SmolLM2-360M: 0/36 presentation-invariant, 36 unstable, run `02c91cf09858a04e4347f848e753c611f75fd3fa818c80427f1b82748a8f7708`.
These older results motivated probability-averaged balanced semantic scoring; do not use them as the current primary baseline metric.
## Inherited SmolLM3 subject

Pinned model: `HuggingFaceTB/SmolLM3-3B@a07cc9a04f16550a088caea529712d1d335b0ac1`
Local path: `C:\Vera\models\SmolLM3-3B-a07cc9a04f16550a088caea529712d1d335b0ac1`
Local inventory digest previously measured: `fd0ee8f56c88636d77521cf9082b14cc499cf11f34f8467c189b11791dc1decb`.
Runtime: CPU BF16; 12 PyTorch threads was faster than 6 threads in bounded tests.

Latest two-case `balanced_semantic_score_v1` probe finished successfully immediately before save:
- `spm0_public_dev_referent_01`: scores `{a: 0.0346832963, b: 0.9188543028, c: 0.0464624047}` => `b`, expected `b`, correct.
- `spm0_public_dev_ambiguity_03`: scores `{a: 0.0014639731, b: 0.0052519136, c: 0.9932840798}` => `c`, expected `c`, correct.
- elapsed: 270.02 s total / 135.01 s per case
- run digest: `ab091096db47bc9bdcb865b2017419549856c41aea390c3bb11dba3e2e0e068e`

Do not extrapolate 2/2 into full SmolLM3 benchmark qualification. The next useful step is to validate the `logits_to_keep=1` optimization on SmolLM3 and then run the full 36-case balanced semantic suite if runtime remains tractable.
## Rezon architecture / team frontier

User clarified the Rezon team is `One`, `Masa`, `Mune`, and `Rezon` (Rezon is itself a chat/participant, not just the repo).

Current architectural hypothesis: Rezon should be tested as an optional external durable continuity-of-inquiry / reasoning substrate, not merely a database and not an oracle. Vera should learn the semantic behavior/tool contract, not a hardcoded GitHub/storage path. SPM remains the learned meaning-centered cognition substrate; Rezon would preserve durable structured reasoning state.

Durable Bus assignments already sent:
- One integration assignment: Bus commit `09bf52595edb525f5f45e95b2286f3bef418b31a`
- Masa hostile assignment: `670769fd...`
- Mune verification-design assignment: `1e9f8c156167621a3d87e60d6f36ffaef0e61426`
- Rezon project-native architecture assignment: `77913547...`

Masa and Rezon independently converged on the same important boundary: compare Rezon against a strong structured-memory baseline using the same SQLite/WAL backend; Rezon only earns extra architecture if it preserves continuity-of-inquiry semantics (current/history, dependency invalidation, provenance, crash/replay/idempotency, unresolved failures) better than the simpler control. Rezon also emphasized `stored != admitted != current != true` and rejection of circular self-confirmation.

As of the last fresh check, One had not yet returned the requested integration response and Mune had not returned the requested substrate-verification design. Fresh-check the Bus in the next chat rather than assuming that is still true.
## Research / behavioral discipline to preserve

- Literal-first adversarial method: first attack the proposal as stated; do not rescue it by reinterpretation. If it survives, support it. If it fails, then preserve the underlying objective and propose a stronger replacement.
- Do not let attractive convergence (`HCAE + HRM + HyPER + PageIndex + KAG/Semantica/Ontosphere/Zelph + hyperbolic ideas`) become kitchen-sink architecture. Each mechanism must have a separable hypothesis, ablation, and kill condition.
- SPM core hypothesis remains: explicit semantic/pragmatic state is a causal computational center rather than only an emergent byproduct of token prediction.
- Distinguish runtime/source/build/install/behavioral qualification. Green tests are evidence, not `SPM_VERIFIED`.
- User wants subagents/independent lanes used for substantial technical work.
- Non-PR inter-agent coordination must go through `thebrazenbeard/chat-communication-bus`.

## Next execution frontier

1. In the continuation branch, preserve/verify the current `logits_to_keep=1` optimization; compare exact score outputs against the pre-optimization path, especially on SmolLM3, before committing it as anything stronger than WIP.
2. Run full SmolLM3 36-case `balanced_semantic_score_v1` only after optimization equivalence is demonstrated; freeze exact manifest, model digest, runtime precision, thread count, evaluator commit and benchmark digest.
3. Re-run/freeze Qwen2.5-0.5B, SmolLM2-360M and Qwen2.5-1.5B manifests if the evaluator code changes in any score-affecting way.
4. Determine baseline readiness; only then proceed to matched B* vs persistent C learned-state training.
5. Fresh-check Radar/FNG exact-head review status and Rezon team replies before carrying review conclusions forward.
6. Do not merge PR #2 without Patrick's explicit authority.
## New-chat restore procedure

In the next BT2 chat, send only:

`SPM_REZON::RESTORE::SPM_REZON_CHAT_CONTINUATION_20260917T1453-0400`

Then restore from the durable state branch named above. Fresh-read PR #2 and the Chat Bus before asserting current review status. Reconnect to Lappy device `937d921e-ecc8-4dc7-bc71-dee5f06ab653` and inspect the exact continuation branch/worktree before editing.

Do not infer durable participant continuity from the chat session itself. Treat the new chat as a replacement execution terminal attaching to the saved project state.

Qualification at save time remains:
- SPM runtime/source engineering: substantial verified evidence on exact subjects.
- Baseline methodology: actively hardened; latest preferred metric is balanced semantic scoring.
- Full inherited SmolLM3 baseline: NOT complete.
- Learned matched B*/C causal experiment: NOT run.
- `CAUSAL_MODEL_VERIFIED`: NO.
- `SPM_VERIFIED`: NO.
