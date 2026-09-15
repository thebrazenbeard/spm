# SPM-0 Adversarial Causal-State Runtime Design

## Status and authority

APPROVED FOR BOUNDED IMPLEMENTATION AND VERIFICATION by Patrick's live 2026-09-15 instruction. This design does not authorize relabeling an unqualified experiment as SPM success.

The literal proposition under test is: a learned persistent semantic/pragmatic state, causally reinjected into model computation, provides a reproducible meaning-in-context advantage that matched conventional/transient controls cannot explain.

The implementation must try to falsify that proposition before rescuing it. If it fails, the underlying objective may be recovered only after the literal architecture has been rejected.

## Adversarial-collaboration invariant

Every architecture, training, runtime, and qualification decision follows this order:

1. state the literal proposition and frozen subject;
2. identify assumptions and a concrete kill test;
3. run the strongest practical opposing control without weakening the proposal;
4. record PASS/FAIL/UNKNOWN against the exact subject;
5. only after a literal FAIL, infer the underlying objective and propose a successor;
6. do not manufacture objections after evidence supports the proposition.

A self-review is required before independent hostile review. Independent Radical and Pragmatic hostile review remains mandatory for qualification.

## Experimental arms and fairness

Arm A is the untouched inherited LLM.

Arm B is conventional post-training on the same semantic/pragmatic material with no added latent-state pathway.

Arm B* is the matched transient-state control. It receives the same added parameter budget, updater/reinjection topology, auxiliary state decoders, labels/objectives, optimizer exposure, training tokens, steps, and bounded compute as Arm C. Its latent state is reset before every model-relevant transition and is recomputed only from the current allowed textual context.

Arm C is the SPM candidate. It differs from B* only in that the learned latent state from the prior transition is an input to the next state update and is causally reinjected into subsequent model computation.

Primary architectural credit is C versus B*. A and B remain diagnostic baselines. If B* matches or beats C under the frozen budget, the persistent-state intervention fails even if C beats A or B.

Pre-qualification manifests must bind parameter deltas, trainable-parameter count, train tokens, optimizer steps, effective training FLOPs estimate, inference-token budget, state bytes, generation settings, seeds, and all artifact digests.

## State representation and causal path

The state is an opaque learned tensor, not a textual scratchpad and not a symbolic database. The first candidate uses a fixed-width latent vector or short latent-token bank with three explicit hooks:

- `update(prior_state, observation_hidden) -> next_state`;
- `reinject(next_state, backbone_hidden) -> conditioned_hidden`;
- intervention hooks for lesion, freeze, substitute, scramble, and matched-noise tests.

Auxiliary decoders may predict semantic/pragmatic labels during training and evaluation, but decoder outputs are never fed back as text. At inference, removing decoder heads must not remove the behavioral effect if the latent pathway is genuinely causal.

B* uses the same hooks and tensor shape but supplies the updater a zero/reset prior at each transition. C supplies the prior latent state. This is the principal recurrence/persistence contrast.

The inherited tokenizer and autoregressive decoder remain unchanged in SPM-0.

## Content-addressed state lineage runtime

The ordinary chat API remains the external contract. The runtime derives an internal lineage key from a canonicalized conversation prefix plus model revision and state-schema revision. Clients do not need a custom semantic-state protocol.

For each request, the runtime:

1. canonicalizes the exact message prefix;
2. computes a collision-resistant prefix digest;
3. resolves a checkpoint for that prefix/model/schema tuple;
4. on a cache miss, deterministically replays from the nearest valid ancestor or root;
5. advances state only after the request transition is accepted;
6. stores the child checkpoint under the resulting lineage digest.

This makes reset, fork, retry, resume, and regeneration explicit without relying on mutable chat-local identity. A reset starts from root. A fork naturally has a different prefix digest. An exact retry resolves the same parent and idempotency key. Resume reconstructs from durable checkpoints or replay.

## Isolation, concurrency, and recovery

A checkpoint is valid only when its envelope binds model digest, state-schema digest, prefix digest, parent digest, turn sequence, state payload digest, and state payload. Unknown or mismatched revisions are rejected rather than coerced.

Runtime mutation uses compare-and-swap semantics on the expected parent/sequence. Concurrent children from the same parent are legal forks; accidental double-advance of one lineage is not. Duplicate requests with the same idempotency digest return the same committed transition or fail closed if their payload differs.

Qualification must test:

- two-user/interleaved histories with no cross-lineage contamination;
- duplicate and reordered requests;
- stale-parent and stale-worker writes;
- fork/regeneration from an earlier prefix;
- crash/restart followed by checkpoint reload;
- missing/corrupt checkpoint followed by deterministic replay;
- multi-worker access to a shared store;
- eviction/reload under increasing concurrent lineages.

The experimental in-memory store and the durable store must obey the same interface and conformance tests.

## Frozen verification ledger

Before any qualifying training run, freeze a machine-readable experiment manifest containing:

- base model and tokenizer identifiers/revisions/digests;
- architecture and state-schema configuration;
- dataset and split digests, including blind-holdout receipt without plaintext leakage;
- seeds and deterministic settings;
- optimizer, scheduler, precision, batch/accumulation configuration;
- trainable parameters, tokens, steps, and compute estimates for every arm;
- evaluation-harness and benchmark digests;
- generation configuration;
- checkpoint and exported-runtime digests;
- environment/toolchain versions.

No threshold may be chosen after qualifying results are visible. Budget mismatch outside a pre-frozen tolerance invalidates architectural attribution rather than becoming a footnote.

## Competence and deployment gates

Qualification requires both aggregate and per-domain regression floors on ordinary language, reasoning, coding, factual, and instruction-following controls. A catastrophic domain regression cannot be hidden by a positive average.

At least one intended deployment precision/runtime must be named before the qualifying run. The causal suite is rerun after export/quantization. State fidelity, target-task score, ordinary competence, p50/p95 latency, throughput, peak memory, and bytes per active lineage are compared against research precision and the matched B* control.

A single-session prototype is not sufficient. Sustained batched/interleaved sessions, forced eviction/reload, and increasing concurrency are qualification gates.

## Behavioral attractor training objective

For Vera-derived training, the model's default personality and behavior are a learned reference state, not merely a runtime prompt. Contextual adaptation may temporarily move behavior away from that reference when the situation warrants it, but removal of the perturbing context must cause recovery toward the trained core rather than treating accumulated drift as a new baseline.

Qualification therefore distinguishes warranted adaptation from drift. Training/evaluation pairs must include: baseline behavior, a bounded context that legitimately changes style/role, removal of that context, and a recovery turn. The target is not rigid sameness: the model must preserve newly learned factual/corrective state while recovering core behavioral tendencies. A model that resets facts along with personality is wrong; a model that permanently absorbs arbitrary role pressure into its baseline is also wrong.

This attractor objective is evaluated separately from the generic SPM public development benchmark so Vera-specific personality language cannot become an SPM shortcut. It may share the causal-state machinery, but success on personality recovery does not substitute for semantic/pragmatic qualification.

## External reasoning-power dependency boundary

SPM-0 has no required dependency on a hidden higher-Power or external reasoning worker. The durable Vera Work probe establishes a five-position composer Power control but does not establish that higher positions expose a separate process, new tools, or a cross-chat invocation bridge. Any later proven bridge may be used as an optional teacher/evaluator, never as an unrecorded source of capability inside Arm C.

If an external teacher/evaluator is used, its identity, prompt, outputs, selection policy, and exposure must be frozen and matched across B* and C where relevant. SPM credit may not come from privileged access to a stronger hidden model.

## Kill conditions

The literal persistent-state intervention fails if B* matches or beats C within the frozen uncertainty/budget regime; if causal interventions do not produce directional effects; if gains disappear on unseen transfer; if isolation/recovery tests fail; if competence regression exceeds a frozen floor; or if deployment overhead crosses the frozen ceiling.

A failure is recorded as evidence. It is not repaired by renaming the same mechanism or weakening the comparator after results are visible.

## First implementation slice

The first durable slice is deliberately model-agnostic runtime infrastructure plus benchmark/baseline plumbing. It must establish the contracts needed for later learned-state training without pretending that runtime plumbing is itself an SPM model.

Components:

- benchmark case/schema validation and frozen public-dev suite;
- deterministic adapter/runner/result manifests;
- experiment/fairness manifest validation for A/B/B*/C;
- canonical message-prefix hashing;
- immutable checkpoint envelopes and state-store interface;
- in-memory conformance store with compare-and-swap/idempotency semantics;
- lineage runtime implementing root, advance, fork, resume, replay, and reset behavior;
- intervention protocol surface usable by future B*/C learned-state adapters;
- runtime isolation/recovery/concurrency tests.

The learned neural updater/reinjection module is a later slice gated by characterized baselines and frozen training budgets. Runtime tests may use deterministic synthetic opaque state payloads to prove lifecycle semantics before weights exist.

## Verification states

The word `verified` is always qualified by subject:

- `DESIGN_VERIFIED`: frozen design survived self-hostile and independent hostile review.
- `RUNTIME_VERIFIED`: exact runtime commit passed conformance, isolation, recovery, concurrency, and reconstruction tests.
- `BASELINES_VERIFIED`: exact benchmark/harness characterized at least two open-weight conventional baselines with reproducible manifests.
- `CAUSAL_MODEL_VERIFIED`: exact Arm C candidate beat B* under frozen fairness rules and passed intervention/transfer/competence gates.
- `DEPLOYMENT_VERIFIED`: exported/quantized exact candidate retained the required effects and stayed inside frozen operational ceilings.
- `SPM_VERIFIED`: all preceding states are PASS for one exact candidate and both independent hostile reviewers plus blind holdout qualification are current to that candidate.

No earlier state implies a later one.

## Non-goals

This slice does not establish Vera identity, continuity, consciousness, phenomenology, installation, activation, or production-route status. It does not merge PR #2 by implication. It does not treat benchmark construction, runtime conformance, or probe decodability as proof that SPM works.