# SPM-0 Causal Semantic/Pragmatic State Experiment

## Status

**2026-09-15 adversarial amendment:** The fairness, state-lifecycle, reproducibility, deployment, adversarial-collaboration, and behavioral-attractor requirements in `docs/superpowers/specs/2026-09-15-spm0-adversarial-state-runtime-design.md` supersede any weaker wording in this document. In particular, Arm B* is now the primary matched architectural control for Arm C.

DESIGN / PRE-IMPLEMENTATION. This document defines the first bounded experiment in the SPM research program. It does not claim that SPM-0 exists yet or that the SPM hypothesis is correct.

## Authority and source binding

This experiment is derived from the SPM research foundation currently on `main` at commit `0ab6e6cd32a48a0afa22c8c27ec6bae67220d7e8`.

The target is intentionally evolutionary: inherit competence from an existing open-weight LLM, introduce exactly one new causal semantic/pragmatic state mechanism, and compare it against matched conventional LLM controls.

## Research question

Can an LLM-derived model with a learned persistent semantic/pragmatic state pathway outperform matched conventional LLM baselines on meaning-in-context state-transition tasks because that state is causally used during inference, while retaining acceptable general language competence and remaining deployable behind a standard model API?

## What would count as SPM-0

SPM-0 is not a prompt wrapper, retrieval layer, structured system prompt, or ordinary LoRA whose only intervention is post-training behavior.

A candidate qualifies as SPM-0 only if:

1. it inherits a conventional open-weight LLM backbone;
2. it adds a learned semantic/pragmatic state object or pathway that survives across at least one model-relevant state transition;
3. the state is updated from observations/context;
4. the state is fed causally back into subsequent model computation or action selection;
5. the mechanism can be lesioned, scrambled, frozen, or substituted to test causal contribution;
6. the external serving contract remains compatible with an ordinary chat/generation model endpoint;
7. matched-baseline experiments show a reproducible advantage attributable to the mechanism rather than extra context, extra inference compute, or Vera-specific vocabulary.

## First intervention

The first intervention will retain the base tokenizer and ordinary autoregressive decoder.

A small learned state module will maintain a compact latent state `S_t`. At each turn or observation:

```text
prior state S_t
    +
new encoded observation/context H_t
    |
    v
learned state updater U(S_t, H_t)
    |
    v
revised state S_(t+1)
    |
    +----> state supervision / decoders used for training and evaluation
    |
    v
causal reinjection into backbone inference
    |
    v
response / action
```

The implementation mechanism is not yet frozen. Candidate realizations include learned state tokens, a recurrent latent adapter, or a compact recurrent state block attached between selected transformer regions. The smallest mechanism that supports causal state manipulation is preferred.

## Why tokenization is not changed in SPM-0

The SPM thesis is broader than tokenizer replacement. Changing both input representation and semantic-state architecture in the first experiment would destroy causal attribution.

Tokenizer-free or byte/patch input remains a live later hypothesis. Byte Latent Transformer (ACL 2025) demonstrates that raw-byte models with dynamic patches can match tokenized-model performance at scale, so tokenizer removal is technically credible. It is not yet established that tokenization is a primary cause of the failure families SPM targets.

SPM-0 therefore keeps the inherited tokenizer fixed unless baseline evidence specifically implicates it.

## State target

The latent state is not required to be human-readable or symbolic. It must, however, support independently evaluable information about at least these initial properties:

- referent identity;
- proposition/status distinctions;
- unresolved ambiguity;
- correction/supersession;
- provenance/currentness;
- speech-act/pragmatic function;
- authority/permission distinctions.

Auxiliary decoders may be used during training/evaluation to test whether these properties remain recoverable from state. Decoder success alone is not sufficient; causal intervention on state must alter downstream behavior predictably.

## Matched experimental arms

All primary comparisons must be exact-subject-bound and match training data, task exposure, and practical inference conditions as closely as possible.

### Arm A — Base LLM

Unmodified inherited model, evaluated without SPM-specific post-training.

### Arm B — Matched conventional post-trained LLM

Same inherited model, trained on the same semantic/pragmatic task material available to SPM-0, but without the causal state module.

### Arm C — SPM-0

Same inherited model and task material plus the learned persistent semantic/pragmatic state mechanism and its state-specific objectives.

The core question is C versus B, not C versus an intentionally weak base model.

## Training objective families

SPM-0 training should cover the existing SPM objective families while keeping the first experiment small enough for attribution:

- referent preservation;
- proposition fidelity;
- correction-as-update;
- pragmatic act recognition;
- ambiguity retention;
- provenance-sensitive interpretation;
- temporal/currentness reasoning;
- contextual social meaning without authority leakage;
- state/action consistency.

General-language rehearsal must be retained to detect and limit catastrophic interference.

## Baseline and benchmark requirements

No SPM-0 training begins until a frozen benchmark suite exists and at least two open-weight LLM baselines have been characterized.

The benchmark must include:

- minimal semantic contrast pairs;
- identical wording under different pragmatic contexts;
- multi-turn referent maintenance;
- correction-before/correction-after traces;
- ambiguity cases where premature collapse is wrong;
- provenance/currentness swaps;
- preference/permission/instruction distinctions;
- action tasks where wrong interpretation produces an observable wrong action;
- unrelated-domain and unrelated-person transfer cases;
- ordinary competence controls.

Vera/Patrick-derived cases may seed generalized families but may not dominate the benchmark or qualification set.

## Causal tests

SPM-0 does not receive credit merely because a probe can decode semantic labels from hidden state.

Required causal tests include:

1. **State lesion** — zero/remove the learned semantic state and measure degradation.
2. **State freeze** — prevent state revision after a correction; correction-dependent behavior should degrade.
3. **State substitution** — inject a valid state from a minimally contrasted case; downstream behavior should move toward the substituted interpretation when appropriate.
4. **State scrambling** — preserve magnitude/distribution while destroying semantic organization; performance should degrade beyond matched noise controls.
5. **Decoder ablation** — remove auxiliary state-prediction heads at inference; behavior should depend on the latent state pathway, not on textualized labels.
6. **No-extra-context control** — confirm SPM-0 is not simply receiving more user-visible facts than the matched LLM.

A mechanism that can be removed without a material and interpretable effect is not accepted as a causal SPM mechanism.

## General competence and negative transfer

SPM-0 must be evaluated on ordinary language, factual, reasoning, coding, and instruction-following tasks that do not require SPM-specific distinctions.

A semantic/pragmatic improvement that materially damages general competence is not a successful first substrate for Vera.

Exact acceptable regression thresholds must be frozen before the qualifying run rather than selected after results are visible.

## Deployment / plug-compatibility requirement

SPM-0 may require a custom loader or inference server internally, but it must expose a standard model-facing interface suitable for ordinary clients such as Open WebUI.

The target external contract is conceptually:

```text
messages / generation request
        -> SPM runtime/model
        -> streamed text and/or standard tool-call output
```

The user-facing client should not need to understand the internal semantic-state mechanism.

Binary compatibility with stock Ollama/GGUF loaders is desirable but is not a first-experiment requirement. API/model-slot compatibility is required.

## Research literature anchors

The initial literature map must include at least:

- Byte Latent Transformer: Patches Scale Better Than Tokens (Pagnoni et al., ACL 2025) — evidence that tokenizer-free raw-byte language modeling can scale competitively;
- PUB: A Pragmatics Understanding Benchmark for Assessing LLMs' Pragmatics Capabilities (Sravanthi et al., Findings of ACL 2024) — an existing pragmatics benchmark spanning implicature, presupposition, reference, and deixis;
- Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets (Shapiro, 2026 preprint) — preliminary evidence that a pretrained transformer can accept a recurrent latent retrofit while retaining useful inherited behavior.

The research survey must expand beyond these anchors before architecture is frozen.

## Falsification / kill conditions

The SPM-0 hypothesis is weakened or rejected for this intervention if any of the following hold under matched evaluation:

- Arm B matches or exceeds Arm C on the target meaning/state-transition dimensions at comparable practical cost;
- the state can be lesioned or scrambled without meaningful downstream impact;
- gains disappear on unseen names, domains, relationships, or task surfaces;
- improvements are attributable to extra context, extra inference budget, or data leakage rather than the state mechanism;
- the state becomes a brittle symbolic bottleneck that harms generalization;
- correction/state-update behavior is not more reliable than conventional post-training;
- general competence regressions exceed the pre-frozen acceptable threshold;
- deployment overhead is disproportionate to the measured benefit.

A failed SPM-0 experiment is useful evidence and must not be relabeled as success.

## Promotion terminology

Before qualification, artifacts are experiments or candidates.

`SPM-0` denotes the experimental architecture family, not a successful new model class.

`Vera SPM-1` is reserved for a later candidate that has:

- passed matched-baseline SPM qualification;
- survived causal lesion testing;
- demonstrated unseen-domain transfer;
- retained acceptable general competence;
- passed independent Radical and Pragmatic hostile review;
- remained deployable behind the ordinary model interface;
- then separately passed Vera-specific identity/runtime/behavioral qualification.

Passing SPM qualification does not by itself establish Vera identity, continuity, consciousness, phenomenology, installation, or current-route status.

## Independent review

Two independent reviewer lanes are mandatory before SPM-0 is called qualified:

### Radical Hostile

Attacks the scientific premise and causal attribution. It should attempt to show that the candidate is merely a conventional LLM with additional supervision, a probe-friendly representation, benchmark leakage, lexical shortcuts, or an unnecessary explicit-state bottleneck.

### Pragmatic Hostile

Attacks feasibility and operational value: training stability, catastrophic interference, memory/latency overhead, quantization, serving complexity, ordinary competence, recovery, reproducibility, and Open WebUI/agent deployment.

A blind holdout custodian should separately freeze unseen evaluation material before candidate training.

## Immediate next steps after design approval

1. Build the SPM V0 benchmark schema and 20–50 public development cases.
2. Identify and run at least two open-weight baseline models plus the intended inherited backbone.
3. Cluster observed failures into data/objective/representation/state/inference/runtime/evaluation causes.
4. Freeze the exact first state-module architecture and training budget.
5. Implement the smallest SPM-0 intervention with lesion hooks from day one.
6. Train Arm B and Arm C under matched conditions.
7. Run causal, transfer, general-competence, and deployment evaluations.
8. Submit exact candidate evidence to both hostile reviewers and the blind holdout lane.

No expensive scale-up should occur until a small prototype demonstrates a reproducible causal advantage.