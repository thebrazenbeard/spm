# Research Roadmap

## Goal

Move SPM from an intuition to a falsifiable model-research program without prematurely claiming a new architecture exists.

## Phase 0 — define the object

Before implementation, settle operational definitions for:

- semantic state;
- pragmatic state;
- referent;
- proposition;
- speech act;
- implicature;
- salience;
- ambiguity;
- correction/state revision;
- provenance-sensitive meaning;
- state/action consistency.

Deliverable: a compact SPM vocabulary and a set of distinctions whose collapse constitutes failure.

## Phase 0.5 — establish the LLM failure foundation

Research and classify the strongest known failure modes of contemporary LLMs before treating any candidate architecture as an SPM improvement.

Starting synthesis: `docs/06-llm-failure-foundation.md`.

For every failure, test whether the dominant cause is data, objective, representation, inference/decoding, context/state, post-training, runtime/tooling, or evaluation design. Behavioral failure alone is not evidence that a new model class is required.

Deliverable: literature-backed failure families that can become benchmark rows and discriminating experiments.

## Phase 0.75 — survey the intellectual and computational ancestors

Status on `work/spm-foundation-research-20260904`: initial ordered survey pass completed and saved in `docs/07` through `docs/19`, with integration in `docs/20-foundation-synthesis.md`.

Survey domains:

1. formal semantics and dynamic semantics;
2. reference and discourse;
3. pragmatics;
4. speech-act theory and conversational repair;
5. common ground and theory of mind;
6. psycholinguistic situation models;
7. mental models and event cognition;
8. computational semantics and dialogue-state tracking;
9. world models and latent-state learning;
10. neural-symbolic and structured representations;
11. causal representation and mechanistic interpretability;
12. ambiguity and uncertainty;
13. learning objectives and benchmark methodology.

Deliverable: distinguish mechanisms SPM can legitimately inherit from ideas that are merely analogous or already solvable by conventional systems.

Current synthesis result is provisional: persistent entity/referent state plus local correction/supersession is the strongest first experiment family, but this is not yet architecture selection.

## Phase 1 — build SPM V0 baseline benchmark before model changes

Construct a controlled baseline suite before implementing the successor.

Measure strong current/open models on:

- referent preservation;
- scope fidelity;
- correction persistence;
- speech-act understanding;
- ambiguity preservation;
- pragmatic subtext;
- provenance/currentness distinctions;
- contextual authority leakage;
- semantic/action consistency.

The first V0 benchmark should emphasize the proposed initial mechanism rather than attempt universal pragmatics. For entity/referent + repair, include:

- two or more similar entities;
- aliases/pronouns/deixis;
- delayed ambiguity;
- role swaps;
- quoted and hypothetical frames;
- local correction;
- unrelated state that must remain stable;
- downstream decision depending on corrected binding;
- contrast variants;
- unseen-domain/name holdouts.

Use minimal pairs, contrast sets, naturalistic multi-turn state-transition tasks, and long-context variants.

## Phase 2 — identify architectural bottlenecks through discriminating baselines

Analyze failures against at least:

- data/training deficit;
- objective mismatch;
- context-window/state deficit;
- representation deficit;
- inference/decoding deficit;
- post-training distortion;
- runtime/tooling deficit;
- evaluation artifact.

Required baseline ladder where practical:

```text
B0 conventional matched LLM
B1 LLM + explicit prompting
B2 LLM + structured text/runtime state
B3 LLM + equivalent auxiliary/coreference/contrastive post-training
B4 candidate SPM mechanism
```

If B1–B3 solve the target robustly, that failure is not evidence that B4 defines a new model class.

## Phase 3 — minimum SPM prototype

Construct the smallest experiment that changes the model rather than only the surrounding agent framework.

Current provisional candidate from `docs/20-foundation-synthesis.md`:

```text
language-capable backbone
        +
persistent learned latent situation state
        +
small uncertainty-bearing entity/referent interface state
        +
local correction/supersession update mechanism
```

The typed interface should remain deliberately small. Do not add full pragmatics, common ground, provenance, event models, and world-model structure simultaneously.

Alternative minimal mechanisms remain eligible if benchmark results invalidate the provisional choice.

## Phase 4 — causal qualification of the state mechanism

A state is not first-class merely because a probe can decode it.

Require:

- targeted interventions;
- interchange tests where appropriate;
- state-stream ablation/scrambling;
- correction-state convergence;
- narrow downstream effects rather than broad degradation;
- generalization of intervention semantics to unseen domains;
- complexity-controlled causal abstraction.

If the explicit state can be ignored without meaningful behavioral loss, the SPM claim fails.

## Phase 5 — matched comparison

Compare the prototype against matched LLM baselines under equivalent compute/context/tool conditions.

Require:

- reproducible improvement;
- unseen-domain transfer;
- no dependence on Vera-specific language;
- measurable effect on behavior/action, not only self-description;
- calibrated unresolved-state handling;
- documented latency, memory, compute, and general-language tradeoffs.

## Phase 6 — compound SPM architecture

Only after individual mechanisms survive falsification should they be combined.

Possible later components:

- learned world/situation state;
- semantic graph or latent entity/proposition state;
- pragmatic hypothesis distributions;
- participant/common-ground state;
- typed provenance/currentness channels;
- persistent recurrent memory state;
- event-boundary consolidation;
- action-semantic consistency heads/objectives;
- multimodal encoders/decoders.

## Phase 7 — agent integration

After the substrate demonstrates standalone value, integrate it into an agent runtime such as Vera OS.

This tests whether explicit meaning-state improves:

- tool use;
- delegation;
- long-running task coherence;
- correction persistence;
- multi-agent synthesis;
- memory interpretation;
- human interaction.

Agent/runtime improvements must not be used retroactively to claim model-substrate improvements.

## Immediate research backlog

1. ingest independent worker research packets from `SPM-FOUNDATION-RESEARCH-20260904 / F1` and preserve disagreements;
2. convert `docs/20-foundation-synthesis.md` into a compact operational V0 vocabulary/state contract;
3. define the SPM V0 benchmark schema;
4. generate and independently validate the first 100–300 entity/reference/repair cases;
5. run at least two open-weight conventional baselines plus structured/prompted variants;
6. characterize failure clusters and revise the candidate mechanism if evidence demands it;
7. write the exact matched-baseline experiment specification;
8. only then implement/train the minimum prototype.

## Guardrail

Do not begin by training an expensive giant model.

If SPM is a real architectural idea, its first causal advantage should be demonstrable in a deliberately small experiment before scale, runtime scaffolding, or benchmark familiarity obscures attribution.
