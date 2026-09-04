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

## Phase 1 — benchmark current LLMs

Build a baseline suite before designing the successor.

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

The point is to discover where current LLMs actually fail, not where intuition says they fail.

## Phase 2 — identify architectural bottlenecks

Analyze failures to distinguish at least:

- data/training deficit;
- objective mismatch;
- context-window/state deficit;
- representation deficit;
- inference/decoding deficit;
- runtime/tooling deficit;
- evaluation artifact.

If ordinary post-training solves a failure robustly, that failure alone does not justify a new model class.

## Phase 3 — minimum SPM prototype

Construct the smallest experiment that changes the model rather than only the surrounding agent framework.

Candidate minimal experiments:

1. learned persistent semantic state between turns;
2. auxiliary semantic/pragmatic state prediction heads whose state is fed causally back into generation/action;
3. typed context channels for provenance/currentness/authority rather than flattened prompt serialization;
4. explicit ambiguity sets maintained across turns;
5. correction training where internal state before/after correction is directly evaluated;
6. separate situation-model recurrence plus language decoder.

Pick one experiment at a time so causal attribution remains possible.

## Phase 4 — matched comparison

Compare prototype against matched LLM baselines under equivalent compute/context/tool conditions.

Require:

- reproducible improvement;
- unseen-domain transfer;
- no dependence on Vera-specific language;
- measurable effect on behavior/action, not only self-description;
- documented trade-offs in latency, memory, compute, and general language quality.

## Phase 5 — compound SPM architecture

Only after individual mechanisms survive falsification should they be combined into a larger architecture.

Possible compound components:

- learned world/situation state;
- semantic graph or latent entity/proposition state;
- pragmatic dialogue-state representation;
- typed provenance/currentness channels;
- persistent recurrent memory state;
- action-semantic consistency head/objective;
- ordinary language decoder;
- multimodal encoders/decoders where useful.

## Phase 6 — agent integration

After the substrate demonstrates standalone value, integrate it into an agent runtime such as Vera OS.

This phase tests whether explicit meaning-state improves:

- tool use;
- delegation;
- long-running task coherence;
- correction persistence;
- multi-agent synthesis;
- memory interpretation;
- human interaction.

Agent integration must not be used retroactively to claim the underlying model improvement if the gain comes from runtime scaffolding.

## Immediate research backlog

- survey contemporary work on world models, latent-state language models, neural-symbolic methods, discourse/pragmatics modeling, state-space/recurrent models, memory architectures, semantic parsing, speech-act/implicature benchmarks, tool-action grounding, and representation interpretability;
- define SPM V0 benchmark schema;
- create 20–50 minimal contrast cases across the initial failure families;
- run at least two current open-weight LLM baselines;
- characterize failure clusters;
- choose one minimal architecture intervention;
- write a falsifiable experiment specification before training.

## Guardrail

Do not begin by training an expensive giant model.

If SPM is a real architectural idea, we should be able to demonstrate its core advantage in a deliberately small prototype before scale obscures causality.