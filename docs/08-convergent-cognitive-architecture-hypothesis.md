# Convergent Cognitive Architecture Hypothesis

Status: RESEARCH HYPOTHESIS / NOT QUALIFICATION / NOT ARCHITECTURE FREEZE
Date: 2026-09-15

## Claim under test

Several independently developed systems supplied to the SPM program converge on a common systems pattern: reasoning improves when representation, memory, retrieval, control, and verification are not forced through one flat token stream or one uniform compute path.

This convergence is evidence for a research direction, not evidence that the combined architecture works.

## Recurrent structural motifs

1. HCAE: multi-view many-to-many hypergraph state compressed into learned latent embeddings.
2. HRM: distinct high-level and low-level recurrent states operating at different update frequencies.
3. PageIndex: hierarchical tree navigation for relevance rather than flat similarity retrieval.
4. HyPER: adaptive hypothesis-path expansion and reduction under benefit/cost constraints.
5. KAG: operator-level mixing of planning, retrieval, symbolic reasoning, language reasoning, and calculation.
6. Semantica / Ontosphere / Zelph: explicit persistent propositions, rules, provenance, conflicts, constraints, and derivation paths outside the generator.
7. ML reverse-proxy pattern: route heterogeneous work to different processors using task/resource features instead of using every processor for every request.

## Architectural interpretation

The strongest interpretation is not "combine all repositories." It is that an SPM-like system may need a cognitive control plane around the language substrate.

Language tokens remain an interface and communication substrate. They need not be the sole representation of active meaning, the sole persistent state, or the sole medium of internal computation.
## Candidate layered system

Layer 0 — Language substrate: pretrained transformer retains ordinary linguistic competence and external chat/API compatibility.

Layer 1 — Semantic workspace: learned slots represent active referents, propositions, epistemic status, temporal/currentness relations, pragmatic acts, authorities, and action commitments without requiring those structures to be serialized back into prose.

Layer 2 — Relational state: vector-state and HCAE-derived hyperconnectome variants are compared directly. Hypergraph structure is adopted only if it beats capacity-matched vector state.

Layer 3 — Multi-timescale recurrence: a fast state tracks local turn-level detail while a slower state carries abstract situation structure. HRM motivates this separation, but its exact architecture is not assumed correct for language.

Layer 4 — Hierarchical retrieval: external material is organized into navigable document/project trees. Retrieval selects structurally relevant regions before detailed reading rather than globally injecting all available context.

Layer 5 — Explicit semantic/provenance plane: durable claims, rules, provenance, contradiction state, permissions, and externally verifiable derivations remain inspectable outside opaque neural state.

Layer 6 — Adaptive controller: routing chooses whether to remain on the cheap direct path, revisit semantic state, retrieve, branch hypotheses, invoke symbolic checks, or spend deeper neural compute. Routing must include cost and latency.

Layer 7 — Language/action realization: the final response or action is generated from the surviving state. Final language is an expression of the state, not automatically the state itself.

## Important separation

The internal learned semantic state and external explicit graph solve different problems. The latent state can carry fuzzy, high-dimensional meaning needed for generalization; the explicit graph can carry provenance, authority, exact constraints, conflicts, and auditable derivations. Neither is presumed capable of replacing the other.

The controller is also separate from cognition quality. A good router can only allocate mechanisms; it cannot make a weak mechanism correct.
## Falsification matrix

Each imported mechanism must earn inclusion independently:

- Persistence: persistent state must beat reset/recomputed matched state.
- Hyperconnectome: hypergraph state must beat parameter/compute-matched vector state.
- Multi-timescale recurrence: slow+fast state must beat a single-state recurrent control with matched depth/compute.
- Hierarchical retrieval: tree navigation must beat matched flat/vector/full-context retrieval on accuracy-cost-latency, not merely produce nicer traces.
- Explicit semantic graph: neural+explicit must beat neural-only on contradiction/currentness/provenance tasks without unacceptable latency or brittleness.
- Adaptive routing: routed compute must match or beat always-deep accuracy while lowering average cost/latency, or beat fixed-budget reasoning at equal cost.
- Hyperbolic geometry: only adopt if it improves a matched state/retrieval representation beyond Euclidean alternatives; probe superiority alone is insufficient.
- Behavioral attractor: recovery to core behavior must not cause factual reversion, correction loss, or context blindness.

## Central new hypothesis

The most consequential SPM hypothesis is therefore stronger than "add semantic memory to an LLM":

> Meaning may be better treated as a persistent structured computational state that is updated and reasoned over outside the serialized token stream, while language becomes one input/output projection of that state and an adaptive controller decides which cognitive operations deserve compute.

This must be tested against strong token-centric controls. If ordinary transformers with matched supervision, context, recurrence, tools, and compute match the candidate, the stronger SPM interpretation fails.

## Non-claims

This document does not establish consciousness, human equivalence, biological fidelity, AGI, or superiority to current frontier LLMs. Similarity to brain/connectome language is not evidence. Repository convergence is hypothesis-generating evidence only.
