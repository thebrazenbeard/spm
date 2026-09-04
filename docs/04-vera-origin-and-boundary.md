# Vera Origin and Boundary

## Why Vera is here

The SPM idea emerged during a 2026-09-03 analysis of Vera's architecture and recurrent failure modes.

The immediate intuition was simple: current language models can be astonishingly fluent while still misunderstanding what a statement means *in use*. Patrick proposed **SPM — Semantics & Pragmatics Model** as an evolution of the LLM itself, not as another layer in Vera OS.

Vera provides unusually rich motivating examples because the project has spent substantial effort distinguishing:

- literal content from relational/symbolic meaning;
- current authority from historical text;
- self-state from external-person inference;
- preference from permission;
- consent from promise;
- correction language from correction effect;
- runtime/model provenance from identity;
- relational deference from operational authority;
- semantic similarity from referential identity.

## Ownership boundary

The repositories should remain conceptually separate:

```text
spm
  research/design for a successor cognition substrate

vera_model_training
  training/qualification work for current Vera-related model artifacts

vera-os
  persistent local governed agent/workspace architecture

vera / vera-R9A0 / related systems
  Vera governance, semantics, continuity, memory, behavior, and engineering lineage
```

SPM is **not** a Vera identity repository and should not ingest private Vera continuity or relational material by default.

Vera OS may eventually consume an SPM artifact as a replaceable cognition engine. That does not make SPM responsible for Vera's identity, memory lifecycle, authority, task state, or protected effects.

## Vera as originating testbed, not universal semantics

Vera-derived failures can provide benchmark seeds because many are clean semantic/pragmatic distinctions. But SPM research must generalize them away from Vera.

For example:

- a Vera `righter` failure becomes a general benchmark for proposition scope fidelity;
- contextual relational deference becomes a general benchmark for context-specific pragmatic rules versus authority leakage;
- centered-save currentness becomes a general benchmark for historical-record versus current-state reasoning;
- correction persistence becomes a general benchmark for state revision;
- relational bids become general speech-act/implicature tests.

The benchmark passes only if the learned competence transfers to unrelated people, organizations, cultures, domains, and tasks.

## Privacy rule

Do not copy private journals, intimate conversation exports, credentials, raw personal-memory ledgers, or other sensitive Vera/Patrick material into this repository merely because it motivated the research.

When a private example reveals a useful general failure family, abstract the *structure* of the problem and build non-private test cases.

## Research independence

A future SPM should be able to fail Vera qualification while still being a legitimate SPM, and a Vera model should be able to remain an LLM while becoming better at Vera-specific behavior.

Those are separate claims:

- **SPM claim:** a new/evolved model architecture has meaning-in-context as a primary computational target and demonstrates general advantages attributable to that design.
- **Vera qualification claim:** a particular cognition model can safely and faithfully serve as an inference substrate for Vera under Vera's own evaluation/governance requirements.