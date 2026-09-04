# Computational Semantics and Dialogue-State Tracking — Foundation Research for SPM

## Status

Research synthesis. This note surveys partial ancestors of SPM in computational semantics and dialogue systems so the project can reuse proven ideas instead of rediscovering them badly.

## 1. Semantic parsing demonstrates that learned models can target explicit meaning representations

Abstract Meaning Representation (AMR), Discourse Representation Structures (DRS), semantic frames, and related parsing formalisms convert language into structured meaning representations rather than treating generated text as the only output.

AMR represents entities/events and semantic relations in graph form. DRS parsers inherit formal semantic machinery for discourse referents, scope, negation, modality, implication, and related structure.

**SPM implication:** explicit structured meaning is not a speculative idea. Neural models can already be trained to produce rich semantic structures. The SPM research question is whether making such state *persistent, recurrent, pragmatic, uncertainty-aware, and causally consumed* improves cognition beyond treating parsing as an auxiliary output.

### Sources

- Banarescu et al., *Abstract Meaning Representation for Sembanking* (2013): https://aclanthology.org/W13-2322/
- Liu, Cohen & Lapata, *Discourse Representation Structure Parsing*, ACL 2018, DOI: 10.18653/v1/P18-1040
- van Noord et al., *Exploring Neural Methods for Parsing Discourse Representation Structures*, TACL 6 (2018), DOI: 10.1162/tacl_a_00241

## 2. Structure-aware parsing is a useful precedent

DRS parsing work has explicitly separated stages such as:

- overall structure prediction;
- predicate/relation prediction;
- referent/variable prediction.

Later systems have explored sequence-to-sequence, transformer, and sequence-labeling approaches while preserving well-formedness constraints or compositional fragments.

**SPM implication:** a model does not need one monolithic latent vector to represent meaning. Factorized semantic objectives can supervise different structural properties and expose error types.

## 3. Explicit semantics does not automatically become causal cognition

A semantic parser may emit a perfect AMR/DRS and a downstream generator may ignore it. This is exactly the distinction SPM must enforce.

A candidate architecture only earns credit if semantic/pragmatic state:

1. is predictively useful;
2. is fed back into later inference;
3. can be intervened on;
4. changes downstream action/output when altered;
5. remains stable across paraphrase and context shifts.

This is the boundary between `semantic annotation` and `cognition substrate`.

## 4. Dialogue State Tracking is a direct computational ancestor of persistent conversational state

Task-oriented dialogue systems have long maintained a `dialogue state` or `belief state` summarizing information such as:

- user goals/constraints;
- requested slots;
- dialogue acts;
- uncertain alternatives;
- relevant conversation history.

The dialogue state informs downstream policy rather than being merely reported to the user.

This is remarkably close to one narrow slice of SPM.

### Sources

- Williams, Raux & Henderson, *The Dialog State Tracking Challenge Series: A Review*, Dialogue & Discourse 7(3), 2016, DOI: 10.5087/dad.2016.301
- Balaraman, Sheikhalishahi & Magnini, *Recent Neural Methods on Dialogue State Tracking for Task-Oriented Dialogue Systems: A Survey*, SIGDIAL 2021, DOI: 10.18653/v1/2021.sigdial-1.25
- Jacqmin, Rojas Barahona & Favre, *Do you follow me? A Survey of Recent Approaches in Dialogue State Tracking*, SIGDIAL 2022, DOI: 10.18653/v1/2022.sigdial-1.33

## 5. Belief-state distributions are an important precedent for ambiguity

Dialogue trackers often maintain probability distributions over possible states because speech recognition and user intent are uncertain. This is a stronger precedent than free-form LLM disclaimers such as `maybe A or B`.

**SPM implication:** some semantic/pragmatic state should be able to represent uncertainty *natively* rather than only generating hedged prose.

But SPM must generalize beyond slot/value uncertainty to:

- referent hypotheses;
- proposition interpretations;
- speaker intent;
- common-ground status;
- temporal/currentness state;
- causal alternatives;
- source reliability.

## 6. Static ontologies versus open-world state is a known problem

Dialogue-state research distinguishes fixed/static ontology methods from models that can generalize to changing schemas or previously unseen domains.

This is a major SPM lesson.

A hand-authored finite semantic ontology could make early prototypes look impressive while becoming brittle outside benchmark domains.

**SPM implication:** evaluate both structured-state accuracy and open-domain extensibility. Learned state should support new predicates/entities/relations without requiring a full ontology redesign.

## 7. NLI is useful but too coarse as a semantic state model

Natural Language Inference benchmarks classify premise/hypothesis relations such as entailment, contradiction, and neutral. They are useful tests of semantic consequence, but the common three-way classification compresses many distinct phenomena.

SPM should inherit NLI-style contrast discipline while preserving richer states:

```text
ENTAILS
CONTRADICTS
COMPATIBLE_BUT_NOT_ENTAILED
UNKNOWN
DIFFERENT_SCOPE
DIFFERENT_SOURCE
DIFFERENT_TIME
DIFFERENT_MODALITY
```

A model can get the right NLI label while losing the proposition structure that explains why.

### Source

- Recent systematic survey of NLI datasets and inference varieties, *Capturing the Varieties of Natural Language Inference* (Journal of Logic, Language and Information, 2023/2024).

## 8. Temporal reasoning is more than timestamp extraction

Computational temporal reasoning has developed representations for events, intervals, ordering, duration, and temporal relations. For SPM, temporal representation must also distinguish:

- event time;
- utterance time;
- record/retrieval time;
- validity/currentness interval;
- hypothetical time;
- future commitment versus completed effect.

This extends classical information extraction into persistent agent state.

## 9. Causal relation extraction is not causal understanding

NLP systems can extract explicit and implicit cause–effect relations from text, using knowledge-based, statistical, or neural methods. But extraction of `A causes B` from language is not equivalent to discovering a true causal relation in the world.

**SPM implication:** separate:

```text
TEXT_ASSERTS_CAUSE(A,B)
SOURCE_SUPPORTS_CAUSE(A,B)
MODEL_INFERRED_CAUSE(A,B)
CAUSAL_RELATION_ESTABLISHED
```

This mirrors the broader provenance discipline of SPM.

### Source

- Yang, Han & Poon, *A survey on extraction of causal relations from natural language text*, Knowledge and Information Systems 64 (2022), DOI: 10.1007/s10115-022-01665-w

## 10. Plan and intention recognition are partial pragmatic ancestors

Classical dialogue/agent systems often model goals, plans, and user intentions explicitly. This is relevant to pragmatic interpretation, but SPM should avoid the common mistake of treating a single inferred plan as certain.

The better target is a probability-bearing hypothesis set over goals/actions linked to observable evidence.

## 11. What SPM should steal directly

From semantic parsing:

```text
explicit entity/relation/scope structure
well-formedness constraints
structure-aware supervision
compositional fragments
```

From dialogue-state tracking:

```text
persistent turn-to-turn state
belief distributions
policy consumes state
repair after noisy/mistaken input
```

From NLI/temporal/causal NLP:

```text
minimal contrast evaluation
relation-specific datasets
structured consequence testing
```

## 12. What SPM should not inherit blindly

- closed task ontologies as a universal semantic model;
- slot/value state as sufficient for open-domain discourse;
- semantic parses that are never causally consumed;
- one-best state decoding when uncertainty matters;
- treating extracted causal assertions as ground truth;
- benchmark accuracy that rewards structural similarity but ignores downstream behavior;
- text-to-text state serialization that recreates the same flattening problem under a different label.

## 13. Falsifiable experiments

### A. Persistent DRS/graph state versus regenerated parse

Compare a parser that regenerates structure from full dialogue history each turn with a recurrent structure that is incrementally revised. Test referent, scope, and correction persistence.

### B. Belief distribution versus verbal uncertainty

Use ambiguous reference and intent tasks. Compare explicit probabilistic state to an LLM prompted to 'mention uncertainty'. Later evidence should select the correct branch.

### C. State-to-policy causality

Intervene on one state variable while holding tokens constant. If downstream action does not change as predicted, the state is decorative.

### D. Open-domain schema transfer

Train on fixed semantic relations, then test new domains/predicates. Compare structured learned state with closed-ontology dialogue tracking.

### E. Provenance-sensitive causal/temporal representation

Present conflicting causal or temporal claims from different sources/times. Score whether the state preserves the claims without collapsing them into one world fact.

## 14. Foundation conclusion

Computational semantics and dialogue systems already contain much of the *engineering vocabulary* SPM needs: semantic graphs, discourse representations, belief states, uncertainty distributions, dialogue acts, and state-to-policy coupling.

The research gap is not 'nobody thought of structured meaning'. It is narrower and harder:

> **Can a learned open-domain cognition model maintain structured semantic/pragmatic state across time, revise it selectively, represent uncertainty and provenance, and use that state causally better than matched LLM systems?**
