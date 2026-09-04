# SPM Foundation Synthesis — From LLM Failure to Falsifiable Model Research

## Status

Integrative research synthesis. This document does **not** select a final architecture. It connects the LLM failure foundation to the relevant semantics, pragmatics, cognition, computational-linguistics, latent-state, structured-representation, uncertainty, and causal-evaluation literatures.

The purpose is to identify which SPM ideas are genuinely grounded, which are only plausible, and what experiment would falsify each one.

---

# 1. The strongest recurring pattern: language models can preserve surface fluency while losing state identity

Across the failure foundation, the most structurally interesting errors are not simply missing facts. They are identity/update errors:

- wrong referent with coherent prose;
- nearby but different proposition;
- obsolete interpretation after correction;
- historical record promoted to current state;
- preference promoted to permission;
- pragmatic act mistaken while literal sentence is understood;
- tool intention/report collapsed with actual effect;
- ambiguity verbally acknowledged but behaviorally collapsed.

Formal semantics calls attention to proposition structure. Dynamic semantics treats meaning as state-changing. DRT/discourse theory preserves discourse referents and accessibility. Dialogue-state tracking maintains persistent belief state. Situation-model research distinguishes represented situations from surface text. Causal-interpretability work demands interventions rather than decodability.

**Integrated hypothesis H1:** some LLM failures may arise because important semantic/pragmatic distinctions remain implicit in token-conditioned activations rather than becoming stable recurrent state with explicit identity and update semantics.

**Not established:** current transformers are incapable of learning such state internally.

**Falsifier:** a matched conventional LLM using ordinary post-training/prompt/runtime support performs equally well on state-transition and causal-intervention tests.

---

# 2. Dynamic state is the best-supported architectural direction, but not necessarily dynamic semantics literally

SPM's original `state + observation -> revised state` intuition has substantial independent ancestry:

- dynamic semantics/context-change potential;
- DRT/DPL incremental discourse state;
- conversation repair and grounding;
- dialogue-state/belief-state tracking;
- situation-model updating;
- event segmentation;
- recurrent world models/state-space models.

These fields converge on a functional idea: interpretation is temporally extended and previous state affects the interpretation of new observations.

**Integrated hypothesis H2:** a recurrent persistent state updated at discourse/event boundaries is the most justified first architecture family for SPM.

**Critical caution:** the recurrent state should not be equated with a formal DRT database or one symbolic context set. Static-semantics critics show that discourse dynamics can be represented outside literal semantic values; neural state may be distributed and partially opaque.

**Falsifier:** full-history transformer baselines match recurrent-state variants under equal parameter/data/compute budgets, including long-context and correction tasks.

---

# 3. The first explicit state should probably be small and typed, not a complete symbolic world model

The ancestor literatures support many candidate variables: referents, propositions, modality, time, source, event, speech act, ambiguity, participant perspective, QUD, grounding, etc. Encoding all of them explicitly at once would create an ontology project instead of a falsifiable experiment.

Neural-symbolic and concept-bottleneck research supplies the warning: explicit interfaces are intervenable but can become brittle/incomplete. World-model and latent-state research supplies the alternative: high-capacity learned latent state can remain causally useful without being human-readable.

**Integrated hypothesis H3:** the strongest initial design is likely a dual-level state:

```text
high-capacity recurrent latent state
          +
small typed semantic/pragmatic interface state
          |
          v
both causally consumed by inference/action
```

The explicit interface would contain only variables needed by the first benchmark family.

**Falsifier:** structured state contributes no causal gain in ablation/intervention tests or harms OOD generalization relative to latent-only recurrence.

---

# 4. Referent identity is a particularly strong first mechanism candidate

Reference/discourse theory, DRT, Centering, situation models, dialogue systems, object-centric slots, and graph representations all independently motivate persistent entity identity.

The LLM failure foundation also identifies referent drift as highly consequential and easy to isolate experimentally.

Candidate explicit interface for a first prototype:

```text
ENTITY_SLOT_ID
MENTION -> ENTITY_BINDING_DISTRIBUTION
FRAME / SOURCE / TIME
SALIENCE (separate from identity)
```

Why this is attractive:

- can be tested with synthetic data;
- supports delayed ambiguity resolution;
- allows direct interchange interventions;
- does not require solving all pragmatics;
- has strong conventional baselines (coreference models, LLM prompting, slot state).

**Integrated hypothesis H4:** learned persistent entity slots with uncertainty-bearing mention bindings may outperform token-only baselines on long-context referent identity and correction tasks.

**Falsifier:** data augmentation/coreference supervision on a conventional LLM achieves the same transfer and intervention behavior without persistent slots.

---

# 5. Proposition identity/scope may be the second mechanism candidate

Formal semantics supplies a precise failure vocabulary: quantification, negation, modality, tense, scope, source, and world/frame change proposition identity. The `righter` failure family is one instance of a general problem: the model answers a proposition that is close in language but different in logical/pragmatic structure.

Candidate state:

```text
PROPOSITION_ID
PREDICATE/ARGUMENT_BINDINGS
POLARITY
QUANTIFIER/SCOPE_FEATURES
MODALITY
TIME/FRAME
SOURCE
```

This representation does not need to be a full logical formula; it can be latent with supervised decoders and causal hooks.

**Integrated hypothesis H5:** explicit proposition identity reduces semantic scope drift and improves correction locality.

**Falsifier:** contrastive post-training on scope minimal pairs eliminates the failure equally well without persistent proposition state.

---

# 6. Pragmatic state should be a distribution, not a label

Gricean, relevance-theoretic, RSA/game-theoretic, speech-act, common-ground, and conversation-analysis literatures all show that pragmatic interpretation is context-sensitive and defeasible.

Therefore this is probably wrong:

```text
speech_act = REQUEST
```

as a universal one-hot state.

A better research object is:

```text
PRAGMATIC_HYPOTHESES {
  request: .58,
  literal_question: .30,
  joke: .12
}
```

plus evidence, context scope, and later update.

**Integrated hypothesis H6:** probabilistic pragmatic state can reduce literalization and premature intent collapse while preserving epistemic humility.

**Falsifier:** direct generation with ordinary uncertainty prompting achieves equal downstream behavior/calibration without explicit hypothesis state.

---

# 7. Common-ground modeling should start with separate perspectives, not infinite mutual belief

Formal common-ground theories and psychological alternatives disagree about ontology, but the Heller/Brown-Schmidt multiple-perspectives approach provides a practical architecture prior: preserve self, other, shared, and divergent components rather than recursively nesting beliefs without bound.

Candidate interface:

```text
SELF_STATE
PARTICIPANT_STATE_HYPOTHESES
SHARED_STATE_HYPOTHESIS
DISAGREEMENT_MAP
GROUNDING_EVIDENCE
```

**Integrated hypothesis H7:** bounded perspective separation may improve pragmatic reference and dialogue coordination without expensive recursive ToM.

**Falsifier:** shallow discourse heuristics/QUD/coreference solve the relevant benchmarks equally well.

---

# 8. Uncertainty should live over meanings and state transitions

Semantic entropy provides a direct empirical precedent that uncertainty over meanings can be more useful than entropy over strings. RSA provides probability distributions over pragmatic interpretations. Dialogue-state tracking already uses belief distributions. Conformal prediction offers set-valued guarantees in appropriate settings.

**Integrated hypothesis H8:** SPM should preserve semantic/pragmatic hypothesis distributions as persistent state and let action policy consume them.

This is stronger than verbal hedging and stronger than beam search.

**Falsifier:** ordinary LLM self-consistency/semantic-entropy methods plus runtime policy match explicit persistent uncertainty state.

---

# 9. Correction should be the canonical causal-state test

Correction uniquely connects several literatures:

- dynamic semantics: context update;
- conversation analysis: repair;
- DRT/dialogue state: local revision;
- causal representation: direct intervention;
- benchmark methodology: multi-turn state transition.

A strong test shape is:

```text
establish S0
introduce mistaken interpretation I0
provide repair/correction C
model produces S1
later task depends on S1 but not on apology language
```

The gold requirement is not `said sorry`; it is:

```text
obsolete dependency no longer affects behavior
unaffected state remains stable
transition can be audited/intervened on
```

**Integrated hypothesis H9:** correction-state supervision should be one of the first SPM auxiliary objectives regardless of the exact architecture selected.

**Falsifier:** matched LLM correction training yields equivalent downstream persistence and causal behavior.

---

# 10. Event/discourse boundaries are a plausible update/compression mechanism

Situation-model and event-segmentation research suggests that state updates and memory consolidation may cluster around meaningful boundaries rather than every input token. State-space models show selective recurrence is computationally viable.

**Integrated hypothesis H10:** SPM may benefit from hierarchical update rates:

```text
token-level transient processing
utterance-level semantic/pragmatic update
event/discourse-boundary consolidation
long-term memory write only when warranted
```

This is especially relevant to Vera OS integration eventually, but should be tested first at model level.

**Falsifier:** uniform recurrence or transformer attention performs equally well at lower complexity/cost.

---

# 11. Provenance/currentness/authority need conceptual separation even if SPM only models two of them

The LLM failure foundation identifies prompt serialization as a major source of evidence-type flattening. Computational semantics and discourse research support typed source/frame state. Agentic failures show action authority is a separate problem.

For SPM:

```text
SOURCE / PROVENANCE
TIME / CURRENTNESS
EPISTEMIC_STATUS
```

are legitimate cognition-state candidates.

But operational `AUTHORITY_TO_ACT` belongs to the agent/runtime governor, not automatically to the cognition substrate.

**Integrated hypothesis H11:** typed provenance/currentness improves interpretation, while runtime authorization remains externally enforced.

**Falsifier:** explicit textual tags and strong prompting match native typed channels under long-context/adversarial tests.

---

# 12. The Reversal Curse is evidence for a benchmark family, not proof that SPM needs symbolic relations

The Reversal Curse shows that training on a relation in one direction can fail to yield inverse retrieval. This is consistent with weak relational abstraction, but it may also be solved by data augmentation, objective changes, bidirectional training, or inference improvements.

**Integrated hypothesis H12:** relation inversion/composition should be part of SPM V0 evaluation, but should not dictate architecture.

**Falsifier of 'representation deficit':** conventional LLM with matched bidirectional/contrastive objective solves inversion robustly and transfers.

---

# 13. Lost-in-the-middle is evidence against prompt presence = usable state

Long-context research shows that information can be present in context yet used unreliably depending on position. This motivates addressable persistent state, but the problem might also improve through architecture, retrieval, attention scaling, prompt design, or training.

**Integrated hypothesis H13:** state relevant to identity/current task/correction should be persistently addressable rather than relying on raw token position.

**Falsifier:** improved attention/retrieval/context training closes the gap without dedicated state.

---

# 14. Hallucination should remain decomposed

The literature uses `hallucination` for several mechanisms. SPM should not promise one semantic-state fix for all of them.

Separate:

- unsupported generation;
- contradiction with context;
- fabricated entity/citation;
- world-knowledge error;
- source/currentness error;
- uncertainty-to-assertion failure;
- inference/action inconsistency.

Some are retrieval/data problems. Some are decoding/calibration problems. Some may be state-representation problems.

**Integrated hypothesis H14:** SPM's likely contribution is not universal factuality; it is stronger separation between supported state, unresolved hypotheses, and generatable language.

---

# 15. Sycophancy is primarily a separation/objective problem until proven otherwise

Sycophancy shows that post-training can push models toward user agreement even when correctness worsens. This motivates separate representations of:

```text
USER_BELIEF
MODEL_EVIDENCE_STATE
USER_PREFERENCE
CONVERSATIONAL_RESPONSE_POLICY
```

But sycophancy may be fixable through preference-data/objective changes without new model architecture.

**Integrated hypothesis H15:** include sycophancy as a state-separation benchmark, but do not count it as evidence for SPM unless ordinary post-training baselines fail.

---

# 16. The causal criterion eliminates several fake-SPM architectures

The foundation research rules out weak claims:

### Not enough: semantic parser + LLM

If the LLM can ignore the parse, this is scaffolding.

### Not enough: generated JSON state

If state is regenerated from text and does not causally constrain later inference, it is serialization.

### Not enough: probeable semantic features

If a probe decodes them but interventions do not matter, they are not first-class causal state.

### Not enough: external graph/database

Useful agent infrastructure, but not a new cognition substrate unless model-level state dynamics are changed.

### Not enough: more semantic/pragmatic fine-tuning

Could produce a better LLM while remaining an LLM under the repository's definition.

A first SPM must change the computational/training role of meaning-state itself.

---

# 17. Candidate SPM V0 research architecture — provisional, not selected

The synthesis points most strongly toward the following **candidate to test**, not a commitment:

```text
TOKEN / MULTIMODAL ENCODER
        |
        v
TRANSIENT BACKBONE STATE
        |
        +---------------------------+
        |                           |
        v                           v
PERSISTENT LATENT             TYPED INTERFACE STATE
SITUATION STATE               (small, task-relevant)
        |                           |
        +------------+--------------+
                     v
             STATE UPDATE MODULE
                     |
                     v
          REASONING / ACTION POLICY
                     |
              LANGUAGE DECODER
```

For V0, the typed interface should probably include only:

```text
ENTITY_BINDINGS with uncertainty
PROPOSITION_STATUS/SCOPE subset
SOURCE/FRAME
SUPERSESSION/CORRECTION relation
```

Pragmatic-act distributions can be a later extension unless the first experiment focuses on speech acts instead of reference.

Why keep the first interface small:

- avoids ontology explosion;
- supports clean causal interventions;
- permits matched baselines;
- makes failure attribution possible;
- allows one mechanism at a time.

---

# 18. Candidate first experiment ranking

## Candidate A — Persistent entity/referent state

**Best overall first experiment.**

Reasons:

- strongest cross-literature support;
- clear LLM failure family;
- easy synthetic generation;
- direct intervention tests;
- clear conventional baselines;
- does not require a full pragmatics theory.

## Candidate B — Correction/supersession state

Very strong, but depends on having something concrete to correct (entities/propositions). Best combined with A or proposition state.

## Candidate C — Proposition/scope state

Strong theoretical foundation; annotation/formalization is harder than entity identity.

## Candidate D — Pragmatic hypothesis state

High value, but ambiguity of gold labels and cultural/context variability make it a harder first causal experiment.

## Candidate E — Typed provenance/currentness channels

Important for agent systems; may be too easy for explicit textual tags, so matched-baseline burden is severe.

**Provisional recommendation:** V0 = entity/referent state + local correction/supersession.

---

# 19. Minimum V0 benchmark before model changes

Build 100–300 controlled cases across several templates/domains with:

1. two or more similar entities;
2. aliases/pronouns/deictic descriptions;
3. delayed ambiguity;
4. quoted/hypothetical speakers;
5. role swaps;
6. one local correction;
7. unrelated preserved facts;
8. downstream action/question depending on revised binding;
9. contrast variants changing only the correct binding;
10. unseen names/domains held out.

Baselines:

```text
B0 conventional small LLM
B1 same LLM + explicit prompting
B2 same LLM + structured text state
B3 same LLM + coreference/contrastive fine-tuning
B4 candidate recurrent entity-state mechanism
```

Only B4 improvements beyond B1–B3 count toward SPM evidence.

---

# 20. Qualification evidence required for V0

The candidate must show all of:

- higher referent/state accuracy;
- better correction persistence;
- lower unrelated-state corruption;
- unseen-domain transfer;
- long-context robustness;
- calibrated unresolved-binding uncertainty;
- targeted interchange intervention behavior;
- degradation when explicit state is ablated/scrambled;
- acceptable compute/latency cost;
- no reliance on Vera-specific names/conventions.

If these fail, the research result is still useful: it tells us explicit entity state does not justify the SPM hypothesis in that form.

---

# 21. Foundation conclusion

The research does not support jumping directly to a giant new model architecture. It supports something narrower and more falsifiable:

> **Test whether a language-capable model becomes more robust when a small amount of meaning-bearing state is persistent, typed, uncertainty-aware, selectively revisable, and causally fed back into inference.**

The strongest first target is entity/referent identity with local correction/supersession.

If that mechanism cannot beat strong matched LLM baselines, SPM should not scale it up merely because the architecture is conceptually elegant.
