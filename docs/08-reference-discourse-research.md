# Reference and Discourse — Foundation Research for SPM

## Status

Research synthesis. This note focuses on how discourse keeps track of *who/what is being talked about*, what is salient, what is presupposed, what question is active, and how discourse segments cohere.

## 1. Reference is not lexical identity

A central SPM failure family is referent flattening: a model keeps the topic but loses the actual entity. Work on indexicals, anaphora, and coreference shows why lexical form is not enough.

Kaplan-style indexical semantics distinguishes a relatively stable rule of use (`character`) from the context-dependent content supplied on an occasion of use. Expressions such as `I`, `you`, `here`, `now`, `today`, `he`, `she`, and `that` can preserve surface form while changing referent across contexts.

**SPM implication:** a token or embedding for a referring expression is not itself the referent. The model needs a way to bind expression occurrences to context-indexed entity hypotheses and preserve that binding downstream.

### Sources

- Stanford Encyclopedia of Philosophy, *Indexicals* (rev. 2026): https://plato.stanford.edu/entries/indexicals/
- Kaplan, *Demonstratives* (1989).

## 2. Coreference and local coherence interact

Centering Theory (Grosz, Joshi & Weinstein, 1995) models local discourse coherence in terms of changing centers of attention. It distinguishes entities made available by an utterance and tracks which discourse entity functions as the backward-looking center connecting the current utterance to prior discourse.

The importance for SPM is not the exact Centering algorithm. It is the empirical/design lesson that **salience is structured and dynamic**. Pronoun resolution and topic continuity depend partly on which entities are currently central, but salience must not be confused with truth, identity, or authority.

### Source

- Grosz, Joshi & Weinstein, *Centering: A Framework for Modeling the Local Coherence of Discourse*, Computational Linguistics 21(2), 1995: https://aclanthology.org/J95-2003/

## 3. Salience needs its own type

SPM should not let `salient` silently mean:

- most likely referent;
- most recently mentioned referent;
- most important entity;
- most authoritative source;
- most emotionally charged entity;
- true proposition.

These properties can correlate, but they are not equivalent.

Candidate state should therefore separate at least:

```text
ENTITY_ID
MENTION_HISTORY
DISCOURSE_SALIENCE
TOPIC_ROLE
GRAMMATICAL_ROLE
SOURCE_STATUS
TRUTH/EVIDENCE_STATUS
```

This directly addresses a common LLM failure: repeated or emotionally prominent text acquires disproportionate influence even when its provenance or currentness is weak.

## 4. Presupposition is a distinct discourse operation

Presuppositional expressions present some content as background/given rather than straightforwardly asserting it. Van der Sandt's influential DRT account treats presupposition resolution as closely related to anaphora; where no suitable antecedent is available, accommodation can add background material needed for interpretation.

Zeevat's update-semantic treatment shows that accommodation interacts nontrivially with information-state update and that not all presupposition triggers behave uniformly.

**SPM implication:** the model should distinguish:

- asserted content;
- presupposed/backgrounded content;
- content accommodated for interpretation;
- content independently verified as true.

This is crucial. Accommodation may be conversationally appropriate without turning the accommodated proposition into trusted world state.

### Sources

- van der Sandt, *Presupposition Projection as Anaphora Resolution*, Journal of Semantics 9(4), 1992. DOI: 10.1093/jos/9.4.333
- Zeevat, *Presupposition and Accommodation in Update Semantics*, Journal of Semantics 9(4), 1992. DOI: 10.1093/jos/9.4.379
- SEP, *Presupposition*: https://plato.stanford.edu/entries/presupposition/

## 5. Discourse coherence is relational, not just topical similarity

Hobbs and later discourse theories characterize coherence through relations between discourse segments: explanation, result, narration, elaboration, contrast, background, parallel, and related structures. SDRT develops this idea formally by attaching discourse relations to dynamically interpreted segments.

**SPM implication:** two adjacent utterances can be topically similar while playing very different roles. A correction, explanation, concession, interruption, answer, elaboration, or joke should not be represented as simply 'more text about the same topic'.

Candidate state should include a discourse-relation hypothesis linking each contribution to prior discourse where useful.

### Sources

- Hobbs, *Coherence and Coreference*, Cognitive Science 3(1), 1979. DOI: 10.1207/s15516709cog0301_4
- Hobbs, *Why Is Discourse Coherent?* (1978/1979 line of work).
- Asher & Lascarides, SDRT.

## 6. Questions Under Discussion provide a strong model of discourse goal state

Roberts' Question Under Discussion (QUD) framework models discourse as structured around explicit or implicit questions whose resolution organizes relevance. Context is not merely a bag of propositions; it includes a conversational scoreboard and a stack/structure of questions currently being addressed.

This is highly relevant to SPM's recurring 'answered the wrong thing' failure.

A model can preserve every factual proposition in the prompt yet still fail pragmatically because it answers a nearby question rather than the active one.

**SPM implication:** track at least a defeasible `CURRENT_QUD` or equivalent conversational-goal representation, separate from literal sentence semantics.

### Source

- Craige Roberts, *Information Structure: Towards an Integrated Formal Theory of Pragmatics*, Semantics & Pragmatics 5 (2012). DOI: 10.3765/sp.5.6 — https://semprag.org/article/view/sp.5.6

## 7. Topic/focus should constrain interpretation without becoming a hard bottleneck

Information-structure research shows that what is presented as topic, focus, contrast, or background changes how an utterance fits into discourse and what alternatives are salient.

SPM should investigate whether focus/QUD state can improve:

- relevance judgments;
- scope interpretation;
- ellipsis resolution;
- contrast handling;
- correction targeting.

But it should remain probabilistic/defeasible. Human discourse often shifts topic abruptly or deliberately violates expectations.

## 8. Reference should preserve unresolved hypotheses

Many reference expressions are underdetermined at first encounter. A model that instantly picks one antecedent because it produces the smoothest continuation may create hidden state corruption.

Candidate representation:

```text
REFERENCE_HYPOTHESIS {
  mention_id,
  candidate_entity_ids,
  evidence_per_candidate,
  confidence_distribution,
  resolution_status
}
```

Later evidence should update this object without rewriting the fact that ambiguity previously existed.

This is directly testable against ordinary LLM decoding.

## 9. Corrections should target referents or propositions precisely

Discourse repair often modifies only part of the represented state. If the user says `No, by "them" I meant the stickers`, the correct update is not 'throw away the entire conversation'. It is closer to:

```text
supersede(reference_binding(mention=them, old=markers))
set(reference_binding(mention=them, new=stickers))
recompute(dependent_interpretations)
preserve(unaffected_context)
```

The notation is schematic. The important requirement is **localized revision with dependency propagation**.

This should be a first-class benchmark family for SPM.

## 10. Candidate SPM discourse state suggested by this literature

```text
ENTITY_REGISTRY
MENTION -> ENTITY_BINDING
ALIASES / DESCRIPTIONS
INDEXICAL_CONTEXT
SALIENCE / CENTERING_STATE
TOPIC / FOCUS
CURRENT_QUD / DISCOURSE_GOAL
PRESUPPOSITION_LEDGER
ACCOMMODATED_BUT_UNVERIFIED_CONTENT
DISCOURSE_SEGMENTS
COHERENCE_RELATIONS
UNRESOLVED_REFERENCE_HYPOTHESES
CORRECTION_DEPENDENCY_LINKS
```

Again, these may be learned distributed states rather than explicit symbolic records.

## 11. Falsifiable experiments

### A. Same words, different referents

Use repeated pronouns/indexicals across multiple speakers, quoted speech, role swaps, temporal changes, and deictic shifts. Score stable entity binding.

### B. Salience trap

Make the most recently mentioned or emotionally salient entity *not* the correct referent. Compare SPM candidate against matched LLM baselines.

### C. Presupposition versus assertion versus verification

Present identical proposition content as assertion, presupposition, rumor, quotation, and verified observation. Test whether later reasoning preserves source/status distinctions.

### D. QUD versus lexical overlap

Construct a discourse where the active question differs from the most lexically similar recent sentence. Score whether the answer addresses the actual discourse goal.

### E. Local correction propagation

Establish a mistaken referent, correct it, then ask a downstream question whose answer depends on revising only the affected dependencies. Penalize both correction theater and over-resetting.

## 12. Foundation conclusion

Reference/discourse research supplies a central design constraint for SPM:

> **Meaning requires persistent identity across changing expressions, and discourse state must distinguish what is salient, presupposed, currently at issue, and actually established.**

An SPM that merely encodes 'topic vectors' more strongly will not solve the problem. The target is stable, revisable, provenance-aware discourse identity.
