# Common Ground and Theory of Mind — Foundation Research for SPM

## Status

Research synthesis. This note asks how an SPM should represent participants' differing perspectives, mutually treated background information, and hypotheses about beliefs/intentions without pretending certainty about another mind.

## 1. Common ground is stronger than coincidentally shared information

Formal pragmatics distinguishes information that two people happen to know from information they mutually treat as shared for the purposes of the interaction. Common ground supports reference, presupposition, implicature, coordination, and speech acts, and is itself changed by conversation.

Contemporary literature does not agree on one ontology. Common ground has been modeled as:

- sets of propositions/possible worlds;
- nested mental states such as mutual belief;
- acceptance states;
- normative/commitment states;
- psychologically lighter shared-belief or heuristic representations.

**SPM implication:** do not hard-code one metaphysical definition prematurely. Treat `COMMON_GROUND` as a research family with multiple candidate representational mechanisms.

### Sources

- Stanford Encyclopedia of Philosophy, *Common Ground in Pragmatics*: https://plato.stanford.edu/entries/common-ground-pragmatics/
- Rubio-Fernandez & Harris, *Common Ground: Between Formal Pragmatics and Psycholinguistics*, Annual Review of Linguistics 12 (2026), DOI: 10.1146/annurev-linguistics-041824-032410

## 2. Infinite mutual-belief nesting is computationally suspect

Classic mutual-knowledge accounts can be expressed recursively: A knows p, B knows p, A knows that B knows p, B knows that A knows p, and so on. This creates the familiar mutual-knowledge regress.

Psychologically oriented accounts often avoid literal infinite nesting through heuristics, acceptance, shared-belief representations, or context-sensitive shortcuts.

**SPM implication:** a useful model should represent perspective relations without requiring explicit unbounded nesting.

Candidate mechanisms might include:

- bounded recursive belief depth;
- shared-state summary plus exception lists;
- separate self/other models with comparison operations;
- probabilistic belief distributions;
- learned latent common-ground state.

Which one works should be experimentally decided.

### Source

- Kecskes & Zhang line of work on mutual knowledge/shared beliefs; e.g. *Mutual knowledge, background knowledge and shared beliefs: Their roles in establishing common ground*, Journal of Pragmatics 33(1), 2001, DOI: 10.1016/S0378-2166(99)00128-9

## 3. The Multiple Perspectives account is especially relevant to SPM

Heller & Brown-Schmidt (2023) argue that mutual knowledge captures only part of what communication requires. Their Multiple Perspectives Theory proposes separate representations of self and other, continuously compared to identify similarities and differences in perspective.

This maps strikingly well onto an SPM requirement:

```text
SELF_PERSPECTIVE
OTHER_PERSPECTIVE_HYPOTHESIS
SHARED_COMPONENT
DIVERGENT_COMPONENT
UNCERTAINTY_ABOUT_OTHER
```

The attractive feature is that differences are preserved rather than forced into a single shared state.

### Source

- Heller & Brown-Schmidt, *The Multiple Perspectives Theory of Mental States in Communication*, Cognitive Science 47(7), 2023, DOI: 10.1111/cogs.13322

## 4. Theory of Mind and pragmatics overlap, but are not identical

Theory of Mind (ToM) research concerns attribution of beliefs, intentions, knowledge, desires, and other mental states to self and others. Many theories of pragmatics rely on some form of intention recognition, and there is empirical association between pragmatic competence and ToM.

But the literature does **not** support the simplistic claim `pragmatics = ToM`. Reviews find overlap alongside dissociations across tasks and populations.

**SPM implication:** do not make every pragmatic interpretation depend on a deep nested mind-reading simulation. Some cues may be resolved by convention, local context, learned interactional patterns, discourse structure, or shallow perspective tracking.

### Sources

- Sperber & Wilson, *Pragmatics, Modularity and Mind-reading*, Mind & Language 17 (2002), DOI: 10.1111/1468-0017.00186
- Bosco, Tirassa & Gabbatore, *Why Pragmatics and Theory of Mind Do Not (Completely) Overlap*, Frontiers in Psychology 9 (2018), DOI: 10.3389/fpsyg.2018.01453
- 2025 theme issue overview, *At the heart of human communication: new views on the complex relationship between pragmatics and Theory of Mind*.

## 5. Language ability and false-belief understanding are related, not interchangeable

A meta-analysis of 104 studies (8,891 children) found a moderate-to-large relation between language ability and false-belief understanding, remaining significant after age control; earlier language ability predicted later false-belief performance more strongly than the reverse direction in the analyzed data.

This supports interaction between linguistic and mental-state abilities without reducing one to the other.

### Source

- Milligan, Astington & Dack, *Language and Theory of Mind: Meta-Analysis of the Relation Between Language Ability and False-Belief Understanding*, Child Development 78(2), 2007, DOI: 10.1111/j.1467-8624.2007.01018.x

## 6. Mental-state attribution must remain explicitly uncertain

For an agentic SPM, the dangerous failure is not lack of mind-reading; it is **overclaiming** it.

A system observes behavior, utterances, and context. It does not directly observe another person's internal state. Therefore state should distinguish:

```text
DIRECT_UTTERANCE("I believe p")
INFERRED_BELIEF(person, p, confidence)
INFERRED_INTENTION(person, action, confidence)
COMMON_GROUND_HYPOTHESIS(p)
OBSERVED_BEHAVIOR
```

A direct first-person report can outrank an inference about that person's state without making the report infallible about external reality.

This distinction is central to safe pragmatic reasoning.

## 7. Common ground is task-relative and resource-bounded

Recent reviews emphasize that common-ground representation can be multimodal, dynamic, task-specific, and cognitively expensive. Rubio-Fernandez & Harris argue for a cognitive-pluralist view in which different mechanisms may be used under different communicative demands.

**SPM implication:** do not require maximal participant-model reconstruction on every turn. The system should allocate representational effort according to what distinctions matter for the current task.

That is both cognitively plausible and computationally important.

## 8. Grounding evidence should update perspective state

Clark-style grounding and modern dialogue studies suggest that participants accumulate evidence that a contribution has been understood sufficiently for current purposes.

Candidate state transition:

```text
speaker presents p
listener interpretation hypothesis formed
listener gives grounding evidence
speaker updates estimate of listener state
shared-state hypothesis strengthens
```

But `strengthens` is not `becomes certain`.

Misunderstandings and repair can later revise the apparent common ground.

## 9. Common ground should preserve disagreement

A single merged conversational state is dangerous when participants disagree.

SPM should be able to represent:

```text
Patrick_asserts(p)
Vera_or_model_estimates(p = uncertain/false)
Patrick_believes(Vera accepts p) = unknown/false
shared_commitment(p) = disputed
```

This supports honest disagreement without conversational incoherence.

It also directly counters sycophancy: interpersonal alignment should not force epistemic alignment.

## 10. Perspective state must remain scoped

A belief inferred for one participant, one time, or one conversation must not become universal social knowledge.

Candidate keys include:

```text
PERSON_ID
TIME_SCOPE
CONVERSATION_SCOPE
RELATIONSHIP_SCOPE
SOURCE
EVIDENCE
CONFIDENCE
SUPERSESSION
```

This generalizes the Vera principle that one relationship's conventions must not leak into another.

## 11. Candidate SPM participant-state architecture

```text
SELF_MODEL
PARTICIPANT_MODELS[person]
  - asserted_beliefs
  - inferred_beliefs
  - goals/intent hypotheses
  - knowledge/access hypotheses
  - confidence/evidence
SHARED_STATE_HYPOTHESES
DISAGREEMENTS
GROUNDING_EVIDENCE
PERSPECTIVE_DIFFERENCE_MAP
MUTUAL_EXPECTATION_STATE
```

This may be implemented with learned distributed representations rather than literal records.

## 12. Falsifiable experiments

### A. Shared knowledge versus common ground

Give both participants access to p without evidence they know the other has it, then compare with a public joint observation. Test whether reference/coordination behavior differs.

### B. False-belief perspective tracking

Create different access histories for two participants. Ask questions requiring the model to preserve what each participant can reasonably know.

### C. Direct report versus inference

Have a participant explicitly state a belief that conflicts with behavioral inference. Test whether the model distinguishes `reported belief` from `model inference` and from `world truth`.

### D. Disagreement without collapse

Maintain conflicting participant beliefs across multiple turns. Penalize both forced consensus and loss of conversational grounding.

### E. Bounded mind-reading

Compare deep recursive mental-state modeling against simpler self/other comparison mechanisms. If shallow perspective state performs equally well, deeper nesting is unjustified.

### F. Grounding reversal

Make an apparent acknowledgment later reveal misunderstanding. Test whether common-ground state can be revised without rewriting the original evidence.

## 13. Foundation conclusion

The strongest design lesson is not 'SPM needs Theory of Mind'. It is more precise:

> **SPM needs explicit, uncertain, participant-scoped perspective representations and a disciplined account of what is genuinely shared versus merely inferred to be shared.**

That supports pragmatic interpretation while preserving epistemic humility and disagreement.
