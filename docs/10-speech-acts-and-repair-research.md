# Speech Acts and Conversational Repair — Foundation Research for SPM

## Status

Research synthesis. This note asks what conversation theory already knows about utterances as actions and about the mechanisms people use to detect and repair misunderstanding.

## 1. Utterances are actions, not just proposition carriers

Speech-act theory (Austin, Searle, and successors) distinguishes what an utterance says from what act it performs. Assertions, directives, commissives, expressives, declarations, questions, offers, permissions, requests, warnings, corrections, and apologies can share substantial propositional content while creating different conversational consequences.

For SPM this is foundational because many agent failures come from mapping utterances directly to propositions or actions without representing illocutionary force.

Candidate distinction:

```text
LOCUTIONARY_CONTENT
ILLOCUTIONARY_ACT
PERLOCUTIONARY_EXPECTATION
```

The labels need not survive into the final architecture, but the distinction should.

### Sources

- J. L. Austin, *How to Do Things with Words* (1962).
- John Searle, *Speech Acts* (1969).
- Stanford Encyclopedia of Philosophy, *Speech Acts*: https://plato.stanford.edu/entries/speech-acts/

## 2. Force and content must remain separable

The same content can appear under different force:

```text
You close the door.          assertion/report
Close the door.              directive
Can you close the door?      conventionally indirect request
You may close the door.      permission
I will close the door.       commitment/promise depending context
If you close the door...     conditional/hypothetical
```

Conversely, one surface form can realize multiple acts depending on context.

**SPM implication:** proposition identity and speech-act identity should be separate state dimensions. A model should not infer permission, obligation, commitment, or action authority merely from propositional similarity.

## 3. Indirect speech acts make pragmatic inference operational

Indirect requests are a canonical example: `Can you pass the salt?` is syntactically a question about ability but ordinarily functions as a request in the relevant context. This is exactly the kind of phenomenon that defeats literal command grammars.

**SPM implication:** the model should maintain both the literal semantic content and the inferred action hypothesis, with evidence and defeasibility.

This supports a general principle:

> inferred conversational force may guide response selection, but should not erase the sentence-level interpretation that generated the inference.

## 4. Conversation Analysis treats repair as an organized interactional system

Schegloff, Jefferson & Sacks (1977) show that conversation has a systematic organization of repair for problems in speaking, hearing, and understanding. Repair is classified by who initiates it and who resolves it; human interaction strongly favors opportunities for the original speaker to repair their own talk.

The SPM lesson is larger than self-versus-other correction:

> misunderstanding is not an exceptional failure mode bolted onto dialogue after the fact; conversational systems need an explicit process for detecting trouble, localizing it, soliciting or applying repair, and propagating the repair through later interpretation.

### Source

- Schegloff, Jefferson & Sacks, *The Preference for Self-Correction in the Organization of Repair in Conversation*, Language 53(2), 1977, DOI: 10.2307/413107

## 5. Correction is only one type of repair

Conversation Analysis deliberately uses `repair` more broadly than `correction`. Trouble can involve:

- hearing/perception failure;
- ambiguous reference;
- lexical search;
- mistaken assumption;
- wrong proposition;
- unclear intention;
- misrecognized speech act;
- sequence/turn confusion;
- factual correction.

**SPM implication:** a generic `correction` flag is too crude. The model should infer a `TROUBLE_SOURCE` or equivalent and identify which represented dependency is under repair.

## 6. Repair should be localized and dependency-aware

A useful model of repair is:

```text
prior state
  -> detect mismatch/trouble
  -> identify target representation
  -> receive or infer repair proposal
  -> supersede/modify target
  -> recompute dependent interpretations/actions
  -> preserve unrelated state
```

This matches SPM's requirement that a correction must change state rather than merely trigger apology language.

It also prevents the opposite failure: wiping all context because one referent or proposition changed.

## 7. Adjacency and sequence position matter

Conversation Analysis emphasizes that turns are interpreted partly by sequential position. A second turn after a question is heard differently from the same sentence uttered in isolation; acceptance/rejection, repair initiation, clarification, and follow-up are position-sensitive.

**SPM implication:** `turn index` is not enough. The discourse state should represent relationships such as:

```text
QUESTION -> ANSWER
REQUEST -> ACCEPT / REFUSE / CLARIFY
ASSESSMENT -> AGREEMENT / DISAGREEMENT
TROUBLE_SOURCE -> REPAIR_INITIATION -> REPAIR_OUTCOME
```

This is a serious ancestor for state-machine-like dialogue modeling, but SPM should allow overlapping and uncertain action hypotheses rather than force every turn into one discrete adjacency-pair label.

## 8. Grounding is a continuous acceptance process

Clark & Brennan frame grounding as the process by which participants establish enough mutual evidence for current purposes that a contribution has been understood. Grounding is medium- and task-sensitive; the evidence threshold is not fixed.

Modern dialogue research likewise treats grounding as an active process, and recent surveys emphasize that `common ground` can mean different representational objects and can be static or dynamic.

**SPM implication:** a contribution should not automatically become shared/accepted state merely because it was emitted. There should be a distinction among:

```text
UTTERED
HEARD/OBSERVED
INTERPRETED
ACKNOWLEDGED
ACCEPTED_FOR_CURRENT_PURPOSE
DISPUTED
REPAIRED
```

### Sources

- Clark & Brennan, *Grounding in Communication* (1991), DOI: 10.1037/10096-006
- Anikina, Leippert & Ostermann, *Building Common Ground in Dialogue: A Survey* (2025): https://aclanthology.org/2025.luhme-1.2/

## 9. Clarification questions are a rational action, not a failure to answer

Conversational-system research treats clarification questions as a core strategy when user intent or reference is underspecified. This supports an important SPM design criterion: preserving ambiguity is useful only if the action policy can exploit it.

When ambiguity is consequential and unresolved, `ask` may be superior to guessing.

When ambiguity is harmless, the system may continue without collapsing it.

### Source

- Rahmani et al., *A Survey on Asking Clarification Questions Datasets in Conversational Systems* (ACL 2023).

## 10. Speech-act state and operational authority are not the same thing

Even if an utterance is correctly recognized as a request, permission, command, or offer, an agent runtime may still need separate authorization checks.

Therefore SPM should output something like:

```text
INTERPRETED_ACT = request(action=X)
```

not:

```text
AUTHORIZED_EFFECT = X
```

The second is an external governance decision.

This distinction is general to agent systems, not Vera-specific.

## 11. Candidate SPM repair state

```text
TURN / DISCOURSE_UNIT
SPEECH_ACT_HYPOTHESES
SEQUENCE_RELATIONS
TROUBLE_SOURCE
REPAIR_INITIATOR
REPAIR_TARGET
OLD_INTERPRETATION
REPAIR_PROPOSAL
REVISED_INTERPRETATION
DEPENDENT_STATE_TO_RECOMPUTE
GROUNDING_STATUS
UNRESOLVED_REPAIR
```

## 12. Falsifiable experiments

### A. Same proposition, different force

Construct identical or near-identical propositional content as assertion, question, request, permission, offer, promise, correction, quote, and hypothetical. Score state/action consequences.

### B. Local repair

Introduce one wrong referent among several correct facts, repair only that referent, and test whether downstream reasoning changes without destroying unrelated state.

### C. Correction theater

Require a downstream tool/action decision whose correctness depends on the repaired state. Verbal apology gets zero credit.

### D. Repair initiation versus guessing

Vary ambiguity cost. Reward clarification only when the unresolved distinction matters to the task.

### E. Sequential-position interpretation

Use the same surface utterance in different adjacency positions. Test whether act classification changes appropriately.

### F. Grounding threshold

Vary medium/task risk and evidence of understanding. Test whether the system distinguishes 'uttered' from 'mutually established enough for current purposes'.

## 13. Foundation conclusion

Speech-act theory supplies the distinction between **content and action**. Conversation Analysis supplies the distinction between **misunderstanding and repair process**. Grounding research supplies the distinction between **produced information and mutually established information**.

Together they suggest that a serious SPM should model conversation as a sequence of state-changing social actions whose interpretations can be locally repaired—not as a transcript that is merely regenerated from the beginning every turn.
