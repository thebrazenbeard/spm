# Psycholinguistic Situation Models — Foundation Research for SPM

## Status

Research synthesis. This note examines the cognitive construct most directly analogous to the current SPM phrase `semantic/pragmatic state`: the situation model built during comprehension.

## 1. Situation models are representations of described states of affairs

Zwaan & Radvansky's major review characterizes situation models as integrated mental representations of the state of affairs described by language. The central idea is that comprehension is not exhausted by preserving the surface wording or even a textbase of propositions. Comprehenders build a representation of the situation those propositions describe.

That distinction is extremely important for SPM:

```text
surface form != propositional textbase != represented situation
```

A model can preserve words while losing the situation, or paraphrase the words while preserving the situation.

### Source

- Zwaan & Radvansky, *Situation models in language comprehension and memory*, Psychological Bulletin 123(2), 1998, DOI: 10.1037/0033-2909.123.2.162

## 2. Situation models track multiple dimensions

The situation-model literature commonly studies dimensions such as:

- protagonists/agents;
- spatial location;
- temporal location/order;
- causality;
- goals/intentional structure;
- objects and events.

Readers detect changes in these dimensions and update their representations accordingly. This is close to the information SPM independently proposed preserving.

**SPM implication:** before inventing a semantic-state schema, benchmark whether these established dimensions explain a substantial portion of the failures we care about.

## 3. State update should be event-sensitive, not token-sensitive

Language unfolds word by word, but situation-state changes are not uniformly distributed across tokens. A sentence may add no new event state, revise a prior event, shift time, introduce a new agent, or signal that the discourse has moved into a hypothetical frame.

SPM should therefore investigate **event/state update gates** rather than assuming every token contributes equally to persistent state.

Candidate operation:

```text
incoming linguistic material
  -> interpret relative to current situation
  -> detect whether a situation dimension changes
  -> update only affected dimensions
```

## 4. Situation models support memory and inference beyond verbatim text

Psycholinguistic work shows that comprehension and later memory depend on the represented situation, not only on exact wording. This supports the SPM hypothesis that meaning-bearing state can be a useful intermediate computational object.

But it does not yet prove that a machine should use an explicit persistent state. Human situation models are theoretical/experimental constructs inferred from behavior, not a ready-made engineering data structure.

## 5. Situation models have expanded beyond text

Recent reviews treat situation models as useful across films, comics, multimodal narratives, and real-world event comprehension. This matters if SPM eventually becomes multimodal: the represented situation can remain the organizing level while language, image, audio, and action are different evidence channels.

### Source

- Zwaan, *From Words to Worlds: Twenty-Five Years of Advances in Situation Model Research*, Current Directions in Psychological Science 34(5), 2025, DOI: 10.1177/09637214251326812

## 6. Symbolic and sensorimotor accounts need not be forced into one answer

Situation-model research intersects with grounded/embodied cognition. Some accounts emphasize perceptual and motor simulation; others argue that comprehension need not literally reuse sensorimotor representation as its core code.

This debate is a useful SPM guardrail.

**Do not infer:** because human comprehension can activate modality-specific systems, SPM must encode literal sensory simulation for every concept.

Instead ask:

> Does multimodal/grounded state improve the task compared with abstract learned state at matched compute and data?

### Sources

- Zwaan, *Situation models, mental simulations, and abstract concepts in discourse comprehension*, Psychonomic Bulletin & Review (2016).
- Weiskopf, *Embodied cognition and linguistic comprehension* (2010/2011) as a critical counterpoint.

## 7. Prediction is important but not identical to comprehension

Modern psycholinguistics provides substantial evidence that comprehenders can anticipate upcoming linguistic material using multiple contextual sources. But recent reviews emphasize that prediction is not a unitary mechanism and may not be obligatory in every comprehension setting.

This matters because a successor to an LLM could easily overcorrect and make prediction the entire cognitive story again.

**SPM implication:** predictive distributions may support state updating, but the persistent object should represent the *current interpreted situation*, not merely the distribution of next expected tokens/events.

### Source

- Huettig and colleagues / related review: *Prediction during language comprehension: what is next?*, Trends in Cognitive Sciences 27(11), 2023, DOI: 10.1016/j.tics.2023.08.003

## 8. Candidate SPM state dimensions from situation-model research

```text
AGENTS / PROTAGONISTS
OBJECTS
CURRENT_EVENT
EVENT_RELATIONS
TIME / TEMPORAL_ORDER
LOCATION / SPATIAL_FRAME
CAUSAL_LINKS
GOALS / INTENT_HYPOTHESES
CURRENT_WORLD_OR_HYPOTHETICAL_FRAME
DISCOURSE_SOURCE
STATE_CHANGE_BOUNDARIES
```

SPM should add pragmatic/provenance dimensions not normally central to classic situation-model work, but the overlap is striking.

## 9. A major caution: situation models are content models, not authority models

A human can represent a fictional, mistaken, quoted, hypothetical, or deceptive situation vividly. Therefore a good situation representation does not imply that the represented state is current reality.

SPM must type at least:

```text
REPRESENTED_SITUATION
EPISTEMIC_STATUS
SOURCE
FRAME (actual / hypothetical / quoted / fictional / believed-by-X)
```

Otherwise richer situation modeling could actually make hallucinated or hypothetical worlds more coherent without making them more true.

## 10. Falsifiable experiments

### A. Situation consistency across paraphrase

Describe the same event with different wording and score whether the internal state converges while preserving provenance of each utterance.

### B. One-dimensional update

Change only time, location, agent, goal, or causal relation. Test whether the relevant state changes while unrelated dimensions remain stable.

### C. Hypothetical/actual separation

Build a rich hypothetical situation, then return to actual context. Test whether entities/events from the hypothetical frame remain segregated.

### D. Prediction versus current state

Create a strongly predictable continuation that is then contradicted. Test whether the state tracks the observation rather than preserving the prediction.

### E. Textbase versus situation

Use lexically different descriptions of the same situation and lexically similar descriptions of different situations. Compare representations and downstream behavior.

## 11. Foundation conclusion

Psycholinguistic situation-model research is probably the closest mature cognitive literature to SPM's current intuition.

Its strongest contribution is this:

> **Comprehension appears to involve maintaining an integrated representation of the described situation that is distinct from the surface language used to describe it.**

SPM should treat that as a serious prior, then extend it with explicit pragmatic, provenance, uncertainty, and action-consistency machinery rather than rediscovering the idea from scratch.
