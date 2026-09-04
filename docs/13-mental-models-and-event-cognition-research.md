# Mental Models and Event Cognition — Foundation Research for SPM

## Status

Research synthesis. This note examines what cognitive theories say about preserving possibilities, events, schemas, memory, prediction, and grounding, while explicitly avoiding the claim that SPM should imitate biological implementation.

## 1. Mental-model theory emphasizes represented possibilities

Johnson-Laird's mental-model theory argues that reasoning depends on constructing representations of possibilities consistent with premises rather than merely applying syntactic proof rules. Multiple models may be required when more than one possibility remains live; reasoning errors can arise when people fail to construct relevant alternatives.

This is highly relevant to SPM's ambiguity requirement.

**SPM implication:** a cognition substrate may benefit from representing a set of situation hypotheses rather than forcing one interpretation into a single hidden state. Counterexamples and alternative models can be computationally meaningful objects.

### Sources

- Johnson-Laird, *Mental models and human reasoning*, PNAS 107(43), 2010, DOI: 10.1073/pnas.1012933107
- Johnson-Laird, *The Mental Models Perspective*, Oxford Handbook of Cognitive Psychology (2013).

## 2. Event cognition provides a natural unit above tokens and below 'the whole world'

Event Segmentation Theory and related research find that people spontaneously divide continuous activity into hierarchically organized events and subevents. Event boundaries correlate with changes in action, goals, causal structure, and predictability; segmentation affects memory and learning.

**SPM implication:** event boundaries may be a useful trigger for persistent-state updates, memory writes, salience resets, and hierarchical compression.

Potential model object:

```text
EVENT {
  participants,
  interval,
  location,
  goals,
  causal_predecessors,
  state_changes,
  parent_event,
  subevents
}
```

This is a research sketch, not a requirement for a symbolic representation.

### Sources

- Zacks & Swallow, *Event Segmentation*, Current Directions in Psychological Science 16(2), 2007, DOI: 10.1111/j.1467-8721.2007.00480.x
- Zacks, *Event Perception and Memory*, Annual Review of Psychology 71 (2020), DOI: 10.1146/annurev-psych-010419-051101

## 3. Prediction error may help identify state-transition boundaries

Event Segmentation Theory proposes that working event models support near-future prediction; when predictions deteriorate, the system updates its event model. This suggests a mechanism for deciding *when* to revise persistent state.

**SPM hypothesis worth testing:** learned semantic/pragmatic state may be updated sparsely or hierarchically when interpretation-relevant prediction error exceeds a threshold, rather than being fully rewritten each token.

But prediction error must not become the only update criterion. Explicit correction, source change, permission, temporal shift, or discourse repair may require update even when the linguistic continuation is predictable.

## 4. Schema can compress recurrent structure but also bias interpretation

Schema-like knowledge allows familiar events and roles to be understood with sparse evidence. This is computationally valuable, but it creates a danger analogous to LLM stereotype completion: expectations can overwrite observation.

**SPM implication:** schema activation should be represented as prior structure, while observation remains separately represented. State revision must be able to violate the schema cleanly.

Candidate distinction:

```text
PRIOR_SCHEMA_EXPECTATION
OBSERVED_EVIDENCE
POSTERIOR_EVENT_STATE
```

## 5. Working memory and semantic memory suggest different persistence timescales

Cognitive research distinguishes temporary working representations from durable semantic knowledge. SPM need not copy human memory systems, but the distinction is a useful design prior:

- transient dialogue/situation state;
- durable learned relational/semantic structure;
- episodic/event records;
- retrieved background knowledge.

Flattening all four into one token history or one mutable vector risks interference and currentness errors.

## 6. Grounded cognition asks what representational vehicles matter

Grounded-cognition theories argue that cognition can recruit perceptual, motor, affective, and introspective systems rather than relying only on amodal symbols. There is substantial evidence for modality-sensitive effects, but strong claims that linguistic understanding *requires* full embodied simulation remain contested.

**SPM implication:** multimodal grounding is a research axis, not a dogma. The key question is whether state tied to perception/action improves semantic and pragmatic generalization compared with abstract state.

### Sources

- Barsalou, *Grounded Cognition*, Annual Review of Psychology 59 (2008), DOI: 10.1146/annurev.psych.59.103006.093639
- Weiskopf, *Embodied cognition and linguistic comprehension* (critical counterpoint).

## 7. Joint attention and grounding matter because meaning can be externally anchored

Reference in real interaction is often grounded by shared perceptual attention, gesture, object manipulation, or task context. A multimodal SPM should therefore be able to bind a linguistic referent to a persistent external object/event identity.

This suggests the model needs an abstraction more general than `text entity`:

```text
REFERENT_ID
  <- linguistic mentions
  <- visual observations
  <- tool/environment object IDs
  <- participant demonstrations
```

The binding should preserve uncertainty when cross-modal identity is not established.

## 8. Predictive processing is relevant but should not be over-generalized

Language-comprehension research shows widespread anticipatory processing, but recent reviews emphasize that prediction varies across people/tasks and that its underlying mechanisms remain debated.

SPM should therefore distinguish:

```text
PREDICTED_NEXT_INPUT
EXPECTED_EVENT_TRANSITION
CURRENT_INTERPRETED_STATE
```

A system should be able to be surprised without treating surprise as contradiction, and should be able to update to an observation that was improbable under its prediction.

## 9. Event boundaries may be ideal memory/compression checkpoints

A promising Vera OS/SPM interface hypothesis is that state should not be serialized indiscriminately every token. Event/discourse boundaries could trigger:

- state consolidation;
- unresolved-hypothesis retention;
- referent/salience summaries;
- commitment/repair checks;
- episodic memory writes;
- long-context compression.

This would connect cognitive event segmentation to a practical persistent-agent architecture while keeping the mechanisms separable for testing.

## 10. Candidate SPM event/state machinery

```text
ACTIVE_EVENT_MODEL
EVENT_BOUNDARY_PROBABILITY
PARENT/SUBEVENT_STRUCTURE
ACTIVE_SCHEMA_PRIORS
OBSERVATION_DELTA
PREDICTION_ERROR
GOAL/CAUSAL_STATE
TRANSIENT_WORKING_STATE
DURABLE_RELATIONAL_KNOWLEDGE
EPISODIC_EVENT_RECORDS
MULTIMODAL_REFERENT_BINDINGS
```

## 11. Falsifiable experiments

### A. Boundary-sensitive memory

Compare token-uniform persistence with event-boundary consolidation on long sequential tasks. Score referent, goal, and causal continuity.

### B. Schema violation

Establish a strong familiar script, then introduce one critical exception. Test whether the model preserves observation over schema completion.

### C. Multiple mental models

Give premises compatible with several possibilities; later evidence selects one. Test whether alternative states remain available without fabricated certainty.

### D. Prediction-error update

Use predictable versus surprising event transitions and measure whether state revisions concentrate at useful boundaries without missing explicit semantic corrections.

### E. Cross-modal reference

Bind names/pronouns to perceived objects across transformations and occlusion. Compare persistent entity state against re-inference from flattened multimodal history.

## 12. Foundation conclusion

Mental-model theory contributes **explicit alternatives**. Event cognition contributes **hierarchical state boundaries**. Memory research contributes **timescale separation**. Grounded cognition contributes **connection to perception/action while warning against purely linguistic abstraction**.

The useful SPM lesson is not 'copy the human brain'. It is:

> **Useful cognition appears to preserve structured possibilities, event identity, and state transitions at levels that do not correspond one-to-one with the incoming word stream.**
