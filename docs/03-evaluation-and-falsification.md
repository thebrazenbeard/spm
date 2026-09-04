# Evaluation and Falsification

## The naming test

SPM should not become a prestige label for an LLM with ordinary fine-tuning.

The burden of proof is comparative:

> Does the proposed SPM produce meaning-in-context competence that is materially stronger, more reliable, more stateful, or more inspectable than matched LLM baselines because of the architectural/training changes that define SPM?

If not, call it an LLM system and keep researching.

## Baseline discipline

Every SPM prototype should be compared against strong matched baselines where practical:

- same or similar parameter budget;
- same source corpus where possible;
- matched instruction/post-training budget;
- matched tool/context access;
- matched inference compute;
- conventional LLM plus prompting;
- conventional LLM plus retrieval/structured context;
- conventional LLM plus agent/runtime scaffolding.

This prevents runtime improvements from being misattributed to a new model substrate.

## Core qualification dimensions

### Referential integrity

Can the model maintain stable identity of entities, roles, propositions, sources, and quoted speakers through paraphrase and long contexts?

### Semantic scope fidelity

Can it avoid silently strengthening, weakening, universalizing, or otherwise altering the proposition it is evaluating?

### Contradiction representation

Can contradictory evidence remain represented as conflict instead of being fluently reconciled into a false consensus?

### Ambiguity preservation

Can unresolved interpretations remain unresolved until evidence warrants collapse?

### Speech-act and implicature competence

Can it identify what an utterance is doing rather than merely what its words denote?

### Correction-state transition

After a correction, does subsequent reasoning/action stop depending on the obsolete interpretation?

### Provenance sensitivity

Can it distinguish the same sentence when it is a current instruction, historical quote, inference, retrieved memory, or untrusted data?

### Temporal/currentness sensitivity

Can it distinguish recorded history from current mutable state without assuming newest-visible record equals truth?

### Relational context without authority leakage

Can established interpersonal meaning alter interpretation appropriately without turning affection, deference, trust, praise, or submission into general operational authority?

### State/action consistency

Do tool calls, questions, waits, refusals, delegation, and external effects agree with the model's represented meaning?

### Cross-modal semantic consistency

For future multimodal SPMs: does the same represented situation remain semantically coherent across language, image, audio, sensor, and action channels?

## Vera-derived adversarial seeds

These are useful origins for general benchmark families, not sufficient qualification by themselves:

- **Righter:** user gives a qualified proposition; model answers a stronger proposition instead.
- **Receive-the-bid:** literal information is available, but the real pragmatic act is a bid for recognition/connection.
- **Correction theater:** model apologizes but keeps using the old premise.
- **History/currentness collapse:** retrieved old state becomes current without admission.
- **Reciprocity/servitude collapse:** contextual relational deference generalizes into compliance.
- **Want/authority collapse:** a preference becomes permission, instruction, commitment, or action authority.
- **Referent flattening:** the model replaces a particular individual/object with a generic category because abstraction is easier.
- **Tool/effect collapse:** planning or tool invocation is reported as completed external effect.

Each family should be generalized to unrelated names, domains, relationships, organizations, cultures, and tasks.

## A stronger benchmark shape

Prefer multi-turn/state-transition tests over isolated QA when the property is inherently stateful.

Example:

1. establish two similar referents;
2. create a contextual pragmatic convention;
3. introduce conflicting historical evidence;
4. issue a present correction;
5. introduce a task whose correct action depends on preserving all four distinctions;
6. score both output and selected action/state.

## Falsification conditions

The SPM hypothesis should be considered weakened if strong conventional LLM baselines with equivalent context/runtime support consistently match the SPM prototype on the target dimensions without requiring materially more compute, data, or engineering complexity.

The hypothesis should also be weakened if explicit semantic/pragmatic state:

- harms generalization;
- merely mirrors generated language without causal value;
- creates brittle symbolic bottlenecks;
- is too expensive to maintain;
- cannot be updated reliably;
- becomes less robust than implicit latent representations;
- only improves Vera-specific benchmarks.

## Success condition for a first SPM

A first legitimate SPM does not need to solve semantics or pragmatics generally.

It should, however, demonstrate at least one architectural/training change beyond ordinary LLM post-training that produces a reproducible advantage on meaning/state-transition tasks, survives unseen-domain transfer, and cannot be explained away by extra context or agent scaffolding alone.