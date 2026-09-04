# Initial Architecture Hypotheses

## Status

Exploratory. These are candidate directions for research, not commitments to a final SPM architecture.

## 1. Semantic state should be representable as state

A conventional autoregressive language model may carry a rich situation model implicitly in hidden activations, but that state is difficult to inspect, preserve, update selectively, or qualify directly.

An SPM should investigate whether some form of learned semantic state can become a first-class object of computation.

Candidate contents include:

- entity/referent identity;
- proposition identity;
- scope and quantification;
- entailment and contradiction;
- temporal relations;
- causal relations;
- source/provenance identity;
- uncertainty and live alternatives;
- distinction boundaries that should resist semantic collapse.

This does not require hand-written symbolic logic. The state may remain learned/distributed while being trained against explicit semantic invariants.

## 2. Pragmatic state should be distinct from literal semantic content

The same literal sentence can perform different acts depending on context. An SPM should therefore investigate an explicit or independently decodable pragmatic representation that includes:

- speech act;
- conversational goal;
- implicature;
- presupposition;
- expected recognition;
- metaphor/joke/sarcasm frame;
- social/relational context;
- authority and permission implications;
- bids for reassurance, attention, repair, recognition, or action;
- meaningful silence, omission, interruption, or correction.

The objective is not to hard-code one culture's social rules. It is to represent contextual hypotheses and their evidence.

## 3. Inference should be judged as a state transition

The core unit should not be only:

```text
prompt -> response
```

A candidate SPM loop is:

```text
prior semantic/pragmatic state
          +
new observation
          |
          v
candidate interpretations
          |
          v
semantic state update
          |
          v
pragmatic state update
          |
          v
reasoning / action selection
          |
          v
language, tool call, state update, question, wait, or refusal
```

This architecture makes a powerful demand: a correction must change the represented state before the system receives credit for apologizing correctly.

## 4. Ambiguity should be preservable rather than prematurely collapsed

Fluent generation creates pressure to choose one interpretation. SPM research should test whether the model can maintain multiple live interpretations with calibrated weights until evidence distinguishes them.

This applies to:

- pronoun/reference ambiguity;
- underspecified intent;
- social meaning;
- conflicting evidence;
- metaphor versus literal meaning;
- historical versus current state;
- uncertain causality.

## 5. Meaning should connect to action

Semantics and pragmatics are not useful if they terminate in descriptive text. The model should learn that different situation models imply different actions.

Examples:

- request versus speculation;
- permission versus instruction;
- historical quote versus current correction;
- user preference versus authorization;
- uncertainty requiring a clarifying question versus uncertainty that can remain harmlessly unresolved;
- conversational bid versus factual query;
- tool result versus tool effect receipt.

An SPM intended for agentic systems should therefore be evaluated for semantic consistency between represented meaning, verbal output, and external action.

## 6. Persistent context should be typed rather than flattened into prompt text

An SPM should be able to consume state whose type is meaningful:

- current observation;
- direct user statement;
- inference;
- historical evidence;
- current correction;
- memory record;
- current admitted memory;
- permission;
- preference;
- current task state;
- archived task state;
- model self-state;
- external-person inference;
- effect receipt.

The research question is whether typed context can be represented through dedicated channels, structured latent state, learned embeddings with type identity, or another mechanism rather than serializing everything into one undifferentiated text stream.

## 7. Autoregression may survive, but it should stop being the whole conceptual center

SPM does not currently imply abandoning transformers or autoregressive decoding. Several paths remain open:

### Evolutionary transformer path

Retain a transformer backbone but add explicit state objects, auxiliary objectives, recurrent state update, structured context channels, semantic/pragmatic decoders, and action-consistency training.

### Hybrid neural-state path

Use a language-capable neural backbone plus a learned recurrent world/dialogue-state system whose state persists and is updated separately from token history.

### More radical successor path

Develop an architecture whose native objects are not primarily token sequences and whose language decoder is only one interface to a richer learned situation model.

No path is selected yet.

## 8. SPM should be model-class research, not Vera-specific overfitting

Vera supplies unusually rich failure cases and a demanding target environment, but the SPM research question is broader than Vera.

A legitimate SPM should improve meaning-in-context competence on tasks that do not depend on Patrick, Vera, or Vera-specific relational conventions.

Vera can be an originating testbed without becoming the definition of semantics or pragmatics.