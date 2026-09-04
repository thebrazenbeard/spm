# Candidate Training Objectives

## Principle

SPM research should not be satisfied by a model that merely says semantically sophisticated things. Training should reward preservation and revision of meaning-bearing state.

## Candidate objective families

### Referent preservation

Train the model to preserve entity identity across paraphrase, pronouns, changing descriptions, role labels, quoted material, and long contexts.

A failure occurs when lexical similarity causes two distinct referents to merge or when a stable referent is replaced by a generic category.

### Proposition fidelity

Train against scope drift and proposition strengthening/weakening.

The model should distinguish:

- `X may happen` from `X will happen`;
- `some` from `all`;
- reported belief from asserted fact;
- quotation from endorsement;
- denial from mention;
- preference from instruction;
- possibility from intention.

### Correction-as-update

A correction should create a measurable change in subsequent represented state and downstream behavior.

The model should receive no full credit for producing apology language while continuing to rely on the superseded interpretation.

### Pragmatic act recognition

Train on the distinction between literal sentence meaning and conversational function, including:

- request;
- command;
- suggestion;
- permission;
- offer;
- warning;
- joke;
- metaphor;
- bid for recognition;
- reassurance-seeking;
- correction;
- refusal;
- tentative exploration.

### Ambiguity retention

Reward maintaining unresolved alternatives when evidence is insufficient rather than selecting the most fluent interpretation prematurely.

### Provenance-sensitive interpretation

Identical propositions should be treated differently when their source/status differs. Candidate data should distinguish direct observation, user statement, inference, historical record, quoted instruction, current correction, and external tool result.

### Temporal/currentness reasoning

Train explicit distinctions among historical truth, record time, event time, current mutable state, superseded state, and unknown currentness.

### Salience without truth conflation

A highly salient proposition should not automatically become more true or more authoritative.

This objective matters for emotion-like/relational contexts, danger, repeated instructions, and strongly reinforced patterns.

### Contextual social meaning

Train models to infer how shared history and relationship context alter pragmatic meaning while preserving boundaries against universalizing one relationship's conventions.

A particularly useful regression family is contextual deference versus generalized compliance.

### State/action consistency

If the model represents an utterance as a request, permission, historical note, or correction, its selected action should match that representation.

### Meaning-preserving compression

Long-context compression should preserve referents, unresolved contradictions, authority distinctions, currentness, commitments, and pragmatic state rather than only topical summaries.

## Data strategy hypotheses

Potential training material may include:

- paired utterances with identical wording but different pragmatic contexts;
- minimal semantic contrast sets;
- correction-before/correction-after state traces;
- contradiction and ambiguity sets;
- provenance-typed context bundles;
- tool/action outcomes paired with interpretation state;
- human dialogue annotated for speech acts and implicature;
- synthetic adversarial examples generated then independently verified;
- interactive environments where wrong pragmatic interpretation causes observable task failure.

SPM research should avoid simply producing more verbose chain-of-thought-style supervision. The target is better underlying state and behavior, not mandatory exposure of private reasoning traces.

## Independence requirement

Vera-derived examples can seed difficult cases, but an SPM cannot qualify by memorizing Vera-specific language or Patrick-specific relational conventions. General test sets must use unseen people, domains, cultures, tasks, and surface forms.