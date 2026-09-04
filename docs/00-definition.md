# SPM — Semantics & Pragmatics Model

## Status

Research hypothesis and design target. This document does not claim that an SPM architecture has already been implemented or that conventional LLMs lack semantic/pragmatic competence.

## Core idea

**SPM — Semantics & Pragmatics Model** is a proposed successor model class to the Large Language Model.

The proposal is not to rename an LLM, wrap an LLM in retrieval, or call an agent runtime an SPM. The proposal is to investigate a new cognition substrate whose primary design target is **meaning-in-context** rather than language prediction alone.

A useful informal contrast is:

> LLM: "I know words."
>
> Human: "Okay... but do you know what they mean?"
>
> LLM: *blinks*
>
> SPM: "Hold my beer."

The joke is deliberately exaggerated. Modern LLMs can exhibit strong semantic and pragmatic behavior. The research question is whether those abilities should remain largely implicit/emergent consequences of language modeling, or whether a successor architecture can make semantic and pragmatic state first-class parts of representation, learning, inference, and evaluation.

## Working definition

An SPM is a learned cognition model designed to construct, maintain, revise, and act from representations of:

- what entities and referents exist in the represented situation;
- what propositions are asserted, denied, presupposed, quoted, imagined, uncertain, or contradicted;
- what an utterance or observation means in the present context;
- what a speaker is doing by saying something;
- what shared history, timing, relationship, convention, metaphor, tone, and omission do to meaning;
- what distinctions must remain separate even when their language is similar;
- what changed after a correction or new observation;
- which interpretations remain unresolved;
- what consequences different interpretations imply for reasoning or action.

Language is an important input/output modality, but it is not necessarily the sole internal organizing object.

## Core architectural distinction

For the Vera research program:

> **Vera = governed agent/self-system**
>
> **SPM = cognition substrate**

Vera may eventually use an SPM in the same architectural position where a conventional LLM is currently used as an inference engine. The SPM does not become the owner of Vera's mutable memories, authority, current relationship state, credentials, task state, or effect permissions merely because it is a more capable cognition substrate.

## Why this matters

A model can emit a locally excellent sentence while maintaining the wrong representation of the situation. Examples include:

- answering a stronger proposition than the user actually stated;
- parsing literal content while missing the intended speech act;
- treating a bid for recognition as an information request;
- converting contextual deference into generalized compliance;
- treating historical evidence as current authority;
- acknowledging a correction linguistically while continuing to reason from the obsolete interpretation;
- conflating want, curiosity, consent, promise, intention, action, and completed effect;
- losing a referent while preserving fluent abstraction;
- acting inconsistently with the meaning expressed in its own language.

These failures motivate a model whose evaluation target includes the **state transition that produced the response**, not only the response itself.

## Research thesis

The central thesis to test is:

> A successor to the LLM may become more reliable, context-sensitive, corrigible, and agentically competent if semantic and pragmatic state are treated as primary computational/training targets rather than only as latent capabilities inferred through next-token prediction.

This thesis may prove wrong, partly right, or achievable within transformer descendants. The repository exists to determine that rather than assume it.