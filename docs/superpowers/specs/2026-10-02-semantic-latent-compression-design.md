# SPM Semantic Latent Compression Design

Status: RESEARCH DESIGN / MODEL-LEVEL EXPERIMENT NOT YET RUN  
Date: 2026-10-02  
Source subject: `thebrazenbeard/spm@d9ea72798ac892ac75f858177b6ed0c5a6b4c37c`

## Purpose

Test whether SPM's primary semantic/pragmatic situation state can also serve as a denser internal computational representation than ordinary token history.

This is not "summarize the prompt." The hypothesis is that a successor model can maintain meaning-bearing state at several resolutions, perform substantial computation over compact state, and request/reconstruct higher-resolution source material only when exact language detail is needed.

## Hypothesis

A useful SPM representation should optimize semantic/pragmatic sufficiency per unit of active model state, not merely linguistic reconstruction score.

Candidate flow:

`language/bytes -> local encoder -> semantic/pragmatic state -> compact latent state -> model computation -> selective resolution upgrade -> language output`

Language remains an input/output surface. It is not required to remain the sole internal coordinate system.

## Required distinctions

- semantic compression is not exact source preservation;
- semantic adequacy is not reconstruction fidelity;
- lower token count is not proof of lower GPU memory;
- a learned latent is not epistemic authority;
- decoder reconstruction is not exact evidence unless the experiment establishes an exactness property it actually measures.

## Experimental ladder

1. deterministic/structured semantic state as a baseline;
2. trainable latent bottleneck over frozen or lightly adapted small open model;
3. query-conditioned selective detail retrieval;
4. only after those survive, investigate model-internal KV/state compression.

Use current SPM small-model infrastructure where feasible rather than introducing a giant training requirement.

## Evaluation families

- proposition preservation;
- referent/coreference preservation;
- correction/supersession;
- pragmatic intent;
- exact number/code/quote recovery;
- ambiguity preservation rather than premature collapse;
- out-of-distribution query after compression;
- long-delay retrieval;
- adversarial irrelevant-detail versus later-relevant-detail cases.

Report quality versus compression ratio and resource cost. No single aggregate score should hide exact-detail failures.

## Research lineage

- BLT: dynamic information-density patches — https://arxiv.org/abs/2412.09871
- ICAE: learned compact memory slots — https://arxiv.org/abs/2307.06945
- DMC: layer/head-specific learned KV compression — https://arxiv.org/abs/2403.09636
- Latent Context Compilation: portable latent buffer tokens — https://arxiv.org/abs/2602.21221

## First implementation slice

Add a benchmark/protocol surface before changing the qualified memory specialist. The existing qualified prototype remains frozen evidence. New compression experiments must have their own subject, training artifact, holdout, and qualification receipt.
