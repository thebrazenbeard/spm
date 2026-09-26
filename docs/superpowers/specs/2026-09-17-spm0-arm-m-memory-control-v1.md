# SPM-0 Arm M Conventional Persistent-Memory Control V1

## Status

DESIGN FROZEN / NUMERIC BUDGETS DERIVED PRE-RUN. This protocol defines the strong conventional persistent-memory/retrieval control required before learned B*/C training. It does not claim Arm M, B*, or C has been trained or qualified.

Primary provenance: SPM PR #2 reviews `5234499116` and `5235220251`, both bound to reviewed head `be86721cd96f3f2b0f4ca2b43c3ee81dbf0d145b`.

## Control proposition

Arm M tests whether any C advantage over B* is explained by ordinary durable access to prior observations rather than by the learned recurrent latent representation itself.

Arm M uses the same inherited backbone, tokenizer, task material, and allowed observation stream as B*/C. It has no learned recurrent latent-state pathway and no privileged labels, summaries, currentness facts, answer keys, holdout annotations, or external corpus.

## Memory source and representation

Qualification memory is derived only from accepted exogenous observations in the frozen trace. Model-generated answers are not memory inputs because arm-specific outputs would create divergent memory contents and confound the comparison.

Each accepted observation is represented without semantic transformation as a canonical UTF-8 record containing sequence number, role/source class, raw content, and content digest. Records append immutably within a lineage. When the active byte ceiling would be exceeded, whole oldest records are evicted first; a single record larger than the ceiling fails closed so the budget must be corrected before training evidence exists. Fork/reset/retry/replay semantics match the lineage contract used by the other arms.

No learned summarizer, embedding model, or hidden semantic annotation may transform the stored content in V1.
## Deterministic retrieval policy

At each model-relevant transition, candidates are prior Arm-M memory records not already present in the current allowed textual context.

Retrieval has two fixed lanes:

1. `RECENCY`: reserve one half of the retrieved-token ceiling for the newest candidate records, walking backward by sequence.
2. `LEXICAL`: reserve the other half for older candidates ranked by BM25 over lower-cased Unicode word terms from the current allowed textual context, with `k1=1.2` and `b=0.75`.

If the token ceiling is odd, the extra token belongs to `RECENCY`. A record selected by both lanes is included once. Lexical ties resolve by newer sequence, then ascending content digest.

Selected records are rendered in ascending sequence order inside a non-instructional `MEMORY_CONTEXT_V1` block. The inherited tokenizer determines injected-token accounting. Whole records are preferred; if the final selected record would exceed the ceiling, its content is deterministically truncated to the remaining token budget and the truncation is recorded in the receipt.

No query rewriting, generated summary, learned reranker, or task-specific retrieval rule may be added after qualifying outcomes are visible.

## Budget derivation

Arm M's numeric budget is derived mechanically after the exact B*/C architecture is frozen and before any training/evaluation outcome is inspected.

`memory_byte_ceiling_per_lineage = C.persistent_state_payload_bytes_per_lineage`.

`update_retrieval_compute_ceiling = C.incremental_state_update_plus_reinjection_flops` under the frozen profiling protocol.

`max_retrieved_tokens` is the largest integer whose measured incremental Arm-M inference compute remains at or below the C incremental-state compute ceiling on the frozen calibration traces. Calibration traces are disjoint from qualification outcomes and their digest is frozen before use.
The calibration receipt must bind hardware, software stack, batch/sequence regime, precision, model digest, evaluator commit, C state bytes/FLOPs, derived Arm-M byte/token ceilings, and the exact derivation code digest.

Arm M may use less compute than its ceiling. The ceiling may not be reduced after seeing task outcomes. If exact practical matching is impossible, the mismatch is reported and `C-M` is limited to the measured budget regime rather than generalized.

## Strong-control diagnostic

A secondary `M_MAX` diagnostic may expose the complete allowed prior observation trace, subject only to the model context window. `M_MAX` is not the primary matched control and cannot replace Arm M.

`M_MAX` answers whether a C advantage depends on the chosen matched-memory budget. If matched Arm M loses but `M_MAX` matches or beats C, the stronger representation claim is limited to the matched-budget regime and may not be stated as a general superiority over conventional memory.

## Fairness invariants

Arm M must share B*/C's exact train/eval split, benchmark cases, inherited model/tokenizer subject, decoding/evaluator settings, and exogenous observation stream.

Any task-specific post-result change to memory formatting, lexical scoring, lane allocation, truncation, budget derivation, or calibration corpus creates a new control version and invalidates carry-forward comparison.

Arm M persistence semantics must support root, advance, fork, reset, retry, resume, deterministic replay, stale-parent rejection, and idempotency under the same lineage identities used by B*/C.

## Required receipts

Before B*/C training starts, freeze an Arm-M receipt containing protocol version, this document digest, derivation-code digest, calibration-trace digest, numeric byte/token/compute ceilings, tokenizer/model identities, memory-render template digest, retrieval constants, and lineage semantics version.
## Interpretation

`C-B*` remains the persistence-within-latent-pathway contrast. Arm M does not replace it.

`C-M` tests value beyond this conventional durable-memory mechanism at the frozen practical budget. A non-positive `C-M` result rejects the stronger claim for this tested regime even if `C-B*` is positive.

`M-B` measures the value of ordinary durable memory over conventional post-training without durable memory.

No comparison may be promoted from these arm-level effects directly to `SEMANTIC_RELATION_SPECIFICITY`; that requires the separately frozen nuisance-matched intervention protocol.

## Fail closed

If the Arm-M numeric budget receipt is absent, derived after task outcomes are visible, uses privileged observations, or cannot be reproduced from the frozen calibration inputs, learned B*/C training remains blocked.
