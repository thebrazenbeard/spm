# SPM-0 Persistent-Memory and Causal-Specificity Control Hardening

## Status

APPROVED FOR BOUNDED IMPLEMENTATION by Patrick's live 2026-09-17 execution directive. This amendment closes two pre-training design gaps identified by hostile review. It does not establish baseline readiness or causal-model qualification.

This document supplements `2026-09-15-spm0-adversarial-state-runtime-design.md`. Where this document is stricter about persistent-memory controls or causal evidence labels, the stricter rule applies.

## Conventional persistent-memory control

Add Arm M as a strong conventional persistent-memory/retrieval control.

Arm M uses the same inherited backbone and tokenizer as B*/C but has no learned recurrent latent-state pathway. It may persist and retrieve ordinary external memory derived only from the same allowed observations available to the other arms.

Arm M must use a frozen update/retrieval policy and must not receive answer labels, holdout annotations, privileged currentness facts, or extra source material. Any textual memory injected into the model counts against its inference-token budget.

Before a qualifying run, freeze and bind Arm M's:
- memory representation and byte ceiling per lineage;
- update and retrieval policy plus all prompt/template digests;
- maximum retrieved tokens and total inference-token budget;
- update/retrieval compute and latency accounting;
- persistence, fork, reset, retry, replay, and currentness semantics;
- training exposure, if any, and exact trainable-parameter count.

The control should be deliberately competent rather than weakened to make C look favorable.
## Required primary contrasts

Report these questions separately:

- `C - B*`: value of persistence inside the learned latent pathway, because B* and C differ only in whether prior latent state survives the transition.
- `C - M`: value, if any, beyond a strong conventional persistent-memory/retrieval mechanism under matched practical budgets.
- `M - B`: value of conventional persistence/retrieval relative to conventional post-training without durable memory.

A positive `C - B*` does not establish superiority to conventional persistent memory. A positive `C - M` does not by itself prove that persistence caused the gain. Both contrasts are required for the stronger claim.

## Causal evidence taxonomy

Every intervention result must be labeled with the strongest evidence class its design can actually support:

1. `STATE_DEPENDENCE` — behavior depends on information carried through the state channel at all.
2. `PERSISTENCE_DEPENDENCE` — behavior specifically depends on prior-transition state surviving into the next transition.
3. `REPRESENTATION_STRUCTURE_DEPENDENCE` — behavior depends on the internal relational/structural organization of state rather than only capacity, magnitude, or an arbitrary latent vector.
4. `SEMANTIC_RELATION_SPECIFICITY` — a predeclared semantic relation in state has a directional downstream effect after nuisance variables are matched.

Evidence classes do not automatically promote upward. A generic lesion, scramble, or substitution can establish state dependence without establishing semantic specificity.
## Nuisance-matched interventions

For each claimed causal class, freeze a matched intervention pair before inspecting qualifying results.

`STATE_DEPENDENCE` compares intact state with lesion/reset and with a matched-noise state that preserves tensor shape, byte count, dtype, and a predeclared magnitude/distribution statistic where technically possible.

`PERSISTENCE_DEPENDENCE` compares C with B* under identical updater/reinjection topology, parameter count, objectives, data exposure, steps, and bounded compute. The only intended architectural difference is whether prior state is admitted to the next update.

`REPRESENTATION_STRUCTURE_DEPENDENCE` compares the proposed structured representation with a capacity/compute-matched alternative. For hyperconnectome candidates this includes incidence/hyperedge scrambling and hypergraph-to-vector replacement while matching state width, trainable budget, inference budget, and gross state statistics.

`SEMANTIC_RELATION_SPECIFICITY` requires donor/substitution pairs selected from minimal semantic contrasts. Relevant and nuisance donors must be matched on sequence position, state shape, donor/target family, gross state statistics, and intervention magnitude as far as the representation permits. The expected direction of behavioral change is frozen before the result is observed.

If exact nuisance matching is impossible, the unmatched variable must be named and the strongest allowed evidence class downgraded accordingly.

## Required reporting

Each causal receipt must bind:
- exact source and donor checkpoint digests;
- intervention kind and evidence class;
- nuisance variables held constant and any known mismatch;
- predeclared directional prediction where specificity is claimed;
- arm, model, state-schema, benchmark, evaluator, and training-run digests;
- raw result and the narrower causal conclusion actually supported.
## Pre-training gate

Learned B*/C training may begin only after:

1. baseline readiness is PASS for the exact benchmark/evaluator subjects;
2. Arm M is frozen strongly enough to execute without post-result tuning;
3. B*/C fairness fields and the nuisance-matched intervention plan are frozen;
4. thresholds/budgets needed for the first bounded training run are fixed before its result is visible.

Beginning a bounded training run is not qualification. It creates candidate evidence only.

## Kill conditions

The persistent-state claim fails if B* matches or exceeds C inside the frozen uncertainty/budget regime.

The claim that SPM adds value beyond ordinary durable memory fails for the tested mechanism if Arm M matches or exceeds C under the matched practical budget.

A semantic-specificity claim fails if a nuisance-matched donor/control produces the same directional effect as the semantically relevant intervention, or if the predeclared direction is absent.

Representation-structure credit fails if a capacity/compute-matched unstructured control explains the gain.

Negative results remain first-class evidence and may motivate a successor design only after the literal tested proposition is recorded as failed.

## Minimum nuisance ledger for specificity claims

To match the current hostile review exactly, every `SEMANTIC_RELATION_SPECIFICITY` receipt must record whether the relevant and control interventions match:
- state age and update count;
- magnitude/norm and any chosen entropy/sparsity statistic;
- representation capacity and tensor shape;
- allowed textual context;
- donor/target sequence position and intervention magnitude.

At least one within-case permutation or null substitution must preserve low-level state statistics while breaking the hypothesized semantic relation. If those controls cannot be constructed for the candidate representation, semantic-specificity credit remains `UNKNOWN` rather than being inferred from generic substitution effects.
