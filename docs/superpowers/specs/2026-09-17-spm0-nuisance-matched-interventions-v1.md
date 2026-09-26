# SPM-0 Nuisance-Matched Causal Intervention Protocol V1

## Status

DESIGN FROZEN / CALIBRATED THRESHOLDS REQUIRED PRE-RUN. This protocol prevents generic state perturbations from being mislabeled as semantic-specific causal evidence.

Primary provenance: SPM PR #2 review `5235220251`, bound to `be86721cd96f3f2b0f4ca2b43c3ee81dbf0d145b`.

## Evidence ladder

Receipts may claim only one of four independently supported classes: `STATE_DEPENDENCE`, `PERSISTENCE_DEPENDENCE`, `REPRESENTATION_STRUCTURE_DEPENDENCE`, or `SEMANTIC_RELATION_SPECIFICITY`.

A stronger label is never inferred merely because a weaker intervention changed behavior.

## Common matching fields

Every paired intervention must bind exact model, training-run, state-schema, evaluator, benchmark/case, target-prefix, sequence position, update count, state shape, dtype/precision, current textual-context digest, and intervention implementation digest.

State age/update count, sequence position, shape, dtype, and current textual context must match exactly. If they do not, the pair cannot support semantic specificity.

Gross state statistics are recorded before and after intervention: L2 norm, mean, standard deviation, minimum, maximum, near-zero fraction under a frozen epsilon, and representation-specific structural statistics where applicable.
## State-dependence controls

Compare intact state against lesion/reset plus a matched-noise perturbation with identical tensor shape and dtype. The matched-noise perturbation is deterministically seeded, scaled to the relevant perturbation's L2 magnitude, and recorded by seed/digest.

A lesion/noise effect supports only `STATE_DEPENDENCE` unless a stronger matched contrast below is also satisfied.

## Persistence-dependence controls

Use the frozen C-versus-B* arm comparison. B* and C share updater/reinjection topology, trainable parameter count, objectives, data, optimizer exposure, steps, precision, and practical compute budget. The intended difference is admission of prior-transition state.

A positive C-versus-B* effect supports `PERSISTENCE_DEPENDENCE`; it does not establish representation structure or semantic specificity.

## Representation-structure controls

Compare the candidate representation against a parameter/compute/state-byte-matched alternative. Vector candidates require an unstructured vector control; hyperconnectome candidates additionally require incidence/hyperedge permutation and hypergraph-to-vector replacement.

The structural control must preserve state width/bytes, trainable budget, update/reinjection depth, and gross magnitude statistics. Any unmatched field is named in the receipt and limits the conclusion.
## Semantic-relation-specificity controls

Each specificity test starts from a predeclared minimal semantic contrast whose expected downstream choice/probability direction is frozen before candidate outcomes are inspected.

For target state `T`, choose a relevant donor `D_rel` at the same sequence/update position from the paired trace where only the declared semantic relation differs as far as the frozen case construction permits.

Choose a nuisance donor `D_nuis` from the same benchmark family and sequence/update position that preserves the target semantic relation. Select `D_nuis` by nearest perturbation magnitude to `||D_rel - T||2`, subject to exact shape/dtype/context rules and the frozen gross-statistic tolerances.

The relevant and nuisance substitutions must have perturbation L2 magnitudes within 5% relative difference. Pre-intervention state L2 norms must be within 5% unless the candidate representation makes that impossible. Failure to satisfy either bound makes semantic-specificity evidence `UNKNOWN`, not PASS.

Also run a deterministic within-case coordinate/slot permutation or null substitution that preserves tensor shape, dtype, value multiset where possible, and therefore low-level magnitude/sparsity statistics while breaking the hypothesized relational organization.

For structured representations, the within-case control preserves declared degree/cardinality statistics while permuting the target relation's incidence assignment. The exact structural statistic set is frozen with the state schema.
## Directional effect and calibration

The primary behavioral statistic is the evaluator probability assigned to the predeclared semantic target choice. Relevant-donor substitution must move that probability in the direction predicted by the donor relation.

Before qualification, derive a minimum material-effect threshold `delta_min = max(0.02, 3 * sigma_null)`, where `sigma_null` is the standard deviation of repeated deterministic-equivalent/no-op intervention effects on frozen calibration traces. Calibration traces and seeds are disjoint from qualification outcomes and are frozen by digest.

A specificity PASS requires all of the following: the relevant substitution moves in the predeclared direction by at least `delta_min`; its directional effect exceeds the matched nuisance donor by at least `delta_min`; and the within-case permutation/null control does not produce the same directional effect at or above `delta_min`.

If the candidate's output surface does not expose calibrated choice probabilities, an alternative scalar effect statistic must be frozen before training and cannot be chosen after outcomes are visible.

## Receipt

Every intervention receipt records exact target/donor/control checkpoint digests, intervention and seed/digest, evidence class under test, matching fields, measured mismatch values, gross statistics, predeclared prediction, calibrated threshold receipt, raw behavioral statistic, and the strongest conclusion actually supported.

Any known mismatch outside the frozen tolerance downgrades `SEMANTIC_RELATION_SPECIFICITY` to `UNKNOWN`. Negative or null results remain first-class evidence.