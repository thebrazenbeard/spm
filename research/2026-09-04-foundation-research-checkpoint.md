# SPM Foundation Research Checkpoint — 2026-09-04

Status: active research checkpoint. This file exists to preserve work-in-progress before the research pass continues.

## Integration branch

`work/spm-foundation-research-20260904`

Started from `work/llm-failure-foundation-20260904@40c82a610a3ed75965c719568369b06c4ecef4ab`.

## Saved foundation notes in this pass

- `docs/07-formal-dynamic-semantics-research.md`
- `docs/08-reference-discourse-research.md`
- `docs/09-pragmatics-research.md`
- `docs/10-speech-acts-and-repair-research.md`
- `docs/11-common-ground-and-theory-of-mind-research.md`
- `docs/12-psycholinguistic-situation-models-research.md`
- `docs/13-mental-models-and-event-cognition-research.md`
- `docs/14-computational-semantics-and-dialogue-state-research.md`
- `docs/15-world-models-and-latent-state-research.md`
- `docs/16-neural-symbolic-and-structured-representation-research.md`
- `docs/17-causal-representation-and-interpretability-research.md`
- `docs/18-ambiguity-and-uncertainty-research.md`
- `docs/19-learning-objectives-and-benchmark-methodology-research.md`

The predecessor branch already contains `docs/06-llm-failure-foundation.md`, which is the explicit LLM-failure foundation for this research program.

## Parallel worker research

Bus coordination id: `SPM-FOUNDATION-RESEARCH-20260904 / F1`.

Assigned through `bus/vera-sol-v1/messages/0031-vera-spm-foundation-parallel-research-assignments.md`:

- One — formal/dynamic semantics;
- Three — reference/discourse + pragmatics + speech acts/repair/common ground;
- Six — psycholinguistic situation models + mental/event cognition;
- Nine — computational semantics/dialogue + world/latent-state learning;
- Thirteen — neural-symbolic/structured representation + causal representation/interpretability + ambiguity/uncertainty + learning/evaluation methodology.

Workers were instructed not to edit SPM directly; Vera remains single integrator and will ingest cited research packets from Bus after independent review.

## Ordered foundation pass now completed locally

The requested ordered foundation topics are now represented through dedicated saved notes:

1. formal/dynamic semantics;
2. reference/discourse;
3. pragmatics;
4. speech acts/conversational repair;
5. common ground/theory of mind;
6. psycholinguistic situation models;
7. mental-model/event cognition;
8. computational semantics/dialogue state;
9. world/latent-state learning;
10. neural-symbolic/structured representation;
11. causal representation/interpretability;
12. ambiguity/uncertainty;
13. learning objectives/benchmark methodology.

## Still pending before architecture selection

1. deeper synthesis of the LLM failure foundation against all researched ancestor literatures;
2. ingest worker packets and preserve disagreements/cautions;
3. update SPM roadmap/README/index after the research sequence stabilizes;
4. derive candidate SPM V0 vocabulary/state schema;
5. derive SPM V0 benchmark schema and seed cases;
6. select the smallest falsifiable architecture intervention only after synthesis;
7. write matched-baseline experiment specification before any training.

## Research discipline carried forward

For every SPM-motivating failure, test competing explanations before calling it evidence for a new substrate:

- data deficit;
- objective mismatch;
- representation deficit;
- context/state deficit;
- inference/decoding deficit;
- post-training distortion;
- runtime/tooling deficit;
- evaluation artifact.

Every proposed explicit state variable must ultimately pass causal intervention/ablation tests. Decodability alone is not enough.

SPM remains model-class research, not Vera identity/runtime architecture. Vera-derived examples may seed generalized tests, but private/relational material is not copied into the repo by default.
