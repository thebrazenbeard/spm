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

## Still pending in the ordered foundation pass

1. causal representation and interpretability;
2. ambiguity and uncertainty representation;
3. learning objectives and benchmark methodology;
4. deeper synthesis of the LLM failure foundation against all researched ancestor literatures;
5. integrate worker packets and preserve disagreements/cautions;
6. update SPM roadmap/README/index after the research sequence stabilizes;
7. derive candidate SPM V0 state schema, benchmark schema, and smallest falsifiable architecture intervention only after the foundation synthesis.

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
