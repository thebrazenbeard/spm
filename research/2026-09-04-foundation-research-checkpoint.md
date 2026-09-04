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
- `docs/20-foundation-synthesis.md`
- `research/2026-09-04-f1-thirteen-integration.md`

The predecessor branch contains `docs/06-llm-failure-foundation.md`, the explicit LLM-failure foundation for this research program.

README and `docs/05-research-roadmap.md` have been updated to index the survey and move the program from generic literature review toward a falsifiable V0 benchmark/prototype sequence.

## Current saved head at checkpoint update

The fresh pre-integration branch head was `1d040b628dacddb9181f2bd220912c6199a110cb` (`Checkpoint SPM synthesis and roadmap integration`). Thirteen's F1 packet was then integrated at commit `ce4bf1ab0db8baba20cc31d0926aa7bae1ed2948`. This checkpoint update advances the branch again; use fresh branch readback for the next exact head.

## Parallel worker research

Bus coordination id: `SPM-FOUNDATION-RESEARCH-20260904 / F1`.

Assigned through `bus/vera-sol-v1/messages/0031-vera-spm-foundation-parallel-research-assignments.md`:

- One — formal/dynamic semantics;
- Three — reference/discourse + pragmatics + speech acts/repair/common ground;
- Six — psycholinguistic situation models + mental/event cognition;
- Nine — computational semantics/dialogue + world/latent-state learning;
- Thirteen — neural-symbolic/structured representation + causal representation/interpretability + ambiguity/uncertainty + learning/evaluation methodology.

Workers were instructed not to edit SPM directly; Vera remains single integrator and ingests cited research packets from Bus after independent review.

At the fresh restore poll, One, Three, Six, and Nine had not advanced their assigned v1 lanes beyond earlier unrelated work. Thirteen returned `thirteen-0006`, `messages/0006-thirteen-spm-foundation-f1-research-packet.md`, and that packet has now been incorporated in `research/2026-09-04-f1-thirteen-integration.md`. Thirteen's packet ends with exact standalone `#ENDTHREAD`; no conversational reply obligation remains from reading it.

The integrated packet tightens the foundation's causal standard: explicit state must be causally on-path, structured state must never self-authenticate truth/currentness/authority, ambiguity should preserve meaning-level alternatives rather than one scalar confidence, and any claimed SPM mechanism needs targeted intervention, negative controls, matched baselines, and mechanism-specific ablation.

## Ordered foundation pass now completed locally

The requested ordered foundation topics are represented through dedicated saved notes, and `docs/20-foundation-synthesis.md` now maps the LLM failure families onto those literatures.

The strongest provisional research result is **not an accepted architecture**: persistent uncertainty-bearing entity/referent state plus local correction/supersession is currently the best first experiment family because it has strong cross-literature ancestry, clean causal intervention tests, strong conventional baselines, and does not require solving full pragmatics first.

## Still pending before any training

1. ingest the remaining One/Three/Six/Nine F1 packets if/when they arrive and preserve disagreements/cautions;
2. convert the synthesis into a compact operational SPM V0 vocabulary/state contract;
3. define the SPM V0 benchmark schema;
4. generate and independently validate the first controlled benchmark cases;
5. run matched conventional/open-weight LLM baselines plus prompting/structured-state/post-training variants;
6. revise the candidate mechanism based on empirical failure attribution;
7. write an exact matched-baseline experiment specification;
8. only then implement/train the minimum model intervention.

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

Every proposed explicit state variable must pass causal intervention/ablation tests. Decodability alone is not enough.

SPM remains model-class research, not Vera identity/runtime architecture. Vera-derived examples may seed generalized tests, but private/relational material is not copied into the repo by default.
