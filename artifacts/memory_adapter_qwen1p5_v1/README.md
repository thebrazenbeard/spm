# Qwen1.5B Memory Specialist V1

This directory contains the artifact set whose predecessor runtime was qualified as SPM memory-currentness specialist RC1.

The current hardening successor changes runtime artifact admission and identity binding. Its predecessor behavioral results remain historical evidence, but the hardened successor is not behaviorally/runtime qualified until a fresh qualification is bound to that successor identity.

It is not a replacement base model. The predecessor qualified runtime was:

- base: `Qwen/Qwen2.5-1.5B-Instruct`;
- base runtime: 4-bit NF4;
- adapter: rank-8 LoRA, 8.75 MB;
- activation: adapter enabled only when bounded persistent retrieval returns records;
- ordinary/no-memory work: base model with adapter disabled.

## Qualification

Frozen qualification receipt:
`state/adapters/SPM_MEMORY_SPECIALIST_GATED_RUNTIME_V1_QUALIFICATION.json`

Qualified result:

- persistent memory: 23/24;
- stale-memory errors: 1/24;
- full history: 24/24;
- ordinary public benchmark: 33/36;
- all frozen gates: PASS.

## Command-line use

Install the repository in editable mode:

```powershell
python -m pip install -e .
```

Then resolve a current state from prior observations:

```powershell
spm-memory-resolve `
  --base-path "PATH_TO_QWEN2.5_1.5B_SNAPSHOT" `
  --adapter-dir "artifacts/memory_adapter_qwen1p5_v1" `
  --memory "The deployment status was STALE." `
  --memory "Update: the deployment status is now CURRENT. STALE is obsolete." `
  --query "What is the current deployment status?" `
  --choice "STALE" `
  --choice "CURRENT" `
  --choice "UNKNOWN"
```

The command emits JSON with the selected answer, balanced semantic scores, adapter activation, selected record digests, retrieval receipt digest, token count, and truncation status.

## Boundary

This V1 specialist is qualified for bounded memory/currentness resolution under the frozen benchmark and routing contract. It is not evidence that a complete SPM architecture has been built.
