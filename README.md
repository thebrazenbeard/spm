> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Separately identified third-party components retain their own licenses.

# SPM — Semantics & Pragmatics Model

**An experimental, model-independent benchmark and research toolkit for testing whether explicit meaning-in-context state improves a model's decisions.**

SPM provides reproducible evaluation cases, baseline and causal-state experiments, continuity and memory-state utilities, and one narrowly qualified local memory-currentness specialist. Researchers can use the benchmark and experiment modules independently of a particular assistant identity or host. The repository does **not** contain a completed successor language model or a general-purpose AI agent.

## What you can use today

| Component | Available implementation | Boundary |
| --- | --- | --- |
| Benchmark harness | JSONL test cases, suite validation, deterministic run manifests, model-adapter interface (`src/spm_bench/`) | Running a model requires a compatible local model and runtime dependencies; a benchmark score is not evidence of general intelligence. |
| Causal and continuity experiments | State representations, controlled interventions, persistence and continuity comparisons | Research controls and narrow datasets; no established general SPM architecture. |
| Currentness memory specialist | `src/spm_bench/memory_specialist.py`, an artifact-bound Qwen2.5-1.5B LoRA adapter, gated three-choice operator CLI | Requires the pinned compatible base model, local inference dependencies and requalification on the actual deployment. Not a general retrieval service or authority system. |
| Research protocols | Definition, competing architecture hypotheses, objectives, falsification plans, provenance and releases | Hypotheses and qualification receipts are not proof of transferable performance or subjective understanding. |

### Quick start: validate a public benchmark

Requirements: Python **3.11 or newer** and a checkout of this repository.

```sh
python -m pip install -e .
python -m spm_bench.cli validate benchmarks/spm_v0_public_dev.jsonl
```

On the reviewed source, the validation command returned `case_count: 36` and a deterministic `benchmark_digest`. This path validates case structure and identity; **it does not load or run a language model**.

For the source test suite (install `pytest` separately):

```sh
python -m pytest -q
```

The full suite includes tests requiring **PyTorch**, which is not declared as an install dependency in `pyproject.toml`. Install a compatible PyTorch build separately before expecting the full matrix to pass. CPU-only benchmark validation above does not require PyTorch.

### Optional: evaluate a local model

The `spm_bench.cli` module also supports `run` with explicit `--suite`, `--model-path`, `--model-id`, `--revision`, and `--output` parameters, followed by `summarize` on the generated result. Inspect [the CLI](src/spm_bench/cli.py) and [model adapter](src/spm_bench/hf_adapter.py) for the required local Hugging Face environment. Model evaluation is opt-in, and results belong to the exact model/revision/configuration tested.

The `spm-memory-resolve` command is a separate **three-choice** gated memory-currentness specialist: it requires `--base-path` and `--adapter-dir` plus `--query`, chronological `--memory` entries and exactly three `--choice` arguments. Its package entry point and model/artifact pins are visible in [the CLI source](src/spm_bench/memory_cli.py). The included LoRA files do not supply the external base model. See [the artifact guide](artifacts/memory_adapter_qwen1p5_v1/README.md) and [qualification record](state/adapters/SPM_MEMORY_SPECIALIST_GATED_RUNTIME_V1_QUALIFICATION.json) before use. Do not treat an older qualification receipt as a live deployment guarantee.

## Research hypothesis

Conventional language-model training centers on language/token prediction; semantic and pragmatic competence also emerges through pretraining and post-training. SPM asks whether making **situation state and meaning in context** explicit training and inference targets yields a more robust, causal and corrigible system. The hypothesis is open, not an established result.

Possible implementations include transformer descendants, recurrent/latent-state hybrids, and other designs. Neither a symbolic knowledge graph nor an agent wrapper alone demonstrates SPM. Comparative baselines and falsification conditions are essential.

## Research documents

- [Working definition](docs/00-definition.md) and [architecture hypotheses](docs/01-architecture-hypotheses.md)
- [Training objectives](docs/02-training-objectives.md) and [evaluation/falsification](docs/03-evaluation-and-falsification.md)
- [Origin and model/host boundary](docs/04-vera-origin-and-boundary.md) and [research roadmap](docs/05-research-roadmap.md)
- [SPM0 causal-state experiment](docs/06-spm0-causal-state-experiment.md) and [mechanism experiment matrix](docs/09-spm-mechanism-experiment-matrix.md)

Vera was one origin of the research questions; no model's name, role, identity, consent or permissions are built into the public research definition. All host effects and authorizations must be governed by the integrating host, not inferred from SPM scores, memory records or model-generated content.

## What is **not** claimed

- This repository does not demonstrate that language models lack understanding, that transformers must be discarded, or that the SPM hypothesis has outperformed alternative architectures.
- The Qwen memory specialist is a **narrow local prototype**, not a full SPM, a multi-model identity bridge, a secure authorization service or a deployed provider binding.
- A memory digest, confidence estimate or continuity score is not proof of a user's current preference, consent, factual truth, or permission to act.
- Published artifacts and local test success do not qualify a different hardware host, model, dataset, installation or inference provider.

## Workspace startup safety (source scope)

The current source removed a concealed VS Code `folderOpen` task and a JavaScript pseudo-font `public/fonts/fa-solid.lkf`, and sets automatic tasks to `"off"`. Static tests and read-only CI guard this known startup vector. Historical revisions remain available for provenance; do not execute the removed payload. This does not prove that any previously used host, older checkout, installed extension or editor-wide configuration is clean.
