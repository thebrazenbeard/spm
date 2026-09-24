# SPM Initial Literature Map

## Status

Research map, not architecture authority. The purpose is to identify existing evidence, reusable benchmarks, and failure modes that constrain SPM-0 design.

## 1. Tokenization is not a law of nature

### Byte Latent Transformer: Patches Scale Better Than Tokens

Pagnoni et al., ACL 2025, DOI `10.18653/v1/2025.acl-long.453`.

Relevant evidence:

- raw-byte language modeling can scale competitively with tokenized models;
- BLT groups bytes into dynamically sized patches rather than using a fixed vocabulary;
- the published study includes models up to 8B parameters and 4T training bytes;
- the paper reports improved scaling at fixed inference cost and robustness/long-tail benefits.

SPM implication: a later SPM need not inherit fixed BPE tokenization. This does **not** establish that tokenization is the cause of SPM-targeted semantic/pragmatic failures, so SPM-0 keeps the inherited tokenizer fixed.

## 2. Pragmatic competence is measurable and uneven

### PUB: A Pragmatics Understanding Benchmark for Assessing LLMs' Pragmatics Capabilities

Sravanthi et al., Findings of ACL 2024, DOI `10.18653/v1/2024.findings-acl.719`.

PUB contains fourteen tasks across implicature, presupposition, reference, and deixis, with roughly 28k examples. The reported results show substantial variation across pragmatic phenomena and a remaining human-model gap.

SPM implication: SPM V0 should reuse or adapt established pragmatics task families rather than inventing all evaluation categories from Vera-specific experience.

### Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges

Ma et al., ACL 2025, DOI `10.18653/v1/2025.acl-long.425`.

This survey organizes contemporary pragmatics datasets/evaluations and highlights continuing difficulty around nuanced meaning in context.

SPM implication: use the survey as a benchmark-discovery index and explicitly identify gaps between existing mostly response-level pragmatics tests and SPM's desired state-transition evaluation.

### Understand the Implication: Learning to Think for Pragmatic Understanding

Sravanthi et al., Findings of ACL 2025, DOI `10.18653/v1/2025.findings-acl.1218`.

The paper reports that supervised/preference training with explicit reasoning for correct and incorrect pragmatic interpretations improves pragmatic performance and transfers to some unseen pragmatic tasks.

SPM implication: ordinary post-training is a strong baseline. If matched SFT/preference training solves the target failure families, SPM-0 has not earned an architectural claim. The paper strengthens the need for Arm B in the SPM-0 experiment.

## 3. Belief/correction revision remains a real failure family

### Belief Revision: The Adaptability of Large Language Models Reasoning

Wilie et al., EMNLP 2024, DOI `10.18653/v1/2024.emnlp-main.586`.

Belief-R evaluates model reasoning as new evidence changes prior conclusions. The authors report that evaluated LMs generally struggle with appropriate belief revision and identify a trade-off where models that update more readily may also update when they should not.

SPM implication: correction/state revision must test both directions:

- update when new evidence requires it;
- preserve prior state when new evidence does not warrant revision.

A model that merely becomes more compliant with the latest statement is not demonstrating robust semantic-state revision.

### CriticBench: Benchmarking LLMs for Critique-Correct Reasoning

Lin et al., Findings of ACL 2024, DOI `10.18653/v1/2024.findings-acl.91`.

CriticBench separates generation, critique, and correction behavior across several reasoning domains.

SPM implication: correction detection and correction application should be measured separately. SPM should not receive credit for producing correction language if its downstream state remains obsolete.

## 4. Architectural retrofit of pretrained models is plausible but not free

### Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets

Shapiro, 2026 preprint, arXiv `2608.11233`.

The experiment retrofits recurrent depth into Qwen2.5-0.5B-Instruct, including a frozen-base adapter-scale intervention and a larger full-block intervention. The reported results provide preliminary evidence that iterative latent computation can be inserted into a pretrained transformer while retaining useful base capability.

SPM implication: the central engineering strategy of inheriting an LLM and surgically adding state machinery is technically credible enough to test. Because this is a recent preprint, it is evidence for feasibility, not proof of a mature recipe.

Research requirement: inspect the full paper/code before freezing the SPM-0 recurrent-state implementation, especially retention/interference results and initialization strategy.

## 5. Agent/tool evaluation supports stateful trajectory testing

### ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities

Lu et al., Findings of NAACL 2025.

ToolSandbox evaluates stateful tool execution with implicit dependencies, on-policy conversation, and intermediate/final milestones.

### ACEBench: A Comprehensive Evaluation of LLM Tool Usage

Chen et al., Findings of EMNLP 2025, DOI `10.18653/v1/2025.findings-emnlp.697`.

ACEBench includes normal, special/ambiguous, and agent/multi-turn tool-use settings.

### TRAJECT-Bench: A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use

He et al., 2025 preprint, arXiv `2510.04550`.

TRAJECT-Bench emphasizes evaluation of the actual tool trajectory rather than only final answers.

SPM implication: later Open WebUI/Computer qualification should score interpretation-to-action trajectories, not just whether the final prose sounds sensible.

## 6. Unclear instructions expose pragmatic/action risk

### Learning to Ask: When LLM Agents Meet Unclear Instruction

Wang et al., EMNLP 2025.

The work studies tool-use under imperfect instructions and reports a tendency for LLM agents to invent missing arguments rather than resolve uncertainty safely.

SPM implication: ambiguity preservation must connect to action selection. An unresolved pragmatic state should sometimes cause a question, wait, or bounded non-effect rather than fluent completion.

## 7. Current evidence does not yet prove SPM

The literature supports several enabling propositions:

1. conventional tokenization is optional;
2. pragmatics is measurably uneven in current models;
3. ordinary post-training can improve pragmatics and is therefore a required strong control;
4. belief/correction revision remains difficult;
5. structural retrofit of a pretrained transformer is plausible;
6. stateful action/trajectory benchmarks can evaluate downstream consequences.

It does **not** currently establish the SPM thesis that making semantic/pragmatic state an explicit causal organizing mechanism will outperform matched strong LLM baselines.

That gap is the purpose of SPM-0.

## 8. Required next survey areas

Before freezing the implementation architecture, research should cover at least:

- recurrent/latent-state transformers and universal/recurrent depth;
- state-space language models and persistent hidden state;
- learned memory architectures;
- world/situation-model representations;
- neural semantic parsing and structured latent-variable models;
- discourse representation and dialogue-state tracking;
- referent/entity tracking in long context;
- causal representation probing versus correlational probes;
- intervention/activation-patching methods suitable for causal state tests;
- catastrophic forgetting/interference during architectural retrofit;
- distillation into modified input/state architectures;
- byte/character/patch input architectures;
- pragmatic inference datasets across languages and cultures;
- ambiguity/calibration evaluation;
- tool/action grounding and stateful agent evaluation;
- model-serving constraints for custom architectures under OpenAI-compatible APIs, Open WebUI, and local runtimes.

## 9. Selection rule for new literature

A paper belongs in the active SPM evidence base only when it changes at least one of:

- feasibility of an architectural mechanism;
- benchmark/evaluation design;
- known baseline strength;
- training objective design;
- causal-attribution method;
- retention/negative-transfer risk;
- deployment feasibility.

Interesting papers that do not affect an experiment decision may be retained as background but should not inflate the apparent evidence for SPM.