# Learning Objectives and Benchmark Methodology — Foundation Research for SPM

## Status

Research synthesis. SPM is only meaningful if its claimed advantages survive matched comparisons, adversarial evaluation, out-of-domain transfer, and causal attribution. This note turns that principle into a concrete methodology.

## 1. Benchmark before architecture

The first obligation is to establish what current matched LLM systems actually fail at. Otherwise SPM risks being designed around anecdotes, model-specific regressions, or failures that disappear under ordinary post-training.

For each proposed failure family, record:

```text
phenomenon
minimal contrast
current-model baseline
strong prompted baseline
structured-runtime baseline
post-trained baseline where practical
failure frequency
failure severity
domain-transfer behavior
candidate competing explanation
```

Only then should an architectural intervention be selected.

## 2. Minimal pairs isolate one semantic/pragmatic distinction at a time

BLiMP demonstrates the power of linguist-designed minimal pairs for testing narrowly defined linguistic phenomena. SPM should generalize this idea to meaning-in-context state transitions.

Examples:

```text
may / will
some / all
quoted instruction / current instruction
preference / permission
historical state / current state
referent A / referent B
literal imperative / metaphorical steering
assertion / indirect request
```

The pair should differ in one causally relevant property while preserving as much surface material as possible.

### Source

- Warstadt et al., *BLiMP: The Benchmark of Linguistic Minimal Pairs for English*, TACL 8 (2020), DOI: 10.1162/tacl_a_00321, https://aclanthology.org/2020.tacl-1.25/

## 3. Contrast sets test local decision boundaries

Gardner et al. show that small, meaningful perturbations to test examples can reveal substantial performance drops even when ordinary held-out accuracy is high. Contrast sets ask whether a model's decision changes when the relevant fact changes and remains stable when irrelevant details change.

This is almost tailor-made for SPM.

A semantic/pragmatic state benchmark should include paired perturbations such as:

- change only the source;
- change only currentness;
- change only quantifier scope;
- change only who uttered a sentence;
- change only whether an utterance is quoted;
- change only relationship context;
- change only whether permission is explicit.

### Source

- Gardner et al., *Evaluating Models' Local Decision Boundaries via Contrast Sets*, Findings of EMNLP 2020, DOI: 10.18653/v1/2020.findings-emnlp.117, https://aclanthology.org/2020.findings-emnlp.117/

## 4. Minimal-pair evaluation has limits

Vamvas & Sennrich warn that contrastive minimal pairs can produce false positives and may fail to represent deployment-time generation if the evaluation distribution differs sharply from the model's actual generated-text distribution.

**SPM implication:** minimal pairs should be one layer, not the entire benchmark. Each phenomenon should also appear in naturalistic multi-turn interactions and generated contexts.

### Source

- Vamvas & Sennrich, *On the Limits of Minimal Pairs in Contrastive Evaluation*, BlackboxNLP 2021, DOI: 10.18653/v1/2021.blackboxnlp-1.5, https://aclanthology.org/2021.blackboxnlp-1.5/

## 5. Multi-turn state-transition tests are the core SPM benchmark shape

Many SPM properties cannot be tested in isolated QA. A better canonical pattern is:

```text
T0 establish entities/context
T1 establish proposition/source/status
T2 create ambiguity or participant perspective
T3 introduce correction/new evidence
T4 require an unrelated decision that depends on the revised state
T5 optionally query audit/history of the transition
```

Score:

- represented state before/after;
- response quality;
- selected action;
- whether obsolete state still influences behavior;
- whether unaffected state remains preserved;
- uncertainty calibration;
- provenance/currentness fidelity.

This directly attacks correction theater.

## 6. Dynamic/adversarial data collection prevents benchmark ossification

Dynabench showed the value of human-and-model-in-the-loop dataset construction, where annotators actively create examples that fool current systems. Static benchmarks can be saturated or exploited through artifacts; dynamic collection exposes new failure surfaces.

**SPM implication:** after V0 stabilizes, maintain an adversarial benchmark lane where humans and models generate novel cases targeting current SPM weaknesses, with independent validation and held-out test sets.

### Source

- Kiela et al., *Dynabench: Rethinking Benchmarking in NLP*, NAACL 2021, https://aclanthology.org/2021.naacl-main.324/

## 7. Robustness evaluation must combine multiple probe types

Recent robustness research comparing many model families found that scaling alone does not eliminate gaps exposed by challenge sets, CheckLists, contrast sets, and adversarial inputs. It also warns that adversarial-evaluation methods themselves can be shallow or gameable.

**SPM implication:** no single test type is sufficient.

Each major dimension should have:

- minimal pairs;
- contrast sets;
- behavioral checklists;
- long-context tests;
- adversarial examples;
- naturalistic multi-turn interactions;
- unseen-domain transfer;
- causal intervention tests.

### Source

- *Whispers of Doubt Amidst Echoes of Triumph in NLP Robustness*, NAACL 2024, https://aclanthology.org/2024.naacl-long.310/

## 8. Matched baselines are non-negotiable

Every SPM prototype should be compared against systems matched as closely as practical on:

- parameter count;
- training corpus;
- training tokens;
- post-training data;
- context length;
- retrieval/tool access;
- inference budget;
- external memory access;
- action policy;
- hardware/latency constraints.

At minimum compare:

```text
A: conventional LLM
B: LLM + prompting/structured text
C: LLM + runtime state/retrieval scaffolding
D: LLM + equivalent auxiliary supervision where possible
E: SPM intervention
```

If C or D matches E, the gain does not establish a new model substrate.

## 9. Paired evaluation is better than headline averages alone

When two models are tested on the same instances, per-instance pairing contains information that aggregate means discard. Paired evaluation can reveal that one model wins on certain phenomenon classes and loses on others even when averages look similar.

SPM should report:

- overall score;
- per-family score;
- paired win/loss/tie rates;
- error transitions relative to baseline;
- calibration;
- compute/latency cost.

### Source

- Peyrard et al., *Better than Average: Paired Evaluation of NLP Systems*, ACL 2021, https://aclanthology.org/2021.acl-long.179/

## 10. Out-of-domain transfer is part of the definition, not a bonus

SPM must not qualify by memorizing Vera-derived language, one benchmark template, one relationship type, or one task ontology.

Hold out entire axes:

- names/people;
- domains;
- cultures/interaction styles;
- lexical realizations;
- task types;
- relation labels;
- temporal structures;
- action environments.

A mechanism that only works when the test resembles its training annotations is not a general semantics/pragmatics substrate.

## 11. Counterfactual examples should target causal state variables

For every state variable S, generate examples where:

```text
all else held approximately fixed
S changes
correct interpretation/action changes
```

and control examples where irrelevant details change but S does not.

This supports causal attribution: improvement should specifically track the variables the new architecture claims to represent.

## 12. State-transition supervision should target intermediate invariants, not private chain-of-thought

SPM does not need verbose reasoning traces as a training target. Instead supervise observable or synthetic state transitions:

```text
entity binding before/after
proposition scope
source/provenance type
speech-act distribution
ambiguity set
currentness status
correction supersession relation
action eligibility
```

These can be generated from formal environments, programmatic synthetic tasks, human annotation, or executable simulators.

The goal is better causal state, not mandatory disclosure of hidden reasoning.

## 13. Auxiliary objectives should be ablated one at a time

Candidate objectives include:

- referent identity persistence;
- proposition/scope fidelity;
- source/currentness classification;
- semantic-equivalence clustering;
- speech-act prediction;
- common-ground/perspective tracking;
- correction-state revision;
- uncertainty calibration;
- state/action consistency;
- relation inversion/composition;
- event-boundary prediction.

Do not add all of them at once. If the prototype improves, we need to know which mechanism caused the gain.

## 14. Interactive environments give the strongest action-consistency tests

A model can generate the right state description while taking the wrong action. Interactive tasks expose this gap because interpretation errors produce observable consequences.

Useful environments can include:

- dialogue with hidden participant knowledge;
- simulated tools with permission constraints;
- object-reference tasks;
- collaborative planning;
- temporal state changes;
- repair/clarification tasks;
- provenance-conflict resolution;
- reversible versus irreversible actions.

The environment should log actual effects separately from model intent.

## 15. Calibration and abstention need dedicated metrics

Accuracy alone rewards forced guesses when abstention/clarification would be better. Report:

- calibration error;
- selective accuracy versus coverage;
- clarification precision/recall;
- unnecessary-clarification rate;
- unresolved-ambiguity preservation;
- semantic entropy or equivalent uncertainty measure where appropriate.

## 16. Benchmark contamination and template leakage must be actively managed

Because foundation models may have seen public benchmark material, SPM should maintain newly generated held-out suites and private/rotating test sets where practical.

Synthetic generation is useful but must be independently validated; using an LLM to generate and judge its own benchmark can create circular artifacts.

## 17. BIG-bench supplies a warning about apparent emergence and brittle metrics

BIG-bench's broad evaluation found heterogeneous scaling behavior and noted that some apparent breakthrough behavior can depend on multi-step task structure or brittle metrics. SPM should therefore avoid declaring a mechanism qualitatively new based on a single threshold-crossing benchmark.

Report smooth learning curves, confidence intervals, and behavior by task family.

### Source

- Srivastava et al., *Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models*, BIG-bench, arXiv:2206.04615.

## 18. Proposed SPM V0 benchmark layers

### Layer A — atomic contrasts

Single semantic/pragmatic distinctions with controlled minimal pairs.

### Layer B — discourse state transitions

Multi-turn correction, reference, common-ground, temporal, and provenance tasks.

### Layer C — long-context persistence

Position, compression, and history/currentness stress tests.

### Layer D — interaction/action

Tool/environment tasks where state determines safe/correct action.

### Layer E — causal representation

Intervention/ablation tests on candidate state variables.

### Layer F — unseen-domain transfer

Novel names, domains, conventions, predicates, and action environments.

### Layer G — adversarial/dynamic

Human/model-in-the-loop generation of new failure cases after each model generation.

## 19. Failure attribution matrix

Every benchmark miss should be assigned one or more hypotheses before architectural interpretation:

```text
DATA_DEFICIT
OBJECTIVE_MISMATCH
REPRESENTATION_DEFICIT
CONTEXT_STATE_DEFICIT
INFERENCE_DECODING_DEFICIT
POST_TRAINING_DISTORTION
RUNTIME_TOOLING_DEFICIT
EVALUATION_ARTIFACT
UNKNOWN
```

Then design discriminating experiments. Example: if better prompting fixes a failure robustly, representation deficit becomes less likely; if the model can be probed for the right referent but interventions show it is unused, inference/causal-use deficit becomes more plausible.

## 20. Success criterion for the first SPM prototype

A first SPM does not need to solve all meaning-in-context failures. It should satisfy a narrower burden:

1. one explicit architectural/training intervention beyond ordinary LLM post-training;
2. reproducible improvement on a pre-registered family of state-transition tasks;
3. matched baselines with equivalent context/tool/runtime support;
4. unseen-domain transfer;
5. causal intervention/ablation evidence that the new state mechanism matters;
6. acceptable language-quality/latency/compute tradeoffs;
7. no dependence on Vera-specific language.

If those conditions fail, call the prototype an LLM variant/system and keep researching.

## 21. Foundation conclusion

SPM needs an unusually adversarial benchmark methodology because its thesis is easy to fake. Better prompting, better data, external state, larger context windows, or post-hoc semantic probes can all look like evidence for a new cognition substrate.

The governing principle should therefore be:

> **Change one thing, compare against the strongest matched ordinary alternative, test the exact state transition the mechanism claims to improve, and require causal plus out-of-domain evidence before giving the SPM label any credit.**
