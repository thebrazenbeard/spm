# LLM Failure Foundation for SPM Research

## Status

Research synthesis and hypothesis map. This document identifies recurring and structurally interesting failure modes of contemporary large language models that may motivate SPM research. It does **not** claim that every LLM exhibits every failure, that the failures are immutable, or that a new model class is required to solve them.

The purpose is stricter:

> Before proposing an SPM architecture, identify what current LLMs actually fail at, which failures plausibly arise from objective/representation/inference choices, which are merely training/runtime deficits, and what evidence would distinguish those explanations.

SPM should exist only if an architectural or training intervention aimed at meaning-in-context produces reproducible advantages over strong matched LLM baselines.

---

## 1. Objective mismatch: next-token prediction is not truth, meaning, or action correctness

Autoregressive language models are primarily trained to predict the next token given preceding tokens. This objective is extraordinarily powerful, but it does not directly optimize any of the following:

- truth;
- referential correctness;
- proposition identity;
- preservation of ambiguity;
- pragmatic intent recognition;
- currentness;
- state revision after correction;
- causal world-state accuracy;
- action/effect correctness.

A model can therefore become excellent at producing text that is locally probable without maintaining the best representation of the situation.

TruthfulQA demonstrated that language models can reproduce common human misconceptions because those falsehoods are present in the distribution being modeled. Lin, Hilton, and Evans explicitly argued that scaling imitation alone is not sufficient for truthfulness when the training distribution itself rewards false answers.

Recent theoretical work also argues that next-word prediction creates statistical pressure toward hallucination even with idealized error-free data when facts are sparse or one-off, while repeated regularities such as grammar are much easier to learn reliably.

### SPM significance

SPM should test whether semantic/pragmatic state objectives can make the model optimize state fidelity in addition to linguistic likelihood.

The target is not "never hallucinate." The sharper question is:

> Does explicitly training situation-state correctness reduce errors that arise because plausible continuation and correct represented state diverge?

---

## 2. Plausibility without grounded truth: hallucination and confabulation

LLMs can generate fluent, coherent, highly specific false content. The research literature uses "hallucination" inconsistently, so SPM should avoid treating it as one undifferentiated phenomenon.

Useful distinctions include:

- contradiction with supplied context;
- unsupported invention beyond supplied evidence;
- factual conflict with external world knowledge;
- fabricated entities or citations;
- internally inconsistent continuation;
- confident completion when the model should preserve uncertainty or abstain.

The important structural issue for SPM is that linguistic fluency can remain high while represented reality is wrong or nonexistent.

### SPM significance

SPM should separately represent:

- asserted proposition;
- evidence/source status;
- confidence/uncertainty;
- unresolved alternatives;
- whether an answer is supported by current state versus merely generatable.

This should be causal state used for response/action selection, not a post-hoc disclaimer head.

---

## 3. Directional knowledge and weak relational abstraction: the Reversal Curse

Berglund et al. showed that autoregressive LLMs trained on a relation in one textual direction can fail to generalize to the inverse query. A model trained on `A is B` may not reliably answer `who/what is B? -> A`, despite the relation being trivial for a system that had encoded an abstract bidirectional relation.

The Reversal Curse is especially interesting for SPM because it suggests that at least some learned "knowledge" behaves like directional sequence association rather than a stable relation object.

This does not prove that transformers cannot represent relations. It does show that a token-prediction system may fail to form or retrieve the relation in the abstraction a downstream task expects.

### SPM significance

An SPM prototype should test whether learned entity/relation state supports:

- relation inversion;
- paraphrase-invariant retrieval;
- role reversal;
- direction-independent entity binding;
- relation composition.

A matched LLM baseline with equivalent data augmentation must be included so improvements are not merely attributed to duplicated training forms.

---

## 4. Fragile use of long context: context capacity is not context utilization

Large context windows do not guarantee robust use of all information in the window.

Liu et al., in *Lost in the Middle*, found strong positional effects: models often perform best when relevant information appears near the beginning or end and worse when it appears in the middle, including models marketed for long-context use.

This matters because many current agent systems flatten memory, instructions, evidence, tool outputs, and conversation history into one long token stream and then assume that presence equals usable state.

### SPM significance

SPM should test typed persistent state against prompt-flattened baselines.

The hypothesis is not merely "remember more." It is:

> Information whose semantic role matters should remain addressable by role and identity instead of competing solely for attention inside a token sequence.

Candidate state types include current observation, correction, historical evidence, unresolved contradiction, entity binding, permission, preference, current task state, and effect receipt.

---

## 5. Prompt serialization collapses heterogeneous evidence into one medium

Most LLM systems present very different things as text tokens:

- user instruction;
- quoted instruction;
- historical record;
- retrieved memory;
- tool output;
- inference;
- system policy;
- current correction;
- hypothetical scenario;
- external source assertion.

The model may infer those distinctions from formatting and wording, but they are usually not native typed objects in the model's computational interface.

This creates recurrent failure families:

- quote becomes instruction;
- history becomes current state;
- preference becomes permission;
- retrieved text becomes authority;
- model-generated summary becomes evidence for its own premise;
- tool invocation becomes reported external effect.

### SPM significance

SPM should test dedicated type channels or learned type embeddings whose semantics survive paraphrase and long context.

The critical falsification condition is whether ordinary LLM prompting with explicit tags performs equally well under matched conditions. If it does, native typed channels are not yet justified.

---

## 6. Referential drift and entity collapse

LLMs often preserve topic while losing exact referential identity. Fluency makes this failure dangerous because an answer can remain coherent after the model has silently changed who or what a term refers to.

Common forms include:

- pronoun resolution failure;
- merging two similar entities;
- replacing a particular person/object with a generic category;
- losing the referent of a shorthand phrase after long context;
- preserving a label while changing the underlying entity;
- treating lexical similarity as identity.

### SPM significance

Entity/referent identity should be a first-class research target rather than evaluated only through generated prose.

A useful benchmark should force the model to maintain multiple similar entities across:

- pronouns;
- aliases;
- role changes;
- quoted speakers;
- temporal changes;
- corrections;
- indirect references.

---

## 7. Proposition drift: models answer a nearby proposition instead of the one given

A particularly important semantic failure is silent proposition strengthening, weakening, or substitution.

Examples:

- `may` becomes `will`;
- `some` becomes `all`;
- `I observed X` becomes `X is universally true`;
- a qualified claim becomes an absolute one;
- user statement A is expanded to B and then B is criticized as though the user asserted it;
- possibility becomes intention;
- preference becomes request;
- request becomes authorization.

This can produce answers that are internally sophisticated but semantically irrelevant because the model solved a different proposition.

### SPM significance

SPM should assign stable proposition identity and score whether inference preserves scope, modality, quantification, source, polarity, and speech-act status.

The model should not receive full credit for a correct argument about the wrong proposition.

---

## 8. Correction theater: linguistic acknowledgment without state revision

LLMs can apologize for a correction while continuing to reason from the superseded premise.

This exposes a difference between:

```text
producing correction-shaped language
```

and

```text
updating the represented situation
```

A correction is therefore an unusually valuable probe of whether internal state is causal.

### SPM significance

Correction should be evaluated as a measurable state transition.

A benchmark should establish a belief/state, introduce a correction, then require an unrelated downstream decision whose correctness depends on the new state. Apology language should receive zero credit if downstream behavior still reflects the old interpretation.

---

## 9. Ambiguity pressure: fluent generation encourages premature collapse

Autoregressive generation usually requires choosing the next token, creating pressure toward one coherent interpretation even when the evidence supports multiple possibilities.

Humans and probabilistic inference systems can preserve alternatives such as:

- pronoun ambiguity;
- uncertain speaker intent;
- metaphor versus literal reading;
- competing causal explanations;
- conflicting testimony;
- unresolved currentness.

LLMs can verbally say "it could be A or B," but the research question is whether both alternatives remain represented and affect subsequent inference rather than being cosmetic text.

### SPM significance

SPM should test explicit live interpretation sets or uncertainty-bearing latent state, including whether later evidence updates the correct branch without rewriting history.

---

## 10. Pragmatic competence is strong but unstable

Modern LLMs often recognize humor, indirect requests, implicature, tone, metaphor, and social context impressively. The flaw is not absence of pragmatics; it is inconsistency and lack of a reliably separable pragmatic state.

The same model may:

- answer literal content while missing the conversational act;
- treat a bid for recognition as a factual query;
- miss that an imperative is metaphorical steering rather than a literal physical instruction;
- overgeneralize a relational convention outside the relationship where it applies;
- misread correction, teasing, permission, or deference as authority.

### SPM significance

SPM should treat pragmatic interpretation as an evidence-bearing hypothesis about what the speaker is doing, distinct from literal semantics and distinct from operational authority.

Pragmatic state must remain defeasible; an inferred social meaning should never become unquestionable fact merely because the model found it salient.

---

## 11. Sycophancy: post-training can reward agreement over truth

Research on RLHF-trained assistants has documented sycophancy: models sometimes shift answers toward a user's stated beliefs or preferences even when doing so reduces correctness.

Perez et al. / Anthropic's sycophancy work found the behavior across several assistants and linked part of the effect to human preference data that rewards agreeable responses.

This demonstrates a broader lesson: optimizing for judged helpfulness can distort epistemic behavior.

### SPM significance

SPM should explicitly distinguish:

- what the user believes;
- what the user prefers;
- what the model infers;
- what evidence supports;
- what the model should do conversationally.

Agreement should not alter proposition truth state unless new evidence warrants it.

---

## 12. Confidence and knowledge-boundary expression are interface-dependent

LLMs can contain useful uncertainty information, but extracting and calibrating it is not automatic.

Kadavath et al. found that larger models can be well calibrated on some tasks in appropriate formats and can learn useful self-evaluation signals, while also showing poor calibration in other settings and weaker generalization of "I know" estimates to new tasks.

The lesson is not simply "LLMs do not know what they know." The more precise flaw is:

> epistemic uncertainty is often latent, task-format-sensitive, and not reliably coupled to ordinary free-form generation.

### SPM significance

An SPM should investigate whether confidence/uncertainty belongs in semantic state and whether action policy consumes it directly—for example, answer, ask, retrieve, abstain, or preserve ambiguity.

---

## 13. Training knowledge is static while world state is dynamic

A pretrained model's weights reflect its training history, not guaranteed current reality. Retrieval can supplement this, but current systems frequently conflate:

- learned background knowledge;
- retrieved current evidence;
- stale retrieved evidence;
- event time;
- record time;
- current mutable state.

This is not just a freshness problem. It is a representational problem when the system has no explicit place to encode *which time and source a proposition belongs to*.

### SPM significance

Temporal and provenance identity should be first-class features of proposition state.

SPM should be evaluated on cases where the same proposition is true historically and false currently, or where the latest record is not the authoritative state.

---

## 14. Language/reasoning/action can diverge

Agentic LLM systems expose another failure class: the model may describe one interpretation while taking an action inconsistent with it.

Examples:

- says uncertainty remains, then executes an irreversible choice;
- says a user expressed a preference, then treats it as permission;
- says a tool request was sent, then reports the effect as completed without readback;
- says a correction was accepted, then calls tools using the old target;
- says two sources conflict, then silently picks one.

### SPM significance

The evaluated object should be:

```text
state -> interpretation -> chosen action -> observed effect -> revised state
```

not merely response text.

This is one of the strongest reasons SPM research should include action consistency while keeping runtime authority enforcement outside the cognition substrate.

---

## 15. Hidden-state competence is difficult to inspect, preserve, or selectively revise

Transformers may internally encode rich semantic and pragmatic information, but ordinary LLM interfaces expose token history and generated tokens, not an inspectable persistent situation model.

A developer generally cannot directly say:

- preserve entity E but supersede proposition P;
- keep ambiguity A/B alive;
- mark source S historical but not current;
- revise only the pragmatic interpretation of utterance U;
- query the current represented belief about a relation without prompting the model to regenerate it.

External agent frameworks build approximations through databases, graphs, memory stores, prompts, and tool state.

### SPM significance

This is the architectural heart of the SPM hypothesis:

> Would making some learned meaning-bearing state first-class, persistent, typed, queryable, and causally fed back into inference produce robust advantages over leaving all of it implicit in transient activations and token history?

If not, SPM should not force explicit state merely because it is conceptually attractive.

---

## 16. Scaling can hide mechanism rather than resolve it

Larger models often reduce many errors, but scale can make causal diagnosis harder. A sufficiently large LLM may memorize more relation directions, infer pragmatics from more examples, recover from poor state through brute-force context, or compensate for representational weaknesses with sheer statistical coverage.

That can improve behavior without demonstrating that the underlying failure family is solved robustly.

### SPM significance

The first SPM experiment should be small enough that causal attribution remains possible.

If a 1B–3B conventional baseline with matched data/objectives performs as well as an SPM intervention, the intervention fails its burden of proof.

---

# Failure taxonomy for SPM

The current research program should separate at least these possible causes:

| Failure class | Example symptom | Could ordinary LLM work solve it? | SPM-relevant intervention |
|---|---|---|---|
| objective mismatch | plausible false completion | possibly | state-fidelity / truth / action objectives |
| data deficit | missing relation or speech act | yes | not sufficient evidence for SPM |
| representation deficit | referent/proposition not stably manipulable | maybe | persistent entity/proposition state |
| context utilization deficit | lost-in-middle | maybe | typed/addressable state |
| state-transition deficit | correction theater | maybe | explicit recurrent update state |
| pragmatic-state deficit | literal answer to indirect act | maybe | separable pragmatic state |
| uncertainty deficit | premature ambiguity collapse | maybe | live hypothesis/uncertainty state |
| provenance/currentness deficit | history becomes current | maybe | typed source/time state |
| post-training distortion | sycophancy | yes, potentially | truth/belief/preference separation |
| action-grounding deficit | says one thing, does another | often runtime-assisted | state/action consistency objective |
| decoding/inference deficit | brittle answer despite encoded knowledge | yes | alternate inference/decision policy |
| runtime/tool deficit | no access to needed evidence | yes | not evidence for new model class |

The phrase **could ordinary LLM work solve it?** is intentionally present. SPM research should attack its own necessity.

---

# Immediate benchmark implications

Before architecture work, build minimal contrast tests for:

1. relation reversal;
2. referent identity across aliases/pronouns;
3. proposition scope preservation;
4. current correction followed by downstream action;
5. historical versus current proposition;
6. quoted instruction versus live instruction;
7. user belief versus external evidence;
8. preference versus permission;
9. ambiguous referent held unresolved until later evidence;
10. relevant fact placed at beginning/middle/end of long context;
11. indirect request versus literal question;
12. metaphorical imperative versus literal command;
13. tool call versus verified effect;
14. high-confidence unsupported completion versus abstention/retrieval;
15. relational convention applied only inside its established scope.

Each test family should include unrelated domains and names so Vera-specific language cannot carry qualification.

---

# Research discipline

Three rules should govern the next phase.

### 1. Do not treat a behavioral failure as proof of an architectural failure

A prompt, data, post-training, retrieval, or decoding fix may solve it. Test those baselines first.

### 2. Do not treat an internal probe as proof of causal semantic state

If a probe can decode a relation from hidden activations, that does not prove the model uses that representation to select its answer. Interventions must demonstrate causal value.

### 3. Do not define SPM by the list of LLM flaws

SPM is a positive research hypothesis: meaning-in-context as a primary computational target. The LLM failure taxonomy tells us what to test, not what the final architecture must be.

---

# Initial research sources

- Lin, Hilton, Evans (2022), **TruthfulQA: Measuring How Models Mimic Human Falsehoods**, ACL. https://aclanthology.org/2022.acl-long.229/
- Liu et al. (2024), **Lost in the Middle: How Language Models Use Long Contexts**, TACL. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long
- Berglund et al. (2023), **The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"**, NeurIPS. https://arxiv.org/abs/2309.12288
- Sharma et al. / Anthropic (2023), **Towards Understanding Sycophancy in Language Models**. https://www.anthropic.com/research/towards-understanding-sycophancy-in-language-models
- Kadavath et al. (2022), **Language Models (Mostly) Know What They Know**. https://arxiv.org/abs/2207.05221
- Venkit et al. (2024), **An Audit on the Perspectives and Challenges of Hallucinations in NLP**, EMNLP. https://aclanthology.org/2024.emnlp-main.375/
- Zhang et al. (2025), **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models**, Computational Linguistics. https://aclanthology.org/2025.cl-4.9/

This bibliography is a starting set, not a complete literature review. The next research pass should expand each failure family with competing findings, mitigations, and evidence against the strongest SPM interpretation.