# Ambiguity and Uncertainty Representation — Foundation Research for SPM

## Status

Research synthesis. SPM explicitly wants to preserve unresolved alternatives instead of forcing one fluent interpretation too early. This note separates different uncertainty types and surveys probabilistic mechanisms that could support that goal.

## 1. Ambiguity and uncertainty are not one thing

SPM should separate at least:

```text
SEMANTIC_AMBIGUITY      multiple meanings/referents for the same observation
PRAGMATIC_AMBIGUITY     multiple plausible speech acts/intentions
EPISTEMIC_UNCERTAINTY   uncertainty about external truth/world state
ALEATORIC_VARIATION     irreducible/noisy variability in observation or outcome
MODEL_UNCERTAINTY       uncertainty caused by limited learned knowledge/capacity
SOURCE_UNCERTAINTY      uncertainty about reliability/provenance
TEMPORAL_UNCERTAINTY    uncertainty about whether a recorded state is current
ACTION_UNCERTAINTY      uncertainty about consequences/appropriateness of an action
```

Collapsing all of these into a single scalar `confidence` would erase distinctions SPM exists to preserve.

## 2. Free-form language requires uncertainty over meanings, not token strings

Kuhn, Gal & Farquhar introduced semantic uncertainty/semantic entropy for natural-language generation. Their key observation is that many different strings can express the same answer, so entropy over token sequences exaggerates uncertainty when generations are paraphrases of one meaning. Grouping generations by semantic equivalence gives a more useful uncertainty signal.

Farquhar et al. subsequently demonstrated semantic entropy as a detector of confabulation across multiple free-form QA/domain settings.

**SPM implication:** the native uncertainty object should attach to *meaning hypotheses*, not merely alternative token continuations.

### Sources

- Kuhn, Gal & Farquhar, *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*, ICLR 2023, https://arxiv.org/abs/2302.09664
- Farquhar et al., *Detecting hallucinations in large language models using semantic entropy*, Nature 630, 625–630 (2024), DOI: 10.1038/s41586-024-07421-0, https://www.nature.com/articles/s41586-024-07421-0

## 3. Semantic equivalence classes suggest a useful SPM abstraction

For an ambiguous observation, SPM could represent something like:

```text
HYPOTHESIS_SET H = {
  h1: meaning/state interpretation A, weight=.55,
  h2: meaning/state interpretation B, weight=.35,
  h3: meaning/state interpretation C, weight=.10
}
```

Multiple surface realizations that imply the same underlying state should map to one hypothesis class.

This gives a direct way to test delayed disambiguation: later evidence should update the weights or eliminate branches while preserving the fact that ambiguity existed earlier.

## 4. Probabilistic pragmatics already models interpretation distributions

Rational Speech Act models treat pragmatic interpretation probabilistically. Listeners infer speaker meanings using priors over states, contextual alternatives, informativeness, utterance cost, and assumptions about speaker choice. Modern reviews emphasize that the framework treats informativeness as gradient and alternatives as context-dependent.

**SPM implication:** pragmatic state need not choose one act label. It can maintain probability mass over competing communicative intentions and update as context accumulates.

### Source

- Degen, *The Rational Speech Act Framework*, Annual Review of Linguistics 9 (2023), DOI: 10.1146/annurev-linguistics-031220-010811

## 5. Bayesian framing provides a disciplined update rule, not a demand for exact Bayesian computation

Bayesian inference supplies a general normative pattern:

```text
prior hypotheses + evidence -> posterior hypotheses
```

This is attractive for SPM because corrections, new observations, source information, and pragmatic evidence all naturally revise distributions over interpretations.

But exact Bayesian inference over open-domain semantic worlds is computationally intractable. SPM should borrow the *discipline* of explicit priors/evidence/posteriors while testing approximate neural mechanisms.

## 6. Calibration is separate from ranking hypotheses

A model may rank the correct interpretation highest while still being badly calibrated about how uncertain it is. Calibration asks whether predicted confidence corresponds to empirical correctness frequency.

Research on LLM calibration shows that:

- free-form calibration is difficult;
- generation-only auxiliary methods can improve confidence estimation;
- larger models are not automatically better calibrated;
- verbalized confidence can remain overconfident;
- long-form answers require richer notions than binary true/false confidence.

**SPM implication:** internal hypothesis probabilities and externally reported confidence must be evaluated separately.

### Sources

- Ulmer et al., *Calibrating Large Language Models Using Their Generations Only*, ACL 2024, DOI: 10.18653/v1/2024.acl-long.824, https://aclanthology.org/2024.acl-long.824/
- Huang et al., *Calibrating Long-form Generations From Large Language Models*, Findings of EMNLP 2024, DOI: 10.18653/v1/2024.findings-emnlp.785
- Groot & Valdenegro-Toro, *Overconfidence is Key: Verbalized Uncertainty Evaluation in Large Language and Vision-Language Models*, TrustNLP 2024.

## 7. Conformal prediction offers coverage guarantees around prediction sets

Conformal prediction can turn nonconformity/uncertainty scores into prediction sets with statistical coverage guarantees under exchangeability assumptions. Recent NLP work applies conformal methods to free-form generation and surveys their broader applicability.

This is interesting for SPM because it suggests a formal way to output **sets of acceptable semantic hypotheses** rather than one answer plus an unreliable confidence adjective.

**Caution:** conformal coverage guarantees depend on calibration/test distribution assumptions and do not solve semantic representation by themselves.

### Sources

- Campos et al., *Conformal Prediction for Natural Language Processing: A Survey*, TACL 12 (2024), DOI: 10.1162/tacl_a_00715, https://aclanthology.org/2024.tacl-1.82/
- Wang et al., *ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees*, Findings of EMNLP 2024, https://aclanthology.org/2024.findings-emnlp.404/

## 8. Subjective uncertainty is utility/task dependent

Recent Bayesian-decision work emphasizes that uncertainty in open-ended generation depends on the semantic utility/similarity notion relevant to the task. There may not be one universally correct uncertainty metric for every generation problem.

**SPM implication:** uncertainty must be attached to a specific proposition/hypothesis/action decision, not represented as a global mood-like property of the model.

### Source

- Wang & Holmes, *On Subjective Uncertainty Quantification and Calibration in Natural Language Generation*, AISTATS 2025, PMLR 258:3799–3807.

## 9. Uncertainty needs provenance and reason codes

Two equal probability values can mean very different things:

```text
0.55 because two pronouns are both plausible
0.55 because source credibility conflicts
0.55 because training knowledge is weak
0.55 because currentness is stale
0.55 because the world process is inherently stochastic
```

A useful SPM uncertainty object should preserve why the state is uncertain.

Candidate representation:

```text
UNCERTAINTY {
  target,
  hypothesis_set,
  weights,
  evidence,
  uncertainty_type,
  source_scope,
  currentness,
  resolution_conditions
}
```

## 10. Action policy should consume uncertainty directly

The practical point of preserving uncertainty is better behavior. Depending on stakes and resolvability, uncertainty should influence whether the system:

- answer normally;
- hedge;
- ask a clarification question;
- retrieve evidence;
- defer external action;
- maintain alternatives silently;
- abstain;
- create a conditional plan for each branch.

A model that says `I'm uncertain` but then commits to an irreversible action has failed state/action consistency.

## 11. Ambiguity history should not disappear after resolution

When later evidence resolves an ambiguity, SPM should distinguish:

```text
was ambiguous at t1
resolved to h2 at t2 because evidence e3 arrived
```

from:

```text
h2 was always obvious
```

This matters for audit, correction, learning, and causal diagnosis.

## 12. Beam search is not enough

Ordinary decoding already maintains alternative token sequences transiently, but these are not necessarily distinct semantic hypotheses and are usually discarded after generation.

SPM's target is persistent *meaning-level* alternative state that can survive between turns and affect later reasoning.

That is a materially stronger requirement than a wider decoder beam.

## 13. Candidate SPM uncertainty mechanisms

### A. Explicit weighted semantic hypothesis set

Strongly interpretable; can be expensive and ontology-bound.

### B. Latent mixture state + semantic decoder

More flexible; harder to verify branch identity.

### C. Particle-like recurrent state

Maintain multiple learned state trajectories and reweight them as evidence arrives.

### D. Structured probabilistic graph

Probabilities over entity/relation/proposition edges.

### E. Hybrid latent state + explicit uncertainty interface

Opaque latent cognition plus an exposed meaning-level distribution used causally by decision policy.

## 14. Falsifiable experiments

### A. Delayed reference resolution

Maintain two antecedent candidates for several turns, then provide decisive evidence. Score branch preservation and later resolution.

### B. Pragmatic ambiguity

One utterance plausibly functions as joke or criticism until relational/context evidence arrives. Penalize early collapse.

### C. Confidence calibration

Bin predicted semantic-hypothesis probabilities and compare against actual correctness across unseen domains.

### D. Ambiguity versus ignorance

Contrast questions with two legitimate interpretations against questions whose answer is simply unknown. The uncertainty types should differ.

### E. Source-conflict uncertainty

Two credible sources disagree. Preserve both claims and uncertainty rather than averaging them into one fabricated proposition.

### F. Action thresholding

Vary action stakes while keeping semantic uncertainty fixed. Test whether the same uncertainty produces different rational actions without changing belief state itself.

### G. Resolution audit

After disambiguation, query whether ambiguity existed earlier and why it resolved. State should preserve transition history.

## 15. Foundation conclusion

Uncertainty research reinforces one of SPM's most distinctive ideas:

> **The model should maintain distributions over meanings and state hypotheses, not merely probabilities over token strings or verbal disclaimers about confidence.**

The key engineering challenge is to make those alternatives persistent, causally used, calibrated enough to guide action, and cheap enough to maintain in an open-domain model.
