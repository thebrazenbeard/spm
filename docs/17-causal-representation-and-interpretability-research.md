# Causal Representation and Interpretability — Foundation Research for SPM

## Status

Research synthesis. SPM explicitly wants semantic/pragmatic state to be more than something a probe can decode. This note defines what evidence would justify calling a representation *causal* in the sense relevant to SPM.

## 1. Decodability is not causal use

A probe can recover information from a neural activation even when the model does not actually use that information to produce its output. Correlation between a representation and a semantic label is therefore insufficient evidence that the representation is part of the model's operative mechanism.

SPM's burden is stronger:

> If a state variable is claimed to represent a referent, proposition, modality, source, correction, speech act, ambiguity set, or participant belief, changing that state should change downstream behavior in the specific ways the semantics predicts.

This makes intervention, mediation, ablation, and counterfactual behavior central to qualification.

## 2. Causal mediation gives a disciplined intervention vocabulary

Causal mediation analysis asks how much of the effect of an input or treatment on an output passes through a particular mediator. In mechanistic interpretability, the mediator may be a neuron, attention head, MLP, residual-stream subspace, token position, latent vector, feature, or other internal unit.

Stolfo et al. use activation interventions to localize mechanisms involved in arithmetic reasoning. Mueller et al.'s 2026 survey argues that mechanistic-interpretability work benefits from explicitly defining the causal units and search procedures used in mediation analysis rather than relying on ad-hoc interpretability claims.

**SPM implication:** every exposed semantic/pragmatic state needs a declared intervention target and a behavioral consequence test.

### Sources

- Stolfo, Belinkov & Sachan, *A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis*, EMNLP 2023, DOI: 10.18653/v1/2023.emnlp-main.435, https://aclanthology.org/2023.emnlp-main.435/
- Mueller et al., *The Quest for the Right Mediator: Surveying Mechanistic Interpretability for NLP Through the Lens of Causal Mediation Analysis*, Computational Linguistics 52(1), 2026, DOI: 10.1162/coli.a.572, https://aclanthology.org/2026.cl-1.10/

## 3. Interchange interventions are especially relevant to semantic state

Causal-abstraction work tests whether high-level variables can be aligned with low-level neural representations so that interventions at the high level correspond to interventions in the network. Interchange-intervention accuracy asks whether swapping a hypothesized high-level variable produces the behavior predicted by the high-level causal model.

This is close to the ideal SPM test:

```text
keep everything else fixed
replace REFERENT(E1) with REFERENT(E2)
observe whether only E-dependent reasoning/actions change
```

or:

```text
replace MODALITY(P)=possible with MODALITY(P)=certain
observe whether downstream entailments/actions change accordingly
```

## 4. Causal abstraction itself can become vacuous without representation constraints

Recent work warns that causal abstraction is not automatically meaningful. If the mapping from low-level neural state to high-level variables is allowed to be arbitrarily powerful/nonlinear, it may be possible to map almost any network to almost any high-level algorithm, including networks that cannot actually perform the task.

That creates an important SPM guardrail:

> A claimed semantic abstraction must have constrained complexity, generalize across inputs, support interventions, and predict behavior outside the examples used to construct the mapping.

A perfect post-hoc alignment is not enough.

### Source

- Sutter, Minder, Hofmann & Pimentel, *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?*, arXiv:2507.08802 (2025), https://arxiv.org/abs/2507.08802

## 5. Multiple high-level causal models may be needed

Pîslar, Magliacane & Geiger show that one simple high-level algorithm may only partially explain a network and propose combining causal models to obtain more faithful abstractions, exposing a tradeoff between abstraction strength and faithfulness.

**SPM implication:** do not assume one semantic/pragmatic state schema will explain every computation. A model may use different mechanisms under different input regimes. SPM qualification should allow partial abstractions and require explicit scope.

### Source

- Pîslar, Magliacane & Geiger, *Combining Causal Models for More Accurate Abstractions of Neural Networks*, CLeaR 2025, PMLR 275:114–138, https://proceedings.mlr.press/v275/pislar25a.html

## 6. Lossy high-level state is unavoidable and should be modeled explicitly

Any human-readable semantic state is likely to be lossy relative to the full neural computation. Recent causal-abstraction work explicitly studies lossy representations and how observational/interventional/counterfactual queries map between low- and high-level models.

SPM therefore should not demand that its exposed semantic state be a complete description of cognition. It should demand that the exposed state be *sufficient for the semantic/pragmatic distinctions it claims to represent*.

### Source

- Xia & Bareinboim, *Causal Abstraction Inference under Lossy Representations*, ICML 2025, PMLR 267:68225–68235, https://proceedings.mlr.press/v267/xia25a.html

## 7. Representation editing is a stronger test than explanation generation

If a model says `I was tracking Patrick as the referent`, that statement proves little. A stronger test is to modify the internal binding and see whether downstream behavior follows the modification.

Candidate SPM tests:

- entity-binding edits;
- proposition source edits;
- modality edits;
- temporal/currentness edits;
- speech-act edits;
- common-ground edits;
- correction/supersession edits;
- ambiguity-set edits.

The intervention should have a narrow predicted effect. Broad nonspecific degradation is not evidence of semantic control.

## 8. Causal units may be nonlinear and distributed

Recent interpretability work reports that useful instructions/features can be linearly separable yet participate in nonlinear causal interactions. This is important because SPM should not require one concept = one neuron or one linear direction.

The research target is functional causal structure, not simplistic localization.

### Source

- Bigoulaeva et al., *Patches of Nonlinearity: Instruction Vectors in Large Language Models*, ACL 2026, https://aclanthology.org/2026.acl-long.559/

## 9. Actionable interpretability is more relevant than passive interpretation

Recent mechanistic-interpretability surveys distinguish locating internal mechanisms from steering/intervening on them. For SPM, steering is not merely a safety feature; it is evidence that the proposed state variable is functionally real enough to support controlled revision.

A correction should ideally correspond to a controlled state intervention whose downstream behavior changes without full retraining or prompt replay.

## 10. Candidate causal qualification ladder for SPM state

### Level 0 — correlated

A probe predicts the state label from activations.

Insufficient for SPM causal-state claim.

### Level 1 — predictive

The state predicts future output/action better than baseline activations or token features.

Still insufficient.

### Level 2 — mediated

Interventions show that some output effect passes through the state/mediator.

Promising.

### Level 3 — targeted intervention

Changing one state variable causes the specific downstream changes predicted by the semantic model while preserving unrelated behavior.

Strong evidence.

### Level 4 — compositional intervention

Multiple independently meaningful state variables can be changed/composed and the model behaves according to their combined semantics across unseen domains.

This is close to the strongest first-generation SPM claim.

## 11. Causal-state failure modes

SPM qualification should explicitly detect:

- **probe illusion:** state is decodable but unused;
- **post-hoc narration:** model generates the right state description after deciding the answer elsewhere;
- **broad damage:** intervention changes output only because it corrupts general computation;
- **alignment overfit:** mapping works only on calibration examples;
- **ontology overfit:** causal variable works only in one domain;
- **hidden bypass:** model routes around the explicit state channel;
- **state aliasing:** one variable controls multiple unrelated phenomena;
- **causal underdetermination:** several incompatible high-level models explain the same behavior.

## 12. Falsifiable experiments

### A. Referent interchange

Swap two entity-state bindings while preserving the same text/context representation. Score whether only referent-dependent answers/actions swap.

### B. Correction intervention

Take a state before correction, directly apply the intended state edit, and compare downstream behavior with naturally processing the correction. The two should converge.

### C. State-stream ablation

Remove or scramble the explicit semantic/pragmatic stream. If behavior is unchanged, the stream is decorative.

### D. Unseen-domain intervention

Learn an intervention mapping on one domain and test the same variable class elsewhere. This attacks alignment overfit.

### E. Complexity-controlled causal abstraction

Compare simple/linear/sparse mappings with increasingly expressive mappings and report faithfulness-versus-complexity rather than celebrating perfect fit from an arbitrarily powerful mapper.

### F. Counterfactual action prediction

Intervene on source/currentness/permission state and require predicted action changes before external effect execution.

## 13. Foundation conclusion

The causal-representation literature gives SPM a hard standard:

> **A semantic/pragmatic representation is not first-class merely because we can name or decode it. It must participate causally in inference, survive controlled interventions, generalize beyond the examples used to discover it, and change behavior according to the semantics we claim it carries.**

That criterion should be written into SPM qualification from the beginning, not added after a prototype appears to work.
