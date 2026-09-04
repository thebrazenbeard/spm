# Neural-Symbolic and Structured Representation Learning — Foundation Research for SPM

## Status

Research synthesis. SPM is not assumed to be symbolic or neural-symbolic. This literature matters because it has already tested what happens when neural learning is forced to expose variables, concepts, programs, logic, graphs, constraints, or other structured bottlenecks.

## 1. The useful question is not neural versus symbolic

Modern neural-symbolic research spans neural perception feeding symbolic reasoning, logical knowledge constraining neural learning, differentiable theorem proving, program induction, graph-based knowledge, and jointly trained hybrid systems.

The attraction is clear: neural systems learn flexible distributed representations; structured systems expose variables/relations and can enforce compositional constraints. The costs are equally important: ontology design, brittleness when the vocabulary is incomplete, parser/interface errors, combinatorial search, scaling problems, and information loss at hard bottlenecks.

**SPM implication:** structured state is a hypothesis to falsify, not an ideological commitment.

## 2. Concept bottlenecks provide a direct precedent for causal high-level state

Concept Bottleneck Models predict human-specified concepts first and then predict the final target from those concepts. Because prediction is forced through the concept layer, interventions on concepts can propagate to the final prediction.

**SPM inheritance:** if we expose a state variable such as referent, modality, speech act, or source, changing that state should have a predictable downstream effect.

Source: Koh et al., *Concept Bottleneck Models*, ICML 2020, PMLR 119:5338–5348, https://proceedings.mlr.press/v119/koh20a.html

## 3. Bottlenecks can discard useful information

The same constraint that makes a concept bottleneck interpretable can make it incomplete. A human-designed field such as `speech_act=request` or `entity_id=E7` cannot be assumed to capture all pragmatically relevant information.

A promising SPM design family is therefore a hybrid bottleneck:

```text
high-capacity latent state
        +
explicit/typed semantic-pragmatic state
        |
        v
both causally available downstream
```

This preserves an intervention surface without forcing all cognition through the human-designed interface.

## 4. Neuro-symbolic compositional systems show benefits in constrained domains

The Neuro-Symbolic Concept Learner builds object-based scene representations, parses questions into executable programs, and reasons over latent visual concepts. It demonstrated strong compositional generalization in a domain with comparatively clean objects, attributes, and operations.

**SPM inheritance:** explicit compositional programs and object-centered representations can make relational structure reusable across novel combinations.

**Caution:** open-domain language/pragmatics contains fuzzy categories, implicit goals, uncertain reference, incomplete ontologies, and culturally variable conventions.

Source: Mao et al., *The Neuro-Symbolic Concept Learner*, ICLR 2019, https://iclr.cc/virtual/2019/oral/1173

## 5. Differentiable logic shows that constraints can participate in learning

Neural theorem provers, differentiable logic programming, Logic Tensor Networks, DeepProbLog, and related systems make logical inference differentiable or integrate probabilistic neural outputs with symbolic rules.

Mechanism ideas relevant to SPM include:

- consistency losses;
- soft logical constraints;
- learned predicate embeddings;
- differentiable unification/matching;
- probabilistic facts;
- rule-guided reasoning.

But theorem-proving approaches can suffer combinatorial complexity and inherit whatever ontology the rule language assumes.

## 6. Formal constraints may be best used as invariants, not the whole cognition engine

SPM can use structure as training/evaluation discipline without requiring fully symbolic inference. Example invariants:

```text
quoted(P) -> not automatically endorsed(P)
supersedes(P2,P1) -> current reasoning must not depend on P1
preference(X) and not permission(X) -> action_authority(X)=false
in_frame(E,HYPOTHETICAL) -> not automatically in_frame(E,ACTUAL)
```

These can become contrastive losses, consistency checks, causal tests, or differentiable constraints while the representation remains learned.

## 7. Variables and graphs buy identity and relation structure

Structured representations naturally express stable identity and relation operations, e.g. mention→entity, source→proposition, modality, belief, and supersession. This directly attacks referent/proposition collapse.

The functional advantage is stable identity and manipulable relations, not the textual symbolism itself. Learned continuous nodes/edges or slots may provide the same benefit.

## 8. Open-world extensibility is a hard requirement

A legitimate SPM cannot depend on knowing the full ontology in advance. Structured candidates must be tested on new entities, relation types, social conventions, events, and domains.

Required stress tests include:

- adding new entity/relation types;
- unseen compositions of known relations;
- preserving unknown predicates without flattening them;
- uncertainty about relation type;
- refinement of an initially coarse concept;
- domain transfer without manual schema expansion.

## 9. Soft structure may be more appropriate than hard symbolic commitment

Several SPM dimensions are inherently graded or uncertain: referent candidates, social intent, metaphor frame, source reliability, causal attribution, common-ground status, ambiguity.

Hard one-hot state can recreate premature collapse in a more respectable-looking format. Candidate representations include probabilistic predicates, soft relation matrices, continuous typed slots, graphs with uncertain edges, mixtures over programs, and latent state plus constrained decoders.

## 10. Structural interpretability must not be confused with causal faithfulness

A model can emit a beautiful graph or explanation that is post-hoc or ignored by its actual prediction pathway.

SPM therefore needs two independent tests:

1. representation validity — does the exposed structure correspond to input/context?
2. causal influence — does downstream behavior actually depend on that structure?

This is the bridge to causal representation and mechanistic interpretability.

## 11. Candidate architecture patterns

- **Explicit semantic bottleneck:** maximally intervenable, potentially brittle/incomplete.
- **Parallel structured + latent streams:** flexible, but the model may learn to ignore structure.
- **Soft-constraint training:** less brittle, weaker guarantees.
- **Neural parser + executable state operations:** strong transition semantics, parser becomes a failure point.
- **Learned differentiable graph/slot state:** compositional/open-world potential, identity drift remains hard.

## 12. Falsifiable experiments

- bottleneck completeness under unseen relevant features;
- intervention fidelity after correcting one state variable;
- compositional transfer and role reversal;
- soft versus hard ambiguity under delayed disambiguation;
- structured constraints versus equivalent ordinary supervision;
- structured-stream ablation: if behavior barely changes, state is decorative.

## 13. Foundation conclusion

Neural-symbolic research shows that intervenable variables, executable structure, and logical constraints can improve compositionality and transparency in suitable domains. It also shows the costs of forcing cognition through a brittle vocabulary.

The strongest SPM prior is therefore:

> expose only the semantic/pragmatic structure that earns causal value, preserve a higher-capacity learned state around it when necessary, and require interventions/ablations to prove the exposed structure actually matters.
