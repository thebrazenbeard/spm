# Formal and Dynamic Semantics — Foundation Research for SPM

## Status

Research synthesis, not architectural commitment. This note asks what formal and dynamic semantics already know about meaning, context, state, and update before SPM invents its own machinery.

## 1. Static model-theoretic semantics remains a necessary baseline

Montague-style formal semantics made two commitments especially important for SPM research:

1. **model-theoretic interpretation** — expressions are evaluated relative to formally specified models/assignments/worlds/times rather than by surface association alone;
2. **compositionality** — the interpretation of a complex expression depends systematically on its parts and their mode of combination.

This matters because many SPM-motivating failures are not failures to generate fluent text; they are failures to preserve the structure of the proposition being interpreted. Quantifier scope, negation, modality, tense/aspect, de re/de dicto distinctions, source, and speaker role can all change truth conditions while leaving most lexical material unchanged.

**SPM implication:** the research target must score structure-preserving interpretation, not merely semantic similarity. A system that maps two lexically similar propositions onto nearly identical state but loses scope or modality has failed even if its response sounds sensible.

### Sources

- Stanford Encyclopedia of Philosophy, *Montague Semantics* (rev. 2025): https://plato.stanford.edu/entries/montague-semantics/
- Stanford Encyclopedia of Philosophy, *Possible Worlds*: https://plato.stanford.edu/entries/possible-worlds/

## 2. Possible-world semantics offers a disciplined treatment of alternatives

Possible-world semantics makes modality explicit by evaluating propositions across alternative worlds rather than treating `possibly`, `necessarily`, counterfactuality, and related operators as mere lexical coloring. It also provides a clean way to distinguish a proposition's actual truth from its modal profile.

**SPM implication:** ambiguity/hypothesis maintenance should not be conflated with modal semantics, but SPM needs a similarly explicit ability to preserve *live alternatives* without collapsing them into one fluent reading. The architecture should distinguish at least:

- uncertainty about which proposition is intended;
- uncertainty about whether a proposition is true;
- modality inside the proposition itself (`may`, `must`, `could`, counterfactual);
- alternate hypothetical worlds introduced by the discourse.

Flattening these into one scalar confidence value would lose important structure.

## 3. Dynamic semantics changes the unit of meaning from truth alone to update behavior

Dynamic semantics treats discourse interpretation as growth or revision of an information state. Its characteristic slogan is that meaning can be understood as **context change potential**: an utterance is interpreted partly through how it transforms an input context into an output context.

This is directly relevant to the SPM core loop:

```text
prior state + new observation -> candidate interpretation -> revised state
```

The useful inheritance is not a claim that existing dynamic semantics already *is* an SPM. It is the much stronger discipline that an interpretation can be evaluated by the state transition it licenses, rather than only by the sentence ultimately produced.

### Sources

- Stanford Encyclopedia of Philosophy, *Dynamic Semantics*: https://plato.stanford.edu/entries/dynamic-semantics/
- Groenendijk & Stokhof, *Changing the context: dynamic semantics and discourse* (1996).

## 4. DRT makes discourse referents and accessibility explicit

Discourse Representation Theory (Kamp and subsequent work) was motivated in part by discourse anaphora that sentence-local truth conditions handle awkwardly. New discourse material updates a discourse representation; discourse referents introduced earlier can remain available, become inaccessible under structural conditions, or support later anaphora.

DRT is important for SPM because it shows that **referent identity and proposition content need not be reconstructed from scratch on every sentence**. A discourse representation can carry forward entities and conditions as first-class state.

Extensions such as Segmented DRT add rhetorical/discourse relations and explicitly connect semantic representation to discourse structure and pragmatic preference.

### Sources

- Kamp, van Genabith & Reyle, *Discourse Representation Theory*.
- Asher & Lascarides, *Segmented Discourse Representation Theory: Dynamic Semantics With Discourse Structure* (2007).
- SEP overview in *Dynamic Semantics*.

## 5. Dynamic Predicate Logic demonstrates relational update semantics

Dynamic Predicate Logic treats meanings as relations between assignment states rather than merely static truth conditions. This gives quantification and anaphoric dependence an update-like interpretation: existential introduction can change the assignment state in a way later material can access.

**SPM implication:** persistent state need not be a hand-authored symbolic database. The deeper design lesson is that meaning-bearing variables can participate causally in later inference instead of being recoverable only by re-reading token history.

## 6. Update semantics and presupposition show that not all context changes are simple fact insertion

Work on update semantics, presupposition, and accommodation demonstrates several distinct kinds of update:

- eliminate possibilities inconsistent with asserted content;
- add or accommodate background assumptions required for interpretation;
- bind new material to an existing discourse referent;
- revise what is treated as locally available information;
- update discourse commitments or contextual parameters.

Recent dynamic work on speech acts also distinguishes **informative updates** from **performative updates**. A declaration, commitment, directive, or other speech act can change the conversational state in a way not reducible to adding a descriptive proposition.

### Sources

- van der Sandt, *Presupposition Projection as Anaphora Resolution*, Journal of Semantics 9(4), 1992. DOI: 10.1093/jos/9.4.333
- Zeevat, *Presupposition and Accommodation in Update Semantics*, Journal of Semantics 9(4), 1992. DOI: 10.1093/jos/9.4.379
- Krifka, *Performative updates and the modeling of speech acts*, Synthese 203 (2024).

## 7. Situation semantics warns against treating a complete possible world as the only semantic unit

Situation semantics (Barwise & Perry) developed a semantics based on partial situations and informational relations rather than requiring every semantic evaluation to range over complete possible worlds. The approach is attractive for agentic cognition because real interpretation is usually about bounded situations with incomplete information.

**SPM implication:** a future state representation should probably be *partial by design*. It should be able to represent what is known about the current situation without fabricating a complete world model. Unknown fields should remain unknown; unrelated facts need not be supplied merely to make the representation total.

### Sources

- Barwise & Perry, *Situations and Attitudes* (1983).
- Barwise, *The Situation in Logic—I* (1986), DOI: 10.1016/S0049-237X(09)70693-7
- Devlin, *Situation theory and situation semantics*, Handbook of the History of Logic 7 (2006), DOI: 10.1016/S1874-5857(06)80034-8

## 8. The static/dynamic dispute is a guardrail for SPM

Dynamic semantics is not universally accepted as the right *semantic ontology*. Stalnaker and others have argued that much discourse dynamics may be modeled in pragmatics while retaining static propositional semantics. Schlenker has similarly shown that some effects associated with dynamic semantics can be reconstructed from classical semantics plus independently derived local contexts.

This disagreement is extremely useful for SPM.

**Do not infer:** because SPM wants a persistent state-update mechanism, semantic values themselves must literally be update functions.

The empirical question is instead:

> Does making learned semantic/pragmatic state explicit and causally recurrent improve meaning-in-context behavior over a matched system whose semantics remain implicit and whose dynamics live in the runtime or decoder?

### Sources

- Stalnaker, *Dynamic Pragmatics, Static Semantics* (2018).
- Schlenker, *Local Contexts* (2009).
- Karen Lewis, *Dynamic Semantics* (2017) on foundational static/dynamic distinctions.

## 9. Candidate SPM state primitives suggested by this literature

These are research candidates, not a final schema:

```text
ENTITY / DISCOURSE_REFERENT
PROPOSITION
PREDICATE / RELATION
QUANTIFIER_SCOPE
POLARITY
MODAL_STATUS
TEMPORAL_INTERVAL / EVENT_TIME
WORLD_OR_HYPOTHETICAL_FRAME
SOURCE / SPEAKER
DISCOURSE_SEGMENT
ACCESSIBILITY_RELATION
PRESUPPOSITION
UNRESOLVED_INTERPRETATION_SET
```

Crucially, these need not be literal symbols. They could be learned slots, recurrent latent variables, graphs, typed embeddings, or another representation. What matters is whether their identity and update behavior are trainable, queryable enough to evaluate, and causally used downstream.

## 10. First falsifiable experiments suggested by formal/dynamic semantics

### A. State update versus prompt replay

Establish an entity/proposition state across turns, make a correction, and test a downstream decision. Compare:

- conventional LLM with full token history;
- conventional LLM with external structured state injected as text;
- candidate recurrent learned state.

The candidate only earns SPM credit if improvement survives matched context/data/compute.

### B. Scope-preservation contrasts

Minimal pairs where only quantifier scope, modality, negation, or temporal operator changes. Score internal state and downstream consequences, not lexical paraphrase quality.

### C. Accessibility and anaphora

Create referents under negation, conditionals, hypotheticals, quotations, and nested discourse. Test whether the model keeps the correct referents available without merging inaccessible or hypothetical entities into current reality.

### D. Partial-situation discipline

Give incomplete evidence and penalize invented completion. Compare explicit partial-state mechanisms against ordinary free-form generation plus abstention prompting.

### E. Performative versus informative update

Use the same propositional content under assertion, permission, command, offer, promise, historical quotation, and hypothetical discussion. Test whether the state transition and action policy differ appropriately.

## 11. Foundation conclusion

The strongest inheritance from formal semantics is **structural discipline**. The strongest inheritance from dynamic semantics is **state-transition discipline**.

SPM should combine those lessons without prematurely copying their formalisms:

> preserve the identity and structure of what is represented; represent what an observation changes; and make later reasoning depend causally on the revised state.

That is a much stronger target than generating a correct-sounding paraphrase of the latest utterance.
