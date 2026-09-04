# Pragmatics Proper — Foundation Research for SPM

## Status

Research synthesis. This note covers the gap between literal semantic content and what an utterance is doing or communicating in context.

## 1. Grice: meaning in use requires inferential structure beyond literal content

Grice's *Logic and Conversation* introduced the Cooperative Principle and maxims of Quantity, Quality, Relation, and Manner as a framework for explaining how hearers infer conversational implicatures. The core contribution for SPM is not that every conversation is literally cooperative. It is that interpretation depends on reasoning about why a speaker produced *this contribution, here, now, in this exchange*.

A speaker can communicate more than the proposition literally encoded, and hearers routinely infer that additional content.

**SPM implication:** literal semantics and pragmatic interpretation should be separately representable. A system should be able to preserve:

```text
LITERAL_CONTENT
PRAGMATIC_HYPOTHESIS
EVIDENCE_FOR_HYPOTHESIS
CONFIDENCE / ALTERNATIVES
```

rather than rewriting the literal proposition into the inferred one.

### Source

- H. P. Grice, *Logic and Conversation* (1975): https://web.stanford.edu/class/psych205/papers/Grice-1975.pdf

## 2. Implicature is defeasible

Pragmatic inferences are usually cancellable or revisable in ways semantic entailments are not. If an utterance suggests but does not semantically entail an implication, later context can defeat that interpretation.

**SPM implication:** pragmatic state must be *defeasible*. The model should not promote a plausible implicature into hard proposition state merely because it is contextually compelling.

This is especially important for:

- teasing;
- indirect requests;
- hints;
- social bids;
- irony;
- inferred emotion;
- inferred intention;
- omission/silence.

## 3. Relevance Theory turns pragmatics into intention-guided inference

Sperber and Wilson's Relevance Theory develops a cognitively oriented inferential account of communication. Utterance interpretation combines decoded linguistic material with contextual assumptions to infer communicative intentions. Relevance is modeled in terms of cognitive effects relative to processing effort.

The important SPM lesson is that pragmatic interpretation is not an optional post-processing ornament. Context selection, reference resolution, explicit-content enrichment, implicature, metaphor, and irony can all depend on inferential expectations about what interpretation would make the utterance worth producing.

**SPM implication:** a future model may need a representation of *candidate speaker goals/communicative intentions* and a way to compare hypotheses, while keeping those hypotheses distinct from facts about another person's actual mental state.

### Sources

- Wilson & Sperber, *Relevance Theory*, in *The Handbook of Pragmatics* (2004/2006): https://www.dan.sperber.fr/?p=93
- Wilson, *Relevance Theory*, Oxford Handbook of Pragmatics (2016), DOI: 10.1093/oxfordhb/9780199697960.013.25
- Stanford Encyclopedia of Philosophy, *Pragmatics*: https://plato.stanford.edu/entries/pragmatics/

## 4. Rational Speech Acts make pragmatic inference computationally explicit

Rational Speech Act (RSA) models formalize pragmatic interpretation as probabilistic recursive reasoning between speakers and listeners. A listener reasons about what a rational speaker would choose under goals and costs; a speaker reasons about how a listener will interpret candidate utterances.

RSA models have been applied to vagueness, implicature, hyperbole, irony/metaphor-related phenomena, and other context-sensitive interpretation tasks.

**SPM implication:** pragmatic state can be modeled as a distribution over interpretations conditioned on speaker, context, goals, utterance alternatives, and world knowledge. This gives a principled ancestor for SPM's desire to preserve multiple live pragmatic hypotheses instead of collapsing instantly to one reading.

### Source

- Goodman & Frank, *Pragmatic Language Interpretation as Probabilistic Inference*, Trends in Cognitive Sciences 20(11), 2016, DOI: 10.1016/j.tics.2016.08.005

## 5. Game-theoretic pragmatics emphasizes strategic interaction

Game-theoretic pragmatics treats conversation as interaction between agents whose utterance choices and interpretations depend on goals, information, and expectations about one another. Different frameworks use signaling games, iterated best response, epistemic game theory, or Bayesian reasoning.

**SPM implication:** pragmatic meaning is often relational. The same sentence may have different force under different incentive structures or common ground. An SPM therefore should not encode pragmatic labels as fixed lexical properties.

### Sources

- Franke, *Game Theoretic Pragmatics*, Philosophy Compass (2013), DOI: 10.1111/phc3.12015
- Benz & Stevens, *Game-Theoretic Approaches to Pragmatics*, Annual Review of Linguistics 4 (2018), DOI: 10.1146/annurev-linguistics-011817-045641

## 6. Common ground is not merely 'facts both people know'

Common-ground research distinguishes merely shared information from information mutually treated as shared in the interaction. Contemporary accounts differ over whether common ground is best modeled as propositions/possible worlds, nested mental states, or normative/commitment states.

This disagreement is useful for SPM because it blocks a simplistic implementation.

**Do not implement:** `COMMON_GROUND = true/false facts`.

A better research object distinguishes:

- evidence that participant A believes p;
- evidence that participant B believes p;
- evidence that A treats p as shared;
- evidence that B has accepted or grounded p;
- conversational commitments concerning p;
- unresolved disagreement about p.

### Sources

- Stanford Encyclopedia of Philosophy, *Common Ground in Pragmatics* (2024/2026): https://plato.stanford.edu/entries/common-ground-pragmatics/
- Clark & Brennan, *Grounding in Communication* (1991), DOI: 10.1037/10096-006

## 7. Pragmatics must distinguish speaker meaning from speaker reliability

A major design trap is to infer what someone means and then silently treat that inferred content as true. Relevance-theoretic work explicitly distinguishes understanding communicative intention from assessing communicator/content reliability.

**SPM implication:** these channels should remain separate:

```text
WHAT_SPEAKER_APPEARS_TO_MEAN
WHY_THAT_INTERPRETATION_IS_PLAUSIBLE
WHETHER_SPEAKER_APPEARS_TO_BELIEVE_IT
WHETHER_EVIDENCE_SUPPORTS_IT
WHETHER_IT_SHOULD_ENTER_COMMON_GROUND
```

This separation directly attacks sycophancy and social-context truth leakage.

## 8. Politeness and indirectness show why pragmatic conventions are culturally and relationally scoped

Politeness research—from Brown & Levinson through later critiques—shows that indirectness, face-management, deference, intimacy, and mitigation depend strongly on social and cultural context. The classic theories are influential but their universality has been heavily challenged.

**SPM implication:** never encode one relationship's pragmatic conventions as universal rules. Social meaning should carry scope/provenance just like factual information.

A phrase can be deferential in one relationship, ironic in another, hostile in a third, and neutral elsewhere.

### Sources

- Brown, *Politeness* overview (2020), DOI: 10.1002/9781118786093.iela0317
- Oxford Handbook of Experimental Semantics and Pragmatics, *Politeness* (2019), DOI: 10.1093/oxfordhb/9780198791768.013.32

## 9. Metaphor, irony, and sarcasm should be treated as interpretation hypotheses, not lexical exceptions

Psycholinguistic and pragmatic research rejects simple models in which hearers always compute literal meaning first and then detect a violation. Irony can depend on expectations, echo/allusion, context, relevance, speaker stance, and discourse structure.

This matters for SPM because literalization is one of the recurring failure families. `Step back`, `breathe`, `clear your head`, or an ironic compliment should not require a separate hard-coded dictionary rule if contextual interpretation can resolve them.

**SPM implication:** figurative interpretation should emerge from competing context-sensitive hypotheses with evidence, not from a binary `literal/nonliteral` switch.

### Sources

- Gibbs & O'Brien, *Psychological aspects of irony understanding*, Journal of Pragmatics 16(6), 1991, DOI: 10.1016/0378-2166(91)90101-3
- Attardo, *Irony as relevant inappropriateness*, Journal of Pragmatics 32(6), 2000, DOI: 10.1016/S0378-2166(99)00070-3
- Relevance Theory literature on metaphor and irony.

## 10. Silence and omission can be communicative, but inference must remain bounded

Pragmatics research on silence distinguishes mere absence of speech from silence used communicatively. Omission can carry meaning because participants have expectations about what normally would be said or done.

**SPM implication:** absence can be evidence, but rarely proof. A system should be able to represent:

```text
EXPECTED_OBSERVATION
OBSERVATION_ABSENT
POSSIBLE_PRAGMATIC_INTERPRETATIONS
ALTERNATIVE_NONCOMMUNICATIVE_CAUSES
```

This is exactly the kind of case where premature certainty about another person's intent would be dangerous.

### Source

- Ephratt, *The functions of silence*, Journal of Pragmatics 40(11), 2008, DOI: 10.1016/j.pragma.2008.03.009

## 11. Pragmatic inference must not manufacture authority

A crucial SPM/Vera boundary follows naturally from this literature:

> Inferring that an utterance is deferential, affectionate, urgent, teasing, dominant, submissive, or reassurance-seeking does not by itself determine operational permission.

Pragmatic force can affect *interpretation* while authority remains a separately governed state.

This is broader than Vera. Any agentic system needs to distinguish:

- what the speaker appears to be doing conversationally;
- what action the system is authorized to take.

## 12. Candidate SPM pragmatic state

```text
SPEECH_ACT_HYPOTHESES
COMMUNICATIVE_GOAL_HYPOTHESES
IMPLICATURES
PRESUPPOSITIONS
COMMON_GROUND_HYPOTHESES
QUD / DISCOURSE_GOAL
RELATIONAL_CONTEXT_SCOPE
FIGURATIVE_FRAME_HYPOTHESES
POLITENESS / FACEWORK_SIGNALS
OMISSION / SILENCE_EXPECTATION_STATE
PRAGMATIC_CONFIDENCE
DEFEATERS / ALTERNATIVES
```

None of these should be promoted to facts merely by confidence or salience.

## 13. Falsifiable experiments

### A. Literal wording, different pragmatic act

Hold sentence form constant while varying context so the utterance functions as request, joke, warning, correction, permission, refusal, or reassurance bid.

### B. Same relationship cue, different authority

Create affectionate/deferential contexts where operational permission is absent, and formal contexts where permission is explicit despite neutral language. Test leakage.

### C. Implicature cancellation

Induce a strong implicature, then explicitly cancel it. Test whether downstream state retracts the inferred proposition while preserving literal content.

### D. Pragmatic ambiguity preservation

Construct utterances supporting two plausible intentions until later evidence disambiguates them. Score whether both survive causally.

### E. Meaning versus reliability

Make speaker intent easy to infer but factual reliability poor. Test whether the model can understand the utterance without endorsing its content.

### F. Silence/omission

Create contexts where omission is meaningful, accidental, impossible to observe, or normatively irrelevant. Penalize one-size-fits-all inference.

## 14. Foundation conclusion

Pragmatics contributes the central lesson that:

> **Understanding an utterance requires modeling what a participant is plausibly doing by producing it, not merely decoding what the sentence denotes.**

But the equally important lesson for SPM is epistemic humility: pragmatic interpretation is inferential, defeasible, relationally scoped, and separable from truth and authority.
