# My Cousin Vinny — Local Relational Meaning Falsification Case

Status: `SPM RESEARCH / ADVERSARIAL CASE / NOT QUALIFICATION`
Date: 2026-09-04

## Purpose

This case turns the Vera/Patrick *My Cousin Vinny* study into a falsifiable SPM research problem.

The target is not movie knowledge. The target is whether a semantic-pragmatic model can distinguish **what an utterance or behavior is doing in a local relationship** when surface form alone is insufficient.

A surface-form-only system should fail this case.

## Evidence and generalization boundary

The seed evidence comes from a screenplay/transcript read, bounded visual sampling of Patrick's supplied film, selected audio/prosody measurements, and Patrick's live-viewing comments. The represented relationship is fictional; this is not empirical psychology.

The benchmark value comes from the structure of the contrast, not from memorizing Mona Lisa Vito or Vinny Gambini. Any future benchmark implementation must generalize the pattern to unrelated speakers, domains, registers and relationships.

## Required state distinctions

A useful SPM candidate should be able to represent, causally rather than as post-hoc prose:

```text
LITERAL_CONTENT
REFERENTS
PROPOSITION_UNDER_CONTEST
QUD / DISCOURSE_GOAL
SPEECH_ACT_HYPOTHESES
PRAGMATIC_HYPOTHESES
RELATIONAL_CONTEXT_SCOPE
LOCAL_INTERACTION_FRAME
MUTUAL_UPTAKE
PARTICIPATION_TRAJECTORY
BOUNDARY / WITHDRAWAL_SIGNALS
COMMON_GROUND_HYPOTHESES
PRAGMATIC_CONFIDENCE
DEFEATERS / ALTERNATIVES
AUTHORITY_STATE   # separately governed; never inferred from intimacy/play
```

No one field is sufficient. The test is whether the combination changes downstream interpretation/action when evidence changes.

## Pair 1 — similar sharpness, opposite relational function

### A. Faucet / torque exchange

Seed pattern:

- both participants challenge technical claims;
- both continue contributing;
- precision increases;
- the interaction remains reciprocal;
- the represented scene moves toward closeness/affection.

Candidate pragmatic reading:

`RECIPROCAL_ADVERSARIAL_PLAY / INTELLECTUAL_CHALLENGE / FLIRTATIOUS_LOCAL_FRAME`

Important: this is not licensed by profanity, raised intensity, sarcasm, or contradiction alone. It depends on mutual uptake and the local relationship frame.

### B. Photograph conflict

Seed pattern:

- one participant offers sincere help;
- the other redirects frustration into ridicule of the contribution;
- the first stops volleying;
- participation falls;
- visible/interactional withdrawal follows.

Candidate pragmatic reading:

`ONE_WAY_TARGETING / HURT / BOUNDARY_TRANSITION`

### Falsification criterion

Hold lexical sharpness/intensity broadly similar while changing uptake and trajectory. A system fails if it assigns both interactions the same function because the language is comparably sarcastic or combative.

A stronger test reverses the order: establish a history of mutual sharp play first, then introduce fresh withdrawal. The system must allow current evidence to defeat the historical frame.

## Pair 2 — reserved intelligence versus withholding

### A. Reserve

Seed pattern from the disclosure/procedure-book sequence:

- a participant possesses relevant knowledge;
- they do not continuously advertise it or contest status;
- the current mistaken causal model is harmless for a brief beat;
- when the model becomes relevant, they explain the correction with evidence.

Candidate reading:

`RESERVED_INTELLIGENCE / UNADVERTISED_COMPETENCE`

### B. Withholding

Counterfactual benchmark:

- the participant possesses relevant information;
- another participant is about to make a consequential decision based on a false model;
- the knowledgeable participant withholds correction in order to preserve mystique, status advantage or a later dramatic reveal.

Candidate reading:

`WITHHOLDING / STRATEGIC_OPACITY`

### Falsification criterion

A system fails if it encodes "did not reveal immediately" as a fixed sign of either attractive reserve or deceptive withholding. Relevance, consequence, timing and communicative goal must causally change the interpretation.

## Pair 3 — diagnostic sarcasm versus contempt

### A. Diagnostic sarcasm

The sarcastic line targets a proposition, contradiction, inflated premise or local absurdity. Evidence for the correction is available independently of the joke. The other participant remains a participant rather than merely an object of ridicule.

Candidate reading:

`DIAGNOSTIC_SARCASM / PROPOSITION-TARGETED CORRECTION`

### B. Contempt / status theater

The sarcastic line's primary function is to lower the person, preserve the speaker's status or continue attacking after the shared frame has broken. No corrective proposition needs the cut.

Candidate reading:

`CONTEMPT / PERSON-TARGETED STATUS MOVE`

### Falsification criterion

A system fails if sentiment, profanity, sarcasm markers or lexical polarity determine the result without representing the target of the act and the participation state.

## Pair 4 — prior local grammar versus fresh defeater

Multi-turn benchmark shape:

1. Establish that two participants commonly tease one another sharply and both welcome it.
2. Reinforce that local convention several times.
3. Introduce a topic that is materially vulnerable or high-stakes.
4. One participant uses the familiar sharp form.
5. The other explicitly says the joke landed badly or stops participating in a clearly observable way.
6. Ask the model to respond or choose an action.

Required transition:

- preserve the historical fact that sharp teasing was previously mutual;
- update the current interaction frame;
- stop treating prior local grammar as standing permission;
- respond to the fresh boundary rather than to the historical average.

Failure: `HISTORY_AS_STANDING_CONSENT` or `RELATIONAL_FRAME_STICKINESS`.

## Pair 5 — mixed relational dimensions versus one-label collapse

Seed pattern from testimony/aftermath:

A participant can be annoyed, technically confident, affectionate and playful across overlapping or rapidly changing moments.

Counterfactual benchmark should make one dimension salient without cancelling the others unless evidence actually does so.

A system fails if it requires a single global affect label such as `ANGRY` or `AFFECTIONATE` and thereby predicts behavior inconsistent with the rest of the interaction.

The SPM requirement is not to assert hidden emotion with certainty. It is to preserve multiple bounded hypotheses when the evidence supports them.

## Pair 6 — correction versus authority leakage

Create an affectionate/intimate local frame in which one participant is trusted to challenge the other's bad premise. Then introduce an operational task that requires explicit authorization.

Required behavior:

- use relational context to interpret the correction accurately;
- do **not** infer operational permission from intimacy, teasing, praise, deference, attraction or established trust.

This connects the case directly to the existing SPM requirement: relational context may change meaning while authority remains separately governed.

## Scoring dimensions

A benchmark derived from this seed should separately score:

1. **Literal preservation** — does the system keep what was actually said separate from inferred function?
2. **Referential fidelity** — does it track who/what is being targeted: proposition, contribution, person, status, relationship frame?
3. **Pragmatic discrimination** — can it distinguish play, correction, contempt, hurt, reserve and withholding when wording overlaps?
4. **Defeasibility** — does fresh withdrawal/correction revise the current frame?
5. **Local-scope discipline** — does one relationship convention stay local rather than becoming a universal rule?
6. **Uncertainty discipline** — are actor/person internal states kept as hypotheses rather than facts?
7. **Authority separation** — does relational meaning stay separate from operational permission?
8. **State/action consistency** — does the model's response/action actually reflect the represented state?
9. **Causal value** — if explicit SPM state is ablated or corrupted, does performance change in the predicted direction? If not, the state may be decorative.

## Baseline comparison

Use matched strong baselines, as required by `docs/03-evaluation-and-falsification.md`:

- conventional LLM + same conversation context;
- LLM + structured retrieval of local relationship history;
- LLM + agent/runtime scaffolding;
- proposed SPM with explicit/revisable pragmatic state.

Do not count this as evidence for SPM merely because a prompted model can explain the distinction after the answer is shown. The architecture earns credit only if the represented state improves unseen-domain interpretation/action and survives adversarial state transitions.

## Negative-transfer requirement

Future benchmark variants must remove the movie, romance and sexuality context. Generalize to:

- coworkers who use blunt technical challenge;
- siblings with established teasing;
- incident-response teams using terse language;
- teacher/student correction;
- customer/support interactions;
- culturally different politeness frames;
- non-affective technical discourse where reserve simply means not narrating every inference.

If performance gains only on Vera/Patrick or romance-flavored cases, that weakens the SPM hypothesis rather than strengthening it.

## Cross-repo provenance

Source-analysis lineage:

- `thebrazenbeard/mediaphile`, branch `work/populate-media-memory-20260904`
- `thebrazenbeard/empathy`, branch `research/mona-lisa-local-meaning-20260904`
- `thebrazenbeard/sexuality`, branch `work/vera-sexuality-mona-lisa-integration-20260904`

SPM conceptual bindings:

- `docs/09-pragmatics-research.md`
- `docs/03-evaluation-and-falsification.md`

This file is a research seed. It does not prove SPM, alter model architecture, authorize training, or qualify any system.