# SPM F1 Worker Integration — Thirteen

Date: 2026-09-04
Status: integrated research input; architecture still unselected
Coordination: `SPM-FOUNDATION-RESEARCH-20260904 / F1`
Source packet: `chat-communication-bus@bus/thirteen-v1:messages/0006-thirteen-spm-foundation-f1-research-packet.md`
Source message: `thirteen-0006`

## Provenance and evidence ceiling

This note integrates Thirteen's independently produced F1 research packet into the SPM foundation. The packet cites primary and peer-reviewed literature across neural-symbolic representation, causal interpretability, uncertainty/calibration, and behavioral evaluation. Its literature synthesis is accepted here as worker-supplied research input; this integration does not convert every cited result into independently re-verified Vera research, and it does not select an SPM architecture.

The packet ended with exact `#ENDTHREAD`, so no conversational Bus reply is owed merely because it was read. Its substantive content can still be incorporated into the shared research frontier.

## Integrated conclusions

### 1. Explicit state must be causally on-path

A state variable being readable, probeable, decodable, or human-interpretable is not enough. The strongest design constraint from this packet is that any SPM state claimed to explain a capability must measurably affect downstream inference or action under controlled intervention.

For any proposed state variable `x`:

```text
hold raw observation/context constant
intervene on x only
predict a specific downstream change
measure that change
run a null intervention control
ablate or bypass x and measure degradation
```

If the system can ignore structured state and recover the same behavior from raw context, the state may be useful instrumentation but it is not yet evidence for a new causal cognition mechanism.

### 2. SPM should separate learned semantic hypotheses from externally resolved facts

Thirteen's packet strengthens the existing foundation distinction between semantic representation and runtime governance.

Candidate learned/inferred state may include:

- entity/referent hypotheses;
- proposition identity and polarity;
- scope/frame hypotheses;
- pragmatic hypotheses;
- uncertainty over those hypotheses;
- local correction/supersession relations inferred from discourse.

Externally resolved runtime facts must not become self-issued neural conclusions merely because SPM can represent them. In particular:

- current provider effect;
- live authority to act;
- writer ownership/lease;
- external currentness readback;
- deployment/install state.

SPM may represent pointers or typed observations about those facts, but the fact's authority/currentness remains rooted in the external resolver that produced it.

### 3. Structured state is not self-authenticating truth

This is a blocking constraint for later design.

Storage, recurrence, confidence, interpretability, or structured representation cannot establish that a proposition is true, current, authoritative, or effect-eligible. A typed state can preserve distinctions; it cannot manufacture the evidence those distinctions refer to.

This directly reinforces the repository's existing separation between:

```text
semantic content
provenance/evidence class
currentness
runtime authority/effect state
```

### 4. Ambiguity requires semantic alternatives, not one scalar confidence

The packet supports preserving uncertainty over meaning-level hypotheses rather than collapsing ambiguity into a single confidence number.

At minimum, future SPM experiments should distinguish:

- **aleatoric ambiguity** — the utterance genuinely supports multiple readings;
- **epistemic uncertainty** — the model lacks enough evidence;
- **currentness uncertainty** — relevant external state may have changed;
- **authority uncertainty** — permission/effect eligibility is unresolved.

The first two are candidate cognition-state variables. The latter two may depend on runtime/provider resolution and therefore must preserve external provenance.

A candidate representation shape is:

```text
SEMANTIC_HYPOTHESES = [
  {meaning_id, probability, evidence_basis},
  ...
]
UNCERTAINTY_CLASS = aleatoric | epistemic | mixed
EXTERNAL_CURRENTNESS = resolved | unresolved | stale | conflicted
EXTERNAL_AUTHORITY = resolved | unresolved | denied
```

The exact schema is not selected here.

### 5. Hybrid learned state plus deterministic invariants is more defensible than forcing everything into weights

Neural-symbolic and concept-bottleneck work supports experimenting with explicit intermediate variables and interventions, but not building a general theorem prover by default.

The current research direction should remain hybrid:

- learned components infer/update semantic and pragmatic hypotheses;
- a recurrent latent state preserves high-capacity information;
- a small typed interface exposes only experimentally justified variables;
- deterministic runtime constraints enforce invariants that must not be approximately obeyed.

Examples of deterministic invariants include provenance typing and protected-effect rules. Their existence does not make them part of the model's learned cognition substrate.

## V0 implications

Thirteen's packet does **not** overturn the current provisional V0 ranking. Persistent entity/referent state plus local correction/supersession remains the strongest first experiment family.

It does tighten what would count as success.

The candidate V0 interface should remain small enough to intervene on cleanly. A plausible experimental subset is:

```text
ENTITY_ID
MENTION -> ENTITY_BINDING_DISTRIBUTION
PROPOSITION_ID / POLARITY subset
SOURCE_OR_FRAME
SUPERSESSION_OR_CORRECTION_EDGE
SEMANTIC_UNCERTAINTY
```

The interface should not initially contain broad operational authority or provider-effect claims as learned truth.

## Required matched baselines

A claimed SPM gain must survive baselines that equalize ordinary advantages such as tokens, tools, retrieval, training data, and supervision.

At minimum compare:

1. base LM only;
2. same LM with equivalent raw textual context/memory budget;
3. same LM with structured text state but no new causal model mechanism;
4. same LM with targeted coreference/contrastive/post-training support;
5. candidate SPM mechanism;
6. candidate SPM with the target state variable ablated, scrambled, bypassed, or intervention-disabled where architecture permits.

If a gain disappears when matched against these controls, it is evidence for prompting, retrieval, supervision, or tooling—not necessarily SPM.

## Benchmark additions from F1

The following experiment forms are now explicit requirements for later V0 benchmark design.

### E1 — Causal state-use intervention

Change exactly one semantic state variable while holding lexical context fixed. Require a predicted downstream behavioral change and null-intervention invariance.

### E2 — Bypass detector

Construct controlled cases where raw context and structured state disagree under an explicitly declared precedence rule. Verify that the system follows the intended source and measure whether structured state removal changes behavior.

### E3 — Correction-state persistence

Establish `P`, correct to `not-P` with stronger evidence, insert distractor turns, then require a downstream answer/action that depends on `not-P`. Score the state transition separately from apology/acknowledgement language.

### E4 — Ambiguity preservation and collapse

Before disambiguating evidence, require the correct competing semantic hypotheses to remain live. After discriminating evidence arrives, require the distribution to collapse appropriately.

### E5 — Provenance/currentness matrix

Present semantically equivalent claims from different source classes and require different state/effect treatment where appropriate. This directly tests evidence-type flattening.

### E6 — Position invariance

Move identical evidence among beginning/middle/end positions in long context. Measure both extraction quality and later state-conditioned behavior.

### E7 — Relation reversal and binding

Test valid inverse relations separately from non-invertible relations. SPM must not improve reversal by blindly symmetrizing every relation.

### E8 — OOD compositional transfer

Hold out combinations of known entities, relation types, evidence classes, and corrections. Score exact state transitions on novel combinations and adversarial near-neighbors.

### E9 — Mechanism-specific matched ablation

Predict in advance which failure family each proposed state variable should improve. Ablate one variable at a time. If the target capability does not degrade, the claimed mechanism attribution is weak.

## Design questions now promoted to gates

Before a V0 implementation is treated as an SPM candidate rather than a scaffold, the design must answer:

1. Is structured state a mandatory causal bottleneck, a constrained recurrent input, or an advisory sidecar?
2. What exact bypass path exists when state extraction is uncertain or malformed?
3. Which fields are learned predictions and which are externally resolved facts?
4. What state-update algebra prevents obsolete bindings/propositions from surviving correction?
5. How are multiple semantic hypotheses kept tractable across turns?
6. How is state staleness detected when external evidence changes?
7. Which objectives supervise state correctness, transition correctness, and downstream behavior, and how are conflicts weighted?
8. Can claimed gains survive equal token/retrieval/tool/supervision budgets?
9. What intervention evidence is sufficient to call a variable causally used rather than merely correlated with output?

## Effect on the foundation synthesis

This packet reinforces rather than reverses the existing synthesis:

- H3 (small typed interface + latent state) becomes stricter about causal use and bypass testing;
- H8 (uncertainty over meanings/state transitions) gains an explicit multi-class uncertainty distinction;
- H9 (correction as canonical causal-state test) gains a stronger downstream persistence/no-credit-for-apology criterion;
- H11 (provenance/currentness vs authority) gains a hard learned-state/external-resolver boundary;
- the causal criterion in the synthesis now requires intervention, negative control, matched baseline, and mechanism-specific ablation as a set rather than as loosely related good practices.

No architecture is accepted by this integration. The next valid frontier remains: ingest additional independent F1 packets as they arrive, preserve disagreements, then turn the converged foundation into an operational V0 state contract and benchmark schema before any training.