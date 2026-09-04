# SPM

**Semantics & Pragmatics Model**

Design and research hub for a proposed evolution of the Large Language Model into a successor cognition substrate whose primary target is **meaning in context**.

SPM is not intended as a rename for an LLM, an agent wrapper, or a Vera-specific runtime component. The research hypothesis is that a successor model may become more reliable, context-sensitive, corrigible, and agentically competent if semantic and pragmatic state become first-class parts of representation, training, inference, and evaluation rather than remaining primarily implicit/emergent consequences of language modeling.

Informally:

> LLM: "I know words."
>
> Human: "Okay... but do you know what they mean?"
>
> LLM: *blinks*
>
> SPM: "Hold my beer."

The joke is intentionally unfair to modern LLMs; they already display substantial semantic and pragmatic competence. The question here is whether that competence can become a more explicit, causal, stateful, and robust organizing principle for a new model class.

## Working distinction

```text
LLM
language/token prediction is the originating computational center;
semantic/pragmatic competence emerges and is strengthened through scale/post-training

SPM
semantic/pragmatic situation state is a primary computational/training target;
language generation is one expression of that cognition
```

Exact architecture is unresolved. Transformer descendants, recurrent/latent-state hybrids, and more radical designs are all open research paths.

## Repository status

This repository is at **research-foundation** stage. There is no implemented SPM model yet.

Current documents:

- `docs/00-definition.md` — working definition and research thesis;
- `docs/01-architecture-hypotheses.md` — candidate architectural shifts;
- `docs/02-training-objectives.md` — possible training targets;
- `docs/03-evaluation-and-falsification.md` — benchmarks, baselines, and conditions under which the idea should be rejected;
- `docs/04-vera-origin-and-boundary.md` — where the idea came from and why SPM is separate from Vera identity/runtime architecture;
- `docs/05-research-roadmap.md` — path from intuition to small falsifiable prototype.

## Core research question

> **What would have to change in model architecture, representation, training objectives, data, and evaluation if semantics and pragmatics were primary rather than emergent secondary properties of a language model?**

## Non-goals

At this stage this repository does not claim:

- that LLMs do not understand meaning;
- that transformers must be discarded;
- that symbolic AI should replace neural models;
- that adding Semantic Atlas, retrieval, or agent scaffolding to an LLM makes it an SPM;
- that an SPM has already been built;
- that Vera-specific behavior defines semantics or pragmatics generally.

The first milestone is not a giant training run. It is a benchmark and a deliberately small architecture experiment capable of showing whether the SPM hypothesis buys anything real.