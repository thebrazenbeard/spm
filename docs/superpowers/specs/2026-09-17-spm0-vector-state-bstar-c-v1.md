# SPM-0 Vector-State B*/C Candidate V1

## Status

FROZEN DESIGN CANDIDATE / NOT YET TRAINED / NOT QUALIFIED.

This is the smallest learned-state experiment permitted by the approved SPM-0 design. It tests persistent learned state before any hyperconnectome, multi-timescale, explicit graph, or Rezon mechanism is added.

## Literal proposition

With the inherited SmolLM3-3B backbone frozen, a learned 128-dimensional state carried across model-relevant transitions (C) will outperform an otherwise identical state pathway whose prior is reset at every transition (B*) on held-out meaning-in-context tasks.

The stronger claim that learned latent state beats conventional persistent memory is tested separately against Arm M.

## Frozen inherited subject

- model: `HuggingFaceTB/SmolLM3-3B`
- revision: `a07cc9a04f16550a088caea529712d1d335b0ac1`
- inventory digest: `fd0ee8f56c88636d77521cf9082b14cc499cf11f34f8467c189b11791dc1decb`
- hidden width: 2048
- transformer blocks: 36
- tokenizer/backbone weights remain frozen
## Architecture

The intervention boundary is after transformer block 31 (zero-based), leaving blocks 32-35 as subsequent frozen model computation.

For each transition:
1. run the textual transition through embeddings and blocks 0-31 with no learned SPM state;
2. take the last real-token hidden vector as `H_t`;
3. compute `x_t = Linear(2048, 128)(H_t)`;
4. update `S_(t+1) = GRUCell(x_t, prior_state)`;
5. project `delta_t = Linear(128, 2048, bias=False)(S_(t+1))`;
6. add `delta_t` to the last real-token hidden vector;
7. run frozen blocks 32-35, final norm, and inherited LM head.

B* and C instantiate the exact same module class and parameter shapes. B* supplies zero prior state at every transition. C supplies the prior transition's detached-or-connected state according to the frozen training sequence; no textual history is added to C that B* does not receive.

Trainable parameters are only observation projection, GRUCell, and state-to-hidden projection. No LoRA, prompt tuning, changed tokenizer, auxiliary answer head, or learned retrieval is admitted in V1.
## Gradient and transition rule

Within one training case, C's recurrent state remains in the autograd graph across transitions so the final answer loss can train earlier state updates. State is detached/reset between independent cases.

B* executes the same number of updater calls but replaces the prior with an all-zero tensor before every call. This is the only B*/C architectural distinction.

The frozen backbone output entering the state module is treated as an observation, not a trainable path into blocks 0-31. Gradients begin at the learned state module and continue through frozen blocks 32-35 and the inherited output head only as needed to optimize the state-module parameters.

## First bounded data budget

Training material is source-controlled synthetic semantic/pragmatic material separate from `spm_v0_public_dev.jsonl`.

- 96 training cases: 12 each across eight semantic/state families
- 24 validation cases: 3 each across those eight families
- public 36-case suite remains evaluation-only
- fixed data seed: `20260917`
- each semantic case is trained under three cyclic choice/label presentations
- exact benchmark turns, case IDs, and canonical case digests are forbidden from the training/validation corpus

The first bounded run is non-qualifying candidate evidence. Blind-holdout qualification remains separate.
## First bounded optimization budget

For each arm independently:
- initialization seed: `1729`
- data-order seed: `20260917`
- epochs: 2
- effective cases per optimizer step: 4
- optimizer: AdamW
- learning rate: `3e-4`
- weight decay: `0.01`
- gradient-norm ceiling: `1.0`
- objective: mean next-answer-token cross entropy over the three balanced presentations
- backbone parameters: frozen
- adapter/state parameters: all trainable

B* and C start from byte-identical state-module initialization. They see the same cases in the same order and perform the same number of optimizer steps. Any failed run is retained rather than retuned after comparing B* and C.

## Pre-training gates

No optimizer step is permitted until exact baseline readiness is `READY`, Arm M/control artifacts are frozen, this architecture/data/budget manifest is committed, and the generated training/validation corpus passes leakage and digest checks.

## Primary kill test

If C does not beat B* on the frozen validation/public evaluation regime within the predeclared uncertainty rule, persistent-state credit fails for V1. If C beats B* but not Arm M, the stronger learned-representation claim fails even if persistence itself is useful.
