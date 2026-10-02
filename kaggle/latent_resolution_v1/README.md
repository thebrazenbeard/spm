# SPM Latent Resolution V1 — Kaggle campaign

This is the first accelerator experiment for the adaptive latent-memory work.

It binds:

- SPM source: `dbd4cb10be61711b349ccad5c900037d7ac68ea1`
- base model: `Qwen/Qwen2.5-0.5B-Instruct`
- immutable model revision: `7ae557604adf67be50417f59c2c2f167def9a775`

The frozen base model encodes each source chunk independently. A trainable query-conditioned recurrent latent memory updates a fixed number of latent slots after each chunk, then a learned router chooses which exact backing chunk should be rehydrated. The experiment trains and compares slot budgets 2, 4, 8, and 16.

The kernel deliberately does not generate exact answers from lossy latent state. Its first claim target is narrower: can compact learned state preserve enough task information to select the correct higher-resolution backing block?

Outputs include route accuracy, unsafe DIRECT decisions on exact-required cases, safe insufficient-fidelity decisions, router artifacts, and CUDA peak-memory probes comparing one full-context encoder pass with sequential chunk encoding plus latent state.

## Local validation

```bash
python kaggle/latent_resolution_v1/kernel_main.py --dry-run
python scripts/build_kaggle_latent_resolution_v1.py --user YOUR_KAGGLE_SLUG --output dist/kaggle-latent-resolution-v1
```

The builder creates Kaggle metadata only in the staging directory. Credentials are never part of this source package.

The current Kaggle CLI supports an explicit accelerator argument. Push the staged directory using the command printed by the builder so accelerator selection is not dependent on metadata alone.

Claim ceiling: `KAGGLE_EXPERIMENT_ONLY_NOT_VERA_RUNTIME_OR_PRODUCTION_VRAM_PROOF`.
