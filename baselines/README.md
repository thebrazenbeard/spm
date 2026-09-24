# SPM baseline evidence

This directory contains source-controlled identity and policy material for SPM baseline subjects. Generated model outputs and large result artifacts stay outside Git unless a later reviewed decision explicitly admits them.

A qualifying baseline subject must bind both model and tokenizer to immutable identities. A mutable alias such as `main`, `latest`, or a model name without a content digest is insufficient. Local inventories use a SHA-256 digest over the complete model directory; repository-backed subjects use immutable hexadecimal revisions and may additionally record an inventory digest.

Every result must also bind the frozen benchmark digest and deterministic generation configuration. Rerunning a friendly model name against different weights is a different subject, not a replication.

The initial baseline gate requires at least two distinct open-weight conventional subjects plus the intended inherited backbone. Passing the public development suite does not qualify SPM; it only characterizes controls and exercises the harness.

Large generated result files should be stored under a git-ignored local evidence directory and referenced by digest from reviewed source-controlled summaries.