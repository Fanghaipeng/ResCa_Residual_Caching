# ResCa DiT

This directory is a DiT migration of the FLUX ResCa prototype, starting from
`ResCa/dit/clusca-dit`.

Current scope:
- Adds `--mode ResCa` with sparse proxy-denoising for the MLP branch.
- Reuses Taylor/Seer approximation for attention during sparse steps.
- Adds trajectory K-medoids proxy selection with `--resca-proxy-method`.
- Adds `--resca-solver` choices: `ie`, `it`, and `bdf2`.
- Keeps the original `ClusCa` and `Taylor` modes available for comparison.

Notes:
- DiT sampling steps run in reverse order, so the ResCa update uses DiT's
  cached residual orientation rather than copying the FLUX signs directly.
- `k=1` is expected for ResCa, matching one proxy token per cluster.
