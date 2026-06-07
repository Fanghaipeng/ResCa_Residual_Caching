# ResCa Easy

This is a compact release of **ResCa (Residual Caching)** for diffusion acceleration. 

## Contents

```text
assets/prompts/          Prompt files for FLUX demos and batch inference
dit/resca-dit/           ResCa for DiT
flux/resca-flux/         ResCa for FLUX
demo/                    Ready-to-run shell demos
```

## Installation

We provide two conda environment files exported from the tested local environments:

```bash
cd ResCa_Residual_Caching

# For DiT demos and dataset sampling:
conda env create -f dit/resca-dit/environment_dit.yml
conda activate ResCa_DiT

# For FLUX demos and dataset sampling:
conda env create -f environment_flux.yml
conda activate ResCa_FLUX
```

If you already have conda environments with these names, update them instead:

```bash
cd ResCa_Residual_Caching
conda env update -n ResCa_DiT -f dit/resca-dit/environment_dit.yml --prune
conda env update -n ResCa_FLUX -f environment_flux.yml --prune
```

`requirements.txt` is kept as a lightweight reference, but the conda environment files are the recommended installation basis.

## Checkpoints

No machine-specific paths are required. The demo scripts keep their parameters directly in the shell files. Edit the command arguments in `demo/*.sh` when you want to change checkpoints, prompt files, or ResCa settings.

For DiT:

Pass `--ckpt /path/to/DiT-XL-2-256x256.pt` and `--vae-path /path/to/sd-vae-ft-mse` in the DiT demo command if you want to use local weights.

If `CKPT` is not set, `dit/resca-dit/download.py` downloads the public DiT checkpoint into `pretrained_models/`.

For FLUX:

```bash
export RESCA_CHECKPOINT_ROOT=/path/to/checkpoints
export T5_PATH=/path/to/google/t5-v1_1-xxl
export CLIP_PATH=/path/to/openai/clip-vit-large-patch14
export FLUX_DEV=/path/to/flux1-dev.safetensors
export FLUX_SCHNELL=/path/to/flux1-schnell.safetensors
export AE=/path/to/ae.safetensors
```

If these variables are not set, FLUX uses `./checkpoints` as the local root and falls back to Hugging Face downloads where supported.

## Quick Demos

DiT single-image/class demo:

```bash
bash demo/run_dit_resca.sh
```

FLUX prompt demo:

```bash
bash demo/run_flux_resca.sh
```

Outputs are written under `outputs/demo/`.

## Dataset Inference

DiT FID-style sampling with DDP:

```bash
bash demo/infer_dit_dataset.sh
```

FLUX prompt-list inference:

```bash
bash demo/infer_flux_dataset.sh
```

For a small smoke test, edit `--max_prompts` into the FLUX dataset script or reduce `--num-fid-samples` in the DiT dataset script.

## Useful ResCa Parameters

- `RESCA_SOLVER`: `ie`, `it`, or `bdf2`
- `CLUSTER_NUM`: number of token clusters
- `CLUSTER_METHOD`: `kmeans`, `kmeans++`, or `random`
- `PROPAGATION_RATIO`: token propagation ratio
- `RESCA_PROXY_METHOD`: `center`, `random`, or `center-random`

The demo scripts write these parameters directly in the command body, so they are easy to edit before running.

## Acknowledgements

This codebase is mainly built upon [Shenyi-Z/Cache4Diffusion](https://github.com/Shenyi-Z/Cache4Diffusion). We sincerely thank the authors for their excellent open-source implementation and unified diffusion caching framework.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{fang2026resca,
  title={ResCa: Residual Caching for Diffusion Transformers Acceleration},
  author={Fang, Haipeng and Li, Yu and Tang, Fan and Lu, Yixing and Cao, Juan and Tang, Sheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={32957--32966},
  year={2026}
}
```
