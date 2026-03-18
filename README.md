# Dive3D (TMLR 2025)

The official code release of **Dive3D**:

> **Dive3D: Diverse Distillation-based Text-to-3D Generation via Score Implicit Matching** (TMLR 2025)  
> **Authors**: Weimin Bai, Yubo Li, Wenzheng Chen, Weijian Luo, He Sun

## Overview

Distilling pre-trained 2D diffusion models into 3D assets has driven remarkable advances in text-to-3D synthesis. However, existing methods typically rely on Score Distillation Sampling (SDS) loss, which involves asymmetric KL divergence—a formulation that inherently favors mode-seeking behavior and limits generation diversity.

**Dive3D** introduces a text-to-3D framework that **replaces KL-based objectives with Score Implicit Matching (SIM) loss**, a score-based objective that mitigates mode collapse. Dive3D further integrates **diffusion distillation** and **reward-guided optimization** under a unified divergence perspective. In practice, this reformulation yields **more diverse 3D outputs** while improving **text alignment**, **human preference**, and **overall visual fidelity**. We validate Dive3D on diverse prompts and on GPTEval3D against multiple strong baselines, showing consistent improvements in diversity, photorealism, aesthetic appeal, and quantitative metrics such as text-asset alignment and 3D plausibility.

## What’s inside

- `main.py`: primary entry point (train / test / mesh export).
- `nerf/`: rendering + optimization pipeline, including SIM-related training logic.
- `raymarching/`, `gridencoder/`, `freqencoder/`: CUDA extensions for fast rendering/encoding.
- `scripts/`:
  - `build_extensions.sh`: build & install CUDA extensions.
  - `run_all_stages.sh`: a “best quality” multi-stage pipeline (stage1→stage2→stage3).
- `bibtex.txt`: BibTeX citation for the TMLR 2025 paper.

## Environment

### Hardware

- **CUDA GPU is required** for training.
- CUDA extensions require a working CUDA toolchain (NVCC) matching your installed PyTorch CUDA build.

### Python

Tested with Python 3.9+.

## Installation

Create and activate an environment (example with conda):

```bash
conda create -n dive3d python=3.10 -y
conda activate dive3d
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build CUDA extensions:

```bash
bash scripts/build_extensions.sh
```

## Quick start (one-click pipeline)

Run the full 3-stage pipeline (SIM-based stage1, DMTet refinement stage2, texture finetune stage3):

```bash
bash scripts/run_all_stages.sh "A rubbery cactus"
```

Common knobs (environment variables):

- `GPU`: which GPU id to use (default `0`)
- `SEED`: random seed (default `12155`)
- `N_PARTICLES`: number of particles / parallel candidates (default `1`)
- `WORKDIR`: workspace root for outputs (default `exp_weimin_mesh`)

Example:

```bash
GPU=1 SEED=2026 N_PARTICLES=4 WORKDIR=outputs bash scripts/run_all_stages.sh "A refined vase with artistic patterns."
```

## Stable Diffusion weights

This release is **portable by default**:

- If you do nothing, it will download weights from HuggingFace based on `--sd_version`:
  - `2.1` → `stabilityai/stable-diffusion-2-1-base`
  - `2.0` → `stabilityai/stable-diffusion-2-base`
  - `1.5` → `runwayml/stable-diffusion-v1-5`
- If you want to use a local checkpoint or a custom HF model id, pass:

```bash
python main.py --hf_key <local_path_or_hf_model_id> ...
```

## Reward guidance (optional)

The provided `scripts/run_all_stages.sh` enables **PickScore** reward by default (`--use_pickscore`).
It will download models automatically from HuggingFace:

- Processor: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- Model: `yuvalkirstain/PickScore_v1`

ImageReward-based RLHF is supported in the code but requires you to provide local paths via environment variables:

- `IMAGEREWARD_CKPT`
- `IMAGEREWARD_MED_CONFIG`

## Outputs

Outputs are written to the chosen workspace directory (e.g., `exp_weimin_mesh/...`), and typically include:

- `validation/`: periodic renders (RGB/depth/normal/textureless).
- `checkpoints/`: saved model states.
- `results/`: test-time videos / renders.
- `mesh/`: exported meshes when `--save_mesh` is used.

## Mesh export

After training, you can export a mesh (OBJ with texture) by running in test mode with `--save_mesh`, pointing to a checkpoint:

```bash
python main.py --test --save_mesh --dmtet --init_ckpt <path_to_ckpt.pth> --text "your prompt"
```

## Citation

```bibtex
@article{Dive3D2025,
  title   = {Dive3D: Diverse Distillation-based Text-to-3D Generation via Score Implicit Matching},
  author  = {Bai, Weimin and Li, Yubo and Chen, Wenzheng and Luo, Weijian and Sun, He},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2025}
}
```

