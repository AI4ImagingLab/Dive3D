#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-A rubbery cactus}"
SEED="${SEED:-12155}"
GPU="${GPU:-0}"
N_PARTICLES="${N_PARTICLES:-1}"
WORKDIR="${WORKDIR:-exp_weimin_mesh}"

sanitize_dirname () {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -e 's/ /_/g' -e 's/[^a-zA-Z0-9_]//g'
}

base_name="$(sanitize_dirname "$PROMPT")"
filename_for_find="$(echo "$PROMPT" | sed 's/ /-/g')"

stage1_dir="${WORKDIR}/${base_name}_s${SEED}_stage1"
stage2_dir="${WORKDIR}/${base_name}_s${SEED}_stage2"
stage3_dir="${WORKDIR}/${base_name}_s${SEED}_stage3"

echo "========================================================"
echo "[Dive3D] prompt: $PROMPT"
echo "[Dive3D] seed: $SEED | gpu: $GPU | n_particles: $N_PARTICLES"
echo "[Dive3D] stage1: $stage1_dir"
echo "[Dive3D] stage2: $stage2_dir"
echo "[Dive3D] stage3: $stage3_dir"
echo "========================================================"

echo "### Stage 1 (SIM-based optimization; diversity-focused) ###"
CUDA_VISIBLE_DEVICES="$GPU" python main.py \
  --text "$PROMPT" \
  --iters 25000 \
  --use_pickscore \
  --lambda_entropy 10 \
  --scale 7.5 \
  --seed "$SEED" \
  --density_thresh 0.2 \
  --n_particles "$N_PARTICLES" \
  --h 512 --w 512 \
  --t5_iters 5000 \
  --workspace "$stage1_dir/"

echo "### Stage 2 (geometry refinement with DMTet) ###"
recent_ckpt_dir="$(find "$stage1_dir" -type d -name "*${filename_for_find}*" -exec ls -d {}/checkpoints \; | head -n 1 || true)"
if [[ -z "${recent_ckpt_dir}" ]]; then
  echo "[Dive3D][ERROR] Could not find stage1 checkpoint dir under: $stage1_dir"
  exit 1
fi
echo "[Dive3D] stage1 checkpoints: $recent_ckpt_dir"

CUDA_VISIBLE_DEVICES="$GPU" python main.py \
  --text "$PROMPT" \
  --iters 15000 \
  --use_pickscore \
  --scale 5 \
  --dmtet \
  --n_particles "$N_PARTICLES" \
  --init_ckpt "$recent_ckpt_dir/best_df_ep0250.pth" \
  --normal True \
  --sds True \
  --density_thresh 0.2 \
  --lambda_normal 0 \
  --workspace "$stage2_dir/"

echo "### Stage 3 (texture finetuning) ###"
recent_ckpt_dir="$(find "$stage2_dir" -type d -name "*${filename_for_find}*" -exec ls -d {}/checkpoints \; | head -n 1 || true)"
if [[ -z "${recent_ckpt_dir}" ]]; then
  echo "[Dive3D][ERROR] Could not find stage2 checkpoint dir under: $stage2_dir"
  exit 1
fi
echo "[Dive3D] stage2 checkpoints: $recent_ckpt_dir"

CUDA_VISIBLE_DEVICES="$GPU" python main.py \
  --text "$PROMPT" \
  --iters 30000 \
  --use_pickscore \
  --scale 7.5 \
  --dmtet \
  --init_ckpt "$recent_ckpt_dir/best_df_ep0150.pth" \
  --density_thresh 0.2 \
  --finetune True \
  --workspace "$stage3_dir/"

echo "### All stages finished. ###"
