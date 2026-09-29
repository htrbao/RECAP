#!/usr/bin/env bash
# Two-stage RECAP (arXiv:2511.14759) fine-tuning of GR00T N1.7 on the VFE
# vacuum-pick task. Ported from the N1.5 version of this script.
#
# Stage 1 (--recap-stage value_head): trains only the distributional value
#   head + advantage embedding from the base checkpoint.
# Stage 2 (--recap-stage policy): trains the advantage-conditioned policy,
#   starting from the Stage 1 checkpoint.
#
# WANDB_API_KEY must be exported by the caller (e.g. your shell profile or a
# untracked .env file) before running this script. Do not hardcode it here —
# this file is tracked by git.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export WANDB_API_KEY="938a809d45af5c62586335955d205d25d0db5d04"
export HF_HOME="/mnt/data/sftp/data/baoht9/"
export PYTHONWARNINGS="ignore"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
MODALITY_CONFIG_PATH="$REPO_ROOT/examples/VFE/vfe_config.py"
EMBODIMENT_TAG="vfe_recap"
OUTPUT_DIR="/mnt/data/sftp/data/vla/vr_checkpoints/vrh3-vfe-vacuum-pick-recap"
WANDB_PROJECT="GR00T N1.7 RECAP"

# Per-dataset raw modality.json (state/action/video/annotation layout plus the
# RECAP reward column mapping, e.g. {"reward": {"current": {"original_key":
# "next.reward"}}}). See lerobot_episode_loader.py's _reward_column() for how
# this is consumed. UPDATE this path before running.
MODALITY_JSON_PATH="/mnt/data/sftp/data/baoht9/modalities/vfe/modality.json"

DATA_PICK1="/mnt/data/sftp/data/vla/data_sim_ac/20260928_VR_H5D_VFE_sim_pick_fail"
DATA_PICK2="/mnt/data/sftp/data/vla/data_sim_ac/20260929_VR_H5D_VFE_sim_pick_fail"
DATA_PICK3="/mnt/data/sftp/data/vla/data_sim_ac/20260928_VR_H5D_VFE_sim_pick_success"
DATA_PICK4="/mnt/data/sftp/data/vla/data_sim_ac/20260929_VR_H5D_VFE_sim_pick_success"
DATASET_PATH="${DATA_PICK1}:${DATA_PICK2}:${DATA_PICK3}:${DATA_PICK4}"

for dataset_dir in "$DATA_PICK1" "$DATA_PICK2" "$DATA_PICK3" "$DATA_PICK4"; do
    cp "$MODALITY_JSON_PATH" "$dataset_dir/meta/modality.json"
done

# Restrict to a single GPU so HF Trainer doesn't wrap the model in DataParallel
# (see examples/finetune.sh for the same guard on multi-GPU runs via torchrun).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Stage 1 (RECAP Eq. 1): value head only ---
python gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --modality-config-path "$MODALITY_CONFIG_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --num-gpus 1 \
    --output-dir "${OUTPUT_DIR}-value" \
    --wandb-project "$WANDB_PROJECT" \
    --use-wandb \
    --max-steps 10000 \
    --global-batch-size 64 \
    --save-steps 5000 \
    --recap-stage value_head

# --- Stage 2 (RECAP Eq. 3): advantage-conditioned policy, from the Stage 1 checkpoint ---
python gr00t/experiment/launch_finetune.py \
    --base-model-path "${OUTPUT_DIR}-value" \
    --dataset-path "$DATASET_PATH" \
    --modality-config-path "$MODALITY_CONFIG_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --num-gpus 1 \
    --output-dir "${OUTPUT_DIR}-policy-$(date +"%Y-%m-%d_%H-%M-%S")" \
    --wandb-project "$WANDB_PROJECT" \
    --use-wandb \
    --max-steps 45000 \
    --global-batch-size 64 \
    --save-steps 5000 \
    --recap-stage policy
