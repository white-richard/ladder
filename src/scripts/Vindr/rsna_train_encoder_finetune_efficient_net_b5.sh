#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="$HOME/.code/datasets/rsna/mammo_clip"
IMG_DIR="train_images_png"
CSV_PATH="${DATA_DIR}/rsna_w_upmc_concepts_breast_clip.csv"
LABEL="breast_birads"
TASK_TYPE="multiclass"
# Example for BI-RADS multiclass:
# LABEL="breast_birads"
# TASK_TYPE="multiclass"
PRETRAINED_CHECKPOINT="out/ViNDr/fold0/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth"
OUT_DIR="out/Vindr_RSNA/fold0_encoder_finetune"
PRECOMPUTE_FEATURES="y"
FEATURE_CACHE_DIR="${OUT_DIR}/feature_cache"
FORCE_RECOMPUTE_FEATURES="n"

echo "=============train_encoder_finetune==================="
python ./src/codebase/train_encoder_finetune.py \
  --data-dir "${DATA_DIR}" \
  --img-dir "${IMG_DIR}" \
  --csv-path "${CSV_PATH}" \
  --dataset "RSNA" \
  --arch "tf_efficientnet_b5_ns-detect" \
  --label "${LABEL}" \
  --task-type "${TASK_TYPE}" \
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
  --precompute-features "${PRECOMPUTE_FEATURES}" \
  --feature-cache-dir "${FEATURE_CACHE_DIR}" \
  --force-recompute-features "${FORCE_RECOMPUTE_FEATURES}" \
  --epochs 30 \
  --freeze-epochs 30 \
  --batch-size 16 \
  --num-workers 0 \
  --print-freq 10000 \
  --log-freq 500 \
  --running-interactive "n" \
  --lr 1.0e-3 \
  --weighted-BCE "y" \
  --balanced-dataloader "n" \
  --start-fold 0 \
  --n_folds 1 \
  --smoke-test "n" \
  --tensorboard-path "${OUT_DIR}" \
  --checkpoints "${OUT_DIR}" \
  --output_path "${OUT_DIR}"

# AUC-ROC-macro: 0.7782784876091365, acc breast_birads patients: 73.28086164043081
