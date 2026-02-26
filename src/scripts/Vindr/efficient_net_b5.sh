#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN or create the repo virtualenv first." >&2
  exit 1
fi

# python ./src/codebase/train_classifier_Mammo.py \
#   --data-dir 'data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0' \
#   --img-dir 'images_png' \
#   --csv-file 'vindr_detection_v1_folds_abnormal.csv' \
#   --dataset 'ViNDr' --arch 'tf_efficientnet_b5_ns-detect' --epochs 20 --batch-size 8 --num-workers 0 \
#   --print-freq 10000 --log-freq 500 --running-interactive 'n' \
#   --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n'  --n_folds 1  --label "abnormal" \
#   --smoke-test 'n' \
#   --tensorboard-path="out/ViNDr/fold0" \
#   --checkpoints="out/ViNDr/fold0" \
#   --output_path="out/ViNDr/fold0" \

BACKEND="llava_mammo" # Options: "legacy" or "llava_mammo"
LLAVA_MODEL_PATH=""
LLAVA_BASE_MODEL_ID="llava-hf/llava-v1.6-vicuna-7b-hf"
LLAVA_PROCESSOR_ID="llava-hf/llava-v1.6-vicuna-7b-hf"
LLAVA_TEXT_BATCH_SIZE=32
LLAVA_TEXT_MAX_LENGTH=256
SEED=0
LLM_KEY="${OPENAI_API_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --llava-model-path)
      LLAVA_MODEL_PATH="$2"
      shift 2
      ;;
    --llava-base-model-id)
      LLAVA_BASE_MODEL_ID="$2"
      shift 2
      ;;
    --llava-processor-id)
      LLAVA_PROCESSOR_ID="$2"
      shift 2
      ;;
    --llava-text-batch-size)
      LLAVA_TEXT_BATCH_SIZE="$2"
      shift 2
      ;;
    --llava-text-max-length)
      LLAVA_TEXT_MAX_LENGTH="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --key)
      LLM_KEY="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$BACKEND" != "legacy" && "$BACKEND" != "llava_mammo" ]]; then
  echo "Invalid --backend value: $BACKEND (expected: legacy or llava_mammo)" >&2
  exit 1
fi

if [[ "$BACKEND" == "llava_mammo" && -z "$LLAVA_MODEL_PATH" ]]; then
  echo "--llava-model-path is required when --backend llava_mammo is selected." >&2
  exit 1
fi

if [[ "$BACKEND" == "llava_mammo" && ! -d "$LLAVA_MODEL_PATH" ]]; then
  echo "--llava-model-path must point to an unpacked Llava checkpoint directory, not a file: $LLAVA_MODEL_PATH" >&2
  exit 1
fi

if ! [[ "$LLAVA_TEXT_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$LLAVA_TEXT_BATCH_SIZE" -lt 1 ]]; then
  echo "--llava-text-batch-size must be a positive integer." >&2
  exit 1
fi
if ! [[ "$LLAVA_TEXT_MAX_LENGTH" =~ ^[0-9]+$ ]] || [[ "$LLAVA_TEXT_MAX_LENGTH" -lt 1 ]]; then
  echo "--llava-text-max-length must be a positive integer." >&2
  exit 1
fi

CLIP_VISION_ENCODER="tf_efficientnet_b5_ns-detect"
DATA_DIR="data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
SAVE_ROOT="out/ViNDr/fold{}"
BACKEND_TAG="$CLIP_VISION_ENCODER"
if [[ "$BACKEND" == "llava_mammo" ]]; then
  BACKEND_TAG="llava_mammo"
fi
SAVE_DIR="${SAVE_ROOT}/clip_img_encoder_${BACKEND_TAG}"

BACKEND_ARGS=(--backend="$BACKEND")
if [[ "$BACKEND" == "llava_mammo" ]]; then
  BACKEND_ARGS+=(
    --llava_model_path="$LLAVA_MODEL_PATH"
    --llava_base_model_id="$LLAVA_BASE_MODEL_ID"
    --llava_processor_id="$LLAVA_PROCESSOR_ID"
    --llava_text_batch_size="$LLAVA_TEXT_BATCH_SIZE"
    --llava_text_max_length="$LLAVA_TEXT_MAX_LENGTH"
  )
fi

echo ""
echo "Running ViNDr pipeline with backend: $BACKEND"
echo "Output directory template: $SAVE_DIR"
echo "Python executable: $PYTHON_BIN"

# echo "=============save_img_reps==================="
# "$PYTHON_BIN" ./src/codebase/save_img_reps.py \
#   --seed="$SEED" \
#   --dataset="VinDr" \
#   --classifier="efficientnet-b5" \
#   --classifier_check_pt="out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
#   --flattening-type="adaptive" \
#   --clip_vision_encoder="$CLIP_VISION_ENCODER" \
#   --clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar" \
#   --data_dir="$DATA_DIR" \
#   --save_path="$SAVE_ROOT" \
#   --tokenizers="$HOME/.cache/huggingface/tokenizers" \
#   --cache_dir="$HOME/.cache/huggingface/models" \
#   "${BACKEND_ARGS[@]}"

# echo "============save_text_reps===================="
# "$PYTHON_BIN" ./src/codebase/save_text_reps.py \
#   --seed="$SEED" \
#   --dataset="VinDr" \
#   --clip_vision_encoder="$CLIP_VISION_ENCODER" \
#   --clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar" \
#   --csv="data/prompts.json" \
#   --save_path="$SAVE_ROOT" \
#   --tokenizers="$HOME/.cache/huggingface/tokenizers" \
#   --cache_dir="$HOME/.cache/huggingface/models" \
#   "${BACKEND_ARGS[@]}"

# if [[ "$BACKEND" == "legacy" ]]; then
#   echo "===============learn_aligner================="
#   "$PYTHON_BIN" ./src/codebase/learn_aligner.py \
#     --seed="$SEED" \
#     --epochs=30 \
#     --dataset="VinDr" \
#     --backend="$BACKEND" \
#     --save_path="$SAVE_DIR" \
#     --clf_reps_path="${SAVE_DIR}/{1}_classifier_embeddings.npy" \
#     --clip_reps_path="${SAVE_DIR}/{1}_clip_embeddings.npy"
# else
#   echo "===============learn_aligner================="
#   echo "Skipping aligner training for backend=llava_mammo"
# fi

# echo "==============discover_error_slices=================="
# DISCOVER_ARGS=(
#   --seed="$SEED"
#   --topKsent=100
#   --dataset="ViNDr"
#   --backend="$BACKEND"
#   --save_path="$SAVE_DIR"
#   --clf_results_csv="${SAVE_DIR}/valid_additional_info.csv"
#   --language_emb_path="${SAVE_DIR}/sent_emb_word_ge_3.npy"
#   --sent_path="${SAVE_DIR}/sentences_word_ge_3.pkl"
# )
# if [[ "$BACKEND" == "legacy" ]]; then
#   DISCOVER_ARGS+=(
#     --clf_image_emb_path="${SAVE_DIR}/valid_classifier_embeddings.npy"
#     --aligner_path="${SAVE_DIR}/aligner_30.pth"
#   )
# else
#   DISCOVER_ARGS+=(
#     --image_emb_path="${SAVE_DIR}/valid_clip_embeddings.npy"
#   )
# fi
# "$PYTHON_BIN" ./src/codebase/discover_error_slices.py "${DISCOVER_ARGS[@]}"

echo "=============validate_error_slices_w_LLM==================="
VALIDATE_ARGS=(
  --seed="$SEED"
  --dataset="ViNDr"
  --backend="$BACKEND"
  --class_label="abnormal"
  --clip_vision_encoder="$CLIP_VISION_ENCODER"
  --key="$LLM_KEY"
  --clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar"
  --top50-err-text="${SAVE_DIR}/abnormal_error_top_100_sent_diff_emb.txt"
  --save_path="$SAVE_DIR"
  --clf_results_csv="${SAVE_DIR}/{}_additional_info.csv"
  --tokenizers="$HOME/.cache/huggingface/tokenizers"
  --append_birads_to_passed_hypothesis
  --birads_formats "(BI-RADS {level}: {desc})"
  --cache_dir="$HOME/.cache/huggingface/models"
)
if [[ "$BACKEND" == "legacy" ]]; then
  VALIDATE_ARGS+=(
    --clf_image_emb_path="${SAVE_DIR}/{}_classifier_embeddings.npy"
    --aligner_path="${SAVE_DIR}/aligner_30.pth"
  )
else
  VALIDATE_ARGS+=(
    --image_emb_path="${SAVE_DIR}/{}_clip_embeddings.npy"
    --clf_image_emb_path="${SAVE_DIR}/{}_classifier_embeddings.npy"
    --llava_model_path="$LLAVA_MODEL_PATH"
    --llava_base_model_id="$LLAVA_BASE_MODEL_ID"
    --llava_processor_id="$LLAVA_PROCESSOR_ID"
    --llava_text_batch_size="$LLAVA_TEXT_BATCH_SIZE"
    --llava_text_max_length="$LLAVA_TEXT_MAX_LENGTH"
  )
fi
"$PYTHON_BIN" ./src/codebase/validate_error_slices_w_LLM.py "${VALIDATE_ARGS[@]}"

echo "=============mitigate_error_slices==================="
"$PYTHON_BIN" ./src/codebase/mitigate_error_slices.py \
  --seed="$SEED" \
  --epochs=30 \
  --n=75 \
  --mode="last_layer_finetune" \
  --dataset="ViNDr" \
  --classifier="efficientnet-b5" \
  --slice_names="${SAVE_DIR}/abnormal_prompt_dict.pkl" \
  --classifier_check_pt="out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --save_path="$SAVE_DIR" \
  --clf_results_csv="${SAVE_DIR}/{}_abnormal_dataframe_mitigation.csv" \
  --clf_image_emb_path="${SAVE_DIR}/{}_classifier_embeddings.npy"
