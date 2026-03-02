#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN or create the repo virtualenv first." >&2
  exit 1
fi

# Generic pipeline controls
DATASET_KEY="vindr" # Options: vindr, rsna, cbis-ddsm
BACKEND="legacy" # Options: legacy, llava_mammo
SEED=0
LLM_KEY="${OPENAI_API_KEY:-}"

# Model/runtime controls
CLIP_VISION_ENCODER="tf_efficientnet_b5_ns-detect"
TOKENIZERS_DIR="$HOME/.cache/huggingface/tokenizers"
CACHE_DIR="$HOME/.cache/huggingface/models"
TOPK_SENT=100
ALIGNER_EPOCHS=30
MITIGATE_EPOCHS=30
MITIGATE_N=75
APPEND_BIRADS=0
BIRADS_FORMAT="(BI-RADS {level}: {desc})"

# Llava controls
LLAVA_MODEL_PATH=""
LLAVA_BASE_MODEL_ID="llava-hf/llava-v1.6-vicuna-7b-hf"
LLAVA_PROCESSOR_ID="llava-hf/llava-v1.6-vicuna-7b-hf"
LLAVA_TEXT_BATCH_SIZE=32
LLAVA_TEXT_MAX_LENGTH=256

# Step toggles
RUN_SAVE_IMG=1
RUN_SAVE_TEXT=1
RUN_ALIGNER=1
RUN_DISCOVER=1
RUN_VALIDATE=1
RUN_MITIGATE=1

# Optional overrides (if empty, dataset defaults are used)
DATA_DIR=""
SAVE_ROOT=""
CLASSIFIER_CHECKPOINT=""
CLIP_CHECKPOINT=""
CLASS_LABEL=""
PROMPTS_FILE="data/prompts.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_KEY="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
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
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --save-root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    --classifier-checkpoint)
      CLASSIFIER_CHECKPOINT="$2"
      shift 2
      ;;
    --clip-checkpoint)
      CLIP_CHECKPOINT="$2"
      shift 2
      ;;
    --class-label)
      CLASS_LABEL="$2"
      shift 2
      ;;
    --prompts-file)
      PROMPTS_FILE="$2"
      shift 2
      ;;
    --clip-vision-encoder)
      CLIP_VISION_ENCODER="$2"
      shift 2
      ;;
    --tokenizers-dir)
      TOKENIZERS_DIR="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --topk-sent)
      TOPK_SENT="$2"
      shift 2
      ;;
    --aligner-epochs)
      ALIGNER_EPOCHS="$2"
      shift 2
      ;;
    --mitigate-epochs)
      MITIGATE_EPOCHS="$2"
      shift 2
      ;;
    --mitigate-n)
      MITIGATE_N="$2"
      shift 2
      ;;
    --no-append-birads)
      APPEND_BIRADS=0
      shift
      ;;
    --birads-format)
      BIRADS_FORMAT="$2"
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
    --skip-save-img)
      RUN_SAVE_IMG=0
      shift
      ;;
    --skip-save-text)
      RUN_SAVE_TEXT=0
      shift
      ;;
    --skip-aligner)
      RUN_ALIGNER=0
      shift
      ;;
    --skip-discover)
      RUN_DISCOVER=0
      shift
      ;;
    --skip-validate)
      RUN_VALIDATE=0
      shift
      ;;
    --skip-mitigate)
      RUN_MITIGATE=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

DATASET_KEY="$(echo "$DATASET_KEY" | tr '[:upper:]' '[:lower:]')"
case "$DATASET_KEY" in
  vindr)
    DATASET_NAME="ViNDr"
    [[ -n "$DATA_DIR" ]] || DATA_DIR="data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
    [[ -n "$SAVE_ROOT" ]] || SAVE_ROOT="out/ViNDr/fold{}"
    [[ -n "$CLASSIFIER_CHECKPOINT" ]] || CLASSIFIER_CHECKPOINT="out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth"
    [[ -n "$CLIP_CHECKPOINT" ]] || CLIP_CHECKPOINT="model_weights/mammoClip-b5-model-best-epoch-7.tar"
    [[ -n "$CLASS_LABEL" ]] || CLASS_LABEL="abnormal"
    ;;
  rsna)
    DATASET_NAME="RSNA"
    [[ -n "$DATA_DIR" ]] || DATA_DIR="$HOME/.code/datasets/rsna/mammo_clip"
    [[ -n "$SAVE_ROOT" ]] || SAVE_ROOT="out/RSNA/fold{}/aucroc0.89"
    [[ -n "$CLASSIFIER_CHECKPOINT" ]] || CLASSIFIER_CHECKPOINT="out/RSNA/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth"
    [[ -n "$CLIP_CHECKPOINT" ]] || CLIP_CHECKPOINT="out/RSNA/fold0/mammoClip-b5-model-best-epoch-7.tar"
    [[ -n "$CLASS_LABEL" ]] || CLASS_LABEL="cancer"
    ;;
  cbis | cbis-ddsm)
    DATASET_NAME="CBIS-DDSM"
    [[ -n "$DATA_DIR" ]] || DATA_DIR="$HOME/.code/datasets/cbis-ddsm"
    [[ -n "$SAVE_ROOT" ]] || SAVE_ROOT="out/CBIS_DDSM/fold{}"
    [[ -n "$CLASSIFIER_CHECKPOINT" ]] || CLASSIFIER_CHECKPOINT="out/CBIS_DDSM/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth"
    [[ -n "$CLIP_CHECKPOINT" ]] || CLIP_CHECKPOINT="model_weights/mammoClip-b5-model-best-epoch-7.tar"
    [[ -n "$CLASS_LABEL" ]] || CLASS_LABEL="cancer"
    ;;
  *)
    echo "Invalid --dataset value: $DATASET_KEY (expected: vindr, rsna, cbis-ddsm)" >&2
    exit 1
    ;;
esac

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
if ! [[ "$TOPK_SENT" =~ ^[0-9]+$ ]] || [[ "$TOPK_SENT" -lt 1 ]]; then
  echo "--topk-sent must be a positive integer." >&2
  exit 1
fi

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
echo "Running mammography pipeline"
echo "  dataset: $DATASET_NAME"
echo "  backend: $BACKEND"
echo "  save root template: $SAVE_ROOT"
echo "  save dir template:  $SAVE_DIR"
echo "  class label: $CLASS_LABEL"
echo "  python: $PYTHON_BIN"

if [[ "$RUN_SAVE_IMG" -eq 1 ]]; then
  echo "=============save_img_reps==================="
  "$PYTHON_BIN" ./src/codebase/save_img_reps.py \
    --seed="$SEED" \
    --dataset="$DATASET_NAME" \
    --classifier="efficientnet-b5" \
    --classifier_check_pt="$CLASSIFIER_CHECKPOINT" \
    --flattening-type="adaptive" \
    --clip_vision_encoder="$CLIP_VISION_ENCODER" \
    --clip_check_pt="$CLIP_CHECKPOINT" \
    --data_dir="$DATA_DIR" \
    --save_path="$SAVE_ROOT" \
    --tokenizers="$TOKENIZERS_DIR" \
    --cache_dir="$CACHE_DIR" \
    "${BACKEND_ARGS[@]}"
fi

if [[ "$RUN_SAVE_TEXT" -eq 1 ]]; then
  echo "============save_text_reps===================="
  "$PYTHON_BIN" ./src/codebase/save_text_reps.py \
    --seed="$SEED" \
    --dataset="$DATASET_NAME" \
    --clip_vision_encoder="$CLIP_VISION_ENCODER" \
    --clip_check_pt="$CLIP_CHECKPOINT" \
    --csv="$PROMPTS_FILE" \
    --save_path="$SAVE_ROOT" \
    --tokenizers="$TOKENIZERS_DIR" \
    --cache_dir="$CACHE_DIR" \
    "${BACKEND_ARGS[@]}"
fi

if [[ "$RUN_ALIGNER" -eq 1 ]]; then
  if [[ "$BACKEND" == "legacy" ]]; then
    echo "===============learn_aligner================="
    "$PYTHON_BIN" ./src/codebase/learn_aligner.py \
      --seed="$SEED" \
      --epochs="$ALIGNER_EPOCHS" \
      --dataset="$DATASET_NAME" \
      --backend="$BACKEND" \
      --save_path="$SAVE_DIR" \
      --clf_reps_path="${SAVE_DIR}/{}_classifier_embeddings.npy" \
      --clip_reps_path="${SAVE_DIR}/{}_clip_embeddings.npy"
  else
    echo "===============learn_aligner================="
    echo "Skipping aligner training for backend=llava_mammo"
  fi
fi

if [[ "$RUN_DISCOVER" -eq 1 ]]; then
  echo "==============discover_error_slices=================="
  DISCOVER_ARGS=(
    --seed="$SEED"
    --topKsent="$TOPK_SENT"
    --dataset="$DATASET_NAME"
    --backend="$BACKEND"
    --save_path="$SAVE_DIR"
    --clf_results_csv="${SAVE_DIR}/valid_additional_info.csv"
    --language_emb_path="${SAVE_DIR}/sent_emb_word_ge_3.npy"
    --sent_path="${SAVE_DIR}/sentences_word_ge_3.pkl"
  )
  if [[ "$BACKEND" == "legacy" ]]; then
    DISCOVER_ARGS+=(
      --clf_image_emb_path="${SAVE_DIR}/valid_classifier_embeddings.npy"
      --aligner_path="${SAVE_DIR}/aligner_${ALIGNER_EPOCHS}.pth"
    )
  else
    DISCOVER_ARGS+=(
      --image_emb_path="${SAVE_DIR}/valid_clip_embeddings.npy"
    )
  fi
  "$PYTHON_BIN" ./src/codebase/discover_error_slices.py "${DISCOVER_ARGS[@]}"
fi

if [[ "$RUN_VALIDATE" -eq 1 ]]; then
  echo "=============validate_error_slices_w_LLM==================="
  VALIDATE_ARGS=(
    --seed="$SEED"
    --dataset="$DATASET_NAME"
    --backend="$BACKEND"
    --class_label="$CLASS_LABEL"
    --clip_vision_encoder="$CLIP_VISION_ENCODER"
    --key="$LLM_KEY"
    --clip_check_pt="$CLIP_CHECKPOINT"
    --top50-err-text="${SAVE_DIR}/${CLASS_LABEL}_error_top_${TOPK_SENT}_sent_diff_emb.txt"
    --save_path="$SAVE_DIR"
    --clf_results_csv="${SAVE_DIR}/{}_additional_info.csv"
    --tokenizers="$TOKENIZERS_DIR"
    --cache_dir="$CACHE_DIR"
  )
  if [[ "$APPEND_BIRADS" -eq 1 ]]; then
    VALIDATE_ARGS+=(
      --append_birads_to_passed_hypothesis
      --birads_formats "$BIRADS_FORMAT"
    )
  fi
  if [[ "$BACKEND" == "legacy" ]]; then
    VALIDATE_ARGS+=(
      --clf_image_emb_path="${SAVE_DIR}/{}_classifier_embeddings.npy"
      --aligner_path="${SAVE_DIR}/aligner_${ALIGNER_EPOCHS}.pth"
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
fi

if [[ "$RUN_MITIGATE" -eq 1 ]]; then
  echo "=============mitigate_error_slices==================="
  "$PYTHON_BIN" ./src/codebase/mitigate_error_slices.py \
    --seed="$SEED" \
    --epochs="$MITIGATE_EPOCHS" \
    --n="$MITIGATE_N" \
    --mode="last_layer_finetune" \
    --dataset="$DATASET_NAME" \
    --classifier="efficientnet-b5" \
    --slice_names="${SAVE_DIR}/${CLASS_LABEL}_prompt_dict.pkl" \
    --classifier_check_pt="$CLASSIFIER_CHECKPOINT" \
    --save_path="$SAVE_DIR" \
    --clf_results_csv="${SAVE_DIR}/{}_${CLASS_LABEL}_dataframe_mitigation.csv" \
    --clf_image_emb_path="${SAVE_DIR}/{}_classifier_embeddings.npy"
fi

echo "Done."
