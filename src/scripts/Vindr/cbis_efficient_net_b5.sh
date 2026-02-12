# echo "=============save_img_reps===================" \
# && python ./src/codebase/save_img_reps.py \
#   --seed=0 \
#   --dataset="CBIS" \
#   --classifier="efficientnet-b5" \
#   --classifier_check_pt="out/ViNDr_CBIS/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
#   --flattening-type="adaptive" \
#   --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
#   --clip_check_pt "model_weights/mammoClip-b5-model-best-epoch-7.tar" \
#   --data_dir="$HOME/.code/datasets/cbis-ddsm" \
#   --save_path="out/ViNDr_CBIS/fold{}" \
#   --tokenizers="$HOME/.cache/huggingface/tokenizers" \
#   --cache_dir="$HOME/.cache/huggingface/models" \
# && echo "============save_text_reps====================" \
# && python ./src/codebase/save_text_reps.py \
#   --seed=0 \
#   --dataset="CBIS" \
#   --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
#   --clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar" \
#   --csv="mammo_rad_report.csv" \
#   --save_path="out/ViNDr_CBIS/fold{}" \
#   --tokenizers="$HOME/.cache/huggingface/tokenizers" \
#   --cache_dir="$HOME/.cache/huggingface/models" \
# && echo "===============learn_aligner=================" \
# && python ./src/codebase/learn_aligner.py \
#   --seed=0 \
#   --epochs=30 \
#   --dataset="CBIS" \
#   --save_path="out/ViNDr_CBIS/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
#   --clf_reps_path="out/ViNDr_CBIS/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy" \
#   --clip_reps_path="out/ViNDr_CBIS/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy" \
echo "" \
&& echo "===============evaluate=================" \
&& python ./src/codebase/evaluate.py \
  --seed=0 \
  --dataset="CBIS" \
  --save_path="out/ViNDr_CBIS/fold{0}" \
  --clf_results_csv="out/ViNDr_CBIS/fold0/clip_img_encoder_tf_efficientnet_b5_ns-detect/test_additional_info.csv" \
  --split="test" \
  --pred_col="out_put_predict" \
  --threshold=0.5 \
  --precision_k 10

# CHANGING:
# Dataset dir: Vindr -> CBIS
# Any instance of: abnormal -> cancer
# Output dir: out/ViNDr -> out/ViNDr_CBIS
