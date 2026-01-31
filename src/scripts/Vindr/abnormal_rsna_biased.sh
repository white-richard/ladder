#!/bin/bash

# Get model performance on vindr abnormality with vindr biased model
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="VinDr" \
  --classifier="efficientnet-b5" \
  --classifier_check_pt="/home/richw/.code/hyp-mammo/repos/ladder/model_weights/vindr_efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt "model_weights/mammoClip-b5-model-best-epoch-7.tar" \
  --data_dir="data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0" \
  --save_path="out/ViNDr/vindr_abnormal_biased_fold{}" \
  --tokenizers "/home/richw/.cache/huggingface/tokenizers" \
  --cache_dir "/home/abnormal/.cache/huggingface" \
  --vindr_label_mode "abnormal"

# Get model performance on vindr abnormality with rsna biased model
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="VinDr" \
  --classifier="efficientnet-b5" \
  --classifier_check_pt="model_weights/rsna-b5-model-best-epoch-7.tar" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt "model_weights/mammoClip-b5-model-best-epoch-7.tar" \
  --data_dir="data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0" \
  --save_path="out/ViNDr/rsna_abnormal_biased_fold{}" \
  --tokenizers "/home/richw/.cache/huggingface/tokenizers" \
  --cache_dir "/home/richw/.cache/huggingface" \
  --vindr_label_mode "abnormal"