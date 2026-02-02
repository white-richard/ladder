# !/bin/bash

# Rsna debiased heads
python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --epochs=30 \
  --n=75 \
  --mode="last_layer_finetune" \
  --dataset="RSNA" \
  --classifier="efficientnet-b5" \
  --slice_names="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_prompt_dict.pkl" \
  --classifier_check_pt="out/RSNA/fold{}/efficientnetb5_seed_10_best_aucroc0.89_ver084.pth" \
  --save_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_cancer_dataframe_mitigation.csv" \
  --clf_image_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
  --eval_only