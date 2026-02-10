# python ./src/codebase/train_classifier_Mammo.py \
#   --data-dir '$HOME/.code/datasets/rsna/mammo_clip' \
#   --img-dir 'train_images_png' \
#   --csv-file 'rsna_w_upmc_concepts_breast_clip.csv' --start-fold 0 --n_folds 1 \
#   --dataset 'RSNA' --arch 'tf_efficientnet_b5_ns-detect' --epochs 9 --batch-size 6 --num-workers 0 \
#   --print-freq 10000 --log-freq 500 --running-interactive 'n' \
#   --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n' \
#   --tensorboard-path="out/RSNA/fold0" \
#   --checkpoints="out/RSNA/fold0" \
#   --output_path="out/RSNA/fold0" \
#   --label "cancer" \
# && echo "=============save_img_reps===================" \
# && python ./src/codebase/save_img_reps.py \
#   --seed=0 \
#   --dataset="RSNA" \
#   --classifier="efficientnet-b5" \
#   --classifier_check_pt="out/RSNA/fold0/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
#   --flattening-type="adaptive" \
#   --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
#   --clip_check_pt "model_weights/mammoClip-b5-model-best-epoch-7.tar" \
#   --data_dir="$HOME/.code/datasets/rsna/mammo_clip" \
#   --save_path="out/RSNA/fold{}/aucroc0.89" \
#   --tokenizers="$HOME/.cache/huggingface/tokenizers" \
#   --cache_dir="$HOME/.cache/huggingface/models" \
# && echo "============save_text_reps====================" \
# && python ./src/codebase/save_text_reps.py \
#   --seed=0 \
#   --dataset="RSNA" \
#   --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
#   --clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar" \
#   --csv="data/mammo_rad_report.csv" \
#   --save_path="out/RSNA/fold{}/aucroc0.89" \
#   --tokenizers="$HOME/.cache/huggingface/tokenizers" \
#   --cache_dir="$HOME/.cache/huggingface/models" \
# && echo "===============learn_aligner=================" \
# && python ./src/codebase/learn_aligner.py \
#   --seed=0 \
#   --epochs=30 \
#   --dataset="RSNA" \
#   --save_path="out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
#   --clf_reps_path="out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy" \
#   --clip_reps_path="out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy" \
# && echo "==============discover_error_slices==================" \
# && python ./src/codebase/discover_error_slices.py \
#   --seed=0 \
#   --topKsent=100 \
#   --dataset="RSNA" \
#   --save_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
#   --clf_results_csv="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv" \
#   --clf_image_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy" \
#   --language_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/sent_emb_word_ge_3.npy" \
#   --sent_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/sentences_word_ge_3.pkl" \
#   --aligner_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth" \
# && echo "=============validate_error_slices_w_LLM===================" \
# && python ./src/codebase/validate_error_slices_w_LLM.py \
#   --seed=0 \
#   --dataset="RSNA" \
#   --class_label="cancer" \
#   --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
#   --key="" \
#   --clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar" \
#   --top50-err-text="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_error_top_100_sent_diff_emb.txt" \
#   --save_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
#   --clf_results_csv="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_additional_info.csv" \
#   --clf_image_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
#   --aligner_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth" \
#   --tokenizers="$HOME/.cache/huggingface/tokenizers" \
#   --cache_dir="$HOME/.cache/huggingface/models" \
echo "" \
&& echo "=============mitigate_error_slices===================" \
&& python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --epochs=30 \
  --n=75 \
  --mode="last_layer_finetune" \
  --dataset="RSNA" \
  --classifier="efficientnet-b5" \
  --slice_names="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_prompt_dict.pkl" \
  --classifier_check_pt="out/RSNA/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --save_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_cancer_dataframe_mitigation.csv" \
  --clf_image_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy"
