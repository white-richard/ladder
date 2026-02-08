# python ./src/codebase/train_classifier_Mammo.py \
#   --data-dir '/home/richw/.code/datasets/cbis-ddsm' \
#   --img-dir '.' \
#   --csv-file 'processed_png_full/manifest_full_mammograms.csv' \
#   --dataset 'CBIS-DDSM' --arch 'tf_efficientnet_b5_ns-detect' --epochs 20 --batch-size 8 --num-workers 0 \
#   --print-freq 10000 --log-freq 500 --running-interactive 'n' \
#   --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n' --n_folds 1 --label "cancer" \
#   --tensorboard-path="out/CBIS_DDSM/fold0" \
#   --checkpoints="out/CBIS_DDSM/fold0" \
#   --output_path="out/CBIS_DDSM/fold0" \
echo "" \
&& echo "=============save_img_reps===================" \
&& python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="CBIS-DDSM" \
  --classifier="efficientnet-b5" \
  --classifier_check_pt="out/CBIS_DDSM/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt "/home/richw/.code/hyp-mammo/repos/ladder/model_weights/mammoClip-b5-model-best-epoch-7.tar" \
  --data_dir="/home/richw/.code/datasets/cbis-ddsm" \
  --save_path="out/CBIS_DDSM/fold{}" \
  --tokenizers="$HOME/.cache/huggingface/tokenizers" \
  --cache_dir="$HOME/.cache/huggingface/models" \
&& echo "============save_text_reps====================" \
&& python ./src/codebase/save_text_reps.py \
  --seed=0 \
  --dataset="CBIS-DDSM" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt="/home/richw/.code/hyp-mammo/repos/ladder/model_weights/mammoClip-b5-model-best-epoch-7.tar" \
  --csv="/home/richw/.code/hyp-mammo/repos/ladder/mammo_rad_report.csv" \
  --save_path="out/CBIS_DDSM/fold{}" \
  --tokenizers="$HOME/.cache/huggingface/tokenizers" \
  --cache_dir="$HOME/.cache/huggingface/models" \
&& echo "===============learn_aligner=================" \
&& python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="CBIS-DDSM" \
  --save_path="out/CBIS_DDSM/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_reps_path="out/CBIS_DDSM/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy" \
  --clip_reps_path="out/CBIS_DDSM/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy" \
&& echo "==============discover_error_slices==================" \
&& python ./src/codebase/discover_error_slices.py \
  --seed=0 \
  --topKsent=100 \
  --dataset="CBIS-DDSM" \
  --save_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv" \
  --clf_image_emb_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy" \
  --language_emb_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/sent_emb_word_ge_3.npy" \
  --sent_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/sentences_word_ge_3.pkl" \
  --aligner_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth" \
&& echo "=============validate_error_slices_w_LLM===================" \
&& python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --dataset="CBIS-DDSM" \
  --class_label="cancer" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --key="" \
  --clip_check_pt="/home/richw/.code/hyp-mammo/repos/ladder/model_weights/mammoClip-b5-model-best-epoch-7.tar" \
  --top50-err-text="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_error_top_100_sent_diff_emb.txt" \
  --save_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_additional_info.csv" \
  --clf_image_emb_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
  --aligner_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth" \
  --tokenizers="$HOME/.cache/huggingface/tokenizers" \
  --cache_dir="$HOME/.cache/huggingface/models" \
&& echo "=============mitigate_error_slices===================" \
&& python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --epochs=30 \
  --n=75 \
  --mode="last_layer_finetune" \
  --dataset="CBIS-DDSM" \
  --classifier="efficientnet-b5" \
  --slice_names="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_prompt_dict.pkl" \
  --classifier_check_pt="out/CBIS_DDSM/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --save_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_cancer_dataframe_mitigation.csv" \
  --clf_image_emb_path="out/CBIS_DDSM/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy"
