python ./src/codebase/train_classifier_Mammo.py \
  --data-dir 'data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0' \
  --img-dir 'images_png' \
  --csv-file 'vindr_detection_v1_folds_abnormal.csv' \
  --dataset 'ViNDr' --arch 'tf_efficientnet_b5_ns-detect' --epochs 20 --batch-size 8 --num-workers 0 \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n'  --n_folds 1  --label "abnormal" \
  --tensorboard-path="out/ViNDr/fold0" \
  --checkpoints="out/ViNDr/fold0" \
  --output_path="out/ViNDr/fold0" \
&& echo "=============save_img_reps===================" \
&& python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="VinDr" \
  --classifier="efficientnet-b5" \
  --classifier_check_pt="out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt "model_weights/mammoClip-b5-model-best-epoch-7.tar" \
  --data_dir="data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0" \
  --save_path="out/ViNDr/fold{}" \
  --tokenizers="$HOME/.cache/huggingface/tokenizers" \
  --cache_dir="$HOME/.cache/huggingface/models" \
&& echo "============save_text_reps====================" \
&& python ./src/codebase/save_text_reps.py \
  --seed=0 \
  --dataset="VinDr" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar" \
  --csv="mammo_rad_report.csv" \
  --save_path="out/ViNDr/fold{}" \
  --tokenizers="$HOME/.cache/huggingface/tokenizers" \
  --cache_dir="$HOME/.cache/huggingface/models" \
&& echo "===============learn_aligner=================" \
&& python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="VinDr" \
  --save_path="out/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_reps_path="out/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy" \
  --clip_reps_path="out/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy" \
&& echo "==============discover_error_slices==================" \
&& python ./src/codebase/discover_error_slices.py \
  --seed=0 \
  --topKsent=100 \
  --dataset="ViNDr" \
  --save_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv" \
  --clf_image_emb_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy" \
  --language_emb_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/sent_emb_word_ge_3.npy" \
  --sent_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/sentences_word_ge_3.pkl" \
  --aligner_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth" \
&& echo "=============validate_error_slices_w_LLM===================" \
&& python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --dataset="ViNDr" \
  --class_label="abnormal" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --key="" \
  --clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar" \
  --top50-err-text="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/abnormal_error_top_100_sent_diff_emb.txt" \
  --save_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_additional_info.csv" \
  --clf_image_emb_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
  --aligner_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth" \
  --tokenizers="$HOME/.cache/huggingface/tokenizers" \
  --cache_dir="$HOME/.cache/huggingface/models" \
&& echo "=============mitigate_error_slices===================" \
&& python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --epochs=30 \
  --n=75 \
  --mode="last_layer_finetune" \
  --dataset="ViNDr" \
  --classifier="efficientnet-b5" \
  --slice_names="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/abnormal_prompt_dict.pkl" \
  --classifier_check_pt="out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --save_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_abnormal_dataframe_mitigation.csv" \
  --clf_image_emb_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
  && python ./src/codebase/evaluate.py \
  --seed=0 \
  --dataset="ViNDr" \
  --save_path="out/ViNDr/fold{0}" \
  --clf_results_csv="out/ViNDr/fold0/clip_img_encoder_tf_efficientnet_b5_ns-detect/test_additional_info.csv" \
  --split="test" \
  --pred_col="out_put_predict" \
  --threshold=0.5 \
  --precision_k 10 \
  --mean_consistent_wga_slices \
  --slice_names "out/ViNDr/fold0/clip_img_encoder_tf_efficientnet_b5_ns-detect/abnormal_prompt_dict.pkl"