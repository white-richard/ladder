# !/bin/bash

# Pre pre

# python ./src/codebase/train_classifier_Mammo.py \
#   --data-dir 'data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0' \
#   --img-dir 'images_png' \
#   --csv-file 'vindr_detection_v1_folds_abnormal.csv' \
#   --dataset 'ViNDr' --arch 'tf_efficientnet_b5_ns-detect' --epochs 20 --batch-size 8 --num-workers 0 \
#   --print-freq 10000 --log-freq 500 --running-interactive 'n' \
#   --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n'  --n_folds 1  --label "abnormal" \
#   --tensorboard-path="out2/ViNDr/fold0" \
#   --checkpoints="out/ViNDr/fold0" \
#   --output_path="out2/ViNDr/fold0" \
#   --eval-only

# Vindr model debiased vindr heads abnormal
# 0.8562505645619797 acc_cancer: 0.8484848484848485
# python ./src/codebase/mitigate_error_slices.py \
#   --seed=0 \
#   --epochs=30 \
#   --n=75 \
#   --mode="last_layer_finetune" \
#   --dataset="ViNDr" \
#   --classifier="efficientnet-b5" \
#   --slice_names="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/abnormal_prompt_dict.pkl" \
#   --classifier_check_pt="out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
#   --save_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
#   --clf_results_csv="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_abnormal_dataframe_mitigation.csv" \
#   --clf_image_emb_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
#   --eval_only
############################# Overall dataset performance after mitigation: #############################
############################### Ground truth slices ########################################
# Dataset shape: (4095, 34)
# Cancer patients: (288, 35)
# Cancer patients with calc: (102, 35)
# Cancer patients without calc: (186, 35)
# Accuracy for Cancer patients without calc (error slice) after mitigation: 0.8655913978494624
# Accuracy for Cancer overall patients: 0.8993055555555556
# AUROC for overall: 0.8848001123661093
# AUROC for positives disease with calc vs all negatives: 0.9418949612942104
# AUROC for positives disease without calc vs all negatives: 0.8534900339216668

# Vindr model debiased rsna heads abnormal
# python ./src/codebase/mitigate_error_slices.py \
#   --seed=0 \
#   --n=75 \
#   --mode="last_layer_finetune" \
#   --dataset="ViNDr" \
#   --classifier="efficientnet-b5" \
#   --slice_names="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_hypothesis_dict.pkl" \
#   --classifier_check_pt="out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
#   --save_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
#   --clf_results_csv="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_abnormal_dataframe_mitigation.csv" \
#   --clf_image_emb_path="out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
#   --eval_only
############################# Overall dataset performance after mitigation: #############################
############################### Ground truth slices ########################################
# Dataset shape: (4095, 34)
# Cancer patients: (288, 35)
# Cancer patients with calc: (102, 35)
# Cancer patients without calc: (186, 35)
# Accuracy for Cancer patients without calc (error slice) after mitigation: 0.8387096774193549
# Accuracy for Cancer overall patients: 0.8854166666666666
# AUROC for overall: 0.8736200493243441
# AUROC for positives disease with calc vs all negatives: 0.9308111476794553
# AUROC for positives disease without calc vs all negatives: 0.8422571889360573



# rsna model debiased rsna heads abnormal
# Need to run save_img_reps using the rsna model to get the correct embeddings
# aucroc: 0.7333364101025718 acc_cancer: 0.45132743362831856
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="VinDr" \
  --classifier="efficientnet-b5" \
  --classifier_check_pt="out/RSNA/fold0/efficientnetb5_seed_10_best_aucroc0.89_ver084.pth" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt "out/RSNA/fold0/b5-model-best-epoch-7.tar" \
  --data_dir="data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0" \
  --save_path="out/rsnamodel_ViNDr/fold{}" \
  --tokenizers="$(pwd)/src/codebase/outputs/huggingface/tokenizers" \
  --cache_dir="$(pwd)/src/codebase/outputs/huggingface/models" \
  --vindr_abnormal_birads_min=5

# Learn Aligner Vindr
python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="VinDr" \
  --save_path="out/rsnamodel_ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_reps_path="out/rsnamodel_ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy" \
  --clip_reps_path="out/rsnamodel_ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy"


python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --n=75 \
  --mode="last_layer_finetune" \
  --dataset="ViNDr" \
  --classifier="efficientnet-b5" \
  --slice_names="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_hypothesis_dict.pkl" \
  --classifier_check_pt="out/RSNA/fold0/efficientnetb5_seed_10_best_aucroc0.89_ver084.pth" \
  --save_path="out/rsnamodel_ViNDr/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="out/rsnamodel_ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_abnormal_dataframe_mitigation.csv" \
  --clf_image_emb_path="out/rsnamodel_ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
  --eval_only

# birads >=4
############################# Overall dataset performance after mitigation: #############################
############################### Ground truth slices ########################################
# Dataset shape: (4095, 34)
# Cancer patients: (288, 35)
# Cancer patients with calc: (102, 35)
# Cancer patients without calc: (186, 35)
# Accuracy for Cancer patients without calc (error slice) after mitigation: 0.46236559139784944
# Accuracy for Cancer overall patients: 0.5833333333333334
# AUROC for overall: 0.6649291874616934
# AUROC for positives disease with calc vs all negatives: 0.8230117894281431
# AUROC for positives disease without calc vs all negatives: 0.5782387283188015

# birads >=5
############################# Overall dataset performance after mitigation: #############################
############################### Ground truth slices ########################################
# Dataset shape: (4095, 34)
# Cancer patients: (288, 35)
# Cancer patients with calc: (102, 35)
# Cancer patients without calc: (186, 35)
# Accuracy for Cancer patients without calc (error slice) after mitigation: 0.46236559139784944
# Accuracy for Cancer overall patients: 0.5833333333333334
# AUROC for overall: 0.6649282753991186
# AUROC for positives disease with calc vs all negatives: 0.8230246656056697
# AUROC for positives disease without calc vs all negatives: 0.578230254963268
