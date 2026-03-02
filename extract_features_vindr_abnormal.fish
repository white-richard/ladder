#!/usr/bin/env fish

python extract_features.py \
  --data-dir "$HOME/.code/datasets/vindr-mammo" \
  --img-dir "images_png" \
  --csv-file "vindr_detection_v1_folds_abnormal.csv" \
  --clip_chk_pt_path "out_author/ViNDr/fold0/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --arch "efficientnetb5" \
  --dataset "ViNDr" \
  --split "all" \
  --label "abnormal" \
  --output-file "features/vindr_efficientnetb5_features.pt"