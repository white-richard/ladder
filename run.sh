# python ./src/codebase/caption_images_gpt_4.py \
#   --seed=0 \
#   --dataset="Waterbirds" \
#   --img-path="data/waterbirds" \
#   --csv="data/waterbirds/metadata_waterbirds.csv" \
#   --save_csv="data/waterbirds/va_metadata_waterbirds_captioning_GPT.csv" \
#   --split="va" \
#   --model="gpt-4o" \
#   --api_key=""

  ### Step1: Save image representations of the image classifier and vision encoder from vision language representation space
 echo "step 1"
  python ./src/codebase/save_img_reps.py \
    --seed=0 \
    --dataset="Waterbirds" \
    --classifier="resnet_sup_in1k" \
    --classifier_check_pt="waterbird_model.pkl" \
    --flattening-type="adaptive" \
    --clip_vision_encoder="ViT-B/32" \
    --data_dir="data" \
    --save_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}"

  ### Step2: Save text representations text encoder from vision language representation space
 echo "step 2"
  python ./src/codebase/save_text_reps.py \
    --seed=0 \
    --dataset="Waterbirds" \
    --clip_vision_encoder="ViT-B/32" \
    --save_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}" \
    --prompt_sent_type="captioning" \
    --captioning_type="gpt-4o" \
    --prompt_csv="data/waterbirds/va_metadata_waterbirds_captioning_GPT.csv"

  ### Step3: Train aligner to align the classifier and vision language image representations
 echo "step 3"
  python ./src/codebase/learn_aligner.py \
    --seed=0 \
    --epochs=30 \
    --dataset="Waterbirds" \
    --save_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32" \
    --clf_reps_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_classifier_embeddings.npy" \
    --clip_reps_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_clip_embeddings.npy"

  ### Step4: Retrieving sentences indicative of biases
 echo "step 4"
  python ./src/codebase/discover_error_slices.py \
    --seed=0 \
    --topKsent=200 \
    --dataset="Waterbirds" \
    --save_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
    --clf_results_csv="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/test_additional_info.csv" \
    --clf_image_emb_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy" \
    --language_emb_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/sent_emb_captions_gpt-4o.npy" \
    --sent_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/sentences_captions_gpt-4o.pkl" \
    --aligner_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"

  ### Step5: Discovering error slices via LLM-driven hypothesis generation
  echo "step 5"
  python ./src/codebase/validate_error_slices_w_LLM.py \
    --seed=0 \
    --LLM="gpt-4o" \
    --dataset="Waterbirds" \
    --class_label="waterbirds" \
    --clip_vision_encoder="ViT-B/32" \
    --key="" \
    --top50-err-text="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/landbirds_error_top_200_sent_diff_emb.txt" \
    --save_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
    --clf_results_csv="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_additional_info.csv" \
    --clf_image_emb_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy" \
    --aligner_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"

  #### Step6: Mitigate multi-bias w/o annotation
  echo "step 6"
  python ./src/codebase/mitigate_error_slices.py \
    --seed=0 \
    --epochs=9 \
    --lr=0.001 \
    --weight_decay=0.0001 \
    --n=600 \
    --mode="last_layer_finetune" \
    --dataset="Waterbirds" \
    --classifier="resnet_sup_in1k" \
    --slice_names="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_prompt_dict.pkl" \
    --classifier_check_pt="waterbird_model.pkl" \
    --save_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
    --clf_results_csv="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_{}_dataframe_mitigation.csv" \
    --clf_image_emb_path="out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy"
