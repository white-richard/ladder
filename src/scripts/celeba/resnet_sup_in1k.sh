# Step 1a: Captioning using blip captioner
python ./src/codebase/caption_images.py \
  --seed=0 \
  --dataset="CelebA" \
  --img-path="data/celeba/img_align_celeba" \
  --csv="data/celeba/metadata_celeba.csv" \
  --save_csv="data/celeba/va_metadata_celeba_captioning_blip.csv" \
  --split="va" \
  --captioner="blip"

# Step 1b: Captioning using GPT-4o captioner
python ./src/codebase/caption_images_gpt_4.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --img-path="data/celeba/img_align_celeba" \
  --csv="data/celeba/metadata_celeba.csv" \
  --save_csv="data/celeba/va_metadata_celeba_captioning_GPT.csv" \
  --split="va" \
  --model="gpt-4o" \
  --api_key="<open-ai key>"


# Step 2: Save Image Reps ViT-B/32
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="CelebA" \
  --classifier="resnet_sup_in1k" \
  --classifier_check_pt="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/model.pkl" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="ViT-B/32" \
  --data_dir="data" \
  --save_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}"


# Step 3: Save Text Reps ViT-B/32 captioning
python ./src/codebase/save_text_reps.py \
  --seed=0 \
  --dataset="CelebA" \
  --clip_vision_encoder="ViT-B/32" \
  --save_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}" \
  --prompt_sent_type="captioning" \
  --captioning_type="gpt-4o" \
  --prompt_csv="data/celeba/va_metadata_celeba_captioning_GPT.csv"


# Step 4: Train aligner
python /Multimodal-mistakes-debug/src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="CelebA" \
  --save_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32" \
  --clf_reps_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_classifier_embeddings.npy" \
  --clip_reps_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_clip_embeddings.npy"


# Step 5: Retrieving Sentences Indicative of Biases captions
python ./src/codebase/discover_error_slices.py \
  --seed=0 \
  --topKsent=200 \
  --dataset="CelebA" \
  --save_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/test_additional_info.csv" \
  --clf_image_emb_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy" \
  --language_emb_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/sent_emb_captions_gpt-4o.npy" \
  --sent_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/sentences_captions_gpt-4o.pkl" \
  --aligner_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"


# Step 6: Discovering Error Slices via LLM  captions
python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --dataset="CelebA" \
  --class_label="blonde" \
  --clip_vision_encoder="ViT-B/32" \
  --key="<open-ai key>" \
  --top50-err-text="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/celebA_error_top_200_sent_diff_emb.txt" \
  --save_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_additional_info.csv" \
  --clf_image_emb_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy" \
  --aligner_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"


# Step 7: Mitigate error slices valid
python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --lr=0.001 \
  --weight_decay=0.0001 \
  --epochs=13 \
  --n=130 \
  --mode="last_layer_finetune" \
  --dataset="CelebA" \
  --classifier="resnet_sup_in1k" \
  --slice_names="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/blonde_prompt_dict.pkl" \
  --classifier_check_pt="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/model.pkl" \
  --save_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_blonde_dataframe_mitigation.csv" \
  --clf_image_emb_path="out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy"
