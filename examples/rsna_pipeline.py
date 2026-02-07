"""Example: Run the full RSNA pipeline using the Python API (dry-run)."""

from ladder_api import (
    AlignerConfig,
    ErrorSliceConfig,
    ImageRepsConfig,
    LLMValidationConfig,
    LadderPipeline,
    MitigationConfig,
    PipelineConfig,
    TextRepsConfig,
    TrainConfig,
)


def main() -> None:
    train_cfg = TrainConfig(
        data_dir="Data/RSNA_Breast_Imaging/Dataset/",
        img_dir="RSNA_Cancer_Detection/train_images_png",
        csv_file="RSNA_Cancer_Detection/rsna_w_upmc_concepts_breast_clip.csv",
        dataset="RSNA",
        arch="tf_efficientnet_b5_ns-detect",
        epochs=9,
        batch_size=6,
        num_workers=0,
        lr=5.0e-5,
        weighted_BCE="y",
        balanced_dataloader="n",
        n_folds=1,
        label="cancer",
        tensorboard_path="out/RSNA/fold0",
        checkpoints="out/RSNA/fold0",
        output_path="out/RSNA/fold0",
    )

    image_reps_cfg = ImageRepsConfig(
        seed=0,
        dataset="RSNA",
        classifier="efficientnet-b5",
        classifier_check_pt="out/RSNA/fold0/efficientnetb5_seed_10_best_aucroc0.89_ver084.pth",
        flattening_type="adaptive",
        clip_vision_encoder="tf_efficientnet_b5_ns-detect",
        clip_check_pt="out/RSNA/fold0/b5-model-best-epoch-7.tar",
        data_dir="Data/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection",
        save_path="out/RSNA/fold{}",
        tokenizers="src/codebase/outputs/huggingface/tokenizers",
        cache_dir="src/codebase/outputs/huggingface/models",
    )

    text_reps_cfg = TextRepsConfig(
        seed=0,
        dataset="RSNA",
        clip_vision_encoder="tf_efficientnet_b5_ns-detect",
        clip_check_pt="out/RSNA/fold0/b5-model-best-epoch-7.tar",
        csv="data/RSNA_Cancer_Detection/mammo_rad_report.csv",
        save_path="out/RSNA/fold{}",
        tokenizers="src/codebase/outputs/huggingface/tokenizers",
        cache_dir="src/codebase/outputs/huggingface/models",
    )

    aligner_cfg = AlignerConfig(
        seed=0,
        epochs=30,
        dataset="RSNA",
        save_path="out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect",
        clf_reps_path="out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy",
        clip_reps_path="out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy",
    )

    error_slices_cfg = ErrorSliceConfig(
        seed=0,
        dataset="RSNA",
        topKsent=100,
        save_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect",
        clf_results_csv="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv",
        clf_image_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy",
        language_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/sent_emb_word_ge_3.npy",
        sent_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/sentences_word_ge_3.pkl",
        aligner_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth",
    )

    llm_cfg = LLMValidationConfig(
        seed=0,
        dataset="RSNA",
        class_label="cancer",
        clip_vision_encoder="tf_efficientnet_b5_ns-detect",
        key="",
        clip_check_pt="out/RSNA/fold0/b5-model-best-epoch-7.tar",
        top50_err_text="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_error_top_100_sent_diff_emb.txt",
        save_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect",
        clf_results_csv="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_additional_info.csv",
        clf_image_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy",
        aligner_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth",
        tokenizers="src/codebase/outputs/huggingface/tokenizers",
        cache_dir="src/codebase/outputs/huggingface/models",
    )

    mitigation_cfg = MitigationConfig(
        seed=0,
        epochs=30,
        n=75,
        mode="last_layer_finetune",
        dataset="RSNA",
        classifier="efficientnet-b5",
        slice_names="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_prompt_dict.pkl",
        classifier_check_pt="out/RSNA/fold{}/efficientnetb5_seed_10_best_aucroc0.89_ver084.pth",
        save_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect",
        clf_results_csv="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_cancer_dataframe_mitigation.csv",
        clf_image_emb_path="out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy",
    )

    pipeline_cfg = PipelineConfig(
        train=train_cfg,
        image_reps=image_reps_cfg,
        text_reps=text_reps_cfg,
        aligner=aligner_cfg,
        error_slices=error_slices_cfg,
        llm_validation=llm_cfg,
        mitigation=mitigation_cfg,
    )

    pipeline = LadderPipeline(pipeline_cfg)
    artifacts = pipeline.run(dry_run=True)
    print(artifacts)


if __name__ == "__main__":
    main()
