"""Example: Run the full VinDr pipeline using the Python API (dry-run)."""

from src.ladder_api import (
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
        data_dir="data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0",
        img_dir="images_png",
        csv_file="vindr_detection_v1_folds_abnormal.csv",
        dataset="ViNDr",
        arch="tf_efficientnet_b5_ns-detect",
        epochs=20,
        batch_size=8,
        num_workers=0,
        lr=5.0e-5,
        weighted_BCE="y",
        balanced_dataloader="n",
        n_folds=1,
        label="abnormal",
        tensorboard_path="out_api/ViNDr/fold0",
        checkpoints="out_api/ViNDr/fold0",
        output_path="out_api/ViNDr/fold0",
    )

    image_reps_cfg = ImageRepsConfig(
        seed=0,
        dataset="VinDr",
        classifier="efficientnet-b5",
        classifier_check_pt="out_api/ViNDr/fold0/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth",
        flattening_type="adaptive",
        clip_vision_encoder="tf_efficientnet_b5_ns-detect",
        clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar",
        data_dir="data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0",
        save_path="out_api/ViNDr/fold{}",
        tokenizers="src/codebase/outputs/huggingface/tokenizers",
        cache_dir="src/codebase/outputs/huggingface/models",
        eval_only=False,
    )

    text_reps_cfg = TextRepsConfig(
        seed=0,
        dataset="VinDr",
        clip_vision_encoder="tf_efficientnet_b5_ns-detect",
        clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar",
        csv="data/RSNA_Cancer_Detection/prompts.json",
        save_path="out_api/ViNDr/fold{}",
        tokenizers="src/codebase/outputs/huggingface/tokenizers",
        cache_dir="src/codebase/outputs/huggingface/models",
    )

    aligner_cfg = AlignerConfig(
        seed=0,
        epochs=30,
        dataset="VinDr",
        save_path="out_api/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect",
        clf_reps_path="out_api/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy",
        clip_reps_path="out_api/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy",
    )

    error_slices_cfg = ErrorSliceConfig(
        seed=0,
        dataset="ViNDr",
        topKsent=100,
        save_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect",
        clf_results_csv="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv",
        clf_image_emb_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy",
        language_emb_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/sent_emb_word_ge_3.npy",
        sent_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/sentences_word_ge_3.pkl",
        aligner_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth",
    )

    llm_cfg = LLMValidationConfig(
        seed=0,
        dataset="ViNDr",
        class_label="abnormal",
        clip_vision_encoder="tf_efficientnet_b5_ns-detect",
        key="",
        clip_check_pt="model_weights/mammoClip-b5-model-best-epoch-7.tar",
        top50_err_text="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/abnormal_error_top_100_sent_diff_emb.txt",
        save_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect",
        clf_results_csv="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_additional_info.csv",
        clf_image_emb_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy",
        aligner_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth",
        tokenizers="src/codebase/outputs/huggingface/tokenizers",
        cache_dir="src/codebase/outputs/huggingface/models",
    )

    mitigation_cfg = MitigationConfig(
        seed=0,
        epochs=30,
        n=75,
        mode="last_layer_finetune",
        dataset="ViNDr",
        classifier="efficientnet-b5",
        slice_names="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/abnormal_prompt_dict.pkl",
        classifier_check_pt="out_api/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth",
        save_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect",
        clf_results_csv="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_abnormal_dataframe_mitigation.csv",
        clf_image_emb_path="out_api/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy",
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
    artifacts = pipeline.run(dry_run=False)
    print(artifacts)


if __name__ == "__main__":
    main()
