"""Example: Customize training parameters for RSNA training."""

from ladder_api import TrainConfig, train_classifier


def main() -> None:
    config = TrainConfig(
        data_dir="/path/to/RSNA_Breast_Imaging/Dataset",
        img_dir="RSNA_Cancer_Detection/train_images_png",
        csv_file="RSNA_Cancer_Detection/rsna_w_upmc_concepts_breast_clip.csv",
        dataset="RSNA",
        arch="tf_efficientnet_b5_ns-detect",
        epochs=12,
        epochs_warmup=2,
        num_cycles=1.0,
        batch_size=8,
        lr=3.0e-5,
        weight_decay=5.0e-5,
        print_freq=2000,
        log_freq=500,
        tensorboard_path="out/RSNA/custom",
        checkpoints="out/RSNA/custom",
        output_path="out/RSNA/custom",
        label="cancer",
    )

    # Set dry_run=False to execute the training.
    train_classifier(config, dry_run=True)


if __name__ == "__main__":
    main()
