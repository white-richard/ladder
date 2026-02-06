"""Example: Train RSNA classifier using the Python API."""

from ladder_api import TrainConfig, train_classifier


def main() -> None:
    config = TrainConfig(
        data_dir="/path/to/RSNA_Breast_Imaging/Dataset",
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
        tensorboard_path="out_api/RSNA/fold0",
        checkpoints="out_api/RSNA/fold0",
        output_path="out_api/RSNA/fold0",
        label="cancer",
    )

    # Set dry_run=False to execute the training.
    artifacts = train_classifier(config, dry_run=True)
    print(f"Checkpoints: {artifacts.checkpoints}")


if __name__ == "__main__":
    main()
