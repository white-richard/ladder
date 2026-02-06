"""Example: Switch to a different backbone architecture."""

from ladder_api import TrainConfig, train_classifier


def main() -> None:
    config = TrainConfig(
        data_dir="/path/to/RSNA_Breast_Imaging/Dataset",
        img_dir="RSNA_Cancer_Detection/train_images_png",
        csv_file="RSNA_Cancer_Detection/rsna_w_upmc_concepts_breast_clip.csv",
        dataset="RSNA",
        arch="swin_base_custom_norm",
        epochs=5,
        batch_size=4,
        num_workers=0,
        lr=1.0e-4,
        tensorboard_path="out/RSNA/swin",
        checkpoints="out/RSNA/swin",
        output_path="out/RSNA/swin",
        label="cancer",
    )

    # Set dry_run=False to execute the training.
    train_classifier(config, dry_run=True)


if __name__ == "__main__":
    main()
