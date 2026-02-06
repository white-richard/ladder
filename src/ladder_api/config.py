"""Typed configuration objects for the LADDER Python API."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]


def _format_with_seed(value: PathLike, seed: int) -> str:
    if value is None:
        return value
    text = str(value)
    if "{seed}" in text:
        return text.format(seed=seed)
    if "{}" in text:
        try:
            return text.format(seed)
        except Exception:
            return text
    if "{0}" in text and "{1}" not in text:
        try:
            return text.format(seed)
        except Exception:
            return text
    return text


class BaseConfig:
    """Shared helpers for config dataclasses."""

    PATH_FIELDS: Tuple[str, ...] = ()

    def resolve_paths(self):
        """Return a copy of this config with {seed} placeholders resolved."""
        updates = {}
        seed = getattr(self, "seed", 0)
        for name in self.PATH_FIELDS:
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            if value is None:
                continue
            updates[name] = _format_with_seed(value, seed)
        return replace(self, **updates)

    def to_namespace(self):
        """Convert this config into an argparse.Namespace for legacy script compatibility."""
        cfg = self.resolve_paths()
        data: dict[str, Any] = {}
        for f in dataclasses.fields(cfg):
            value = getattr(cfg, f.name)
            if isinstance(value, Path):
                value = str(value)
            data[f.name] = value
        from argparse import Namespace

        return Namespace(**data)


@dataclass
class TrainConfig(BaseConfig):
    """Configuration for RSNA/VinDr classifier training (train_classifier_Mammo.py)."""

    tensorboard_path: PathLike = "Ladder/out/RSNA/log"
    checkpoints: PathLike = "Ladder/out/RSNA/fold0"
    output_path: PathLike = "Ladder/out/RSNA/fold0"
    data_dir: PathLike = "/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/"
    img_dir: str = "RSNA_Cancer_Detection/train_images_png"
    csv_file: str = ""
    detector_threshold: float = 0.1
    pretrained_swin_encoder: str = "y"
    swin_model_type: str = "y"
    dataset: str = "RSNA_breast"
    data_frac: float = 1.0
    label: str = "cancer"
    VER: str = "084"
    arch: str = "tf_efficientnet_b5_ns"
    epochs_warmup: float = 0
    num_cycles: float = 0.5
    alpha: float = 10
    sigma: float = 15
    p: float = 1.0
    mean: float = 0.3089279
    std: float = 0.25053555408335154
    focal_alpha: float = 0.6
    focal_gamma: float = 2.0
    num_classes: int = 1
    n_folds: int = 4
    start_fold: int = 0
    seed: int = 10
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 9
    lr: float = 5.0e-5
    weight_decay: float = 1e-4
    warmup_epochs: float = 1
    img_size: Sequence[int] = field(default_factory=lambda: [1520, 912])
    device: str = "cuda"
    apex: str = "y"
    print_freq: int = 5000
    log_freq: int = 1000
    running_interactive: str = "n"
    inference_mode: str = "n"
    eval_only: bool = False
    model_type: str = "Classifier"
    weighted_BCE: str = "n"
    balanced_dataloader: str = "n"

    PATH_FIELDS = (
        "tensorboard_path",
        "checkpoints",
        "output_path",
        "data_dir",
        "img_dir",
        "csv_file",
    )


@dataclass
class ImageRepsConfig(BaseConfig):
    """Configuration for image representation saving (save_img_reps.py)."""

    dataset: str = "NIH"
    data_dir: PathLike = "./data"
    classifier: str = "resnet_sup_in1k"
    classifier_check_pt: PathLike = (
        "./out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/model.pkl"
    )
    clip_check_pt: PathLike = ""
    save_path: PathLike = (
        "./out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}"
    )
    flattening_type: str = "adaptive"
    clip_vision_encoder: str = "RN50"
    device: str = "cuda"
    seed: int = 0
    tokenizers: PathLike = ""
    cache_dir: PathLike = ""
    debug: bool = False
    vindr_label_mode: str = "abnormal"
    vindr_abnormal_birads_min: Optional[int] = None
    eval_only: bool = False

    PATH_FIELDS = (
        "data_dir",
        "classifier_check_pt",
        "clip_check_pt",
        "save_path",
        "tokenizers",
        "cache_dir",
    )


@dataclass
class TextRepsConfig(BaseConfig):
    """Configuration for text representation saving (save_text_reps.py)."""

    dataset: str = "NIH"
    save_path: PathLike = "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}"
    csv: PathLike = ""
    clip_check_pt: PathLike = ""
    clip_vision_encoder: str = "RN50"
    device: str = "cuda"
    prompt_sent_type: str = "zero-shot or captioning"
    captioning_type: str = "blip"
    prompt_csv: Optional[PathLike] = None
    seed: int = 0
    tokenizers: PathLike = ""
    cache_dir: PathLike = ""
    report_word_ge: int = 3

    PATH_FIELDS = (
        "save_path",
        "csv",
        "clip_check_pt",
        "prompt_csv",
        "tokenizers",
        "cache_dir",
    )


@dataclass
class AlignerConfig(BaseConfig):
    """Configuration for aligner training (learn_aligner.py)."""

    dataset: str = "NIH"
    save_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/aligner"
    )
    clf_reps_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_RN50/{1}_classifier_embeddings.npy"
    )
    clip_reps_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_RN50/{1}_clip_embeddings.npy"
    )
    fold: int = 0
    seed: int = 0
    epochs: int = 50
    lr: float = 0.01

    PATH_FIELDS = ("save_path", "clf_reps_path", "clip_reps_path")


@dataclass
class ErrorSliceConfig(BaseConfig):
    """Configuration for error slice discovery (discover_error_slices.py)."""

    dataset: str = "Waterbirds"
    save_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32"
    )
    clf_results_csv: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_additional_info.csv"
    )
    clf_image_emb_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy"
    )
    language_emb_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/sent_emb_word.npy"
    )
    sent_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/sentences.pkl"
    )
    aligner_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/aligner/aligner_50.pth"
    )
    topKsent: int = 20
    prediction_col: str = "out_put_predict"
    seed: int = 0

    PATH_FIELDS = (
        "save_path",
        "clf_results_csv",
        "clf_image_emb_path",
        "language_emb_path",
        "sent_path",
        "aligner_path",
    )


@dataclass
class LLMValidationConfig(BaseConfig):
    """Configuration for LLM-based error slice validation (validate_error_slices_w_LLM.py)."""

    dataset: str = "Waterbirds"
    clip_check_pt: PathLike = ""
    LLM: str = "gpt-4o"
    key: str = ""
    clip_vision_encoder: str = "swin-tiny-cxr-clip"
    class_label: str = ""
    device: str = "cuda"
    prediction_col: str = "out_put_predict"
    top50_err_text: PathLike = (
        "./Ladder/out/NIH_Cxrclip/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip/pneumothorax_error_top_50_sent_diff_emb.txt"
    )
    save_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32"
    )
    clf_results_csv: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_additional_info.csv"
    )
    clf_image_emb_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy"
    )
    aligner_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/aligner/aligner_50.pth"
    )
    tokenizers: PathLike = ""
    cache_dir: PathLike = ""
    azure_api_version: str = ""
    azure_endpoint: str = ""
    azure_deployment_name: str = ""
    seed: int = 0

    PATH_FIELDS = (
        "clip_check_pt",
        "top50_err_text",
        "save_path",
        "clf_results_csv",
        "clf_image_emb_path",
        "aligner_path",
        "tokenizers",
        "cache_dir",
    )


@dataclass
class MitigationConfig(BaseConfig):
    """Configuration for error slice mitigation (mitigate_error_slices.py)."""

    dataset: str = "Waterbirds"
    n: int = 200
    batch_size: int = 32
    classifier: str = "ResNet50"
    slice_names: PathLike = ""
    classifier_check_pt: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/model.pkl"
    )
    save_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32"
    )
    clf_results_csv: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_dataframe_mitigation.csv"
    )
    clf_image_emb_path: PathLike = (
        "./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy"
    )
    mode: str = "last_layer_retrain"
    default_hypothesis: Optional[List[str]] = None
    epochs: int = 10
    lr: float = 5.0e-5
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"
    eval_only: bool = False

    PATH_FIELDS = (
        "slice_names",
        "classifier_check_pt",
        "save_path",
        "clf_results_csv",
        "clf_image_emb_path",
    )
