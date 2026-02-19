import argparse
import gc
import math
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from Classifiers.experiments_RSNA import _build_cbis_splits, stratified_sample
from Classifiers.models.Efficient_net_custom import EfficientNet
from mammo_metrics import normalize_mammo_dataset_name
from med_img_datasets_clf.feature_label_dataset import Feature_label_dataset
from med_img_datasets_clf.dataset_utils import get_dataloader_mammo
from utils import seed_all

warnings.filterwarnings("ignore")


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard-path", metavar="DIR", default="out/RSNA/log")
    parser.add_argument("--checkpoints", metavar="DIR", default="out/RSNA/fold0")
    parser.add_argument("--output_path", "--output-path", dest="output_path", metavar="DIR", default="out/RSNA/fold0")
    parser.add_argument("--data-dir", default="", type=str)
    parser.add_argument(
        "--img-dir",
        default="RSNA_Cancer_Detection/train_images_png",
        type=str,
        help="Relative image root under --data-dir for rsna/vindr format.",
    )
    parser.add_argument(
        "--csv-path",
        "--csv-file",
        dest="csv_path",
        required=True,
        type=str,
        help="Path to manifest CSV.",
    )
    parser.add_argument(
        "--label",
        "--label-column",
        dest="label",
        required=True,
        type=str,
        help="Target CSV column name.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        "--classifier_check_pt",
        dest="pretrained_checkpoint",
        required=True,
        type=str,
        help="Mammography pretrained classifier checkpoint (.pth) used to initialize encoder weights.",
    )
    parser.add_argument(
        "--task-type",
        default="auto",
        choices=["auto", "binary", "multiclass"],
        help="`auto` infers from label cardinality; use `multiclass` for BI-RADS-style targets.",
    )
    parser.add_argument("--dataset", default="rsna", type=str)
    parser.add_argument(
        "--rsna-train-folds",
        default="1,2",
        type=str,
        help="Comma-separated RSNA fold values to train on (column: `fold`).",
    )
    parser.add_argument(
        "--rsna-eval-folds",
        default="3",
        type=str,
        help="Comma-separated RSNA fold values to evaluate on (column: `fold`).",
    )
    parser.add_argument(
        "--vindr-train-splits",
        default="train",
        type=str,
        help="Comma-separated ViNDr split labels to train on (column: `split_new`).",
    )
    parser.add_argument(
        "--vindr-eval-splits",
        default="val",
        type=str,
        help="Comma-separated ViNDr split labels to evaluate on (column: `split_new`).",
    )
    parser.add_argument(
        "--cbis-train-splits",
        default="train",
        type=str,
        help="Comma-separated CBIS splits to train on from {train,val,test}. Alias `valid` is accepted.",
    )
    parser.add_argument(
        "--cbis-eval-splits",
        default="val",
        type=str,
        help="Comma-separated CBIS splits to evaluate on from {train,val,test}. Alias `valid` is accepted.",
    )
    parser.add_argument("--data-frac", default=1.0, type=float)
    parser.add_argument("--VER", default="084", type=str)
    parser.add_argument("--arch", default="tf_efficientnet_b5_ns", type=str)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument(
        "--num-classes",
        default=0,
        type=int,
        help="If >0, must match inferred label cardinality exactly.",
    )
    parser.add_argument("--n_folds", "--n-folds", dest="n_folds", default=4, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--freeze-epochs", default=1, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--img-size", nargs="+", default=[1520, 912])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--print-freq", default=5000, type=int)
    parser.add_argument("--log-freq", default=1000, type=int)
    parser.add_argument("--running-interactive", default="n", type=str)
    parser.add_argument("--inference-mode", default="n", type=str)
    parser.add_argument("--model-type", default="Classifier", type=str)
    parser.add_argument("--weighted-BCE", default="y", type=str)
    parser.add_argument("--balanced-dataloader", default="n", type=str)
    parser.add_argument("--smoke-test", default="n", type=str)
    parser.add_argument("--cbis-val-ratio", default=0.1, type=float)
    parser.add_argument(
        "--precompute-features",
        default="n",
        type=str,
        choices=["y", "n"],
        help="If `y`, precompute pooled encoder features and train/eval using cached feature dataloaders during frozen epochs.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default="",
        type=str,
        help="Directory for saved feature caches. Defaults to <output_path>/feature_cache when empty.",
    )
    parser.add_argument(
        "--force-recompute-features",
        default="n",
        type=str,
        choices=["y", "n"],
        help="If `y`, ignore existing feature cache files and recompute.",
    )
    return parser.parse_args()


def _validate_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {df_name}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _parse_csv_tokens(raw_value):
    if raw_value is None:
        return []
    tokens = [tok.strip() for tok in str(raw_value).split(",")]
    return [tok for tok in tokens if tok]


def _parse_int_selector(raw_value, arg_name):
    tokens = _parse_csv_tokens(raw_value)
    if not tokens:
        raise ValueError(f"`{arg_name}` must include at least one integer value.")
    try:
        values = [int(tok) for tok in tokens]
    except ValueError as exc:
        raise ValueError(f"`{arg_name}` must be a comma-separated list of integers. Got: {raw_value}") from exc
    return list(dict.fromkeys(values))


def _parse_choice_selector(raw_value, arg_name, allowed_values):
    tokens = [tok.lower() for tok in _parse_csv_tokens(raw_value)]
    if not tokens:
        raise ValueError(f"`{arg_name}` must include at least one split name.")
    unknown = sorted(set(tokens) - set(allowed_values))
    if unknown:
        raise ValueError(
            f"`{arg_name}` has unsupported split(s): {unknown}. Allowed values: {sorted(set(allowed_values))}"
        )
    return list(dict.fromkeys(tokens))


def _assert_disjoint_train_eval(train_selector, eval_selector, train_arg_name, eval_arg_name):
    overlap = sorted(set(train_selector).intersection(set(eval_selector)))
    if overlap:
        raise ValueError(
            f"Train/eval selectors overlap: {overlap}. "
            f"Please provide disjoint values for `{train_arg_name}` and `{eval_arg_name}` to avoid leakage."
        )


def _normalize_cbis_split_name(split_name):
    key = str(split_name).strip().lower()
    return "val" if key == "valid" else key


def _select_and_concat_named_splits(split_map, selected_names):
    selected_frames = [split_map[name] for name in selected_names]
    if len(selected_frames) == 1:
        return selected_frames[0].copy().reset_index(drop=True)
    return pd.concat(selected_frames, axis=0, ignore_index=True).reset_index(drop=True)


def _resolve_label_column(df, requested_label):
    # Exact match first.
    if requested_label in df.columns:
        return requested_label

    # Case-insensitive exact match.
    lower_to_cols = {}
    for col in df.columns:
        lower_to_cols.setdefault(str(col).lower(), []).append(col)
    key = str(requested_label).lower()
    if key in lower_to_cols and len(lower_to_cols[key]) == 1:
        return lower_to_cols[key][0]

    # Common BI-RADS aliases across RSNA/VinDr/CBIS manifests.
    if key in {"breast_birads", "birads"}:
        for candidate in [
            "breast_birads",
            "BIRADS",
            "birads",
            "birads_assessment",
            "birads_category",
            "assessment",
        ]:
            if candidate in df.columns:
                return candidate
        for col in df.columns:
            if "birads" in str(col).lower():
                return col

    # Common breast-density aliases.
    if key in {"breast_density", "density"}:
        for candidate in ["breast_density", "density"]:
            if candidate in df.columns:
                return candidate

    # Generic prefix fallback: breast_x -> x.
    if key.startswith("breast_"):
        short_key = key.replace("breast_", "", 1)
        if short_key in lower_to_cols and len(lower_to_cols[short_key]) == 1:
            return lower_to_cols[short_key][0]

    raise ValueError(
        f"Label column `{requested_label}` was not found. Available columns: {list(df.columns)}"
    )


def _encode_label_column(df, label_col, df_name):
    resolved_label_col = _resolve_label_column(df, label_col)
    if resolved_label_col != label_col:
        print(f"Resolved label column `{label_col}` -> `{resolved_label_col}`")
    label_col = resolved_label_col

    raw = df[label_col]
    if raw.isna().any():
        bad_count = int(raw.isna().sum())
        raise ValueError(f"Label column `{label_col}` contains {bad_count} missing values in {df_name}.")

    df[f"{label_col}_raw"] = raw.copy()

    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().all():
        if not np.allclose(numeric.values, np.round(numeric.values)):
            raise ValueError(
                f"Label column `{label_col}` must be integer-like for classification in {df_name}."
            )
        values = numeric.astype(int)
    else:
        values = raw.astype(str).str.strip()

    unique_values = sorted(values.unique().tolist())
    if len(unique_values) < 2:
        raise ValueError(f"Label column `{label_col}` must have at least 2 distinct classes. Found: {unique_values}")

    value_to_index = {v: i for i, v in enumerate(unique_values)}
    df[label_col] = values.map(value_to_index).astype(np.int64)
    return label_col, unique_values, value_to_index


def _resolve_task_type(args):
    inferred_binary = args.num_classes == 2
    if args.task_type == "auto":
        return "binary" if inferred_binary else "multiclass"
    if args.task_type == "binary" and not inferred_binary:
        raise ValueError(
            f"--task-type binary requires exactly 2 classes. Found {args.num_classes}: {args.label_values}"
        )
    if args.task_type == "multiclass" and args.num_classes < 2:
        raise ValueError("--task-type multiclass requires at least 2 classes.")
    return args.task_type


def _validate_split_class_coverage(args):
    expected = set(range(args.num_classes))
    train_classes = set(args.train_folds[args.label].astype(int).unique().tolist())
    valid_classes = set(args.valid_folds[args.label].astype(int).unique().tolist())
    if train_classes != expected:
        raise ValueError(
            f"Training split class coverage mismatch. Expected {sorted(expected)}, got {sorted(train_classes)}."
        )
    if valid_classes != expected:
        missing = sorted(expected - valid_classes)
        print(
            "Warning: validation split is missing encoded class indices "
            f"{missing}. Metrics will be computed on classes present in validation."
        )


def _macro_auc_ovr_present_classes(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    present_classes = sorted(np.unique(y_true).tolist())
    if len(present_classes) < 2:
        raise ValueError(
            "Validation labels have fewer than 2 classes after aggregation; AUROC is undefined."
        )

    aucs = []
    for cls_idx in present_classes:
        binary_true = (y_true == cls_idx).astype(int)
        if binary_true.min() == binary_true.max():
            continue
        aucs.append(roc_auc_score(binary_true, y_score[:, cls_idx]))

    if not aucs:
        raise ValueError("Could not compute multiclass AUROC on present classes.")
    return float(np.mean(aucs)), present_classes


def _compute_pos_weight(train_df, label_col):
    pos = int((train_df[label_col] == 1).sum())
    neg = int((train_df[label_col] == 0).sum())
    if pos == 0:
        raise ValueError("Training split has zero positive samples; cannot compute BCE pos_weight.")
    if neg == 0:
        raise ValueError("Training split has zero negative samples; cannot compute BCE pos_weight.")
    return float(neg / pos)


def _load_cbis_manifest_strict(args):
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "patient_id" not in df.columns and "subject_id" in df.columns:
        subject_parts = (
            df["subject_id"]
            .astype(str)
            .str.strip()
            .str.extract(r"^(?:Calc|Mass)-(?:Training|Test)_(.+)_(LEFT|RIGHT)_(CC|MLO)$")
        )
        df["patient_id"] = subject_parts[0]
        if "laterality" not in df.columns:
            df["laterality"] = subject_parts[1]

    if "laterality" in df.columns:
        mapping = {"L": 0, "R": 1, "LEFT": 0, "RIGHT": 1, 0: 0, 1: 1}
        df["laterality"] = df["laterality"].map(lambda x: mapping.get(str(x).strip().upper(), x))
        df["laterality"] = pd.to_numeric(df["laterality"], errors="coerce")
    else:
        raise ValueError("CBIS CSV must include `laterality` or parseable `subject_id`.")

    _validate_columns(df, ["png_path", "patient_id", "laterality"], str(csv_path))
    if "fold" not in df.columns:
        df["fold"] = 0
    return df


def _load_and_split_manifest(args):
    dataset_name = normalize_mammo_dataset_name(args.dataset)
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if dataset_name == "rsna":
        df = pd.read_csv(csv_path)
        _validate_columns(df, ["patient_id", "image_id", "laterality", "fold"], str(csv_path))
        df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
        # MammoDataset appends ".png" for RSNA paths; normalize if CSV already includes extension.
        image_id_series = df["image_id"].astype(str).str.strip()
        had_png_suffix = image_id_series.str.lower().str.endswith(".png")
        if had_png_suffix.any():
            df["image_id"] = image_id_series.str.replace(r"\.png$", "", regex=True)
            print(f"Normalized RSNA image_id by stripping .png suffix for {int(had_png_suffix.sum())} rows.")
        resolved_label, label_values, value_to_index = _encode_label_column(df, args.label, str(csv_path))
        args.label = resolved_label
        args.df = df
        available_fold_values = sorted(df["fold"].unique().tolist())
        train_fold_values = _parse_int_selector(args.rsna_train_folds, "--rsna-train-folds")
        eval_fold_values = _parse_int_selector(args.rsna_eval_folds, "--rsna-eval-folds")
        _assert_disjoint_train_eval(train_fold_values, eval_fold_values, "--rsna-train-folds", "--rsna-eval-folds")

        missing_train = sorted(set(train_fold_values) - set(available_fold_values))
        missing_eval = sorted(set(eval_fold_values) - set(available_fold_values))
        if missing_train or missing_eval:
            raise ValueError(
                "Requested RSNA fold(s) not present in CSV. "
                f"Missing train folds: {missing_train}, missing eval folds: {missing_eval}, "
                f"available folds: {available_fold_values}"
            )

        args.train_folds = df[df["fold"].isin(train_fold_values)].reset_index(drop=True)
        args.valid_folds = df[df["fold"].isin(eval_fold_values)].reset_index(drop=True)
        print(f"RSNA split selection -> train folds: {train_fold_values}, eval folds: {eval_fold_values}")

    elif dataset_name == "vindr":
        df = pd.read_csv(csv_path)
        _validate_columns(df, ["patient_id", "image_id", "split_new"], str(csv_path))
        if "ImageLateralityFinal" in df.columns and "laterality" not in df.columns:
            df = df.rename(columns={"ImageLateralityFinal": "laterality"})
        _validate_columns(df, ["laterality"], str(csv_path))
        split_series = df["split_new"].astype(str).str.strip().str.lower()
        available_splits = sorted(split_series.unique().tolist())
        train_splits = _parse_choice_selector(args.vindr_train_splits, "--vindr-train-splits", available_splits)
        eval_splits = _parse_choice_selector(args.vindr_eval_splits, "--vindr-eval-splits", available_splits)
        _assert_disjoint_train_eval(train_splits, eval_splits, "--vindr-train-splits", "--vindr-eval-splits")

        resolved_label, label_values, value_to_index = _encode_label_column(df, args.label, str(csv_path))
        args.label = resolved_label
        args.df = df
        args.train_folds = df[split_series.isin(train_splits)].reset_index(drop=True)
        args.valid_folds = df[split_series.isin(eval_splits)].reset_index(drop=True)
        args.test_folds = df[split_series == "test"].reset_index(drop=True)
        print(f"ViNDr split selection -> train: {train_splits}, eval: {eval_splits}")

    elif dataset_name == "cbis":
        df = _load_cbis_manifest_strict(args)
        resolved_label, label_values, value_to_index = _encode_label_column(df, args.label, str(csv_path))
        args.label = resolved_label
        args.df = df
        train_df, valid_df, test_df = _build_cbis_splits(
            df,
            seed=getattr(args, "seed", 42),
            val_ratio=getattr(args, "cbis_val_ratio", 0.1),
        )
        allowed_cbis = ["train", "val", "valid", "test"]
        selected_train = _parse_choice_selector(args.cbis_train_splits, "--cbis-train-splits", allowed_cbis)
        selected_eval = _parse_choice_selector(args.cbis_eval_splits, "--cbis-eval-splits", allowed_cbis)
        selected_train = [_normalize_cbis_split_name(name) for name in selected_train]
        selected_eval = [_normalize_cbis_split_name(name) for name in selected_eval]
        _assert_disjoint_train_eval(selected_train, selected_eval, "--cbis-train-splits", "--cbis-eval-splits")

        cbis_split_map = {
            "train": train_df,
            "val": valid_df,
            "test": test_df,
        }
        args.train_folds = _select_and_concat_named_splits(cbis_split_map, selected_train)
        args.valid_folds = _select_and_concat_named_splits(cbis_split_map, selected_eval)
        args.test_folds = test_df.reset_index(drop=True)
        print(f"CBIS split selection -> train: {selected_train}, eval: {selected_eval}")
    else:
        raise ValueError(
            f"Unsupported mammography dataset `{args.dataset}`. Use one of: rsna, vindr, cbis, cbis-ddsm."
        )

    if args.train_folds.empty:
        raise ValueError("Training split is empty after CSV parsing.")
    if args.valid_folds.empty:
        raise ValueError("Validation split is empty after CSV parsing.")

    if "fold" not in args.train_folds.columns:
        args.train_folds["fold"] = int(args.cur_fold)
    if "fold" not in args.valid_folds.columns:
        args.valid_folds["fold"] = int(args.cur_fold)
    if hasattr(args, "test_folds") and "fold" not in args.test_folds.columns:
        args.test_folds["fold"] = int(args.cur_fold)
    args.label_values = label_values
    args.label_value_to_index = value_to_index
    inferred_num_classes = len(label_values)
    if args.num_classes > 0 and args.num_classes != inferred_num_classes:
        raise ValueError(
            f"--num-classes ({args.num_classes}) does not match inferred label classes "
            f"({inferred_num_classes}): {label_values}"
        )
    args.num_classes = inferred_num_classes
    args.task_type = _resolve_task_type(args)

    _validate_split_class_coverage(args)


def _set_encoder_requires_grad(model, requires_grad):
    for name, param in model.named_parameters():
        if name.startswith("_fc."):
            continue
        param.requires_grad = requires_grad


def _load_pretrained_encoder_weights(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    encoder_state = {k: v for k, v in state_dict.items() if not k.startswith("_fc.")}
    load_res = model.load_state_dict(encoder_state, strict=False)

    missing = set(load_res.missing_keys)
    unexpected = set(load_res.unexpected_keys)
    allowed_missing = {"_fc.weight", "_fc.bias"}
    illegal_missing = missing - allowed_missing
    if illegal_missing:
        raise ValueError(
            f"Checkpoint is missing required encoder keys: {sorted(illegal_missing)} (file: {checkpoint_path})"
        )
    if unexpected:
        raise ValueError(f"Checkpoint has unexpected keys: {sorted(unexpected)} (file: {checkpoint_path})")

    print(f"Loaded mammography-pretrained encoder weights from: {checkpoint_path}")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _cuda_stats_for_postfix(device):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return "n/a", "n/a"
    return f"{torch.cuda.memory_usage(device)}%", f"{torch.cuda.utilization(device)}%"


def _forward_from_pooled_features(model, pooled_features):
    x = pooled_features
    if hasattr(model, "_dropout"):
        x = model._dropout(x)
    return model._fc(x)


def _cache_file_path(args, split):
    label_tag = str(args.label).replace("/", "_")
    ckpt_tag = Path(args.pretrained_checkpoint).stem
    return (
        args.feature_cache_dir
        / f"fold{args.cur_fold}_{label_tag}_{args.task_type}_k{args.num_classes}_{ckpt_tag}_{split}_features.pt"
    )


def _precompute_features_if_needed(args, model, image_loader, split, device):
    cache_path = _cache_file_path(args, split)
    if cache_path.exists() and not args.force_recompute_features:
        payload = torch.load(cache_path, map_location="cpu")
        if "features" not in payload or "labels" not in payload or "meta" not in payload:
            raise ValueError(f"Invalid cached feature file format: {cache_path}")
        meta = payload["meta"]
        expected_meta = {
            "task_type": args.task_type,
            "num_classes": args.num_classes,
            "label": args.label,
            "pretrained_checkpoint": str(args.pretrained_checkpoint),
            "split": split,
        }
        if meta != expected_meta:
            raise ValueError(
                f"Feature cache metadata mismatch for {cache_path}.\nExpected: {expected_meta}\nFound: {meta}"
            )
        print(f"Loaded cached {split} features from {cache_path}")
        return payload["features"], payload["labels"]

    # Feature extraction should cover full split once; do not drop samples.
    extraction_loader = DataLoader(
        image_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    features = []
    labels = []
    progress_iter = tqdm(
        enumerate(extraction_loader),
        desc=f"[feature-precompute {split}]",
        total=len(extraction_loader),
    )
    for _, data in progress_iter:
        inputs = data["x"].to(device)
        inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        with torch.no_grad():
            encoded = model.extract_features(inputs)
            pooled = model._avg_pooling(encoded).flatten(start_dim=1)
        features.append(pooled.to("cpu"))
        if args.task_type == "binary":
            labels.append(data["y"].float().to("cpu"))
        else:
            labels.append(data["y"].long().to("cpu"))

    feature_tensor = torch.cat(features, dim=0).contiguous()
    label_tensor = torch.cat(labels, dim=0).contiguous()
    if feature_tensor.shape[0] != label_tensor.shape[0]:
        raise ValueError(
            f"Precomputed feature count ({feature_tensor.shape[0]}) and label count ({label_tensor.shape[0]}) mismatch."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": feature_tensor,
            "labels": label_tensor,
            "meta": {
                "task_type": args.task_type,
                "num_classes": args.num_classes,
                "label": args.label,
                "pretrained_checkpoint": str(args.pretrained_checkpoint),
                "split": split,
            },
        },
        cache_path,
    )
    print(f"Saved cached {split} features to {cache_path}")
    return feature_tensor, label_tensor


def _get_feature_dataloaders(args, model, train_loader, valid_loader, device):
    if args.balanced_dataloader == "y":
        raise ValueError("`--balanced-dataloader y` is not supported with `--precompute-features y`.")

    train_features, train_labels = _precompute_features_if_needed(args, model, train_loader, "train", device)
    valid_features, valid_labels = _precompute_features_if_needed(args, model, valid_loader, "valid", device)

    train_dataset = Feature_label_dataset(train_features, train_labels)
    valid_dataset = Feature_label_dataset(valid_features, valid_labels)

    train_feature_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_feature_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print("---------------------------------------")
    print("Train Feature Loader: ", len(train_feature_loader))
    print("Valid Feature Loader: ", len(valid_feature_loader))
    print("Train feature dataset", len(train_dataset))
    print("Valid feature dataset", len(valid_dataset))
    print("---------------------------------------")
    return train_feature_loader, valid_feature_loader


def train_fn(train_loader, model, criterion, optimizer, epoch, args, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = time.time()

    progress_iter = tqdm(
        enumerate(train_loader),
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
        total=len(train_loader),
    )
    for step, data in progress_iter:
        inputs = data["x"].to(device)
        inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        batch_size = inputs.size(0)

        with torch.cuda.amp.autocast(enabled=args.apex):
            y_preds = model(inputs)
            if args.task_type == "binary":
                labels = data["y"].float().to(device)
                loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
            else:
                labels = data["y"].long().to(device)
                loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        cuda_mem, cuda_util = _cuda_stats_for_postfix(device)
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]["lr"]],
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": cuda_mem,
                "CUDA-Util": cuda_util,
            }
        )

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "LR: {lr:.8f}".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar("train/epoch", epoch, index)
            logger.add_scalar("train/iter_loss", losses.avg, index)
            logger.add_scalar("train/iter_lr", optimizer.param_groups[0]["lr"], index)

    return losses.avg


def valid_fn(valid_loader, model, criterion, args, device, epoch=1, logger=None):
    losses = AverageMeter()
    model.eval()
    start = time.time()

    binary_scores = []
    multi_probs = []

    progress_iter = tqdm(
        enumerate(valid_loader),
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
        total=len(valid_loader),
    )
    for step, data in progress_iter:
        inputs = data["x"].to(device)
        batch_size = inputs.size(0)
        inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        with torch.no_grad():
            y_preds = model(inputs)
            if args.task_type == "binary":
                labels = data["y"].float().to(device)
                loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
                binary_scores.append(y_preds.squeeze(1).sigmoid().to("cpu").numpy())
            else:
                labels = data["y"].long().to(device)
                loss = criterion(y_preds, labels)
                probs = torch.softmax(y_preds, dim=1).to("cpu").numpy()
                multi_probs.append(probs)

        losses.update(loss.item(), batch_size)

        cuda_mem, cuda_util = _cuda_stats_for_postfix(device)
        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": cuda_mem,
                "CUDA-Util": cuda_util,
            }
        )

        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
            )

        if (step % args.log_freq == 0 or step == (len(valid_loader) - 1)) and logger is not None:
            index = step + len(valid_loader) * epoch
            logger.add_scalar("valid/iter_loss", losses.avg, index)

    if args.task_type == "binary":
        scores = np.concatenate(binary_scores)
        return losses.avg, scores, scores[:, None]

    probs = np.concatenate(multi_probs, axis=0)
    pred_class = np.argmax(probs, axis=1)
    return losses.avg, pred_class, probs


def train_fn_features(feature_loader, model, criterion, optimizer, epoch, args, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = time.time()

    progress_iter = tqdm(
        enumerate(feature_loader),
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train cached]",
        total=len(feature_loader),
    )
    for step, (features, labels) in progress_iter:
        features = features.to(device)
        batch_size = features.size(0)
        with torch.cuda.amp.autocast(enabled=args.apex):
            y_preds = _forward_from_pooled_features(model, features)
            if args.task_type == "binary":
                labels = labels.float().to(device)
                loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
            else:
                labels = labels.long().to(device)
                loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        cuda_mem, cuda_util = _cuda_stats_for_postfix(device)
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]["lr"]],
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": cuda_mem,
                "CUDA-Util": cuda_util,
            }
        )

        if step % args.print_freq == 0 or step == (len(feature_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "LR: {lr:.8f}".format(
                    epoch + 1,
                    step,
                    len(feature_loader),
                    remain=timeSince(start, float(step + 1) / len(feature_loader)),
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

        if step % args.log_freq == 0 or step == (len(feature_loader) - 1):
            index = step + len(feature_loader) * epoch
            logger.add_scalar("train_cached/epoch", epoch, index)
            logger.add_scalar("train_cached/iter_loss", losses.avg, index)
            logger.add_scalar("train_cached/iter_lr", optimizer.param_groups[0]["lr"], index)

    return losses.avg


def valid_fn_features(feature_loader, model, criterion, args, device, epoch=1, logger=None):
    losses = AverageMeter()
    model.eval()
    start = time.time()
    binary_scores = []
    multi_probs = []

    progress_iter = tqdm(
        enumerate(feature_loader),
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid cached]",
        total=len(feature_loader),
    )
    for step, (features, labels) in progress_iter:
        features = features.to(device)
        batch_size = features.size(0)

        with torch.no_grad():
            y_preds = _forward_from_pooled_features(model, features)
            if args.task_type == "binary":
                labels = labels.float().to(device)
                loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
                binary_scores.append(y_preds.squeeze(1).sigmoid().to("cpu").numpy())
            else:
                labels = labels.long().to(device)
                loss = criterion(y_preds, labels)
                multi_probs.append(torch.softmax(y_preds, dim=1).to("cpu").numpy())

        losses.update(loss.item(), batch_size)

        cuda_mem, cuda_util = _cuda_stats_for_postfix(device)
        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": cuda_mem,
                "CUDA-Util": cuda_util,
            }
        )

        if step % args.print_freq == 0 or step == (len(feature_loader) - 1):
            print(
                f"EVAL: [{step}/{len(feature_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(feature_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
            )

        if (step % args.log_freq == 0 or step == (len(feature_loader) - 1)) and logger is not None:
            index = step + len(feature_loader) * epoch
            logger.add_scalar("valid_cached/iter_loss", losses.avg, index)

    if args.task_type == "binary":
        scores = np.concatenate(binary_scores)
        return losses.avg, scores, scores[:, None]

    probs = np.concatenate(multi_probs, axis=0)
    pred_class = np.argmax(probs, axis=1)
    return losses.avg, pred_class, probs


def _aggregate_binary(df, args):
    dataset_name = normalize_mammo_dataset_name(args.dataset)
    if dataset_name in {"vindr", "cbis"}:
        return df.copy()
    if dataset_name == "rsna":
        return (
            df[["patient_id", "laterality", args.label, "prediction", "fold"]]
            .groupby(["patient_id", "laterality"])
            .mean()
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _aggregate_multiclass(df, args, prob_cols):
    dataset_name = normalize_mammo_dataset_name(args.dataset)
    if dataset_name in {"vindr", "cbis"}:
        return df[[args.label] + prob_cols].copy()
    if dataset_name == "rsna":
        probs = df[["patient_id", "laterality"] + prob_cols].groupby(["patient_id", "laterality"]).mean()
        labels = (
            df[["patient_id", "laterality", args.label]]
            .groupby(["patient_id", "laterality"])[args.label]
            .agg(lambda x: x.mode().iloc[0])
        )
        merged = probs.copy()
        merged[args.label] = labels
        return merged
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def train_loop(args, device):
    print(f"\n================== fold: {args.cur_fold} training ======================")
    args.train_folds = args.train_folds.sample(frac=args.data_frac, random_state=1, ignore_index=True)
    args.image_encoder_type = None

    if args.running_interactive and args.task_type == "binary":
        args.train_folds = stratified_sample(args.train_folds, 100, label=args.label)
        args.valid_folds = stratified_sample(args.valid_folds, 100, label=args.label)

    train_loader, valid_loader = get_dataloader_mammo(args)

    out_dim = 1 if args.task_type == "binary" else args.num_classes
    model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=out_dim)
    _load_pretrained_encoder_weights(model, args.pretrained_checkpoint)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.epochs_warmup,
        num_training_steps=args.epochs,
        num_cycles=args.num_cycles,
    )

    model = model.to(device)
    logger = SummaryWriter(args.tb_logs_path / f"fold{args.cur_fold}")

    if args.task_type == "binary":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]], device=device)
        print(f"pos_wt: {pos_wt}")
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_wt)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    if args.freeze_epochs > 0:
        _set_encoder_requires_grad(model, requires_grad=False)
        print(f"Encoder frozen for first {args.freeze_epochs} epoch(s).")

    train_feature_loader = None
    valid_feature_loader = None
    if args.precompute_features:
        if args.freeze_epochs <= 0:
            raise ValueError("`--precompute-features y` requires `--freeze-epochs > 0`.")
        print(
            "Precompute-features enabled: cached feature dataloaders will be used during frozen epochs only."
        )
        train_feature_loader, valid_feature_loader = _get_feature_dataloaders(
            args, model, train_loader, valid_loader, device
        )

    best_score = -1.0
    best_metric_name = "AUC-ROC"
    prob_cols = [f"prob_class_{i}" for i in range(args.num_classes)] if args.task_type == "multiclass" else None

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            _set_encoder_requires_grad(model, requires_grad=True)
            print("Encoder unfrozen for fine-tuning.")

        start_time = time.time()
        use_cached = args.precompute_features and epoch < args.freeze_epochs
        if use_cached:
            avg_loss = train_fn_features(train_feature_loader, model, criterion, optimizer, epoch, args, logger, device)
        else:
            avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, args, logger, device)
        scheduler.step()
        if use_cached:
            avg_val_loss, pred_values, prob_values = valid_fn_features(
                valid_feature_loader, model, criterion, args, device, epoch, logger
            )
        else:
            avg_val_loss, pred_values, prob_values = valid_fn(
                valid_loader, model, criterion, args, device, epoch, logger
            )

        args.valid_folds = args.valid_folds.copy()
        args.valid_folds["prediction"] = pred_values
        if args.task_type == "multiclass":
            for i, col in enumerate(prob_cols):
                args.valid_folds[col] = prob_values[:, i]

        logger.add_scalar(f"valid/{args.label}/train_loss", avg_loss, epoch + 1)
        logger.add_scalar(f"valid/{args.label}/val_loss", avg_val_loss, epoch + 1)

        if args.task_type == "binary":
            valid_agg = _aggregate_binary(args.valid_folds, args)
            aucroc = roc_auc_score(valid_agg[args.label].values, valid_agg["prediction"].values)
            valid_pos = valid_agg[valid_agg[args.label] == 1].copy()
            valid_pos["prediction"] = valid_pos["prediction"].apply(lambda x: 1 if x >= 0.5 else 0)
            acc_pos = np.mean(valid_pos[args.label].values == valid_pos["prediction"].values)
            metric_value = float(aucroc)
            best_metric_name = "AUC-ROC"
            logger.add_scalar(f"valid/{args.label}/AUC-ROC", aucroc, epoch + 1)
            logger.add_scalar(f"valid/{args.label}/+ve Acc Score", acc_pos, epoch + 1)
            print(
                f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  "
                f"time: {time.time() - start_time:.0f}s"
            )
            print(f"Epoch {epoch + 1} - AUC-ROC Score: {aucroc:.4f}, Acc +ve {args.label}: {acc_pos * 100:.4f}")
        else:
            valid_agg = _aggregate_multiclass(args.valid_folds, args, prob_cols)
            probs = valid_agg[prob_cols].values
            y_true = valid_agg[args.label].astype(int).values
            aucroc, present_classes = _macro_auc_ovr_present_classes(y_true, probs)
            pred_class = np.argmax(probs, axis=1)
            acc = np.mean(y_true == pred_class)
            metric_value = float(aucroc)
            best_metric_name = "AUC-ROC-macro"
            logger.add_scalar(f"valid/{args.label}/AUC-ROC-macro", aucroc, epoch + 1)
            logger.add_scalar(f"valid/{args.label}/Acc", acc, epoch + 1)
            print(
                f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  "
                f"time: {time.time() - start_time:.0f}s"
            )
            print(
                f"Epoch {epoch + 1} - AUC-ROC-macro: {aucroc:.4f}, Acc {args.label}: {acc * 100:.4f} "
                f"(encoded classes present in val: {present_classes}, raw classes: {args.label_values})"
            )

        if not np.isfinite(metric_value):
            raise ValueError(f"Validation metric is not finite at epoch {epoch + 1}.")

        if best_score < metric_value:
            best_score = metric_value
            model_name = f"{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth"
            print(f"Epoch {epoch + 1} - Save {best_metric_name}: {best_score:.4f} Model")
            torch.save(
                {
                    "model": model.state_dict(),
                    "predictions": pred_values,
                    "probabilities": prob_values,
                    "epoch": epoch,
                    "metric_name": best_metric_name,
                    "metric_value": metric_value,
                    "task_type": args.task_type,
                    "label_values": args.label_values,
                },
                args.chk_pt_path / model_name,
            )

        model_name = f"{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth"
        best_ckpt = torch.load(args.chk_pt_path / model_name, map_location="cpu")
        args.valid_folds["prediction"] = best_ckpt["predictions"]
        if args.task_type == "multiclass":
            best_probs = best_ckpt["probabilities"]
            for i, col in enumerate(prob_cols):
                args.valid_folds[col] = best_probs[:, i]
        print(f"[Fold{args.cur_fold}], {best_metric_name}: {best_score:.4f}")

    torch.cuda.empty_cache()
    gc.collect()
    return args.valid_folds


def do_finetune_experiments(args, device):
    if "efficientnetv2" in args.arch:
        args.model_base_name = "efficientv2_s"
    elif "efficientnet_b5_ns" in args.arch:
        args.model_base_name = "efficientnetb5"
    else:
        args.model_base_name = args.arch

    if args.freeze_epochs < 0:
        raise ValueError("--freeze-epochs must be >= 0")
    if args.freeze_epochs > args.epochs:
        raise ValueError("--freeze-epochs must be <= --epochs.")
    if args.freeze_epochs == args.epochs and not args.precompute_features:
        raise ValueError("--freeze-epochs must be < --epochs so fine-tuning can occur.")

    args.data_dir = Path(args.data_dir)
    args.csv_path = Path(args.csv_path)
    args.pretrained_checkpoint = Path(args.pretrained_checkpoint)
    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
    if args.feature_cache_dir:
        args.feature_cache_dir = Path(args.feature_cache_dir)
    else:
        args.feature_cache_dir = args.output_path / "feature_cache"
    if args.precompute_features:
        args.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.DataFrame()
    args.BCE_weights = {}
    for fold in range(args.start_fold, args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)
        _load_and_split_manifest(args)

        fold_key = f"fold{args.cur_fold}"
        if args.task_type == "binary":
            if args.weighted_BCE == "y":
                args.BCE_weights[fold_key] = _compute_pos_weight(args.train_folds, args.label)
            else:
                args.BCE_weights[fold_key] = 1.0

        print(f"df shape: {args.df.shape}")
        print(args.df.columns)
        print(f"train_folds shape: {args.train_folds.shape}")
        print(f"valid_folds shape: {args.valid_folds.shape}")
        if hasattr(args, "test_folds"):
            print(f"test_folds shape: {args.test_folds.shape}")
        print(f"task_type: {args.task_type}")
        print(f"label classes: {args.label_values}")
        if args.task_type == "binary":
            print(f"{fold_key} pos_weight: {args.BCE_weights[fold_key]}")

        if args.smoke_test:
            print("Smoke test enabled: skipping training and aggregation.")
            return

        _oof_df = train_loop(args, device)
        oof_df = pd.concat([oof_df, _oof_df])

    oof_df = oof_df.reset_index(drop=True)
    print("================ CV ================")

    if args.task_type == "binary":
        _validate_columns(oof_df, ["patient_id", "laterality", args.label, "prediction"], "OOF dataframe")
        oof_df["prediction_bin"] = oof_df["prediction"].apply(lambda x: 1 if x >= 0.5 else 0)
        oof_df_agg = (
            oof_df[["patient_id", "laterality", args.label, "prediction"]]
            .groupby(["patient_id", "laterality"])
            .mean()
        )
        oof_df_agg[args.label] = oof_df_agg[args.label].round().astype(int)
        aucroc = roc_auc_score(y_true=oof_df_agg[args.label].values, y_score=oof_df_agg["prediction"].values)
        oof_df_agg_pos = oof_df_agg[oof_df_agg[args.label] == 1].copy()
        oof_df_agg_pos["prediction"] = oof_df_agg_pos["prediction"].apply(lambda x: 1 if x >= 0.5 else 0)
        acc_pos = np.mean(oof_df_agg_pos[args.label].values == oof_df_agg_pos["prediction"].values)
        print(f"AUC-ROC: {aucroc}, acc +ve {args.label} patients: {acc_pos * 100}")
    else:
        prob_cols = [f"prob_class_{i}" for i in range(args.num_classes)]
        _validate_columns(oof_df, ["patient_id", "laterality", args.label] + prob_cols, "OOF dataframe")
        probs = oof_df[["patient_id", "laterality"] + prob_cols].groupby(["patient_id", "laterality"]).mean()
        labels = (
            oof_df[["patient_id", "laterality", args.label]]
            .groupby(["patient_id", "laterality"])[args.label]
            .agg(lambda x: x.mode().iloc[0])
        )
        oof_df_agg = probs.copy()
        oof_df_agg[args.label] = labels
        y_true = oof_df_agg[args.label].astype(int).values
        y_score = oof_df_agg[prob_cols].values
        aucroc, present_classes = _macro_auc_ovr_present_classes(y_true, y_score)
        pred_class = np.argmax(y_score, axis=1)
        acc = np.mean(y_true == pred_class)
        print(
            f"AUC-ROC-macro: {aucroc}, acc {args.label} patients: {acc * 100}, "
            f"encoded classes present in CV: {present_classes}"
        )

    print("\n")
    print(oof_df.head(10))
    print(f"Results shape: {oof_df.shape}")
    print("\n")
    print(args.output_path)
    oof_df.to_csv(args.output_path / f"seed_{args.seed}_n_folds_{args.n_folds}_outputs.csv", index=False)


def main(args):
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.apex = args.apex == "y"
    args.running_interactive = args.running_interactive == "y"
    args.smoke_test = args.smoke_test == "y"
    args.precompute_features = args.precompute_features == "y"
    args.force_recompute_features = args.force_recompute_features == "y"

    args.chk_pt_path = Path(args.checkpoints)
    args.output_path = Path(args.output_path)
    args.tb_logs_path = Path(args.tensorboard_path)

    os.makedirs(args.chk_pt_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.tb_logs_path, exist_ok=True)

    print("====================> Paths <====================")
    print(f"checkpoint_path: {args.chk_pt_path}")
    print(f"output_path: {args.output_path}")
    print(f"tb_logs_path: {args.tb_logs_path}")
    print("device:", device)
    print("torch version:", torch.__version__)
    print("====================> Paths <====================")

    pickle.dump(args, open(os.path.join(args.output_path, f"seed_{args.seed}_train_configs.pkl"), "wb"))
    torch.cuda.empty_cache()
    do_finetune_experiments(args, device)


if __name__ == "__main__":
    args = config()
    if torch.cuda.is_available():
        print("CUDA is available on this system.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA is not available on this system.")
    main(args)
