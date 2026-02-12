import argparse
import os
import warnings
from pathlib import Path
from typing import List, Optional

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

from mammo_metrics import is_mammo_dataset
from metrics import auroc
from metrics_factory.calculate_worst_group_acc import (
    calculate_worst_group_acc_celebA,
    calculate_worst_group_acc_metashift,
    calculate_worst_group_acc_rsna_mammo,
    calculate_worst_group_acc_waterbirds,
    calculate_rsna_consistent_aucroc,
)
from utils import seed_all

warnings.filterwarnings("ignore")


def _format_path(path_template: str, seed: int, split: str) -> str:
    try:
        return path_template.format(seed, split)
    except (IndexError, KeyError):
        return path_template


def _default_attribute_col(dataset: str) -> str:
    dataset_name = str(dataset).lower()
    if dataset_name in {"waterbirds", "metashift", "celeba"}:
        return "attribute_bg_predict"
    if is_mammo_dataset(dataset):
        return "calc"
    return "attribute_bg_predict"


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("precision@k requires k > 0")
    if y_true.size == 0:
        return float("nan")
    k = min(k, y_true.size)
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].mean())


def _compute_standard_metrics(df: pd.DataFrame, pred_col: str, threshold: float, precision_ks: List[int]):
    if "out_put_GT" not in df.columns:
        raise ValueError("Required ground-truth column 'out_put_GT' is missing.")

    y_true = df["out_put_GT"].to_numpy()
    y_score = df[pred_col].to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "auroc": auroc(gt=y_true, pred=y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision_at_k": {
            str(k): _precision_at_k(y_true, y_score, k) for k in precision_ks
        },
    }
    return metrics


def _load_slice_attributes(slice_names_path: Optional[str]) -> List[str]:
    if not slice_names_path:
        return []
    slice_path = Path(slice_names_path)
    if not slice_path.exists():
        raise FileNotFoundError(f"Slice names file not found: {slice_path}")
    attrs = pickle.load(open(slice_path, "rb"))
    return list(attrs.keys())


def _mean_consistent_wga_aucroc(
    df: pd.DataFrame,
    slice_attributes: List[str],
    score_col: str,
    verbose: bool = True,
) -> float:
    if not slice_attributes:
        return float("nan")
    available_attributes = [attr for attr in slice_attributes if attr in df.columns]
    missing_attributes = [attr for attr in slice_attributes if attr not in df.columns]
    if verbose and missing_attributes:
        print(
            "Skipping missing slice attributes (not in CSV columns): "
            + ", ".join(missing_attributes)
        )
    if not available_attributes:
        return float("nan")
    wga_values = []
    for attribute in available_attributes:
        metrics = calculate_rsna_consistent_aucroc(
            df, score_col=score_col, attribute_col=attribute
        )
        wga_values.append(metrics["consistent_wga_auroc"])
    return float(np.mean(wga_values))


def evaluate(args):
    csv_path = _format_path(args.clf_results_csv, args.seed, args.split)
    df = pd.read_csv(csv_path)

    attribute_col = args.attribute_col or _default_attribute_col(args.dataset)
    pred_col = args.pred_col

    if pred_col not in df.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in {csv_path}")

    standard_metrics = _compute_standard_metrics(
        df,
        pred_col=pred_col,
        threshold=args.threshold,
        precision_ks=args.precision_k,
    )
    print("#################################### Overall Metrics ####################################")
    print(f"AUROC: {standard_metrics['auroc']}")
    print(f"Accuracy: {standard_metrics['accuracy']}")
    print(f"Recall: {standard_metrics['recall']}")
    print(f"Precision@k: {standard_metrics['precision_at_k']}")

    if is_mammo_dataset(args.dataset):
        if "out_put_predict" not in df.columns:
            df["out_put_predict"] = df[pred_col]
        print("#################################### RSNA/VinDr/CBIS Metrics ####################################")
        wga = calculate_worst_group_acc_rsna_mammo(df, pred_col=pred_col, attribute_col=attribute_col)
        consistent_metrics = calculate_rsna_consistent_aucroc(
            df, score_col=pred_col, attribute_col=attribute_col)
        print("#################################### Consistent AUROC Metrics ####################################")
        print(f"Consistent Mean AUROC: {consistent_metrics['consistent_mean_auroc']}")
        print(f"Consistent WGA AUROC: {consistent_metrics['consistent_wga_auroc']}")
        if args.mean_consistent_wga_slices:
            if not args.slice_names:
                raise ValueError("--slice_names is required when --mean_consistent_wga_slices is set.")
            slice_attributes = _load_slice_attributes(args.slice_names)
            if attribute_col not in slice_attributes:
                slice_attributes = [attribute_col] + slice_attributes
            mean_consistent_wga = _mean_consistent_wga_aucroc(
                df, slice_attributes, score_col=pred_col
            )
            print("#################################### Slice-Mean Consistent WGA AUROC ####################################")
            print(f"Mean Consistent WGA AUROC: {mean_consistent_wga}")
            return {
                "standard": standard_metrics,
                "wga": wga,
                "consistent_aucroc": consistent_metrics,
                "mean_slice_consistent_wga": mean_consistent_wga,
            }
        return {"standard": standard_metrics, "wga": wga, "consistent_aucroc": consistent_metrics}

    dataset_name = str(args.dataset).lower()
    if dataset_name == "waterbirds":
        print("#################################### Waterbirds Metrics ####################################")
        wga = calculate_worst_group_acc_waterbirds(
            df,
            pred_col=pred_col,
            attribute_col=attribute_col,
            log_file=args.out_file,
        )
        return {"standard": standard_metrics, "wga": wga}

    if dataset_name == "celeba":
        print("#################################### CelebA Metrics ####################################")
        pos_pred_col = args.pos_pred_col or pred_col
        neg_pred_col = args.neg_pred_col or pred_col
        wga = calculate_worst_group_acc_celebA(
            df,
            pos_pred_col=pos_pred_col,
            neg_pred_col=neg_pred_col,
            attribute_col=attribute_col,
            log_file=args.out_file,
        )
        return {"standard": standard_metrics, "wga": wga}

    if dataset_name == "metashift":
        print("#################################### MetaShift Metrics ####################################")
        wga = calculate_worst_group_acc_metashift(
            df,
            pred_col=pred_col,
            attribute_col=attribute_col,
        )
        return {"standard": standard_metrics, "wga": wga}

    raise ValueError(
        f"Unsupported dataset '{args.dataset}'. Provide a supported dataset or add handling in evaluate()."
    )

def config():
    parser = argparse.ArgumentParser(description="Evaluate dataset predictions and report group metrics.")
    parser.add_argument(
        "--dataset", default="Waterbirds", type=str,
        help="Name of the dataset (e.g., Waterbirds, CelebA, RSNA, NIH).")
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32",
        help="Directory to save logs (optional).")
    parser.add_argument(
        "--clf_results_csv", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_dataframe_mitigation.csv",
        help="Path to CSV containing classifier predictions and ground truth.")
    parser.add_argument(
        "--split", default="test", type=str,
        help="Data split name used when formatting --clf_results_csv (e.g., test, valid).")
    parser.add_argument(
        "--pred_col", default="out_put_predict", type=str,
        help="Prediction column to evaluate (e.g., out_put_predict or Predictions_bin).")
    parser.add_argument(
        "--threshold", default=0.5, type=float,
        help="Threshold used to binarize predictions for accuracy/recall.")
    parser.add_argument(
        "--precision_k", default=[10], type=int, nargs="+",
        help="One or more k values for precision@k.")
    parser.add_argument(
        "--pos_pred_col", default=None, type=str,
        help="Positive prediction column for CelebA-style metrics (optional).")
    parser.add_argument(
        "--neg_pred_col", default=None, type=str,
        help="Negative prediction column for CelebA-style metrics (optional).")
    parser.add_argument(
        "--attribute_col", default=None, type=str,
        help="Attribute column for group metrics (optional; dataset default is used if omitted).")
    parser.add_argument(
        "--mean_consistent_wga_slices", action="store_true",
        help="Also compute mean Consistent WGA AUROC across all slice attributes and the default attribute.")
    parser.add_argument(
        "--slice_names", default=None, type=str,
        help="Path to the .pkl file containing discovered slice (attribute) names (required with --mean_consistent_wga_slices).")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for reproducibility.")
    return parser.parse_args()


def main(args):
    seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.save_path = Path(args.save_path.format(args.seed))
    args.save_path.mkdir(parents=True, exist_ok=True)
    args.out_file = args.save_path / "ladder_evaluate.txt"
    print("\n")
    print(args.save_path)
    evaluate(args)
    print(f"log saved at: {args.out_file}")


if __name__ == "__main__":
    _args = config()
    main(_args)