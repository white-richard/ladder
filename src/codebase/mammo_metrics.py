import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def normalize_mammo_dataset_name(dataset):
    name = str(dataset).lower()
    return "cbis" if name == "cbis-ddsm" else name


def is_mammo_dataset(dataset):
    return normalize_mammo_dataset_name(dataset) in {"rsna", "vindr", "cbis", "embed"}


def safe_binary_auroc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    unique_true = np.unique(y_true)
    if not np.isin(unique_true, [0, 1]).all():
        y_true = (y_true >= 0.5).astype(int)

    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def aggregate_mammo_predictions(
        df: pd.DataFrame,
        label_col: str,
        prediction_col: str = "prediction",
        patient_col: str = "patient_id",
        laterality_col: str = "laterality",
        fold_col: str = "fold",
) -> pd.DataFrame:
    required_cols = {patient_col, laterality_col, label_col, prediction_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for mammography aggregation: {sorted(missing)}")

    agg_dict = {
        label_col: "max",
        prediction_col: "mean",
    }
    if fold_col in df.columns:
        agg_dict[fold_col] = "first"

    grouped = (
        df[[c for c in [patient_col, laterality_col, label_col, prediction_col, fold_col] if c in df.columns]]
        .groupby([patient_col, laterality_col], as_index=False)
        .agg(agg_dict)
    )
    grouped[label_col] = (grouped[label_col] >= 0.5).astype(int)
    return grouped

