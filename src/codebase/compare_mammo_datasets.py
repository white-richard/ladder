import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

np = None
pd = None
plt = None


DEFAULT_METADATA_COLUMNS = [
    "laterality",
    "view",
    "view_position",
    "age",
    "breast_birads",
    "breast_density",
    "pathology",
    "Mass",
    "Suspicious_Calcification",
    "CLIP_V1_bin",
    "SCAR_V1_bin",
    "MARK_V1_bin",
    "MOLE_V1_bin",
]

ID_CANDIDATE_COLUMNS = ["patient_id", "study_id", "subject_id"]
SPLITS = ["train", "val", "test"]
CANONICAL_LABEL_COL = "cancer_or_abnormal"

"""
python src/codebase/compare_mammo_datasets.py \
  --rsna-data-dir /home/richw/.code/datasets/rsna/mammo_clip \
  --vindr-data-dir /home/richw/.code/hyp-mammo/repos/ladder/data/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0 \
  --cbis-data-dir /home/richw/.code/datasets/cbis-ddsm \
  --out-dir out/dataset_comparison

"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare RSNA, VinDr, and CBIS-DDSM metadata/class distributions and write plots + tables."
        )
    )
    parser.add_argument("--rsna-data-dir", type=Path, required=True, help="Root directory for RSNA CSV files.")
    parser.add_argument("--vindr-data-dir", type=Path, required=True, help="Root directory for VinDr CSV files.")
    parser.add_argument("--cbis-data-dir", type=Path, required=True, help="Root directory for CBIS metadata/manifest.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out/dataset_comparison"), help="Output directory for tables/plots."
    )
    parser.add_argument(
        "--cbis-manifest-csv",
        type=Path,
        default=None,
        help="Optional override for CBIS manifest path.",
    )
    parser.add_argument(
        "--cbis-label-col",
        type=str,
        default="cancer",
        help="CBIS label column used for class distribution.",
    )
    parser.add_argument(
        "--cbis-val-ratio",
        type=float,
        default=0.1,
        help="Validation patient ratio for CBIS train split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for CBIS split generation.")
    parser.add_argument(
        "--vindr-label-mode",
        type=str,
        default="abnormal",
        help="Label column/mode for VinDr (default mirrors dataloader behavior).",
    )
    parser.add_argument(
        "--vindr-abnormal-birads-min",
        type=float,
        default=4,
        help="If VinDr label mode is abnormal, threshold on breast_birads.",
    )
    return parser.parse_args()


def _configure_matplotlib():
    global plt
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for this script. Install it with `pip install matplotlib`."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt = _plt


def _configure_data_libs():
    global np, pd
    try:
        import numpy as _np
        import pandas as _pd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "numpy and pandas are required for this script. Install with `pip install numpy pandas`."
        ) from exc
    np = _np
    pd = _pd


def _check_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _pick_patient_col(df):
    for col in ID_CANDIDATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def _normalize_laterality(series):
    mapping = {
        "L": "L",
        "R": "R",
        "LEFT": "L",
        "RIGHT": "R",
        "0": "L",
        "1": "R",
        0: "L",
        1: "R",
    }
    return series.map(lambda x: mapping.get(str(x).strip().upper(), str(x).strip().upper()))


def _escape_markdown_cell(value):
    if value is None:
        text = ""
    else:
        text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _dataframe_to_markdown(df):
    if df is None or len(df.columns) == 0:
        return "_(empty table)_"

    headers = [_escape_markdown_cell(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        cells = [_escape_markdown_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _to_binary_label(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric

    mapped = (
        series.astype(str).str.strip().str.upper().map(
            {
                "MALIGNANT": 1,
                "BENIGN": 0,
                "ABNORMAL": 1,
                "NORMAL": 0,
                "TRUE": 1,
                "FALSE": 0,
                "YES": 1,
                "NO": 0,
                "POSITIVE": 1,
                "NEGATIVE": 0,
            }
        )
    )
    return pd.to_numeric(mapped, errors="coerce")


def _add_canonical_label_column(df, source_col):
    if source_col not in df.columns:
        df[CANONICAL_LABEL_COL] = pd.Series(index=df.index, dtype=float)
        return
    df[CANONICAL_LABEL_COL] = _to_binary_label(df[source_col])


def _extract_birads_numeric(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    extracted = series.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def _find_birads_column(df):
    preferred = [
        "breast_birads",
        "birads",
        "birads_assessment",
        "birads_category",
        "assessment",
    ]
    lower_map = {col.lower(): col for col in df.columns}
    for name in preferred:
        if name in df.columns:
            return name
        if name in lower_map:
            return lower_map[name]
    for col in df.columns:
        if "birads" in col.lower():
            return col
    return None


def _write_table(df, csv_path, txt_path):
    df.to_csv(csv_path, index=False)
    txt_path.write_text(_dataframe_to_markdown(df) + "\n")


def _plot_table(df, out_path, title):
    if df.empty:
        return
    display_df = df.copy()
    if len(display_df) > 30:
        display_df = display_df.head(30)
    fig_h = max(2.0, 0.4 * (len(display_df) + 2))
    fig_w = max(8.0, 0.4 * len(display_df.columns) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)
    table = ax.table(
        cellText=display_df.astype(str).values,
        colLabels=list(display_df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_rsna(data_dir):
    path = data_dir / "rsna_w_upmc_concepts_breast_clip.csv"
    _check_exists(path, "RSNA CSV")
    df = pd.read_csv(path)
    if "laterality" in df.columns:
        df["laterality"] = df["laterality"].map({"L": 0, "R": 1})
    splits = {
        "train": df[(df["fold"] == 1) | (df["fold"] == 2)].reset_index(drop=True),
        "val": df[df["fold"] == 3].reset_index(drop=True),
        "test": df[df["fold"] == 0].reset_index(drop=True),
    }
    for split_name in SPLITS:
        _add_canonical_label_column(splits[split_name], "cancer")
    return {"dataset": "rsna", "label_col": CANONICAL_LABEL_COL, "splits": splits}


def _load_vindr(data_dir, label_mode, abnormal_birads_min):
    path = data_dir / "vindr_detection_v1_folds_abnormal.csv"
    _check_exists(path, "VinDr CSV")
    df = pd.read_csv(path)
    if "laterality" in df.columns:
        df["laterality"] = df["laterality"].map({"L": 0, "R": 1})
    splits = {
        "train": df[df["split_new"] == "train"].reset_index(drop=True),
        "val": df[df["split_new"] == "val"].reset_index(drop=True),
        "test": df[df["split_new"] == "test"].reset_index(drop=True),
    }

    label_col = label_mode
    if label_mode == "abnormal":
        if abnormal_birads_min is None:
            raise ValueError("vindr_abnormal_birads_min must be set when --vindr-label-mode abnormal is used.")
        for split_name in SPLITS:
            split_df = splits[split_name]
            if "abnormal" in split_df.columns:
                split_df["abnormal"] = _to_binary_label(split_df["abnormal"]).fillna(0).astype(int)
            else:
                split_df["breast_birads_num"] = _extract_birads_numeric(split_df["breast_birads"])
                split_df["abnormal"] = (split_df["breast_birads_num"] >= abnormal_birads_min).astype(int)
    elif label_mode not in df.columns:
        raise ValueError(f"VinDr label column `{label_mode}` not found in {path}.")

    for split_name in SPLITS:
        split_df = splits[split_name]
        source_col = "abnormal" if label_col == "abnormal" else label_col
        _add_canonical_label_column(split_df, source_col)
    return {"dataset": "vindr", "label_col": CANONICAL_LABEL_COL, "splits": splits}


def _load_cbis(data_dir, cbis_manifest_csv, cbis_label_col, seed, val_ratio):
    from dataset_factory import _build_cbis_splits, _load_cbis_manifest

    args = SimpleNamespace(cbis_manifest_csv=cbis_manifest_csv, cbis_label_col=cbis_label_col)
    df, label_col = _load_cbis_manifest(data_dir, args)
    train_df, val_df, test_df = _build_cbis_splits(df, seed=seed, val_ratio=val_ratio)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split_name in SPLITS:
        _add_canonical_label_column(splits[split_name], label_col)
    return {"dataset": "cbis-ddsm", "label_col": CANONICAL_LABEL_COL, "splits": splits}


def _summarize_splits(dataset_infos):
    rows = []
    for info in dataset_infos:
        dataset = info["dataset"]
        label_col = info["label_col"]
        for split_name in SPLITS:
            split_df = info["splits"][split_name]
            n = len(split_df)
            patient_col = _pick_patient_col(split_df)
            n_patients = split_df[patient_col].nunique(dropna=True) if patient_col else np.nan
            label_series = (
                pd.to_numeric(split_df[label_col], errors="coerce")
                if label_col in split_df.columns
                else pd.Series(dtype=float)
            )
            known = label_series.dropna()
            positives = int((known == 1).sum())
            negatives = int((known == 0).sum())
            unknown = int(n - known.shape[0])
            pos_rate = float(positives / known.shape[0]) if known.shape[0] else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "samples": n,
                    "patients": n_patients,
                    "label_col": label_col,
                    "positives": positives,
                    "negatives": negatives,
                    "unknown_label": unknown,
                    "positive_rate": pos_rate,
                }
            )
    split_summary = pd.DataFrame(rows).sort_values(["dataset", "split"]).reset_index(drop=True)
    return split_summary


def _summarize_overall(dataset_infos):
    rows = []
    for info in dataset_infos:
        dataset = info["dataset"]
        label_col = info["label_col"]
        full_df = pd.concat([info["splits"][s] for s in SPLITS], ignore_index=True)
        patient_col = _pick_patient_col(full_df)
        n_patients = full_df[patient_col].nunique(dropna=True) if patient_col else np.nan
        label_series = (
            pd.to_numeric(full_df[label_col], errors="coerce")
            if label_col in full_df.columns
            else pd.Series(dtype=float)
        )
        known = label_series.dropna()
        positives = int((known == 1).sum())
        negatives = int((known == 0).sum())
        rows.append(
            {
                "dataset": dataset,
                "samples": len(full_df),
                "patients": n_patients,
                "label_col": label_col,
                "positives": positives,
                "negatives": negatives,
                "positive_rate": float(positives / known.shape[0]) if known.shape[0] else np.nan,
                "num_columns": len(full_df.columns),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _select_metadata_columns(dataset_infos):
    counts = Counter()
    label_cols = {info["label_col"] for info in dataset_infos}
    for info in dataset_infos:
        full_df = pd.concat([info["splits"][s] for s in SPLITS], ignore_index=True)
        for col in full_df.columns:
            counts[col] += 1

    selected = [col for col in DEFAULT_METADATA_COLUMNS if counts[col] > 0]
    excluded_tokens = ["path", "id", "fold", "split", "xmin", "xmax", "ymin", "ymax", "image"]
    extras = []
    for col, present_in in counts.items():
        if present_in < 2 or col in selected or col in label_cols:
            continue
        lower = col.lower()
        if any(token in lower for token in excluded_tokens):
            continue
        extras.append(col)
    extras = sorted(extras)[:12]
    return selected + extras


def _metadata_coverage_table(dataset_infos, metadata_columns):
    rows = []
    for info in dataset_infos:
        dataset = info["dataset"]
        full_df = pd.concat([info["splits"][s] for s in SPLITS], ignore_index=True)
        row = {"dataset": dataset, "samples": len(full_df)}
        for col in metadata_columns:
            if col not in full_df.columns:
                row[col] = "N/A"
                continue
            non_null = int(full_df[col].notna().sum())
            ratio = non_null / len(full_df) if len(full_df) else 0
            row[col] = f"{non_null}/{len(full_df)} ({ratio:.1%})"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _metadata_missing_matrix(dataset_infos, metadata_columns):
    datasets = []
    matrix = []
    for info in dataset_infos:
        dataset = info["dataset"]
        full_df = pd.concat([info["splits"][s] for s in SPLITS], ignore_index=True)
        datasets.append(dataset)
        row = []
        for col in metadata_columns:
            if col not in full_df.columns or len(full_df) == 0:
                row.append(np.nan)
            else:
                row.append(1.0 - float(full_df[col].notna().sum() / len(full_df)))
        matrix.append(row)
    return datasets, np.array(matrix, dtype=float)


def _plot_missingness_heatmap(dataset_infos, metadata_columns, out_path):
    if not metadata_columns:
        return
    datasets, matrix = _metadata_missing_matrix(dataset_infos, metadata_columns)
    fig_w = max(8, 0.6 * len(metadata_columns))
    fig_h = max(3, 0.8 * len(datasets))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metadata_columns)))
    ax.set_xticklabels(metadata_columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("Metadata Missingness Rate")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Missing rate")
    for i in range(len(datasets)):
        for j in range(len(metadata_columns)):
            val = matrix[i, j]
            text = "N/A" if np.isnan(val) else f"{val:.0%}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_class_distribution(split_summary, out_path):
    ordered = split_summary.copy()
    ordered["dataset_split"] = ordered["dataset"] + ":" + ordered["split"]
    x = np.arange(len(ordered))
    pos = ordered["positives"].to_numpy()
    neg = ordered["negatives"].to_numpy()
    unk = ordered["unknown_label"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, neg, label="Negative", color="#4c78a8")
    ax.bar(x, pos, bottom=neg, label="Positive", color="#f58518")
    ax.bar(x, unk, bottom=neg + pos, label="Unknown", color="#bab0ac")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["dataset_split"], rotation=30, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Class Distribution by Dataset/Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_positive_rate(split_summary, out_path):
    ordered = split_summary.copy()
    ordered["dataset_split"] = ordered["dataset"] + ":" + ordered["split"]
    x = np.arange(len(ordered))
    y = ordered["positive_rate"].fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, y, color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["dataset_split"], rotation=30, ha="right")
    ax.set_ylim(0, min(1.0, max(0.05, y.max() * 1.2)))
    ax.set_ylabel("Positive Rate")
    ax.set_title("Positive Rate by Dataset/Split")
    for idx, val in enumerate(y):
        ax.text(idx, val + 0.01, f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _candidate_categorical_columns(dataset_infos, metadata_columns):
    selected = []
    for col in metadata_columns:
        present = 0
        global_unique = set()
        for info in dataset_infos:
            full_df = pd.concat([info["splits"][s] for s in SPLITS], ignore_index=True)
            if col not in full_df.columns:
                continue
            present += 1
            values = full_df[col].dropna()
            if col == "laterality":
                values = _normalize_laterality(values)
            global_unique.update(values.astype(str).str.strip().tolist())
        if present >= 2 and 2 <= len(global_unique) <= 15:
            selected.append(col)
    return selected


def _categorical_distribution(dataset_infos, column):
    frames = []
    for info in dataset_infos:
        dataset = info["dataset"]
        full_df = pd.concat([info["splits"][s] for s in SPLITS], ignore_index=True)
        if column not in full_df.columns:
            continue
        values = full_df[column].dropna()
        if column == "laterality":
            values = _normalize_laterality(values)
        values = values.astype(str).str.strip()
        if values.empty:
            continue
        counts = values.value_counts()
        table = counts.rename_axis("value").reset_index(name="count")
        table["dataset"] = dataset
        table["fraction"] = table["count"] / table["count"].sum()
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_categorical_distribution(dist_df, column, out_path):
    if dist_df.empty:
        return
    pivot = dist_df.pivot_table(index="dataset", columns="value", values="fraction", fill_value=0.0)
    pivot = pivot.loc[:, sorted(pivot.columns)]
    if pivot.shape[1] > 8:
        top_cols = pivot.sum(axis=0).sort_values(ascending=False).head(7).index
        other_col = [c for c in pivot.columns if c not in set(top_cols)]
        reduced = pivot[top_cols].copy()
        reduced["Other"] = pivot[other_col].sum(axis=1)
        pivot = reduced

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(f"{column} distribution (normalized)")
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Dataset")
    ax.legend(title=column, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _birads_distribution(dataset_infos):
    rows = []
    for info in dataset_infos:
        dataset = info["dataset"]
        full_df = pd.concat([info["splits"][s] for s in SPLITS], ignore_index=True)
        birads_col = _find_birads_column(full_df)
        if birads_col is None:
            continue
        values = _extract_birads_numeric(full_df[birads_col]).dropna()
        if values.empty:
            continue
        values = values.astype(int)
        counts = values.value_counts().sort_index()
        total = float(counts.sum())
        for birads_val, count in counts.items():
            rows.append(
                {
                    "dataset": dataset,
                    "birads": int(birads_val),
                    "count": int(count),
                    "fraction": float(count / total) if total else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _plot_birads_distribution(dist_df, out_path):
    if dist_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No BI-RADS data found", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return
    pivot = dist_df.pivot_table(index="dataset", columns="birads", values="fraction", fill_value=0.0)
    pivot = pivot.loc[:, sorted(pivot.columns)]
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c")
    ax.set_title("BI-RADS distribution (normalized, int)")
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Dataset")
    ax.legend(title="BI-RADS", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _numeric_metadata_summary(dataset_infos, metadata_columns):
    rows = []
    for info in dataset_infos:
        dataset = info["dataset"]
        full_df = pd.concat([info["splits"][s] for s in SPLITS], ignore_index=True)
        for col in metadata_columns:
            if col not in full_df.columns:
                continue
            values = pd.to_numeric(full_df[col], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "column": col,
                    "count": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["column", "dataset"]).reset_index(drop=True)


def main():
    args = parse_args()
    _configure_data_libs()
    _configure_matplotlib()
    out_dir = args.out_dir
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    cat_tables_dir = tables_dir / "categorical_distributions"
    cat_plots_dir = plots_dir / "categorical_distributions"
    for path in [out_dir, tables_dir, plots_dir, cat_tables_dir, cat_plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    dataset_infos = [
        _load_rsna(args.rsna_data_dir),
        _load_vindr(args.vindr_data_dir, args.vindr_label_mode, args.vindr_abnormal_birads_min),
        _load_cbis(
            args.cbis_data_dir,
            args.cbis_manifest_csv,
            args.cbis_label_col,
            args.seed,
            args.cbis_val_ratio,
        ),
    ]

    split_summary = _summarize_splits(dataset_infos)
    overall_summary = _summarize_overall(dataset_infos)
    metadata_columns = _select_metadata_columns(dataset_infos)
    coverage_table = _metadata_coverage_table(dataset_infos, metadata_columns)
    numeric_summary = _numeric_metadata_summary(dataset_infos, metadata_columns)

    _write_table(split_summary, tables_dir / "split_summary.csv", tables_dir / "split_summary.md")
    _write_table(overall_summary, tables_dir / "overall_summary.csv", tables_dir / "overall_summary.md")
    _write_table(coverage_table, tables_dir / "metadata_coverage.csv", tables_dir / "metadata_coverage.md")
    if not numeric_summary.empty:
        _write_table(numeric_summary, tables_dir / "numeric_metadata_summary.csv", tables_dir / "numeric_metadata_summary.md")

    _plot_table(split_summary, plots_dir / "split_summary_table.png", "Split Summary")
    _plot_table(overall_summary, plots_dir / "overall_summary_table.png", "Overall Summary")
    _plot_table(coverage_table, plots_dir / "metadata_coverage_table.png", "Metadata Coverage")
    _plot_class_distribution(split_summary, plots_dir / "class_distribution_by_split.png")
    _plot_positive_rate(split_summary, plots_dir / "positive_rate_by_split.png")
    _plot_missingness_heatmap(dataset_infos, metadata_columns, plots_dir / "metadata_missingness_heatmap.png")
    birads_dist = _birads_distribution(dataset_infos)
    _plot_birads_distribution(birads_dist, plots_dir / "birads_distribution.png")

    categorical_columns = _candidate_categorical_columns(dataset_infos, metadata_columns)
    for col in categorical_columns:
        dist_df = _categorical_distribution(dataset_infos, col)
        if dist_df.empty:
            continue
        safe_col = col.lower().replace(" ", "_").replace("/", "_")
        _write_table(
            dist_df,
            cat_tables_dir / f"{safe_col}_distribution.csv",
            cat_tables_dir / f"{safe_col}_distribution.md",
        )
        _plot_categorical_distribution(
            dist_df,
            col,
            cat_plots_dir / f"{safe_col}_distribution.png",
        )

    report_lines = [
        "# Mammography Dataset Comparison",
        "",
        "Generated datasets: RSNA, VinDr, CBIS-DDSM",
        "",
        "## Output files",
        f"- Tables: `{tables_dir}`",
        f"- Plots: `{plots_dir}`",
        "",
        "## Top-level summaries",
        "",
        "### Overall",
    ]
    report_lines.append(_dataframe_to_markdown(overall_summary))
    report_lines.extend(["", "### By split"])
    report_lines.append(_dataframe_to_markdown(split_summary))

    (out_dir / "README.md").write_text("\n".join(report_lines) + "\n")
    print(f"Comparison complete. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
