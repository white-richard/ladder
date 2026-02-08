from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations import *
from torch.utils.data import DataLoader

from dataset_factory import datasets
from dataset_factory.custom_datasets import SubsetDataset
from dataset_factory.datasets import RSNADataset, Dataset_NIH, collate_NIH, EmbedDataset
from utils import get_hparams


def _pick_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def _build_cbis_splits(df, seed=42, val_ratio=0.1):
    df = df.copy()
    if "split" in df.columns:
        split_series = df["split"].astype(str).str.strip().str.lower()
        train_pool = df[split_series == "train"].copy()
        test_df = df[split_series == "test"].copy()
    elif "split_group" in df.columns:
        split_series = df["split_group"].astype(str).str.strip().str.lower()
        train_pool = df[split_series.str.endswith("_train")].copy()
        test_df = df[split_series.str.endswith("_test")].copy()
    else:
        # Fallback: use all rows as train pool if split metadata is absent.
        train_pool = df.copy()
        test_df = df.iloc[0:0].copy()

    if train_pool.empty:
        return train_pool, train_pool.copy(), test_df

    rng = np.random.RandomState(seed)
    patient_ids = train_pool["patient_id"].astype(str).fillna("unknown").unique()
    if len(patient_ids) <= 1:
        valid_df = train_pool.sample(n=min(1, len(train_pool)), random_state=seed).copy()
        train_df = train_pool.drop(valid_df.index).copy()
        if train_df.empty:
            train_df = train_pool.copy()
    else:
        n_val = max(1, int(round(len(patient_ids) * val_ratio)))
        n_val = min(n_val, len(patient_ids) - 1)
        val_patients = set(rng.choice(patient_ids, size=n_val, replace=False))
        valid_df = train_pool[train_pool["patient_id"].astype(str).isin(val_patients)].copy()
        train_df = train_pool[~train_pool["patient_id"].astype(str).isin(val_patients)].copy()
        if train_df.empty:
            train_df = train_pool.copy()
            valid_df = train_pool.iloc[0:0].copy()

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _load_cbis_manifest(data_dir, args):
    manifest_override = getattr(args, "cbis_manifest_csv", None)
    candidates = []
    if manifest_override:
        candidates.append(Path(manifest_override))
    candidates.extend(
        [
            data_dir / "processed_png_full" / "manifest_full_mammograms.csv",
            data_dir / "metadata" / "manifest_full_mammograms.csv",
            data_dir / "manifest_full_mammograms.csv",
            data_dir / "metadata.csv",
            data_dir / "metadata" / "metadata.csv",
        ]
    )
    manifest_path = _pick_existing_file(candidates)
    if manifest_path is None:
        raise FileNotFoundError(
            "Could not find CBIS manifest CSV. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "patient_id" not in df.columns and "subject_id" in df.columns:
        subject_parts = (
            df["subject_id"].astype(str).str.strip().str.extract(
                r"^(?:Calc|Mass)-(?:Training|Test)_(.+)_(LEFT|RIGHT)_(CC|MLO)$"
            )
        )
        df["patient_id"] = subject_parts[0]
        if "laterality" not in df.columns:
            df["laterality"] = subject_parts[1]
        if "view" not in df.columns:
            df["view"] = subject_parts[2]

    if "laterality" in df.columns:
        mapping = {"L": 0, "R": 1, "LEFT": 0, "RIGHT": 1, 0: 0, 1: 1}
        df["laterality"] = df["laterality"].map(lambda x: mapping.get(str(x).strip().upper(), x))
        df["laterality"] = pd.to_numeric(df["laterality"], errors="coerce").fillna(0).astype(int)
    else:
        df["laterality"] = 0

    if "pathology" in df.columns and "cancer" not in df.columns:
        pathology = df["pathology"].astype(str).str.strip().str.upper()
        df["cancer"] = pathology.str.contains("MALIGNANT", na=False).astype(int)

    if "task" in df.columns:
        task = df["task"].astype(str).str.strip().str.lower()
        if "Mass" not in df.columns:
            df["Mass"] = (task == "mass").astype(int)
        if "Suspicious_Calcification" not in df.columns:
            df["Suspicious_Calcification"] = (task == "calc").astype(int)
    else:
        df["Mass"] = df.get("Mass", 0)
        df["Suspicious_Calcification"] = df.get("Suspicious_Calcification", 0)

    for col in ["CLIP_V1_bin", "SCAR_V1_bin", "MARK_V1_bin", "MOLE_V1_bin"]:
        if col not in df.columns:
            df[col] = 0

    if "png_path" not in df.columns:
        raise ValueError(
            f"CBIS manifest `{manifest_path}` is missing required column `png_path`."
        )

    label_col = getattr(args, "cbis_label_col", "cancer")
    if label_col not in df.columns:
        if "cancer" in df.columns:
            label_col = "cancer"
        elif "pathology" in df.columns:
            df["cancer"] = df["pathology"].astype(str).str.upper().str.contains("MALIGNANT").astype(int)
            label_col = "cancer"
        else:
            raise ValueError(
                f"CBIS label column `{label_col}` not found and no fallback pathology/cancer column available."
            )

    df = df[df[label_col].notna()].copy()
    return df, label_col


def get_natural_images_dataloaders(args):
    hparams = get_hparams(args.dataset, args.classifier)
    if args.dataset in vars(datasets):
        train_dataset = vars(datasets)[args.dataset](args.data_dir, 'tr', hparams, train_attr="yes")
        val_dataset = vars(datasets)[args.dataset](args.data_dir, 'va', hparams)
        test_dataset = vars(datasets)[args.dataset](args.data_dir, 'te', hparams)
        print(f"Dataset sizes => train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
        num_workers = train_dataset.N_WORKERS

        if hparams['group_balanced']:
            train_weights = np.asarray(train_dataset.weights_g)
            train_weights /= np.sum(train_weights)
        else:
            train_weights = None

        train_loader = DataLoader(
            train_dataset, batch_size=hparams['batch_size'], shuffle=args.shuffle, num_workers=num_workers,
            pin_memory=True, drop_last=True
        )

        valid_loader = DataLoader(
            val_dataset, batch_size=max(128, hparams['batch_size'] * 2), shuffle=args.shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=max(128, hparams['batch_size'] * 2), shuffle=args.shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        return {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "test_loader": test_loader
        }


def get_rsna_dataloaders(args):
    train_tfms = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
        ElasticTransform(alpha=10, sigma=15)
    ], p=1.0)
    val_tfms = None

    data_dir = Path(args.data_dir)
    train_df = None
    valid_df = None
    test_df = None
    label_col = None
    dataset_name = args.dataset.lower()
    mammo_dataset = "cbis" if dataset_name == "cbis-ddsm" else dataset_name

    if mammo_dataset == "rsna":
        df = pd.read_csv(data_dir / "rsna_w_upmc_concepts_breast_clip.csv")
        mapping = {'L': 0, 'R': 1}
        df['laterality'] = df['laterality'].map(mapping)
        train_df = df[(df['fold'] == 1) | (df['fold'] == 2)].reset_index(drop=True)
        valid_df = df[df['fold'] == 3].reset_index(drop=True)
        test_df = df[df['fold'] == 0].reset_index(drop=True)
        label_col = "cancer"
    elif mammo_dataset == "vindr":
        df = pd.read_csv(data_dir / "vindr_detection_v1_folds_abnormal.csv")
        mapping = {'L': 0, 'R': 1}
        df['laterality'] = df['laterality'].map(mapping)
        train_df = df[df["split_new"] == "train"].reset_index(drop=True)
        valid_df = df[df["split_new"] == "val"].reset_index(drop=True)
        test_df = df[df["split_new"] == "test"].reset_index(drop=True)
        vindr_label_mode = getattr(args, "vindr_label_mode", "abnormal")

        vindr_abnormal_birads_min = getattr(args, "vindr_abnormal_birads_min", 4)
        if vindr_label_mode == "abnormal" and vindr_abnormal_birads_min is not None:
            for _df in (train_df, valid_df, test_df):
                _df["breast_birads_num"] = pd.to_numeric(_df["breast_birads"], errors="coerce")
                _df["abnormal"] = (_df["breast_birads_num"] >= vindr_abnormal_birads_min).astype(int)
        label_col = "abnormal"
    elif mammo_dataset == "cbis":
        df, label_col = _load_cbis_manifest(data_dir, args)
        train_df, valid_df, test_df = _build_cbis_splits(
            df, seed=getattr(args, "seed", 42), val_ratio=getattr(args, "cbis_val_ratio", 0.1)
        )

    train_dataset = RSNADataset(
        train_df, data_dir, train_tfms, mean=0.3089279, std=0.25053555408335154, dataset=mammo_dataset,
        label_col=label_col)
    valid_dataset = RSNADataset(
        valid_df, data_dir, val_tfms, mean=0.3089279, std=0.25053555408335154, dataset=mammo_dataset,
    label_col=label_col)
    test_dataset = RSNADataset(
        test_df, data_dir, val_tfms, mean=0.3089279, std=0.25053555408335154, dataset=mammo_dataset,
    label_col=label_col)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
        drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )
    print("\n ==========================> [Shapes] Dataset <==========================")
    print("train: ", len(train_dataset), "val: ", len(valid_dataset), "test: ", len(test_dataset))
    print("\n ==========================> [Shapes] Dataloaders <==========================")
    print("train: ", len(train_loader), "val: ", len(valid_loader), "test: ", len(test_loader))
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
    }

def get_nih_dataloaders(args):
    column_name_split = "val_train_split"
    tfms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_file = Path(args.data_dir) / "nih_processed_v2.csv"
    df = pd.read_csv(data_file)
    dataset = Dataset_NIH(df=df, class_names=["Pneumothorax"], transform=tfms, seed=args.seed)
    try:
        df_train = df.loc[(df[column_name_split] == 1)]
        train_inds = np.asarray(df_train.index)
        df_test = df.loc[(df[column_name_split] == 0)]
        test_inds = np.asarray(df_test.index)
        df_val = df.loc[(df[column_name_split] == 2)]
        val_inds = np.asarray(df_val.index)
        print("train: ", train_inds.shape, "test: ", test_inds.shape, "val: ", val_inds.shape)
    except:
        print(
            "The data_file doesn't have a train column, "
            "hence we will randomly split the entire dataset to have 15% samples as validation set.")
        train_inds = np.empty([])
        test_inds = np.empty([])
        val_inds = np.empty([])

    train_dataset = SubsetDataset(dataset, train_inds)
    valid_dataset = SubsetDataset(dataset, val_inds)
    test_dataset = SubsetDataset(dataset, test_inds)
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=args.shuffle, num_workers=4, pin_memory=True, collate_fn=collate_NIH
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=128, shuffle=args.shuffle, num_workers=4, pin_memory=True, collate_fn=collate_NIH
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=args.shuffle, num_workers=4, pin_memory=True, collate_fn=collate_NIH
    )

    print("\n ==========================> [Shapes] NIH Dataset <==========================")
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    print("\n ==========================> [Shapes] NIH Dataloaders <==========================")
    print(len(train_loader), len(valid_loader), len(test_loader))

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
    }


def create_dataloaders(args):
    if args.dataset.lower() == "waterbirds" or args.dataset.lower() == "celeba" or args.dataset.lower() == "metashift":
        return get_natural_images_dataloaders(args)
    elif args.dataset.lower() == "nih":
        return get_nih_dataloaders(args)
    elif args.dataset.lower() in {"rsna", "vindr", "cbis", "cbis-ddsm"}:
        return get_rsna_dataloaders(args)
