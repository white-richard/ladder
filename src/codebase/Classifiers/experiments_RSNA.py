import gc
import time
from pathlib import Path
import random
import os

import math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from Classifiers.models.Efficient_net_custom import EfficientNet

from med_img_datasets_clf.dataset_utils import get_dataloader_mammo
from mammo_metrics import aggregate_mammo_predictions, normalize_mammo_dataset_name
from metrics import auroc

def compute_accuracy_np_array(gt, pred):
    return np.mean(gt == pred)


def _pick_existing_file(candidates):
    for path in candidates:
        path = Path(path)
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
        train_pool = df.copy()
        test_df = df.iloc[0:0].copy()

    if train_pool.empty:
        return train_pool, train_pool.copy(), test_df

    rng = np.random.RandomState(seed)
    if "patient_id" in train_pool.columns:
        patient_series = train_pool["patient_id"].astype(str).fillna("unknown")
    else:
        patient_series = pd.Series(
            [f"row-{idx}" for idx in train_pool.index], index=train_pool.index
        )

    patient_ids = patient_series.unique()
    if len(patient_ids) <= 1:
        valid_df = train_pool.sample(n=min(1, len(train_pool)), random_state=seed).copy()
        train_df = train_pool.drop(valid_df.index).copy()
        if train_df.empty:
            train_df = train_pool.copy()
    else:
        n_val = max(1, int(round(len(patient_ids) * val_ratio)))
        n_val = min(n_val, len(patient_ids) - 1)
        val_patients = set(rng.choice(patient_ids, size=n_val, replace=False))
        valid_mask = patient_series.isin(val_patients)
        valid_df = train_pool[valid_mask].copy()
        train_df = train_pool[~valid_mask].copy()
        if train_df.empty:
            train_df = train_pool.copy()
            valid_df = train_pool.iloc[0:0].copy()

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _load_cbis_manifest(args):
    data_dir = Path(args.data_dir)
    csv_override = Path(args.csv_file) if args.csv_file else None
    csv_candidates = []
    if csv_override is not None:
        if csv_override.is_absolute():
            csv_candidates.append(csv_override)
        else:
            csv_candidates.append(data_dir / csv_override)
    csv_candidates.extend(
        [
            data_dir / "processed_png_full" / "manifest_full_mammograms.csv",
            data_dir / "metadata" / "manifest_full_mammograms.csv",
            data_dir / "manifest_full_mammograms.csv",
            data_dir / "metadata.csv",
        ]
    )
    manifest_path = _pick_existing_file(csv_candidates)
    if manifest_path is None:
        raise FileNotFoundError(
            "Could not find CBIS manifest CSV. Checked: "
            + ", ".join(str(p) for p in csv_candidates)
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

    if "png_path" not in df.columns:
        raise ValueError(
            f"CBIS manifest `{manifest_path}` is missing required column `png_path`."
        )

    label_col = args.label if args.label in df.columns else "cancer"
    if label_col not in df.columns:
        raise ValueError(
            f"CBIS label column `{args.label}` not found and no fallback `cancer` column available."
        )

    for col in ["CLIP_V1_bin", "SCAR_V1_bin", "MARK_V1_bin", "MOLE_V1_bin"]:
        if col not in df.columns:
            df[col] = 0
    if "fold" not in df.columns:
        df["fold"] = 0

    df = df[df[label_col].notna()].copy()
    return df, label_col


def stratified_sample(df, n, label):
    df_0 = df[df[label] == 0]
    df_1 = df[df[label] == 1]

    # Sample n/2 from each class
    df_0_sampled = df_0.sample(n=n // 2, random_state=42)
    df_1_sampled = df_1.sample(n=n // 2, random_state=42)

    return pd.concat([df_0_sampled, df_1_sampled])


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def do_experiments(args, device):
    if 'efficientnetv2' in args.arch:
        args.model_base_name = 'efficientv2_s'
    elif 'efficientnet_b5_ns' in args.arch:
        args.model_base_name = 'efficientnetb5'
    else:
        args.model_base_name = args.arch

    args.data_dir = Path(args.data_dir)

    dataset_name = normalize_mammo_dataset_name(args.dataset)
    oof_df = pd.DataFrame()
    for fold in range(args.start_fold, args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)
        if dataset_name == "rsna":
            args.df = pd.read_csv(args.data_dir / args.csv_file)
            args.df = args.df.fillna(0)
            print(f"df shape: {args.df.shape}")
            print(args.df.columns)
            args.train_folds = args.df[(args.df['fold'] == 1) | (args.df['fold'] == 2)].reset_index(drop=True)
            args.valid_folds = args.df[args.df['fold'] == 3].reset_index(drop=True)
            print(f"train_folds shape: {args.train_folds.shape}")
            print(f"valid_folds shape: {args.valid_folds.shape}")

        elif dataset_name == "vindr":
            args.df = pd.read_csv(args.data_dir / args.csv_file)
            args.df = args.df.fillna(0)
            print(f"df shape: {args.df.shape}")
            print(args.df.columns)
            args.train_folds = args.df[args.df['split_new'] == "train"].reset_index(drop=True)
            args.valid_folds = args.df[args.df['split_new'] == "val"].reset_index(drop=True)
            args.test_folds = args.df[args.df['split_new'] == "test"].reset_index(drop=True)

            print(f"train_folds shape: {args.train_folds.shape}")
            print(f"valid_folds shape: {args.valid_folds.shape}")
            print(f"test_folds shape: {args.test_folds.shape}")
            print(args.train_folds.columns)

            args.train_folds = args.train_folds.rename(columns={'ImageLateralityFinal': 'laterality'})
            args.valid_folds = args.valid_folds.rename(columns={'ImageLateralityFinal': 'laterality'})

            args.BCE_weights = {}
            args.BCE_weights[f"fold{args.cur_fold}"] = args.train_folds[args.train_folds[args.label] == 0].shape[0] / \
                                                       args.train_folds[args.train_folds[args.label] == 1].shape[0]
        elif dataset_name == "cbis":
            args.df, label_col = _load_cbis_manifest(args)
            args.label = label_col
            train_df, valid_df, test_df = _build_cbis_splits(
                args.df,
                seed=getattr(args, "seed", 42),
                val_ratio=getattr(args, "cbis_val_ratio", 0.1),
            )
            args.train_folds = train_df.fillna(0).reset_index(drop=True)
            args.valid_folds = valid_df.fillna(0).reset_index(drop=True)
            args.test_folds = test_df.fillna(0).reset_index(drop=True)

            print(f"df shape: {args.df.shape}")
            print(args.df.columns)
            print(f"train_folds shape: {args.train_folds.shape}")
            print(f"valid_folds shape: {args.valid_folds.shape}")
            print(f"test_folds shape: {args.test_folds.shape}")

            neg = args.train_folds[args.train_folds[args.label] == 0].shape[0]
            pos = args.train_folds[args.train_folds[args.label] == 1].shape[0]
            pos_weight = float(neg / pos) if pos > 0 else 1.0
            if not hasattr(args, "BCE_weights"):
                args.BCE_weights = {}
            args.BCE_weights[f"fold{args.cur_fold}"] = pos_weight

        _oof_df = train_loop(args, device)
        oof_df = pd.concat([oof_df, _oof_df])

    oof_df = oof_df.reset_index(drop=True)
    oof_df['prediction_bin'] = oof_df['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)

    oof_df_agg = aggregate_mammo_predictions(
        oof_df[["patient_id", "laterality", args.label, "prediction", "fold"]],
        label_col=args.label,
    )

    print('================ CV ================')
    aucroc = auroc(y_true=oof_df_agg[args.label].values, y_score=oof_df_agg['prediction'].values)
    oof_df_agg['prediction'] = oof_df_agg['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)

    oof_df_agg_cancer = oof_df_agg[oof_df_agg[args.label] == 1]
    oof_df_agg_cancer = oof_df_agg_cancer.copy()
    oof_df_agg_cancer['prediction'] = oof_df_agg_cancer['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
    if oof_df_agg_cancer.empty:
        acc_cancer = float("nan")
    else:
        acc_cancer = compute_accuracy_np_array(
            oof_df_agg_cancer[args.label].values,
            oof_df_agg_cancer['prediction'].values
        )

    print(f'AUC-ROC: {aucroc}, acc +ve {args.label} patients: {acc_cancer * 100}')
    print('\n')
    print(oof_df.head(10))
    print(f"Results shape: {oof_df.shape}")
    print('\n')
    print(args.output_path)
    oof_df.to_csv(args.output_path / f'seed_{args.seed}_n_folds_{args.n_folds}_outputs.csv', index=False)


def train_loop(args, device):
    print(f'\n================== fold: {args.cur_fold} training ======================')
    args.train_folds = args.train_folds.sample(frac=args.data_frac, random_state=1, ignore_index=True)
    args.image_encoder_type = None

    if args.running_interactive:
        args.train_folds = stratified_sample(args.train_folds, 100, label=args.label)
        args.valid_folds = stratified_sample(args.valid_folds, 100, label=args.label)

    train_loader, valid_loader = get_dataloader_mammo(args)

    n_class = 1

    attr_embs = None
    model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=n_class)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.epochs_warmup, num_training_steps=args.epochs, num_cycles=args.num_cycles
    )

    model = model.to(device)
    logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')
    pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to('cuda')
    print(f'pos_wt: {pos_wt}')
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_wt)

    best_aucroc = 0.
    for epoch in range(args.epochs):
        start_time = time.time()
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, attr_embs, logger, device
        )
        scheduler.step()
        avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, args, device, epoch, attr_embs=attr_embs, logger=logger
        )
        args.valid_folds['prediction'] = predictions
        logger.add_scalar(f'valid/{args.label}/train_loss', avg_loss, epoch + 1)
        logger.add_scalar(f'valid/{args.label}/val_loss', avg_val_loss, epoch + 1)

        valid_agg = None
        dataset_name = normalize_mammo_dataset_name(args.dataset)
        if dataset_name == "vindr":
            valid_agg = args.valid_folds
        elif dataset_name in {"rsna", "cbis"}:
            valid_agg = aggregate_mammo_predictions(
                args.valid_folds[["patient_id", "laterality", args.label, "prediction", "fold"]],
                label_col=args.label,
            )

        aucroc = auroc(valid_agg[args.label].values, valid_agg['prediction'].values)
        valid_agg_cancer = valid_agg[valid_agg[args.label] == 1]
        valid_agg_cancer = valid_agg_cancer.copy()
        valid_agg_cancer['prediction'] = valid_agg_cancer['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
        if valid_agg_cancer.empty:
            acc_cancer = float("nan")
        else:
            acc_cancer = compute_accuracy_np_array(
                valid_agg_cancer[args.label].values,
                valid_agg_cancer['prediction'].values
            )

        valid_agg['prediction'] = valid_agg['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
        elapsed = time.time() - start_time
        print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s'
            )
        print(f'Epoch {epoch + 1} - AUC-ROC Score: {aucroc:.4f}, Acc +ve {args.label}: {acc_cancer * 100:.4f}')
        logger.add_scalar(f'valid/{args.label}/AUC-ROC', aucroc, epoch + 1)

        logger.add_scalar(f'valid/{args.label}/+ve Acc Score', acc_cancer, epoch + 1)


        if best_aucroc < aucroc:
            best_aucroc = aucroc
            model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
            print(f'Epoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model')
            torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'auroc': aucroc
                    }, args.chk_pt_path / model_name
            )


        model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
        predictions = torch.load(args.chk_pt_path / model_name, map_location='cpu')['predictions']
        args.valid_folds['prediction'] = predictions
        print(f'[Fold{args.cur_fold}], AUC-ROC Score: {best_aucroc:.4f}')
    torch.cuda.empty_cache()
    gc.collect()
    return args.valid_folds


def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, attr_embs, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = end = time.time()

    progress_iter = tqdm(enumerate(train_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
                         total=len(train_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        batch_size = inputs.size(0)

        with torch.cuda.amp.autocast(enabled=args.apex):
                y_preds = model(inputs)

        labels = data['y'].float().to(device)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]['lr']],
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}'
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr']))

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar('train/epoch', epoch, index)
            logger.add_scalar('train/iter_loss', losses.avg, index)
            logger.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'], index)

    return losses.avg


def valid_fn(valid_loader, model, criterion, args, device, epoch=1, attr_embs=None, logger=None):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()

    progress_iter = tqdm(enumerate(valid_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
                         total=len(valid_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        batch_size = inputs.size(0)
        inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        with torch.no_grad():
            y_preds = model(inputs)


        labels = data['y'].float().to(device)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)
        preds.append(y_preds.squeeze(1).sigmoid().to('cpu').numpy())

        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))

        if (step % args.log_freq == 0 or step == (len(valid_loader) - 1)) and logger is not None:
            index = step + len(valid_loader) * epoch
            logger.add_scalar('valid/iter_loss', losses.avg, index)

    if (
            args.label.lower() == "density" or args.label.lower() == "birads" or args.label.lower() == "race" or
            args.label.lower() == "age_group"):
        predictions = np.array(preds)
    else:
        predictions = np.concatenate(preds)
    return losses.avg, predictions
