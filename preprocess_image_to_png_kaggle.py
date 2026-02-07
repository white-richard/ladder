import argparse
import ctypes
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pydicom
from pydicom.filebase import DicomBytesIO
from tqdm.auto import tqdm

try:
    import dicomsdl  # type: ignore

    HAS_DICOMSDL = True
except Exception:
    dicomsdl = None
    HAS_DICOMSDL = False

try:
    import torch

    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    from nvidia.dali.backend import TensorGPU, TensorListGPU
    from nvidia.dali.types import DALIDataType

    HAS_DALI = True
except Exception:
    fn = None
    types = None
    pipeline_def = None
    TensorGPU = None
    TensorListGPU = None
    DALIDataType = None
    HAS_DALI = False

try:
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    HAS_VOI_LUT = True
except Exception:
    apply_voi_lut = None
    HAS_VOI_LUT = False

JPEG2000_UID = "1.2.840.10008.1.2.4.90"

to_torch_type = {}
if HAS_DALI and HAS_TORCH:
    to_torch_type = {
        types.DALIDataType.FLOAT: torch.float32,
        types.DALIDataType.FLOAT64: torch.float64,
        types.DALIDataType.FLOAT16: torch.float16,
        types.DALIDataType.UINT8: torch.uint8,
        types.DALIDataType.INT8: torch.int8,
        types.DALIDataType.UINT16: torch.int16,
        types.DALIDataType.INT16: torch.int16,
        types.DALIDataType.INT32: torch.int32,
        types.DALIDataType.INT64: torch.int64,
    }


def feed_ndarray(dali_tensor, arr, cuda_stream=None):
    dali_type = to_torch_type[dali_tensor.dtype]
    assert dali_type == arr.dtype, (
        "The element type of DALI tensor doesn't match the target PyTorch tensor: "
        f"{dali_type} vs {arr.dtype}"
    )
    assert dali_tensor.shape() == list(arr.size()), (
        f"Shapes do not match: DALI tensor has size {dali_tensor.shape()}, "
        f"but target has size {list(arr.size())}"
    )
    cuda_stream = types._raw_cuda_stream(cuda_stream)
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


if HAS_DALI:

    @pipeline_def
    def j2k_decode_pipeline(j2kfiles):
        jpegs, _ = fn.readers.file(files=j2kfiles)
        images = fn.experimental.decoders.image(
            jpegs,
            device="mixed",
            output_type=types.ANY_DATA,
            dtype=DALIDataType.UINT16,
        )
        return images


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    obj_cols = out.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        out[col] = out[col].replace("", np.nan)
    return out


def sanitize_subject_id(subject_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(subject_id))


def np_CountUpContinuingOnes(b_arr):
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]
    return right - left - 1


def np_ExtractBreast(img):
    if img.size == 0:
        return img
    if img.ndim != 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        return img

    img_copy = img.copy()
    max_value = int(np.max(img_copy)) if img_copy.size else 0
    threshold = 40 if max_value <= 255 else int(round(max_value * (40.0 / 255.0)))
    img = np.where(img_copy <= threshold, 0, img_copy)
    height, _ = img.shape

    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    if continuing_ones.size == 0 or continuing_ones.max() <= 0:
        return img_copy
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    if col_ind.size == 0:
        return img_copy
    img = img[:, col_ind]

    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    if continuing_ones.size == 0 or continuing_ones.max() <= 0:
        return img_copy
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]
    if row_ind.size == 0:
        return img_copy
    return img_copy[row_ind][:, col_ind]


def torch_CountUpContinuingOnes(b_arr):
    left = torch.arange(len(b_arr), device=b_arr.device)
    left[b_arr > 0] = 0
    left = torch.cummax(left, dim=-1)[0]

    rev_arr = torch.flip(b_arr, [-1])
    right = torch.arange(len(rev_arr), device=b_arr.device)
    right[rev_arr > 0] = 0
    right = torch.cummax(right, dim=-1)[0]
    right = len(rev_arr) - 1 - torch.flip(right, [-1])
    return right - left - 1


def torch_ExtractBreast(img_ori):
    if img_ori.numel() == 0:
        return img_ori
    max_value = int(img_ori.max().item())
    threshold = 40 if max_value <= 255 else int(round(max_value * (40.0 / 255.0)))
    img = torch.where(img_ori <= threshold, torch.zeros_like(img_ori), img_ori)
    height, _ = img.shape

    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].to(torch.float32).std(dim=0) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)
    if continuing_ones.numel() == 0 or continuing_ones.max().item() <= 0:
        return img_ori
    col_ind = torch.where(continuing_ones == continuing_ones.max())[0]
    if col_ind.numel() == 0:
        return img_ori
    img = img[:, col_ind]

    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].to(torch.float32).std(dim=1) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)
    if continuing_ones.numel() == 0 or continuing_ones.max().item() <= 0:
        return img_ori
    row_ind = torch.where(continuing_ones == continuing_ones.max())[0]
    if row_ind.numel() == 0:
        return img_ori
    return img_ori[row_ind][:, col_ind]


def parse_subject_id(subject_id: str) -> Dict[str, Optional[str]]:
    sid = str(subject_id).strip()
    match = re.match(
        r"^(?P<task>Calc|Mass)-(?P<split>Training|Test)_(?P<patient>.+)_(?P<laterality>LEFT|RIGHT)_(?P<view>CC|MLO)$",
        sid,
    )
    if not match:
        return {
            "task": None,
            "split": None,
            "patient_id": None,
            "laterality": None,
            "view": None,
        }
    split = "train" if match.group("split").lower() == "training" else "test"
    return {
        "task": match.group("task").lower(),
        "split": split,
        "patient_id": match.group("patient"),
        "laterality": match.group("laterality").upper(),
        "view": match.group("view").upper(),
    }


def normalize_image(arr: np.ndarray, bitdepth: int) -> Tuple[np.ndarray, float, float]:
    arr = arr.astype(np.float32)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val <= min_val:
        if bitdepth == 8:
            return np.zeros(arr.shape, dtype=np.uint8), min_val, max_val
        return np.zeros(arr.shape, dtype=np.uint16), min_val, max_val

    arr = (arr - min_val) / (max_val - min_val)
    if bitdepth == 8:
        return np.rint(arr * 255.0).astype(np.uint8), min_val, max_val
    return np.rint(arr * 65535.0).astype(np.uint16), min_val, max_val


def convert_dicom_to_j2k(dicom_path: str, out_jp2_path: str) -> str:
    os.makedirs(os.path.dirname(out_jp2_path), exist_ok=True)
    with open(dicom_path, "rb") as fp:
        raw = DicomBytesIO(fp.read())
        ds = pydicom.dcmread(raw)

    pixel_data = ds.PixelData
    offset = pixel_data.find(b"\x00\x00\x00\x0C")
    if offset >= 0:
        bitstream = pixel_data[offset:]
    else:
        bitstream = pixel_data

    with open(out_jp2_path, "wb") as binary_file:
        binary_file.write(bytearray(bitstream))
    return out_jp2_path


def decode_jp2_with_dali(jp2_path: str) -> np.ndarray:
    if not (HAS_DALI and HAS_TORCH and torch.cuda.is_available()):
        raise RuntimeError(
            "GPU JPEG2000 decode requested, but DALI/Torch CUDA is unavailable."
        )
    pipe = j2k_decode_pipeline(
        [jp2_path], batch_size=1, num_threads=2, device_id=0, debug=False
    )
    pipe.build()
    out = pipe.run()
    img = out[0][0]
    img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
    feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))
    arr = img_torch.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def read_dicom_pixels(
    dicom_path: str,
    subject_id: str,
    use_gpu_j2k: bool = False,
    voi_lut: bool = False,
    j2k_folder: Optional[str] = None,
) -> Tuple[np.ndarray, str, Dict[str, str]]:
    header = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
    transfer_syntax = str(getattr(getattr(header, "file_meta", None), "TransferSyntaxUID", ""))
    photometric = str(getattr(header, "PhotometricInterpretation", ""))
    meta = {"transfer_syntax_uid": transfer_syntax}

    if transfer_syntax == JPEG2000_UID and use_gpu_j2k:
        if j2k_folder is None:
            raise RuntimeError("`j2k_folder` is required when `use_gpu_j2k` is enabled.")
        safe_id = sanitize_subject_id(subject_id)
        jp2_path = os.path.join(j2k_folder, f"{safe_id}.jp2")
        convert_dicom_to_j2k(dicom_path, jp2_path)
        try:
            arr = decode_jp2_with_dali(jp2_path)
        finally:
            if os.path.exists(jp2_path):
                os.remove(jp2_path)
        return arr, photometric, meta

    if HAS_DICOMSDL:
        try:
            dcm = dicomsdl.open(dicom_path)
            arr = dcm.pixelData()
            info = dcm.getPixelDataInfo()
            photometric = str(info.get("PhotometricInterpretation", photometric))
            return np.asarray(arr), photometric, meta
        except Exception:
            pass

    ds = pydicom.dcmread(dicom_path)
    try:
        arr = ds.pixel_array
    except Exception as exc:
        if transfer_syntax == JPEG2000_UID:
            raise RuntimeError(
                "JPEG2000 decode failed via pydicom handlers. Install GDCM/pylibjpeg "
                "or rerun with --use_gpu_j2k when CUDA+DALI are available."
            ) from exc
        raise
    if voi_lut and HAS_VOI_LUT:
        try:
            if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
    photometric = str(getattr(ds, "PhotometricInterpretation", photometric))
    return np.asarray(arr), photometric, meta


def load_metadata_full_images(metadata_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv, dtype=str)
    meta = strip_object_columns(meta)
    meta = meta.rename(columns={c: normalize_colname(c) for c in meta.columns})

    required_cols = ["subject_id", "file_location", "series_description"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Missing required column `{col}` in metadata CSV: {metadata_csv}")

    full_df = meta[
        meta["series_description"].fillna("").str.lower() == "full mammogram images"
    ].copy()
    full_df = full_df.reset_index(drop=True)
    return full_df


def resolve_file_location(cbis_root: str, file_location: str) -> str:
    loc = str(file_location).strip()
    loc = loc[2:] if loc.startswith("./") else loc
    if os.path.isabs(loc):
        return loc
    return os.path.join(cbis_root, loc)


def choose_dicom(dcm_files: List[str]) -> str:
    return max(dcm_files, key=lambda p: (os.path.getsize(p), p))


def resolve_full_mammo_dicoms(cbis_root: str, full_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    warnings = []

    for _, row in full_df.iterrows():
        subject_id = str(row["subject_id"]).strip()
        full_dir = resolve_file_location(cbis_root, row["file_location"])
        dcm_files = sorted(glob.glob(os.path.join(full_dir, "*.dcm")))
        if not dcm_files:
            dcm_files = sorted(
                glob.glob(os.path.join(full_dir, "**", "*.dcm"), recursive=True)
            )

        if not dcm_files:
            warnings.append(f"[missing] {subject_id}: no DICOM found in {full_dir}")
            continue

        selected = choose_dicom(dcm_files)
        if len(dcm_files) > 1:
            warnings.append(
                f"[multiple] {subject_id}: {len(dcm_files)} DICOMs in {full_dir}, selected largest {selected}"
            )

        rows.append(
            {
                "subject_id": subject_id,
                "file_location": row["file_location"],
                "dicom_dir": full_dir,
                "dicom_path": selected,
            }
        )

    resolved_df = pd.DataFrame(rows)
    return resolved_df, warnings


def standardize_case_description(
    path: str, task: str, split: str
) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)
    df = strip_object_columns(df)
    df = df.rename(columns={c: normalize_colname(c) for c in df.columns})

    for required in ["patient_id", "left_or_right_breast", "image_view"]:
        if required not in df.columns:
            raise ValueError(f"Missing `{required}` in case description CSV: {path}")

    df["patient_id"] = df["patient_id"].fillna("").astype(str).str.strip()
    df["laterality"] = (
        df["left_or_right_breast"].fillna("").astype(str).str.strip().str.upper()
    )
    df["view"] = df["image_view"].fillna("").astype(str).str.strip().str.upper()

    split_token = "Training" if split == "train" else "Test"
    task_token = "Calc" if task == "calc" else "Mass"
    df["subject_id"] = (
        task_token
        + "-"
        + split_token
        + "_"
        + df["patient_id"]
        + "_"
        + df["laterality"]
        + "_"
        + df["view"]
    )
    df["task"] = task
    df["split"] = split
    df["source_csv"] = os.path.basename(path)

    keep_cols = [
        "subject_id",
        "task",
        "split",
        "source_csv",
        "patient_id",
        "laterality",
        "view",
        "pathology",
        "assessment",
        "subtlety",
        "breast_density",
        "abnormality_id",
        "abnormality_type",
        "calc_type",
        "calc_distribution",
        "mass_shape",
        "mass_margins",
        "image_file_path",
        "cropped_image_file_path",
        "roi_mask_file_path",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    return df[existing_cols].copy()


def load_all_case_descriptions(args) -> pd.DataFrame:
    frames = []
    specs = [
        (args.calc_train_csv, "calc", "train"),
        (args.calc_test_csv, "calc", "test"),
        (args.mass_train_csv, "mass", "train"),
        (args.mass_test_csv, "mass", "test"),
    ]
    for path, task, split in specs:
        if path and os.path.exists(path):
            frame = standardize_case_description(path, task, split)
            if not frame.empty:
                frames.append(frame)
        elif path:
            print(f"[warn] Case description CSV not found, skipping: {path}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_worklist(args) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    full_df = load_metadata_full_images(args.metadata_csv)
    resolved_df, resolve_warnings = resolve_full_mammo_dicoms(args.cbis_root, full_df)

    labels_df = load_all_case_descriptions(args)
    if labels_df.empty:
        work_df = resolved_df.copy()
    else:
        work_df = resolved_df.merge(labels_df, on="subject_id", how="left")

    for col in ["patient_id", "laterality", "view", "task", "split"]:
        if col not in work_df.columns:
            work_df[col] = np.nan

    parsed = pd.DataFrame([parse_subject_id(s) for s in work_df["subject_id"].tolist()])
    for col in ["patient_id", "laterality", "view", "task", "split"]:
        work_df[col] = work_df[col].fillna(parsed[col])

    work_df["task"] = work_df["task"].fillna("unknown").str.lower()
    work_df["split"] = work_df["split"].fillna("unknown").str.lower()
    work_df["split_group"] = work_df["task"] + "_" + work_df["split"]
    work_df["safe_subject_id"] = work_df["subject_id"].map(sanitize_subject_id)
    work_df["png_path"] = work_df.apply(
        lambda row: os.path.join(
            args.out_dir, row["split_group"], f"{row['safe_subject_id']}.png"
        ),
        axis=1,
    )

    label_probe_cols = [
        c
        for c in ["pathology", "assessment", "subtlety", "breast_density", "abnormality_id"]
        if c in work_df.columns
    ]
    joined_with_labels = 0
    if label_probe_cols:
        joined_with_labels = int(work_df[label_probe_cols].notna().any(axis=1).sum())

    counts = {
        "full_mammo_rows": int(len(full_df)),
        "resolved_dicom_rows": int(len(resolved_df)),
        "joined_with_labels": joined_with_labels,
    }
    return work_df, resolve_warnings, counts


def process_one_image(record: Dict, args, j2k_folder: str) -> Dict:
    subject_id = record["subject_id"]
    dicom_path = record["dicom_path"]
    png_path = record["png_path"]

    if (
        (not args.overwrite)
        and os.path.exists(png_path)
        and os.path.getsize(png_path) > 0
    ):
        return {
            "subject_id": subject_id,
            "dicom_path": dicom_path,
            "png_path": png_path,
            "status": "skipped_existing",
        }

    try:
        arr, photometric, meta = read_dicom_pixels(
            dicom_path=dicom_path,
            subject_id=subject_id,
            use_gpu_j2k=args.use_gpu_j2k,
            voi_lut=args.voi_lut,
            j2k_folder=j2k_folder,
        )
        arr = np.asarray(arr)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected pixel array shape: {arr.shape}")

        pre_min = float(arr.min())
        pre_max = float(arr.max())

        border = max(int(args.border_crop), 0)
        if border > 0 and arr.shape[0] > 2 * border and arr.shape[1] > 2 * border:
            arr = arr[border:-border, border:-border]

        if str(photometric).upper() == "MONOCHROME1":
            arr = arr.max() - arr

        img, norm_min, norm_max = normalize_image(arr, args.bitdepth)
        zero_dynamic = bool(norm_max <= norm_min)

        if not args.disable_breast_crop:
            img = np_ExtractBreast(img)

        if not args.no_resize:
            img = cv2.resize(
                img, (args.width, args.height), interpolation=cv2.INTER_AREA
            )

        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        ok = cv2.imwrite(png_path, img)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")

        return {
            "subject_id": subject_id,
            "dicom_path": dicom_path,
            "png_path": png_path,
            "status": "ok",
            "pre_min": pre_min,
            "pre_max": pre_max,
            "post_min": float(img.min()) if img.size else 0.0,
            "post_max": float(img.max()) if img.size else 0.0,
            "zero_dynamic_range": zero_dynamic,
            "transfer_syntax_uid": meta.get("transfer_syntax_uid", ""),
        }
    except Exception as exc:
        return {
            "subject_id": subject_id,
            "dicom_path": dicom_path,
            "png_path": png_path,
            "status": "error",
            "error": str(exc),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CBIS-DDSM full mammogram DICOMs to normalized PNGs."
    )
    parser.add_argument("--cbis_root", type=str, required=True, help="CBIS-DDSM root directory")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--calc_train_csv", type=str, default=None)
    parser.add_argument("--calc_test_csv", type=str, default=None)
    parser.add_argument("--mass_train_csv", type=str, default=None)
    parser.add_argument("--mass_test_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--width", type=int, default=912)
    parser.add_argument("--height", type=int, default=1520)
    parser.add_argument("--no_resize", action="store_true")

    parser.add_argument("--save_manifest", type=bool, default=True)
    parser.add_argument("--manifest_path", type=str, default=None)

    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--use_gpu_j2k", action="store_true")
    parser.add_argument("--disable_breast_crop", action="store_true")
    parser.add_argument("--bitdepth", type=int, choices=[8, 16], default=8)
    parser.add_argument("--voi_lut", action="store_true")
    parser.add_argument("--border_crop", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def apply_default_paths(args):
    metadata_dir = os.path.join(args.cbis_root, "metadata")

    if args.metadata_csv is None:
        args.metadata_csv = os.path.join(metadata_dir, "metadata.csv")
    if args.calc_train_csv is None:
        args.calc_train_csv = os.path.join(metadata_dir, "calc_case_description_train_set.csv")
    if args.calc_test_csv is None:
        args.calc_test_csv = os.path.join(metadata_dir, "calc_case_description_test_set.csv")
    if args.mass_train_csv is None:
        args.mass_train_csv = os.path.join(metadata_dir, "mass_case_description_train_set.csv")
    if args.mass_test_csv is None:
        args.mass_test_csv = os.path.join(metadata_dir, "mass_case_description_test_set.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.cbis_root, "processed_png_full")
    if args.manifest_path is None:
        args.manifest_path = os.path.join(args.out_dir, "manifest_full_mammograms.csv")


def main():
    args = parse_args()
    apply_default_paths(args)

    os.makedirs(args.out_dir, exist_ok=True)
    j2k_folder = os.path.join(args.out_dir, "_tmp_j2k")
    os.makedirs(j2k_folder, exist_ok=True)

    if args.use_gpu_j2k:
        if not HAS_DALI or not HAS_TORCH or not torch.cuda.is_available():
            raise RuntimeError(
                "You set --use_gpu_j2k, but DALI + Torch CUDA are unavailable. "
                "Disable --use_gpu_j2k to use pydicom decoding path instead."
            )
        if args.n_jobs != 1:
            print("[warn] --use_gpu_j2k enabled; forcing --n_jobs=1 for stable GPU decode.")
            args.n_jobs = 1

    work_df, resolve_warnings, counts = build_worklist(args)

    print(f"Full mammogram rows in metadata: {counts['full_mammo_rows']}")
    print(f"Rows with resolved DICOM on disk: {counts['resolved_dicom_rows']}")
    print(f"Rows with joined labels: {counts['joined_with_labels']}")

    if resolve_warnings:
        print("\nResolution warnings (up to first 20):")
        for msg in resolve_warnings[:20]:
            print(msg)
        if len(resolve_warnings) > 20:
            print(f"... {len(resolve_warnings) - 20} more warnings omitted")

    if args.limit is not None:
        work_df = work_df.head(args.limit).copy()
        print(f"\nApplying --limit={args.limit}; worklist rows now: {len(work_df)}")

    manifest_cols_prefix = [
        "subject_id",
        "dicom_path",
        "png_path",
        "patient_id",
        "laterality",
        "view",
        "task",
        "split",
        "split_group",
    ]
    manifest_cols = manifest_cols_prefix + [
        c for c in work_df.columns if c not in manifest_cols_prefix
    ]
    manifest_df = work_df[manifest_cols].copy()

    if args.save_manifest:
        os.makedirs(os.path.dirname(args.manifest_path), exist_ok=True)
        manifest_df.to_csv(args.manifest_path, index=False)
        print(f"\nManifest saved: {args.manifest_path}")

    print("\nManifest preview:")
    preview_cols = [c for c in manifest_cols_prefix if c in manifest_df.columns]
    print(manifest_df[preview_cols].head(5).to_string(index=False))

    image_df = manifest_df[["subject_id", "dicom_path", "png_path"]].drop_duplicates()
    records = image_df.to_dict("records")
    print(f"\nUnique images to process: {len(records)}")

    if args.n_jobs == 1:
        results = [
            process_one_image(record, args, j2k_folder)
            for record in tqdm(records, total=len(records))
        ]
    else:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_one_image)(record, args, j2k_folder)
            for record in tqdm(records, total=len(records))
        )

    result_df = pd.DataFrame(results)
    n_ok = int((result_df["status"] == "ok").sum()) if not result_df.empty else 0
    n_skip = (
        int((result_df["status"] == "skipped_existing").sum()) if not result_df.empty else 0
    )
    n_err = int((result_df["status"] == "error").sum()) if not result_df.empty else 0
    n_zero = (
        int(result_df.get("zero_dynamic_range", pd.Series(dtype=bool)).fillna(False).sum())
        if not result_df.empty
        else 0
    )

    print("\nProcessing summary:")
    print(f"ok: {n_ok}")
    print(f"skipped_existing: {n_skip}")
    print(f"errors: {n_err}")
    print(f"zero_dynamic_range_warnings: {n_zero}")

    if not result_df.empty:
        ok_stats = result_df[result_df["status"] == "ok"][
            ["subject_id", "pre_min", "pre_max", "post_min", "post_max"]
        ]
        if not ok_stats.empty:
            print("\nPixel stats preview:")
            print(ok_stats.head(5).to_string(index=False))

        if n_err > 0:
            print("\nErrors (up to first 20):")
            for _, row in result_df[result_df["status"] == "error"].head(20).iterrows():
                print(f"[error] {row['subject_id']} :: {row.get('error', 'unknown error')}")

    if os.path.isdir(j2k_folder) and not os.listdir(j2k_folder):
        os.rmdir(j2k_folder)


if __name__ == "__main__":
    main()
"""
python preprocess_image_to_png_kaggle.py \
  --cbis_root /home/richw/.code/datasets/cbis-ddsm \
  --limit 20 \
  --n_jobs 4 \
  --bitdepth 8

python preprocess_image_to_png_kaggle.py \
  --cbis_root /home/richw/.code/datasets/cbis-ddsm \
  --out_dir /home/richw/.code/datasets/cbis-ddsm/processed_png_full \
  --manifest_path /home/richw/.code/datasets/cbis-ddsm/processed_png_full/manifest_full_mammograms.csv \
  --n_jobs 8

"""