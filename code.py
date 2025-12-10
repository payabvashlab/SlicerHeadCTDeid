import os
from datetime import datetime
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from nibabel.processing import resample_to_output

# =========================
# CONFIG
# =========================
FILTER_OUT_BONE_KERNEL = False     # if True, skip series with "BONE" kernel/description
NIFTI_OUT_DIRNAME = None           # e.g. "nifti" to also save iso NIfTI, or None to skip
ISO_SPACING = (1.0, 1.0, 1.0)      # mm
OUTPUT_FILENAME = "output.csv"
PROCESSED_FILENAME = "process.csv"

# =========================
# DICOM utils
# =========================
def is_dicom_strict(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        return hasattr(ds, "file_meta") and hasattr(ds.file_meta, "TransferSyntaxUID")
    except Exception:
        return False

def list_direct_dicoms(folder_path: str):
    try:
        with os.scandir(folder_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if name.startswith("."):
                    continue
                if name.lower().endswith((".json", ".txt", ".xml", ".csv", ".ini", ".nfo")):
                    continue
                fp = entry.path
                if is_dicom_strict(fp):
                    yield fp
    except FileNotFoundError:
        return

def has_direct_dicoms(folder_path: str) -> bool:
    for _ in list_direct_dicoms(folder_path):
        return True
    return False

def load_scan(path: str):
    return pydicom.dcmread(path, force=True)

def get_pixels_hu(ds: pydicom.Dataset) -> np.ndarray:
    img = ds.pixel_array.astype(np.int16)
    slope = getattr(ds, "RescaleSlope", 1)
    intercept = getattr(ds, "RescaleIntercept", 0)
    if slope != 1:
        img = (img.astype(np.float64) * slope).astype(np.int16)
    img = img + np.int16(intercept)
    return img

def normalize_time_str(t: str) -> str:
    if not t:
        return ""
    if isinstance(t, bytes):
        t = t.decode(errors="ignore")
    t = str(t)
    if "." in t:
        t = t.split(".")[0]
    t = t.strip()
    if len(t) in (3, 4, 5):
        t = t.zfill(6)
    if len(t) == 6:
        try:
            return datetime.strptime(t, "%H%M%S").strftime("%H:%M:%S")
        except Exception:
            return ""
    return ""

def normalize_date_str(d: str) -> str:
    if not d:
        return ""
    if isinstance(d, bytes):
        d = d.decode(errors="ignore")
    d = str(d).strip()
    if len(d) == 8:
        try:
            return datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return ""
    return ""

def is_localizer(ds: pydicom.Dataset) -> bool:
    it = [s.upper() for s in getattr(ds, "ImageType", [])]
    desc = str(getattr(ds, "SeriesDescription", "")).upper()
    return any(x in it for x in ("LOCALIZER", "SCOUT")) or ("LOCALIZER" in desc or "SCOUT" in desc)

def kernel_is_bone(ds: pydicom.Dataset) -> bool:
    desc = str(getattr(ds, "SeriesDescription", "")).upper()
    kern = str(getattr(ds, "ConvolutionKernel", "")).upper()
    return ("BONE" in desc) or ("BONE" in kern)

# =========================
# Safe affine helpers
# =========================
def _safe_float(x, default):
    try:
        v = float(x)
        if not np.isfinite(v) or v <= 0:
            return default
        return v
    except Exception:
        return default

def get_px_py(ds):
    px, py = 1.0, 1.0
    ps = getattr(ds, "PixelSpacing", None)
    if ps and len(ps) >= 2:
        px = _safe_float(ps[0], 1.0)
        py = _safe_float(ps[1], 1.0)
    return px, py

def get_dz_fallback(ds):
    sbs = getattr(ds, "SpacingBetweenSlices", None)
    dz = _safe_float(sbs, np.nan)
    if np.isnan(dz):
        dz = _safe_float(getattr(ds, "SliceThickness", None), 1.0)
    return dz

def get_row_col(ds):
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) < 6:
        return np.array([1., 0., 0.]), np.array([0., 1., 0.])
    row = np.asarray(iop[:3], dtype=float)
    col = np.asarray(iop[3:6], dtype=float)
    if not np.isfinite(row).all() or not np.isfinite(col).all():
        row, col = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    rnorm = np.linalg.norm(row)
    cnorm = np.linalg.norm(col)
    if rnorm == 0 or cnorm == 0:
        row, col = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    else:
        row /= rnorm
        col /= cnorm
    return row, col

def get_origin(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None or len(ipp) < 3:
        return np.array([0., 0., 0.], dtype=float)
    o = np.asarray(ipp[:3], dtype=float)
    if not np.isfinite(o).all():
        return np.array([0., 0., 0.], dtype=float)
    return o

def slice_normal(ds: pydicom.Dataset) -> np.ndarray:
    row, col = get_row_col(ds)
    n = np.cross(row, col)
    norm = np.linalg.norm(n)
    return n / norm if norm != 0 else np.array([0., 0., 1.])

def z_along_normal(ds: pydicom.Dataset, n: np.ndarray) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None or len(ipp) < 3:
        return 0.0
    ippv = np.asarray(ipp[:3], dtype=float)
    if not np.isfinite(ippv).all():
        return 0.0
    return float(np.dot(ippv, n))

def compute_affine_from_series(sorted_ds, zvals) -> np.ndarray:
    """
    Build a valid 4x4 affine with strictly positive zooms.
    Uses median Δz; guards single-slice and degenerate z-spacing.
    """
    ds0 = sorted_ds[0]
    row, col = get_row_col(ds0)
    normal = np.cross(row, col)
    n_norm = np.linalg.norm(normal)
    normal = (normal / n_norm) if n_norm != 0 else np.array([0., 0., 1.])

    px, py = get_px_py(ds0)

    dz = np.diff(zvals) if len(zvals) > 1 else np.array([])
    if dz.size == 0 or not np.isfinite(dz).all():
        dz_med = get_dz_fallback(ds0)
    else:
        dz_med = float(np.median(np.abs(dz)))
        if dz_med <= 0 or not np.isfinite(dz_med):
            dz_med = get_dz_fallback(ds0)

    eps = 1e-3
    px = max(px, eps)
    py = max(py, eps)
    dz_med = max(dz_med, eps)

    origin = get_origin(ds0)

    M = np.eye(4, dtype=float)
    M[:3, 0] = row * px
    M[:3, 1] = col * py
    M[:3, 2] = normal * dz_med
    M[:3, 3] = origin

    if not np.isfinite(M).all():
        M = np.array([[1.,0.,0.,0.],
                      [0.,1.,0.,0.],
                      [0.,0.,1.,0.],
                      [0.,0.,0.,1.]], dtype=float)
    return M

# =========================
# Stacking & resampling
# =========================
def build_3d_stack(ds_list):
    """Return HU volume (H,W,Z) and robust affine; resilient to degenerate tags."""
    if len(ds_list) == 0:
        return None, None
    n = slice_normal(ds_list[0])
    zvals = [z_along_normal(ds, n) for ds in ds_list]
    order = np.argsort(zvals)
    sorted_ds = [ds_list[i] for i in order]
    zvals_sorted = np.array([z_along_normal(ds, n) for ds in sorted_ds], dtype=float)

    imgs = []
    for ds in sorted_ds:
        try:
            hu = get_pixels_hu(ds).astype(np.float32)
        except Exception:
            return None, None
        imgs.append(hu)
    vol = np.stack(imgs, axis=-1)  # H x W x Z

    affine = compute_affine_from_series(sorted_ds, zvals_sorted)
    return vol, affine

def to_nifti(vol: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    return nib.Nifti1Image(vol, affine)

def resample_iso(nii: nib.Nifti1Image, voxel_sizes=(1.0,1.0,1.0), order=1) -> nib.Nifti1Image:
    """Resample NIfTI to isotropic voxel size in world space (order=1 → linear for HU)."""
    return resample_to_output(nii, voxel_sizes=voxel_sizes, order=order)

# =========================
# Masking (after resampling!)
# =========================
def make_intra_mask_from_hu_iso(vol_iso: np.ndarray) -> np.ndarray:
    """
    Intracranial-ish mask after resampling:
    - Suppress bone by setting HU>300 to 0
    - Mask = vol_iso_no_bone > 0
    """
    vol = vol_iso.copy()
    vol[vol > 300] = 0
    mask = vol > 0
    return mask.astype(np.uint8)

def make_brain_mask_from_hu_iso(vol_iso: np.ndarray, intra_mask: np.ndarray) -> np.ndarray:
    brain = (vol_iso > 10) & (vol_iso < 200)
    return (brain & (intra_mask.astype(bool))).astype(np.uint8)

# =========================
# One subfolder → one row
# =========================
def process_one_direct_folder(folder_path: str):
    """
    Build HU stack from DICOMs directly in folder, resample to 1mm iso, then apply HU>300.
    Returns (row_df, slice_count, optional_nii_tuple)
    """
    ds_all = []
    for fp in list_direct_dicoms(folder_path):
        try:
            ds = load_scan(fp)
            mod = str(getattr(ds, "Modality", "")).upper()
            if mod and mod != "CT":
                continue
            if is_localizer(ds):
                continue
            if FILTER_OUT_BONE_KERNEL and kernel_is_bone(ds):
                continue
            _ = ds.pixel_array  # ensure pixels
            ds_all.append(ds)
        except Exception:
            continue

    if len(ds_all) == 0:
        return pd.DataFrame(), 0, None

    vol_native, aff_native = build_3d_stack(ds_all)
    if vol_native is None:
        return pd.DataFrame(), 0, None

    # Resample to 1x1x1 BEFORE thresholding
    nii_native = to_nifti(vol_native, aff_native)
    try:
        nii_iso = resample_iso(nii_native, voxel_sizes=ISO_SPACING, order=1)
    except Exception:
        # last resort: use identity affine, then resample
        ident = np.array([[1.,0.,0.,0.],
                          [0.,1.,0.,0.],
                          [0.,0.,1.,0.],
                          [0.,0.,0.,1.]], dtype=float)
        nii_native = nib.Nifti1Image(vol_native, ident)
        nii_iso = resample_iso(nii_native, voxel_sizes=ISO_SPACING, order=1)

    vol_iso = np.asanyarray(nii_iso.dataobj).astype(np.float32)
    aff_iso = nii_iso.affine

    intra_mask_iso = make_intra_mask_from_hu_iso(vol_iso)
    brain_mask_iso = make_brain_mask_from_hu_iso(vol_iso, intra_mask_iso)

    intra_vol_mm3 = float(intra_mask_iso.sum())   # 1 mm³ per voxel
    brain_vol_mm3 = float(brain_mask_iso.sum())

    pid = str(getattr(ds_all[0], "PatientID", ""))
    pname = str(getattr(ds_all[0], "PatientName", ""))
    tstamp = normalize_time_str(
        getattr(ds_all[0], "AcquisitionTime", "") or
        getattr(ds_all[0], "SeriesTime", "") or
        getattr(ds_all[0], "StudyTime", "")
    )
    dstamp = normalize_date_str(
        getattr(ds_all[0], "AcquisitionDate", "") or
        getattr(ds_all[0], "SeriesDate", "") or
        getattr(ds_all[0], "StudyDate", "")
    )

    row = pd.DataFrame([{
        "PatientID": pid,
        "PatientName": pname,
        "BrainVolume": brain_vol_mm3,        # mm^3
        "IntraSkullVolume": intra_vol_mm3,   # mm^3
        "TimeStamp": tstamp,
        "DateStamp": dstamp,
        "SlicesCount": len(ds_all),
        "IsoSpacing": "1x1x1",
    }])

    if NIFTI_OUT_DIRNAME:
        return row, len(ds_all), (vol_iso, intra_mask_iso, brain_mask_iso, aff_iso)
    else:
        return row, len(ds_all), None

# =========================
# Orchestration
# =========================
def process(folder1: str, extract_path: str):
    """
    Recurse each patient folder; for any subfolder containing DICOMs directly,
    build a stack, RESAMPLE to 1x1x1, then apply HU>300 bone suppression.
    Save per-subfolder row to output.csv (adds Rate, PatientFolder, RelativeSubfolder).
    Optionally save iso NIfTIs if NIFTI_OUT_DIRNAME is set.
    """
    out_dir = folder1 or "."
    PROCESSED_FILE = os.path.join(out_dir, PROCESSED_FILENAME)
    OUTPUT_FILE = os.path.join(out_dir, OUTPUT_FILENAME)
    os.makedirs(out_dir, exist_ok=True)

    if NIFTI_OUT_DIRNAME:
        NIFTI_DIR = os.path.join(out_dir, NIFTI_OUT_DIRNAME)
        os.makedirs(NIFTI_DIR, exist_ok=True)
    else:
        NIFTI_DIR = None

    patient_roots = [d for d in os.listdir(extract_path)
                     if os.path.isdir(os.path.join(extract_path, d))]

    if os.path.exists(PROCESSED_FILE):
        processed_df = pd.read_csv(PROCESSED_FILE)
        if "key" not in processed_df.columns:
            processed_df.columns = ["key"][:len(processed_df.columns)]
        processed_set = set(processed_df["key"].astype(str))
    else:
        processed_df = pd.DataFrame(columns=["key"])
        processed_set = set()

    if os.path.exists(OUTPUT_FILE):
        all_results = pd.read_csv(OUTPUT_FILE)
        print(f"Loaded {len(all_results)} rows from {OUTPUT_FILE}")
    else:
        all_results = pd.DataFrame()

    for patient in patient_roots:
        patient_path = os.path.join(extract_path, patient)

        for root, dirs, files in os.walk(patient_path):
            if root == patient_path:
                continue
            if not has_direct_dicoms(root):
                continue

            rel_sub = os.path.relpath(root, start=patient_path)
            key = f"{patient}/{rel_sub}"
            if key in processed_set:
                print(f"Skipping already processed: {key}")
                continue

            print(f"Processing: {key}")
            row, n_slices, payload = process_one_direct_folder(root)

            if n_slices > 0 and not row.empty:
                row["Rate"] = (row["BrainVolume"] / row["IntraSkullVolume"]).replace([np.inf, -np.inf], np.nan)
                row["PatientFolder"] = patient
                row["RelativeSubfolder"] = rel_sub

                all_results = pd.concat([all_results, row], ignore_index=True)
                all_results.to_csv(OUTPUT_FILE, index=False)
                print(f"→ Appended {len(row)} row(s) for {key} to {OUTPUT_FILE}")

                # Save NIfTIs if requested
                if payload and NIFTI_DIR:
                    vol_iso, intra_mask_iso, brain_mask_iso, aff_iso = payload
                    def _safe_name(s: str) -> str:
                        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)
                    base = f"{_safe_name(patient)}__{_safe_name(rel_sub)}"
                    nib.save(nib.Nifti1Image(vol_iso, aff_iso), os.path.join(NIFTI_DIR, f"{base}__iso_orig.nii.gz"))
                    nib.save(nib.Nifti1Image(intra_mask_iso.astype(np.uint8), aff_iso), os.path.join(NIFTI_DIR, f"{base}__iso_mask.nii.gz"))
                    # Optional: skull-stripped HU
                    vol_strip = vol_iso * intra_mask_iso.astype(np.float32)
                    nib.save(nib.Nifti1Image(vol_strip, aff_iso), os.path.join(NIFTI_DIR, f"{base}__iso_stripped.nii.gz"))
                    print(f"→ Saved isotropic NIfTI set for {key} in {NIFTI_DIR}")

            else:
                print(f"→ DICOM(s) present but no usable slices in {key}")

            processed_df = pd.concat([processed_df, pd.DataFrame([{"key": key}])], ignore_index=True)
            processed_df.to_csv(PROCESSED_FILE, index=False)
            processed_set.add(key)

    print("Done.")

# =========================
# Post-process helper
# =========================
def summarize_no_bone(output_csv_path="output.csv", out_filtered_path="output_no_bone.csv"):
    """
    From output.csv, drop rows whose RelativeSubfolder contains 'bone' (case-insensitive),
    save to output_no_bone.csv, and print summary stats.
    """
    df = pd.read_csv(output_csv_path)
    df_no_bone = df[~df["RelativeSubfolder"].str.contains("bone", case=False, na=False)].copy()
    df_no_bone.to_csv(out_filtered_path, index=False)

    mean_brain = df_no_bone["BrainVolume"].mean()
    mean_intra = df_no_bone["IntraSkullVolume"].mean()
    mean_rate  = df_no_bone["Rate"].mean()
    std_rate   = df_no_bone["Rate"].std()

    print("Summary after excluding 'bone' subfolders:")
    print(f"Mean BrainVolume      = {mean_brain:.2f} mm^3")
    print(f"Mean IntraSkullVolume = {mean_intra:.2f} mm^3")
    print(f"Mean Rate             = {mean_rate:.6f}")
    print(f"Std  Rate             = {std_rate:.6f}")


!pip install nibabel python-gdcm pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg pydicom scikit-image==0.24.0

import os
import csv
import pandas as pd
from utilsv3 import *
import zipfile

folder = '/home/sagemaker-user/files/workspace_files/'
extract_path = '/home/sagemaker-user/files/workspace_files/extract_data/Processed_for_AHA_20250912_160259'

#extract_new_zips(folder, extract_path)
process(folder, extract_path)
summarize_no_bone("output.csv", "output_no_bone.csv")
print("Done")