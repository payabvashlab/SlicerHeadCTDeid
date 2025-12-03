# -*- coding: utf-8 -*-
"""
head_ct_deid_cli.py

Head CT De-identification (no GUI)
- Reads an Excel/CSV mapping of original_folder_name -> new_folder_name
- Walks input DICOM tree, filters for head CT (optionally includes CTA/perfusion)
- Anonymizes DICOM tags and pixel data (face-drowning; optional OCR text blackout)
- Mirrors the input structure into an output root
- Then, at every directory level under the output root, renames each level's
  immediate subfolders to <new_folder_name>_<1..N> (deterministic order),
  preserving the depth.

Dependencies (install ahead of time, or use --auto-install):
  pip install numpy pandas openpyxl pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg opencv-python scikit-image easyocr torch --index-url https://download.pytorch.org/whl/cpu
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

# ------------------------------- Constants -----------------------------------

FACE_MAX_VALUE = 50
FACE_MIN_VALUE = -125
AIR_THRESHOLD = -800

# --------------------------- Optional auto-install ----------------------------

def ensure_dependencies(auto_install: bool = False):
    """
    Best-effort import of required packages. If --auto-install is provided,
    tries to pip install missing dependencies into the current environment.
    """
    def _pip_install(pkgs):
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install"] + pkgs
        subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)

    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas==2.2.3"),
        ("openpyxl", "openpyxl"),
        ("pydicom", "pydicom"),
        ("cv2", "opencv-python"),
        ("skimage", "scikit-image"),
        ("easyocr", "easyocr"),
        # JPEG2000 & compressed transfer syntax support
        ("pylibjpeg", "pylibjpeg"),
        ("pylibjpeg_libjpeg", "pylibjpeg-libjpeg"),
        ("pylibjpeg_openjpeg", "pylibjpeg-openjpeg"),
    ]

    missing = []
    for mod_name, pkg in required:
        try:
            __import__(mod_name)
        except Exception:
            missing.append(pkg)

    if missing and auto_install:
        _pip_install(missing)
        # easyocr requires torch; prefer CPU wheels
        try:
            __import__("torch")
        except Exception:
            try:
                _pip_install(["torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
            except Exception:
                pass

    # final import check (will raise if still missing)
    import numpy  # noqa
    import pandas  # noqa
    import openpyxl  # noqa
    import pydicom  # noqa
    import cv2  # noqa
    import skimage  # noqa
    try:
        import easyocr  # noqa
    except Exception:
        # It's okay to proceed without OCR; text blackout will be skipped
        pass


# ------------------------------- Core Logic -----------------------------------

class DicomProcessor:
    """
    Processing pipeline:
      Phase 1: Walk input tree, mirror structure to out_path and write anonymized DICOMs.
      Phase 2: For every directory level under out_path, rename its immediate subfolders
               in sorted order to <id>_<1..N>, preserving the nesting depth.
    """

    def __init__(self, logger: logging.Logger = None):
        self.error = ""
        self.net = ""
        self.study_uid_map = defaultdict(str)
        self.series_uid_map = defaultdict(str)
        self.sop_uid_map = defaultdict(str)
        self._ocr_reader = None  # lazy-initialized easyocr.Reader
        self.logger = logger or logging.getLogger("HeadCTDeid")

    # ----------------------------- OCR helper ---------------------------------

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"])
            except Exception:
                self._ocr_reader = None
        return self._ocr_reader

    # ------------------------------- DICOM IO ---------------------------------

    def is_dicom(self, file_path, remove_CTA=False):
        import pydicom
        try:
            ds = pydicom.dcmread(file_path, force=True)
            try:
                ds.decompress()
            except Exception:
                pass
            return self.checkCTmeta(ds, remove_CTA) == 1
        except Exception as e:
            try:
                with open("log.txt", "a") as f:
                    f.write(f"Error reading DICOM {file_path}: {e}\n")
            except Exception:
                pass
            return False

    def load_scan(self, path):
        import pydicom
        p = Path(path)
        if p.is_file():
            return pydicom.dcmread(str(p), force=True)
        raise FileNotFoundError(f"Not a file: {path}")

    # ---------------------------- Image utilities -----------------------------

    def get_pixels_hu(self, ds):
        import numpy as np
        image = ds.pixel_array.astype(np.int16)
        image[image <= -2000] = 0
        intercept = getattr(ds, "RescaleIntercept", 0)
        slope = getattr(ds, "RescaleSlope", 1)
        if slope != 1:
            image = (image.astype(np.float64) * slope).astype(np.int16)
        image += np.int16(intercept)
        return image

    def binarize_volume(self, volume, air_hu=AIR_THRESHOLD):
        import numpy as np
        out = np.zeros_like(volume, dtype=np.uint8)
        out[volume <= air_hu] = 1
        return out

    def largest_connected_component(self, binary_image):
        import numpy as np
        import cv2
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(binary_image, dtype=np.uint8)
        largest_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        lcc = np.zeros(labels.shape, dtype=np.uint8)
        lcc[labels == largest_idx] = 1
        return lcc

    def get_largest_component_volume(self, volume):
        return self.largest_connected_component(volume)

    def dilate_volume(self, volume, kernel_size=None):
        import cv2
        k = kernel_size or random.randint(30, 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(volume.astype(np.uint8), kernel)

    def apply_mask_and_get_values(self, image_volume, mask_volume):
        import numpy as np
        masked = image_volume * mask_volume
        vals = np.unique(masked)
        vals = vals[(vals > FACE_MIN_VALUE) & (vals < FACE_MAX_VALUE)]
        return vals.tolist()

    def apply_random_values_optimized(self, pixels_hu, dilated_volume, unique_values_list):
        import numpy as np
        new_vol = np.copy(pixels_hu)
        new_vol[dilated_volume == 1] = -1000  # replace with air
        return new_vol

    # ---------------------------- Tag scrub helpers ---------------------------

    def person_names_callback(self, ds, elem):
        if elem.VR == "PN":
            elem.value = "anonymous"

    def curves_callback(self, ds, elem):
        if elem.tag.group & 0xFF00 == 0x5000:
            del ds[elem.tag]

    def is_substring_in_list(self, substring, string_list):
        return any(substring in str(s) for s in string_list)

    # ---------------------------- CT selection --------------------------------

    def checkCTmeta(self, ds, remove_CTA=False):
        """
        Accept only CT head (original/primary/axial). By default, exclude CTA/perfusion.
        If remove_CTA=True -> do not exclude CTA/perfusion (i.e., include such series).
        """
        try:
            # Modality check
            modality = ds.get((0x08, 0x60), "")
            modality = [modality.value] if hasattr(modality, "value") else [modality]
            modality = [str(x).lower().replace(" ", "") for x in modality]
            status1 = any(self.is_substring_in_list(c, modality) for c in ["ct", "computedtomography", "ctprotocal"])

            # ImageType check
            imageType = ds.get((0x08, 0x08), "")
            imageType = [imageType.value] if hasattr(imageType, "value") else [imageType]
            imageType = [str(x).lower().replace(" ", "") for x in imageType]
            status2 = all(self.is_substring_in_list(c, imageType) for c in ["original", "primary", "axial"])

            # Study description includes head terms; optionally exclude CTA/perfusion
            studyDes = None
            for tag in [(0x08, 0x1030), (0x08, 0x103e), (0x18, 0x0015), (0x18, 0x1160)]:
                if tag in ds:
                    studyDes = ds[tag].value
                    break
            studyDes = [studyDes] if isinstance(studyDes, str) else [studyDes]
            studyDes = [str(x).lower().replace(" ", "") for x in studyDes if x is not None]

            include = ["head", "brain", "skull"]
            exclude = ["angio", "cta", "perfusion"]

            status3 = any(self.is_substring_in_list(c, studyDes) for c in include)

            status4 = True
            if not remove_CTA:
                if any(self.is_substring_in_list(e, studyDes) for e in exclude):
                    status4 = False

            return int(status1 and status2 and status3 and status4)
        except Exception as e:
            self.error = str(e)
            return 0

    # --------------------------- Anonymization core ---------------------------

    def save_new_dicom_files(
        self,
        original_dir,
        out_dir,
        replacer="face",
        id="new_folder_name",
        patient_id="0",
        new_patient_id="Processed for anonymization",
        remove_text=False,
        remove_CTA=False,
    ):
        import os
        import pydicom
        from pydicom.datadict import keyword_for_tag
        from pydicom.uid import generate_uid

        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(original_dir) if self.is_dicom(os.path.join(original_dir, f), remove_CTA)]
        errors = []

        # Robust sort by InstanceNumber if present; fallback to filename
        def _instnum(path):
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                return int(getattr(ds, "InstanceNumber", 1))
            except Exception:
                return sys.maxsize

        files.sort(key=lambda fn: (_instnum(os.path.join(original_dir, fn)), fn))

        for i, fname in enumerate(files, start=1):
            try:
                ds = self.load_scan(os.path.join(original_dir, fname))
                try:
                    ds.decompress()
                except Exception:
                    pass

                # Remove private tags & PHI-like content
                ds.remove_private_tags()
                if "OtherPatientIDs" in ds:
                    delattr(ds, "OtherPatientIDs")
                if "OtherPatientIDsSequence" in ds:
                    del ds.OtherPatientIDsSequence
                ds.walk(self.person_names_callback)
                ds.walk(self.curves_callback)

                ANON = "anonymous"

                # Minimal required IDs
                if (0x0008, 0x0050) not in ds:
                    ds.add_new((0x0008, 0x0050), "SH", id)
                else:
                    ds[0x0008, 0x0050].value = id

                if (0x0010, 0x0020) not in ds:
                    ds.add_new((0x0010, 0x0020), "LO", ANON)
                else:
                    ds[0x0010, 0x0020].value = ANON

                if (0x0010, 0x0010) not in ds:
                    ds.add_new((0x0010, 0x0010), "PN", new_patient_id)
                else:
                    ds[0x0010, 0x0010].value = new_patient_id

                # Scrub a set of tags
                requirement_tags = [
                    (0x0010, 0x1000), (0x0010, 0x1001), (0x0010, 0x1005), (0x0010, 0x1040),
                    (0x0010, 0x2154), (0x0010, 0x2295), (0x0012, 0x0020), (0x0012, 0x0030),
                    (0x0012, 0x0040), (0x0012, 0x0042), (0x0012, 0x0071), (0x0018, 0x9445),
                    (0x0020, 0x0010), (0x0020, 0x9056), (0x0032, 0x000A), (0x0032, 0x000C),
                    (0x0032, 0x0012), (0x0038, 0x0008), (0x0038, 0x0010), (0x0038, 0x0400),
                    (0x0040, 0x0031), (0x0040, 0x0032), (0x0040, 0x0033), (0x0040, 0x2016),
                    (0x0040, 0x2017), (0x0040, 0xA123), (0x0070, 0x0080), (0x0400, 0x0005),
                    (0x0400, 0x0020), (0x0400, 0x0564), (0x300A, 0x0182), (0x4008, 0x0040),
                    (0x4008, 0x0119), (0x4008, 0x011A), (0x4008, 0x0210), (0x4008, 0x0212),
                    (0x0010, 0x0030), (0x0010, 0x2298), (0x0010, 0x0201), (0x0012, 0x0060),
                    (0x0038, 0x0011), (0x0040, 0x0001), (0x0040, 0x0010), (0x0040, 0x0035),
                    (0x0040, 0x0241), (0x0040, 0x0242), (0x0040, 0x1010), (0x0040, 0x2008),
                    (0x0040, 0x2009), (0x0040, 0x2010), (0x0040, 0xA075), (0x0070, 0x0084),
                    (0x0088, 0x0130), (0x0400, 0x0115), (0x0400, 0x0120), (0x3006, 0x00A6),
                    (0x4008, 0x010A), (0x4008, 0x010C), (0x4008, 0x0114), (0x0032, 0x1033),
                ]
                for tag in requirement_tags:
                    if tag in ds:
                        tag_name = keyword_for_tag(tag)
                        tag_vr = ds[tag].VR
                        if "ID" in tag_name:
                            ds[tag].value = "0"
                        elif tag_vr == "DA":
                            ds[tag].value = "00010101"
                        else:
                            ds[tag].value = ANON

                # Remap key UIDs consistently per-study/series/instance
                from pydicom.uid import generate_uid
                if (0x0020, 0x000E) in ds:
                    series_uid = str(ds[0x0020, 0x000E].value)
                    if series_uid and series_uid not in self.series_uid_map:
                        self.series_uid_map[series_uid] = generate_uid()
                    ds[0x0020, 0x000E].value = self.series_uid_map.get(series_uid, generate_uid())

                if (0x0020, 0x000D) in ds:
                    study_uid = str(ds[0x0020, 0x000D].value)
                    if study_uid and study_uid not in self.study_uid_map:
                        self.study_uid_map[study_uid] = generate_uid()
                    ds[0x0020, 0x000D].value = self.study_uid_map.get(study_uid, generate_uid())

                if (0x0008, 0x0018) in ds:
                    sop_uid = str(ds[0x0008, 0x0018].value)
                    if sop_uid and sop_uid not in self.sop_uid_map:
                        self.sop_uid_map[sop_uid] = generate_uid()
                    ds[0x0008, 0x0018].value = self.sop_uid_map.get(sop_uid, generate_uid())

                # Normalize common tags
                TAGS = [
                    (0x0008, 0x0201),  # TimezoneOffsetFromUTC
                    (0x0010, 0x2150),  # CountryOfResidence
                    (0x0010, 0x2152),  # RegionOfResidence
                    (0x0038, 0x0300),  # CurrentPatientLocation
                    (0x0008, 0x0080),  # InstitutionName
                    (0x0008, 0x0081),  # InstitutionAddress
                ]
                for tag in TAGS:
                    if tag in ds:
                        ds[tag].value = ANON

                if (0x0010, 0x2160) in ds and str(ds[(0x0010, 0x2160)].value).lower() == "unknown":
                    ds[(0x0010, 0x2160)].value = ""

                # Simplify race
                RACE_TAG = (0x0010, 0x2201)
                RACE_MAPPING = {
                    "WHITE": "White",
                    "BLACK OR AFRICAN AMERICAN": "Black",
                    "BLACK": "Black",
                    "ASIAN": "Asian",
                    "PACIFIC ISLANDER": "Asian",
                    "AMERICAN INDIAN": "Asian",
                    "NATIVE INDIAN": "Asian",
                }
                if RACE_TAG in ds:
                    rv = str(ds[RACE_TAG].value).strip().upper() if ds[RACE_TAG].value else "Other"
                    ds[RACE_TAG].value = RACE_MAPPING.get(rv, "Other")

                # --- pixel operations ---
                import numpy as np
                import cv2

                pixels_hu = self.get_pixels_hu(ds)
                bin_mask = self.binarize_volume(pixels_hu)
                lcc = self.get_largest_component_volume(bin_mask)
                dilated = self.dilate_volume(lcc)

                if replacer == "face":
                    vals = self.apply_mask_and_get_values(pixels_hu, (dilated - lcc))
                elif replacer == "air":
                    vals = [0]
                else:
                    try:
                        vals = [int(replacer)]
                    except Exception:
                        vals = self.apply_mask_and_get_values(pixels_hu, (dilated - lcc))

                new_volume = self.apply_random_values_optimized(pixels_hu, dilated, vals)

                # Optional OCR blackout
                if remove_text:
                    try:
                        reader = self._get_ocr_reader()  # may be None
                        if reader is not None:
                            mn, mx = int(np.min(pixels_hu)), int(np.max(pixels_hu))
                            rng = max(1, mx - mn)
                            gray8 = np.uint8(((pixels_hu - mn) / rng) * 255.0)
                            img = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
                            results = reader.readtext(img)
                            for (bbox, text, prob) in results:
                                if prob > 0.8:
                                    tl = tuple(map(int, bbox[0]))
                                    br = tuple(map(int, bbox[2]))
                                    cv2.rectangle(new_volume, tl, br, 0, thickness=cv2.FILLED)
                    except Exception as e:
                        errors.append((fname, f"OCR blackout error: {e}"))

                # Write back pixels (re-apply DICOM rescale)
                slope = float(getattr(ds, "RescaleSlope", 1)) or 1.0
                intercept = float(getattr(ds, "RescaleIntercept", 0))
                new_slice = (new_volume - intercept) / slope
                ds.PixelData = new_slice.astype(np.int16).tobytes()
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1

                out_name = f"{id}_{i:05d}.dcm"
                ds.save_as(os.path.join(out_dir, out_name), write_like_original=False)

                del ds, pixels_hu, new_volume
            except Exception as e:
                errors.append((fname, str(e)))

        if errors:
            try:
                with open(os.path.join(out_dir, "log.txt"), "a") as error_file:
                    for dicom_file, err in errors:
                        error_file.write(f"File: {dicom_file}, Error: {err}\n")
            except Exception:
                pass

        return errors

    # ---------------------------- Directory driver ----------------------------

    def drown_volume(
        self,
        in_path,
        out_path,
        replacer="face",
        id="new_folder_name",
        patient_id="0",
        name="",
        remove_text=False,
        remove_CTA=False,
    ):
        """
        Phase 1: process while mirroring input structure.
        Phase 2: rename, at each level, immediate subdirectories to <id>_<n> (1-based),
                 preserving nesting; uses only os.rename (safe two-step).
        """
        try:
            # Phase 1: mirror + write anonymized slices
            for root, dirs, files in os.walk(in_path):
                rel = os.path.relpath(root, in_path)
                out_dir = os.path.join(out_path, rel)
                dicom_files = [f for f in files if self.is_dicom(os.path.join(root, f), remove_CTA)]
                if dicom_files:
                    os.makedirs(out_dir, exist_ok=True)
                    self.save_new_dicom_files(
                        original_dir=root,
                        out_dir=out_dir,
                        replacer=replacer,
                        id=id,
                        patient_id=patient_id,
                        new_patient_id="Processed for anonymization",
                        remove_text=remove_text,
                        remove_CTA=remove_CTA,
                    )

            # Phase 2: per-level renaming to <id>_<n>
            for curr, subdirs, files in os.walk(out_path, topdown=True):
                if not subdirs:
                    continue
                subdirs_sorted = sorted(subdirs)

                # temp rename to avoid collisions
                tmp_map = []
                for i, d in enumerate(subdirs_sorted, start=1):
                    src = os.path.join(curr, d)
                    tmp = os.path.join(curr, f"__TMP__RENAME__{i:04d}__")
                    if os.path.exists(src):
                        os.rename(src, tmp)
                        tmp_map.append(tmp)

                # final rename to <id>_<i>
                new_names = []
                for i, tmp in enumerate(tmp_map, start=1):
                    dst_name = f"{id}_{i}"
                    dst = os.path.join(curr, dst_name)
                    os.rename(tmp, dst)
                    new_names.append(dst_name)

                # update traversal list so os.walk continues into renamed dirs
                subdirs[:] = new_names

        except Exception as e:
            try:
                os.makedirs(out_path, exist_ok=True)
                with open(os.path.join(out_path, "log.txt"), "a") as f:
                    f.write(f"Error: {e}\n")
            except Exception:
                pass
            return 0

        return 1


# ---------------------------- Processing wrapper -----------------------------

def _ensure_logger(output_folder: str) -> logging.Logger:
    os.makedirs(output_folder, exist_ok=True)
    logger = logging.getLogger("HeadCTDeid")
    logger.setLevel(logging.INFO)

    log_file = os.path.join(output_folder, "patient_processing.log")
    already = any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_file
                  for h in logger.handlers)
    if not already:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # also log to console
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    logger.info(f"Initialized patient processing module: {log_file}")
    return logger


def load_mapping_table(excel_or_csv_path: str):
    import pandas as pd
    if not os.path.exists(excel_or_csv_path):
        raise ValueError(f"Mapping file does not exist: {excel_or_csv_path}")

    ext = os.path.splitext(excel_or_csv_path)[1].lower()
    dtype = {"original_folder_name": str, "new_folder_name": str}

    if ext == ".csv":
        df = pd.read_csv(excel_or_csv_path, dtype=dtype)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(excel_or_csv_path, dtype=dtype)
    else:
        raise ValueError(f"Unsupported mapping file type: {ext}")

    if ("original_folder_name" not in df.columns) or ("new_folder_name" not in df.columns):
        raise ValueError("Mapping must contain columns: 'original_folder_name' and 'new_folder_name'")

    df["original_folder_name"] = df["original_folder_name"].astype(str).str.strip()
    df["new_folder_name"] = df["new_folder_name"].astype(str).str.strip()
    return dict(zip(df["original_folder_name"], df["new_folder_name"]))


def process_all(input_root: str,
                mapping_path: str,
                output_root: str,
                deidentify_text: bool,
                include_cta: bool):
    """
    Mirrors your Slicer logic.process() in a CLI context.
    """
    if not os.path.exists(input_root):
        raise ValueError(f"Input folder does not exist: {input_root}")
    id_mapping = load_mapping_table(mapping_path)
    os.makedirs(output_root, exist_ok=True)

    logger = _ensure_logger(output_root)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_root, f"Processed for Anonymization_{current_time}")
    os.makedirs(out_path, exist_ok=True)

    dicom_folders = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    total_rows = max(1, len(id_mapping))
    total_time = 0.0
    successful = 0

    # include_cta=True means we DO NOT exclude CTA -> original flag was remove_CTA
    remove_CTA_flag = include_cta  # reuse semantics used inside methods

    for i, foldername in enumerate(sorted(dicom_folders)):
        if foldername not in id_mapping:
            continue
        dst_folder = ""
        try:
            start_time = time.time()
            dst_folder = os.path.join(out_path, id_mapping[foldername])
            processor = DicomProcessor(logger=logger)
            src_folder = os.path.join(input_root, foldername)
            _ = processor.drown_volume(
                in_path=src_folder,
                out_path=dst_folder,
                replacer="face",
                id=id_mapping[foldername],
                patient_id="0",
                name=f"Processed for Anonymization {id_mapping[foldername]}",
                remove_text=deidentify_text,
                remove_CTA=remove_CTA_flag,
            )
            elapsed = time.time() - start_time
            total_time += elapsed
            successful += 1
            logger.info(f"[{i+1}/{total_rows}] Finished processing folder: {foldername} -> {id_mapping[foldername]}  "
                        f"({elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"Error processing folder {foldername}: {str(e)}")
            if dst_folder and os.path.exists(dst_folder):
                shutil.rmtree(dst_folder)

    if successful > 0:
        avg = total_time / successful
        logger.info(f"Average time per folder: {avg:.2f}s")
    else:
        logger.info("No folders were processed successfully.")

    # Report missing folders from mapping that weren't in input
    try:
        requested = set(load_mapping_table(mapping_path).keys())
        actual = set(dicom_folders)
        missing = [f for f in requested if f not in actual]
        if missing:
            logger.error(f"Missing Folders {missing}")
    except Exception as e:
        logger.error(f"Post-check error: {str(e)}")

    logger.info(f"All done. Output root: {out_path}")
    return out_path


# --------------------------------- CLI ---------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Head CT De-identification (no GUI). "
                    "Anonymize tags & pixels, mirror input tree, then rename subfolders per mapping."
    )
    p.add_argument("-i", "--input", required=True, help="Input root folder containing original DICOM subfolders")
    p.add_argument("-m", "--mapping", required=True, help="Excel/CSV with columns: original_folder_name,new_folder_name")
    p.add_argument("-o", "--output", required=True, help="Output root folder")
    p.add_argument("--deidentify-text", action="store_true",
                   help="Run OCR to detect and blackout overlaid text on slices (slow; requires easyocr/torch)")
    p.add_argument("--include-cta", action="store_true",
                   help="Include CTA/perfusion series (default excludes them)")
    p.add_argument("--auto-install", action="store_true",
                   help="Attempt to pip install any missing dependencies into current environment")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dependencies(auto_install=args.auto_install)
    process_all(
        input_root=args.input,
        mapping_path=args.mapping,
        output_root=args.output,
        deidentify_text=args.deidentify_text,
        include_cta=args.include_cta,
    )


if __name__ == "__main__":
    main()