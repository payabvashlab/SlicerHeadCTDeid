"""
HeadCTDeid (3D Slicer scripted module)
"""

import csv
import json
import logging
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import defaultdict
from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

warnings.filterwarnings("ignore")


ENABLE_TEXT_DETECTION = True

FLORENCE_RUN_IN_SUBPROCESS = True

FLORENCE_WORKER_PYTHON = ""

FLORENCE_CUDA_VISIBLE_DEVICES = "0"

FLORENCE_PYTORCH_CUDA_ALLOC_CONF = ""

FLORENCE_RETRY_CUDA_WITH_MINIMAL_ENV = True

FLORENCE_EXTRA_WORKER_ENV = {}

FLORENCE_FALLBACK_TO_CPU_ON_CUDA_ERROR = True

FLORENCE_USE_SLICER_STARTUP_ENV = True

FLORENCE_WORKER_LOAD_TIMEOUT_SEC = 3600
FLORENCE_WORKER_CALL_TIMEOUT_SEC = 300
FLORENCE_WORKER_MAX_RESTARTS = 3

FLORENCE_SHUTDOWN_WORKER_AFTER_RUN = True

FLORENCE_MODEL_ID = "florence-community/Florence-2-large"

FLORENCE_PREFER_NATIVE = True

FLORENCE_NATIVE_EQUIVALENT = {
    "microsoft/Florence-2-large": "florence-community/Florence-2-large",
    "microsoft/Florence-2-base": "florence-community/Florence-2-base",
    "microsoft/Florence-2-large-ft": "florence-community/Florence-2-large-ft",
    "microsoft/Florence-2-base-ft": "florence-community/Florence-2-base-ft",
}

FLORENCE_HF_CACHE_DIR = ""

FLORENCE_TASK = "<OCR_WITH_REGION>"
FLORENCE_FALLBACK_PLAIN_OCR = True

FLORENCE_MAX_NEW_TOKENS = 1024
FLORENCE_NUM_BEAMS = 3
FLORENCE_DTYPE = "float16"
FLORENCE_DEVICE = "auto"
FLORENCE_ATTN_IMPL = "sdpa"
FLORENCE_LOCAL_FILES_ONLY = False

FLORENCE_UPSCALE = 2.0

FLORENCE_EMPTY_CACHE_EVERY_N_CALLS = 50

FLORENCE_SHARE_MODEL_ACROSS_PATIENTS = True

FLORENCE_MIN_CONFIDENCE = 0.4

FLORENCE_KEEP_WHEN_CONFIDENCE_UNKNOWN = True

FLORENCE_MIN_ALNUM = 2
FLORENCE_MIN_BOX_SIDE = 6

FLORENCE_RESTRICT_TO_BORDER_BAND = False
FLORENCE_BORDER_FRAC = 0.30

FLORENCE_APPLY_ALLOWLIST_FILTER = True
FLORENCE_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:-_/(). "

FLORENCE_MAX_PLAUSIBLE_BLOCK_CHARS = 300
FLORENCE_REPEAT_FLAG_COUNT = 6
FLORENCE_BLANK_IMAGE_STD = 3.0
FLORENCE_DROP_SUSPECTED_HALLUCINATIONS = False

PRESCREEN_MODE = "on"

PRESCREEN_MIN_CHAR_H = 5
PRESCREEN_MAX_CHAR_H = 48
PRESCREEN_MIN_CHAR_W = 2
PRESCREEN_MAX_CHAR_W = 48
PRESCREEN_MIN_CHAR_AREA = 6
PRESCREEN_MIN_COMPONENTS = 3
PRESCREEN_TOPHAT_KERNEL = 15

PRESCREEN_ALWAYS_RUN_ON_GROUND_TRUTH = True

RESPECT_MONOCHROME1 = True

OCR_DEBUG_DRAW_BOXES = True
OCR_DEBUG_DRAW_LABELS = True
OCR_DEBUG_BOX_THICKNESS = 2
OCR_DEBUG_FONT_SCALE = 0.5
OCR_DEBUG_FONT_THICKNESS = 1
OCR_DEBUG_COLOR_OK = (0, 255, 0)
OCR_DEBUG_COLOR_FLAGGED = (0, 0, 255)

SAVE_DETECTED_DEBUG_PNG = True
SAVE_NO_TEXT_DEBUG_PNG = True
SAVE_REDACTED_DEBUG_PNG = True

SAVE_PRESCREEN_SKIPPED_DEBUG_PNG = True

SAVE_NOT_EXAMINED_DEBUG_PNG = True


TEXT_ACTION = "redact"

NEVER_DROP_SLICES = True

REDACT_PAD_PX = 6
REDACT_PAD_FRAC = 0.6

REDACT_FILL = "air"
REDACT_AIR_HU = -1000.0

REDACT_BOXLESS_STRATEGY = "border_band"
REDACT_BORDER_BAND_FRAC = 0.18

REDACT_GEOMETRY = "exact"

REDACT_EXACT_PAD_PX = 0

OCR_DEBUG_DRAW_MASK_RECTS = True

REDACT_SWEEP_TEXTLIKE_COMPONENTS = True

REDACT_SWEEP_BORDER_FRAC = 0.30
REDACT_SWEEP_NEAR_HIT_PX = 60

REDACT_SWEEP_LINE_GAP_X = 26
REDACT_SWEEP_LINE_GAP_Y = 8
REDACT_SWEEP_MIN_CHARS_PER_LINE = 2

REDACT_EXPAND_TO_LINE = True
REDACT_LINE_EXTRA_PX = 12

REDACT_VERIFY_WITH_SECOND_PASS = False

REDACT_VERIFY_ON_FAILURE = "warn"
OCR_DEBUG_VERIFY_FAIL_DIRNAME = "text_after_redaction"

OCR_DEBUG_ROOT_DIRNAME = "only_for_debug"
OCR_DEBUG_DETECTED_DIRNAME = "detected_text"
OCR_DEBUG_NO_TEXT_DIRNAME = "no_text_detected"
OCR_DEBUG_REDACTED_DIRNAME = "redacted_text"
OCR_DEBUG_PRESCREEN_DIRNAME = "not_examined_prescreen_skipped"
OCR_DEBUG_NOT_EXAMINED_DIRNAME = "not_examined_detection_off"

DEID_FIX_INVALID_UIDS = True

DEID_SYNC_FILE_META = True

DEID_CLEAR_SOURCE_AE_TITLE = True

UID_TAGS_NEVER_REMAPPED = {
    (0x0002, 0x0002),
    (0x0002, 0x0010),
    (0x0008, 0x0016),
}

FACE_MAX_VALUE = 50
FACE_MIN_VALUE = -125
AIR_THRESHOLD = -800
BONE_STOP_HU = 250
FRONT_BOOST_KERNEL = (3, 3)

FACE_KERNEL_MIN_MM = 50.0
FACE_KERNEL_MAX_MM = 60.0

PDF_TAGS_TO_DEID = {
    (0x0008, 0x0014),
    (0x0008, 0x0018),
    (0x0008, 0x0050),
    (0x0008, 0x0054),
    (0x0008, 0x0080),
    (0x0008, 0x0081),
    (0x0008, 0x0090),
    (0x0008, 0x0092),
    (0x0008, 0x0094),
    (0x0008, 0x010C),
    (0x0008, 0x010D),
    (0x0008, 0x0201),
    (0x0008, 0x1010),
    (0x0008, 0x1048),
    (0x0008, 0x1050),
    (0x0008, 0x1060),
    (0x0008, 0x1070),
    (0x0008, 0x1150),
    (0x0008, 0x1155),
    (0x0008, 0x3010),
    (0x0008, 0x9123),
    (0x0010, 0x0010),
    (0x0010, 0x0020),
    (0x0010, 0x0021),
    (0x0010, 0x0030),
    (0x0010, 0x0032),
    (0x0010, 0x0033),
    (0x0010, 0x0034),
    (0x0010, 0x0035),
    (0x0010, 0x1000),
    (0x0010, 0x1001),
    (0x0010, 0x1005),
    (0x0010, 0x1040),
    (0x0010, 0x2150),
    (0x0010, 0x2152),
    (0x0010, 0x2154),
    (0x0010, 0x2295),
    (0x0010, 0x2299),
    (0x0012, 0x0010),
    (0x0012, 0x0020),
    (0x0012, 0x0030),
    (0x0012, 0x0031),
    (0x0012, 0x0040),
    (0x0012, 0x0042),
    (0x0012, 0x0060),
    (0x0012, 0x0071),
    (0x0018, 0x1000),
    (0x0018, 0x1250),
    (0x0018, 0x1251),
    (0x0020, 0x000D),
    (0x0020, 0x000E),
    (0x0020, 0x0010),
    (0x0020, 0x0052),
    (0x0020, 0x0200),
    (0x0020, 0x1000),
    (0x0020, 0x9056),
    (0x0020, 0x9164),
    (0x0032, 0x000A),
    (0x0032, 0x000C),
    (0x0032, 0x0012),
    (0x0038, 0x0008),
    (0x0038, 0x0010),
    (0x0038, 0x0011),
    (0x0038, 0x0300),
    (0x0038, 0x0400),
    (0x0040, 0x0001),
    (0x0040, 0x0010),
    (0x0040, 0x0031),
    (0x0040, 0x0032),
    (0x0040, 0x0033),
    (0x0040, 0x0035),
    (0x0040, 0x0241),
    (0x0040, 0x0242),
    (0x0040, 0x1010),
    (0x0040, 0x2008),
    (0x0040, 0x2009),
    (0x0040, 0x2010),
    (0x0040, 0x2016),
    (0x0040, 0x2017),
    (0x0040, 0xA075),
    (0x0040, 0xA123),
    (0x0040, 0xA124),
    (0x0070, 0x0080),
    (0x0070, 0x0084),
    (0x0088, 0x0130),
    (0x0088, 0x0140),
    (0x0400, 0x0005),
    (0x0400, 0x0010),
    (0x0400, 0x0020),
    (0x0400, 0x0100),
    (0x0400, 0x0115),
    (0x0400, 0x0120),
    (0x0400, 0x0564),
    (0x3006, 0x0024),
    (0x3006, 0x00A6),
    (0x3006, 0x00C2),
    (0x300A, 0x0182),
    (0x4008, 0x0040),
    (0x4008, 0x010A),
    (0x4008, 0x010C),
    (0x4008, 0x0114),
    (0x4008, 0x0119),
    (0x4008, 0x011A),
    (0x4008, 0x0200),
    (0x4008, 0x0210),
    (0x4008, 0x0212),
}

GLOBAL_DROPPED_CSV_NAME = "global_burned_in_text_actions.csv"

def _to_str(x):
    try:
        if x is None:
            return ""
        if hasattr(x, "value"):
            x = x.value
        if isinstance(x, (list, tuple)):
            if not x:
                return ""
            x = x[0]
        return str(x).strip().lower()
    except Exception:
        return ""


def dicom_has_burned_in(ds) -> bool:
    try:
        bia_val = getattr(ds, "BurnedInAnnotation", None)
        if bia_val is None:
            bia_val = ds.get((0x0028, 0x0301), None)
        v = _to_str(bia_val)
        return v in {"yes", "y", "true", "1"}
    except Exception:
        return False


def _subprocess_import_ok(module_name: str, timeout_sec: int = 25) -> bool:
    try:
        cmd = [sys.executable, "-c", f"import {module_name}; print('OK')"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return (p.returncode == 0) and ("OK" in (p.stdout or ""))
    except Exception:
        return False


def _safe_show_status(msg: str, ms: int = 2000):
    try:
        slicer.util.showStatusMessage(str(msg), int(ms))
    except Exception:
        pass


_UID_COMPONENT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")


def _uid_is_valid(value) -> bool:
    """DICOM PS3.5 section 9.1 UID syntax check.

    Valid: digits in dot-separated components, at most 64 characters, and no
    component with a leading zero unless that component is a single "0".
    """
    s = str(value if value is not None else "").strip()
    if not s or len(s) > 64:
        return False
    return all(_UID_COMPONENT_RE.match(part) for part in s.split("."))


_SAVE_AS_SUPPORTS_ENFORCE = None


def _dcm_save_as(ds, path, enforce_file_format=True):
    """Write a DICOM dataset across the pydicom 2.x / 3.x keyword change.

    pydicom 3.0 deprecated `write_like_original` in favour of
    `enforce_file_format`, with the sense inverted:
        write_like_original=False  ==  enforce_file_format=True
    Both mean "write a proper Part 10 file, adding File Meta Information as
    needed", which is what this pipeline wants for its de-identified output.
    """
    global _SAVE_AS_SUPPORTS_ENFORCE

    if _SAVE_AS_SUPPORTS_ENFORCE is None:
        try:
            import inspect
            _SAVE_AS_SUPPORTS_ENFORCE = (
                "enforce_file_format" in inspect.signature(ds.save_as).parameters)
        except Exception:
            _SAVE_AS_SUPPORTS_ENFORCE = False

    if _SAVE_AS_SUPPORTS_ENFORCE:
        return ds.save_as(path, enforce_file_format=bool(enforce_file_format))
    return ds.save_as(path, write_like_original=(not enforce_file_format))


def _safe_filename(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.replace("\\", "__").replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "unknown"


def _apply_allowlist(text):
    if not FLORENCE_APPLY_ALLOWLIST_FILTER:
        return str(text or "")
    allowed = set(FLORENCE_ALLOWLIST)
    return "".join(ch for ch in str(text or "") if ch in allowed)


def _alnum_count(text) -> int:
    return len(re.findall(r"[A-Za-z0-9]", str(text) if text is not None else ""))


def _text_plausible(text) -> bool:
    s = _apply_allowlist(text).strip()
    return bool(s) and _alnum_count(s) >= int(FLORENCE_MIN_ALNUM)


def _box_big_enough(points) -> bool:
    if points is None:
        return True
    try:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return True
        w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        return min(w, h) >= float(FLORENCE_MIN_BOX_SIDE)
    except Exception:
        return True


def _in_border_band(points, shape_hw) -> bool:
    """True when the box centre lies in the outer band of the image."""
    if points is None:
        return True
    try:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return True
        h, w = int(shape_hw[0]), int(shape_hw[1])
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        fx = float(FLORENCE_BORDER_FRAC) * w
        fy = float(FLORENCE_BORDER_FRAC) * h
        return (cx <= fx) or (cx >= w - fx) or (cy <= fy) or (cy >= h - fy)
    except Exception:
        return True


def _hallucination_flags(text, image):
    """Heuristics for text the model invented rather than read."""
    flags = []
    s = str(text or "")

    if len(s) > int(FLORENCE_MAX_PLAUSIBLE_BLOCK_CHARS):
        flags.append("too_long")

    tokens = re.findall(r"\S+", s)
    if tokens:
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        if max(counts.values()) >= int(FLORENCE_REPEAT_FLAG_COUNT):
            flags.append("repetitive")

    if re.search(r"(.)\1{9,}", s):
        flags.append("char_run")

    try:
        if float(np.std(np.asarray(image, dtype=np.float32))) < float(FLORENCE_BLANK_IMAGE_STD):
            flags.append("blank_source_image")
    except Exception:
        pass

    return flags


def _strip_special(text) -> str:
    """<OCR_WITH_REGION> labels usually carry </s>, <s> and <loc_*> markers."""
    s = str(text or "")
    s = re.sub(r"</?s>", "", s)
    s = re.sub(r"<pad>", "", s)
    s = re.sub(r"<loc_\d+>", "", s)
    return s.strip()


def _coerce_points(raw):
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 4:
        x1, y1, x2, y2 = [float(v) for v in arr]
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if arr.size >= 8 and arr.size % 2 == 0:
        return arr.reshape(-1, 2)
    return None


def _count_textlike_components(gray8) -> int:
    """Count connected components whose size looks like a burned-in character.

    Top-hat morphology highlights small BRIGHT regions on a darker background,
    which is what burned-in annotation looks like on CT. This is plain image
    processing, not OCR.
    """
    try:
        import cv2

        k = int(PRESCREEN_TOPHAT_KERNEL)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat = cv2.morphologyEx(gray8, cv2.MORPH_TOPHAT, kernel)
        if float(np.max(tophat)) < 1.0:
            return 0
        _, bw = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        num, _labels, stats, _cent = cv2.connectedComponentsWithStats(bw, 8)
        count = 0
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if (PRESCREEN_MIN_CHAR_H <= h <= PRESCREEN_MAX_CHAR_H
                    and PRESCREEN_MIN_CHAR_W <= w <= PRESCREEN_MAX_CHAR_W
                    and area >= PRESCREEN_MIN_CHAR_AREA):
                count += 1
        return count
    except Exception:
        return int(PRESCREEN_MIN_COMPONENTS)


def _prescreen_says_maybe_text(gray8):
    n = _count_textlike_components(gray8)
    return (n >= int(PRESCREEN_MIN_COMPONENTS)), n


def _fixed_get_imports(filename):
    """Drop the flash_attn requirement from Florence-2's remote code.

    modeling_florence2.py imports flash_attn at module scope, so transformers
    treats it as mandatory even though the model runs fine with sdpa
    (huggingface/transformers#31793).
    """
    from transformers.dynamic_module_utils import get_imports

    imports = get_imports(filename)
    if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
        imports = [im for im in imports if im != "flash_attn"]
    return imports


def _resolve_florence_device():
    if str(FLORENCE_CUDA_VISIBLE_DEVICES) != "":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(FLORENCE_CUDA_VISIBLE_DEVICES))
    if str(FLORENCE_DEVICE).lower() != "auto":
        return str(FLORENCE_DEVICE)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_florence_dtype(device):
    try:
        import torch
    except Exception:
        return None
    if device == "cpu":
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(str(FLORENCE_DTYPE).lower(), torch.float16)


class Florence2Engine:
    """Thin wrapper around microsoft/Florence-2-* for burned-in text OCR."""

    def __init__(self):
        import torch
        from unittest.mock import patch
        from transformers import AutoModelForCausalLM, AutoProcessor

        try:
            from transformers import Florence2ForConditionalGeneration
            has_native = True
        except Exception:
            Florence2ForConditionalGeneration = None
            has_native = False

        self.torch = torch
        self.device = _resolve_florence_device()
        self.cuda_error = None

        if self.device == "cuda":
            try:
                probe = torch.zeros(1, device="cuda")
                del probe
                torch.cuda.synchronize()
            except Exception as exc:
                self.cuda_error = "CUDA probe failed: %s" % exc
                if not FLORENCE_FALLBACK_TO_CPU_ON_CUDA_ERROR:
                    raise
                self.device = "cpu"

        self.dtype = _resolve_florence_dtype(self.device)
        self._calls_done = 0

        if FLORENCE_HF_CACHE_DIR:
            os.environ["HF_HOME"] = str(FLORENCE_HF_CACHE_DIR)
            os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(str(FLORENCE_HF_CACHE_DIR), "hub")

        model_id = FLORENCE_MODEL_ID
        if FLORENCE_PREFER_NATIVE and has_native and model_id in FLORENCE_NATIVE_EQUIVALENT:
            model_id = FLORENCE_NATIVE_EQUIVALENT[model_id]
        self.model_id = model_id

        local_only = bool(FLORENCE_LOCAL_FILES_ONLY)
        use_native = bool(FLORENCE_PREFER_NATIVE and has_native
                          and not str(model_id).startswith("microsoft/"))

        def _load_native():
            base = {"attn_implementation": FLORENCE_ATTN_IMPL, "local_files_only": local_only}
            for dtype_key in ("dtype", "torch_dtype"):
                kw = dict(base)
                if self.dtype is not None:
                    kw[dtype_key] = self.dtype
                try:
                    mdl = Florence2ForConditionalGeneration.from_pretrained(model_id, **kw)
                    prc = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
                    return mdl, prc, f"native({dtype_key})"
                except TypeError:
                    continue
            mdl = Florence2ForConditionalGeneration.from_pretrained(model_id, **base)
            prc = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
            return mdl, prc, "native(no-dtype)"

        def _load_remote_code():
            kw = {
                "trust_remote_code": True,
                "attn_implementation": FLORENCE_ATTN_IMPL,
                "local_files_only": local_only,
                "low_cpu_mem_usage": True,
            }
            if self.dtype is not None:
                kw["torch_dtype"] = self.dtype
            with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
                try:
                    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kw)
                except TypeError:
                    kw.pop("attn_implementation", None)
                    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kw)
                prc = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True, local_files_only=local_only)
            return mdl, prc, "remote_code"

        try:
            if use_native:
                try:
                    self.model, self.processor, self.impl = _load_native()
                except Exception:
                    self.model, self.processor, self.impl = _load_remote_code()
            else:
                try:
                    self.model, self.processor, self.impl = _load_remote_code()
                except Exception:
                    if not has_native:
                        raise
                    self.model, self.processor, self.impl = _load_native()
        except Exception as exc:
            text = str(exc)
            if ("forced_bos_token_id" in text or "_supports_sdpa" in text
                    or "Unrecognized configuration class" in text):
                raise RuntimeError(
                    f"{exc} -- known Florence-2 remote-code break on transformers "
                    f">= 4.50. Set FLORENCE_MODEL_ID = 'florence-community/Florence-2-large' "
                    f"(transformers >= 4.56), or pin transformers==4.49.0."
                ) from exc
            raise

        try:
            self.model = self.model.to(self.device)
        except Exception as exc:
            if self.device != "cuda" or not FLORENCE_FALLBACK_TO_CPU_ON_CUDA_ERROR:
                raise
            self.cuda_error = "moving the model to CUDA failed: %s" % exc
            self.device = "cpu"
            self.dtype = torch.float32
            self.model = self.model.to(torch.float32).to("cpu")
        self.model.eval()

    @staticmethod
    def to_pil(gray8):
        import cv2
        from PIL import Image

        scale = float(FLORENCE_UPSCALE) if FLORENCE_UPSCALE and FLORENCE_UPSCALE > 0 else 1.0
        work = gray8
        if abs(scale - 1.0) > 1e-6:
            work = cv2.resize(
                gray8,
                (max(1, int(round(gray8.shape[1] * scale))),
                 max(1, int(round(gray8.shape[0] * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
        rgb = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb), scale

    def _generate(self, prompt, images):
        torch = self.torch

        try:
            inputs = self.processor(text=[prompt] * len(images), images=images,
                                    return_tensors="pt", padding=True)
        except Exception:
            inputs = self.processor(text=[prompt] * len(images), images=images,
                                    return_tensors="pt")
        try:
            inputs = inputs.to(self.device)
        except Exception:
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                      for k, v in dict(inputs).items()}
        if self.dtype is not None and self.device != "cpu" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(FLORENCE_MAX_NEW_TOKENS),
                num_beams=int(FLORENCE_NUM_BEAMS),
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequences = getattr(out, "sequences", out)
        texts = self.processor.batch_decode(sequences, skip_special_tokens=False)
        confs = self._sequence_confidences(out, len(texts))
        return texts, confs

    def _sequence_confidences(self, out, n):
        """Sequence-level probability in 0..1 (exp of mean token log-prob).

        Beam search provides `sequences_scores` (length-normalised log-prob)
        whenever output_scores=True; otherwise it is recomputed from `scores`.
        Returns None per item when nothing can be derived, so "unknown" is
        distinguishable from "certain".
        """
        torch = self.torch
        default = [None] * n

        try:
            seq_scores = getattr(out, "sequences_scores", None)
            if seq_scores is not None:
                vals = [float(v) for v in seq_scores.detach().float().cpu().tolist()]
                return [float(np.clip(np.exp(v), 0.0, 1.0)) for v in vals][:n] or default
        except Exception:
            pass

        try:
            scores = getattr(out, "scores", None)
            sequences = getattr(out, "sequences", None)
            if scores is None or sequences is None:
                return default
            trans = self.model.compute_transition_scores(
                sequences, scores, normalize_logits=True
            ).detach().float().cpu()
            finite = torch.isfinite(trans)
            totals = torch.where(finite, trans, torch.zeros_like(trans)).sum(dim=-1)
            counts = finite.sum(dim=-1).clamp(min=1)
            means = (totals / counts).tolist()
            return [float(np.clip(np.exp(v), 0.0, 1.0)) for v in means][:n] or default
        except Exception:
            return default

    def _generate_with_oom_retry(self, prompt, images):
        """Halve the batch and retry on CUDA OOM instead of failing the run."""
        try:
            return self._generate(prompt, images)
        except Exception as exc:
            is_oom = "out of memory" in str(exc).lower()
            if not is_oom or len(images) == 1:
                raise
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
            mid = len(images) // 2
            left_t, left_c = self._generate_with_oom_retry(prompt, images[:mid])
            right_t, right_c = self._generate_with_oom_retry(prompt, images[mid:])
            return left_t + right_t, left_c + right_c

    def _parse(self, decoded, task, size):
        """Turn a Florence-2 completion into [(points|None, text), ...]."""
        try:
            parsed = self.processor.post_process_generation(decoded, task=task, image_size=size)
        except Exception:
            return []

        payload = parsed.get(task) if isinstance(parsed, dict) else None
        out = []
        if payload is None:
            return out

        if isinstance(payload, str):
            txt = _strip_special(payload)
            if txt:
                out.append((None, txt))
            return out

        if isinstance(payload, dict):
            labels = payload.get("labels") or []
            quads = payload.get("quad_boxes")
            if quads is None:
                quads = payload.get("bboxes")

            for i, lab in enumerate(labels):
                txt = _strip_special(str(lab))
                if not txt:
                    continue
                pts = None
                if quads is not None and i < len(quads):
                    pts = _coerce_points(quads[i])
                out.append((pts, txt))
        return out

    def readtext_batch(self, grays):
        """Return, per image, a list of (points|None, text, confidence)."""
        if not grays:
            return []

        prepared = [self.to_pil(g) for g in grays]
        images = [pr[0] for pr in prepared]
        scales = [pr[1] for pr in prepared]
        sizes = [im.size for im in images]

        decoded, confs = self._generate_with_oom_retry(FLORENCE_TASK, images)

        if len(decoded) != len(images):
            raise RuntimeError(f"Florence-2 returned {len(decoded)} results for {len(images)} images")
        if len(confs) != len(images):
            confs = [None] * len(images)

        results = []
        need_fallback = []

        for i, txt in enumerate(decoded):
            hits = self._parse(txt, FLORENCE_TASK, sizes[i])
            if not hits and FLORENCE_FALLBACK_PLAIN_OCR and FLORENCE_TASK != "<OCR>":
                need_fallback.append(i)
            c_i = None if confs[i] is None else float(confs[i])
            results.append([(pnt, t, c_i) for pnt, t in hits])

        if need_fallback:
            try:
                sub = [images[i] for i in need_fallback]
                decoded2, confs2 = self._generate_with_oom_retry("<OCR>", sub)
                for j, i in enumerate(need_fallback):
                    if j < len(decoded2):
                        c = float(confs2[j]) if (j < len(confs2) and confs2[j] is not None) else None
                        results[i] = [(pnt, t, c) for pnt, t in self._parse(decoded2[j], "<OCR>", sizes[i])]
            except Exception:
                pass

        final = []
        for hits, scale in zip(results, scales):
            if abs(scale - 1.0) > 1e-6:
                hits = [((pnt / scale) if pnt is not None else None, t, c) for pnt, t, c in hits]
            final.append(hits)

        self._calls_done += 1
        if (FLORENCE_EMPTY_CACHE_EVERY_N_CALLS
                and self.device == "cuda"
                and self._calls_done % int(FLORENCE_EMPTY_CACHE_EVERY_N_CALLS) == 0):
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass

        return final

    def readtext(self, gray8):
        out = self.readtext_batch([gray8])
        return out[0] if out else []


FLORENCE_WORKER_SOURCE = r"""
import sys, os, json, traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set here; the parent decides,
# because expandable_segments forces PyTorch down the NVML code path.


def emit(obj):
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def log(msg):
    try:
        sys.stderr.write(str(msg) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def main():
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

    cache_dir = cfg.get("hf_cache_dir") or ""
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")

    try:
        import numpy as np
        import torch
        import transformers
        from PIL import Image
        from unittest.mock import patch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        emit({"ready": False, "error": "import failed: %s" % exc})
        log(traceback.format_exc())
        return 1

    tf_version = getattr(transformers, "__version__", "?")

    # Native Florence-2 support landed in transformers on 2025-08-20. When the
    # class exists we use it and skip trust_remote_code entirely.
    try:
        from transformers import Florence2ForConditionalGeneration
        HAS_NATIVE = True
    except Exception:
        Florence2ForConditionalGeneration = None
        HAS_NATIVE = False

    def fixed_get_imports(filename):
        # Florence-2's remote code imports flash_attn at module scope even
        # though it runs fine with sdpa (huggingface/transformers#31793).
        from transformers.dynamic_module_utils import get_imports
        imports = get_imports(filename)
        if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
            imports = [im for im in imports if im != "flash_attn"]
        return imports

    allow_cpu_fallback = bool(cfg.get("cpu_fallback", True))
    cuda_error = None

    device = str(cfg.get("device", "auto"))
    if device == "auto":
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:
            cuda_error = "torch.cuda.is_available() failed: %s" % exc
            device = "cpu"

    # torch.cuda.is_available() can report True while the CUDA context is
    # actually unusable (driver/NVML mismatch, driver updated without a reboot,
    # container without the NVIDIA runtime). Probe with a real allocation so the
    # failure happens here, cheaply, rather than mid-run.
    if device == "cuda":
        try:
            probe = torch.zeros(1, device="cuda")
            del probe
            torch.cuda.synchronize()
        except Exception as exc:
            cuda_error = "CUDA probe failed: %s" % exc
            log(cuda_error)
            if not allow_cpu_fallback:
                emit({"ready": False, "error": cuda_error})
                return 1
            log("falling back to CPU")
            device = "cpu"

    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}.get(str(cfg.get("dtype", "float16")).lower(),
                                               torch.float16)

    model_id = cfg.get("model_id", "florence-community/Florence-2-large")
    local_only = bool(cfg.get("local_files_only", False))
    native_map = cfg.get("native_equivalent") or {}
    prefer_native = bool(cfg.get("prefer_native", True))

    # Remap legacy microsoft/* ids onto their native equivalents.
    if prefer_native and HAS_NATIVE and model_id in native_map:
        log("remapping %s -> %s (native transformers implementation)"
            % (model_id, native_map[model_id]))
        model_id = native_map[model_id]

    use_native = bool(prefer_native and HAS_NATIVE
                      and not str(model_id).startswith("microsoft/"))

    def load_native():
        # transformers v5 renamed torch_dtype to dtype; try both.
        base = {"attn_implementation": cfg.get("attn_impl", "sdpa"),
                "local_files_only": local_only}
        for dtype_key in ("dtype", "torch_dtype"):
            kw = dict(base)
            kw[dtype_key] = dtype
            try:
                mdl = Florence2ForConditionalGeneration.from_pretrained(model_id, **kw)
                prc = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
                return mdl, prc, "native(%s)" % dtype_key
            except TypeError as exc:
                log("native load with %s failed: %s" % (dtype_key, exc))
                continue
        # Last resort: no dtype argument at all.
        mdl = Florence2ForConditionalGeneration.from_pretrained(model_id, **base)
        prc = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
        return mdl, prc, "native(no-dtype)"

    def load_remote_code():
        kw = {"trust_remote_code": True,
              "attn_implementation": cfg.get("attn_impl", "sdpa"),
              "local_files_only": local_only,
              "low_cpu_mem_usage": True,
              "torch_dtype": dtype}
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            try:
                mdl = AutoModelForCausalLM.from_pretrained(model_id, **kw)
            except TypeError:
                kw.pop("attn_implementation", None)
                mdl = AutoModelForCausalLM.from_pretrained(model_id, **kw)
            prc = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=local_only)
        return mdl, prc, "remote_code"

    impl = "?"
    errors = []
    try:
        if use_native:
            try:
                model, processor, impl = load_native()
            except Exception as exc:
                errors.append("native: %s" % exc)
                log(traceback.format_exc())
                log("native load failed; falling back to remote code")
                model, processor, impl = load_remote_code()
        else:
            try:
                model, processor, impl = load_remote_code()
            except Exception as exc:
                errors.append("remote_code: %s" % exc)
                log(traceback.format_exc())
                if not HAS_NATIVE:
                    raise
                log("remote code failed; falling back to the native implementation")
                model, processor, impl = load_native()

        try:
            model = model.to(device)
        except Exception as exc:
            if device != "cuda" or not allow_cpu_fallback:
                raise
            cuda_error = "moving the model to CUDA failed: %s" % exc
            log(cuda_error)
            log("falling back to CPU")
            device = "cpu"
            dtype = torch.float32
            model = model.to(torch.float32).to("cpu")
        model.eval()
    except Exception as exc:
        hint = ""
        text = " ".join(errors) + " " + str(exc)
        if ("forced_bos_token_id" in text or "_supports_sdpa" in text
                or "Unrecognized configuration class" in text):
            hint = (" -- this is the known Florence-2 remote-code break on "
                    "transformers >= 4.50. Use FLORENCE_MODEL_ID = "
                    "'florence-community/Florence-2-large' (needs transformers "
                    ">= 4.56), or pin transformers==4.49.0.")
        elif "nvml" in text.lower() or "NVML_SUCCESS" in text:
            hint = (" -- CUDA/NVML could not be initialised. Check `nvidia-smi`; "
                    "a driver update without a reboot is the usual cause. Set "
                    "FLORENCE_DEVICE = 'cpu' to run without the GPU.")
        emit({"ready": False,
              "error": "model load failed: %s%s" % (exc, hint),
              "transformers": tf_version,
              "has_native": HAS_NATIVE})
        log(traceback.format_exc())
        return 1

    emit({"ready": True, "device": device, "dtype": str(dtype), "model_id": model_id,
          "impl": impl, "transformers": tf_version, "cuda_error": cuda_error})

    task = cfg.get("task", "<OCR_WITH_REGION>")
    fallback_plain = bool(cfg.get("fallback_plain_ocr", True))
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    num_beams = int(cfg.get("num_beams", 3))
    upscale = float(cfg.get("upscale", 1.0) or 1.0)
    empty_cache_every = int(cfg.get("empty_cache_every", 50) or 0)

    calls = [0]

    def to_pil(gray):
        img = Image.fromarray(gray).convert("RGB")
        scale = upscale if upscale and upscale > 0 else 1.0
        if abs(scale - 1.0) > 1e-6:
            img = img.resize((max(1, int(round(img.width * scale))),
                              max(1, int(round(img.height * scale)))),
                             Image.BICUBIC)
            return img, scale
        return img, 1.0

    def sequence_confidences(out, n):
        # exp of the length-normalised log-prob, clipped to 0..1.
        # Returns None per item when nothing can be derived, so the caller can
        # tell "confident" apart from "unknown" instead of silently seeing 1.0.
        default = [None] * n
        try:
            seq_scores = getattr(out, "sequences_scores", None)
            if seq_scores is not None:
                vals = [float(v) for v in seq_scores.detach().float().cpu().tolist()]
                return [float(np.clip(np.exp(v), 0.0, 1.0)) for v in vals][:n] or default
        except Exception:
            pass
        try:
            scores = getattr(out, "scores", None)
            sequences = getattr(out, "sequences", None)
            if scores is None or sequences is None:
                return default
            trans = model.compute_transition_scores(
                sequences, scores, normalize_logits=True).detach().float().cpu()
            finite = torch.isfinite(trans)
            totals = torch.where(finite, trans, torch.zeros_like(trans)).sum(dim=-1)
            counts = finite.sum(dim=-1).clamp(min=1)
            means = (totals / counts).tolist()
            return [float(np.clip(np.exp(v), 0.0, 1.0)) for v in means][:n] or default
        except Exception:
            return default

    def generate(prompt, images):
        texts_in = [prompt] * len(images)
        try:
            inputs = processor(text=texts_in, images=images,
                               return_tensors="pt", padding=True)
        except Exception:
            inputs = processor(text=texts_in, images=images, return_tensors="pt")

        # Forward everything the processor produced (input_ids, pixel_values and
        # attention_mask if present) rather than hand-picking keys.
        try:
            inputs = inputs.to(device)
        except Exception:
            inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                      for k, v in dict(inputs).items()}
        if device != "cpu" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        # output_scores=True is REQUIRED for any confidence at all: both
        # `scores` and beam search's `sequences_scores` are only returned when
        # it is set. Turning it off silently makes every confidence unknown,
        # which makes FLORENCE_MIN_CONFIDENCE a no-op. With max_new_tokens=64
        # the cost is roughly 40 MB per call, which is worth paying.
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        sequences = getattr(out, "sequences", out)
        texts = processor.batch_decode(sequences, skip_special_tokens=False)
        return texts, sequence_confidences(out, len(texts))

    def generate_oom_safe(prompt, images):
        try:
            return generate(prompt, images)
        except Exception as exc:
            if "out of memory" not in str(exc).lower() or len(images) == 1:
                raise
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            mid = len(images) // 2
            lt, lc = generate_oom_safe(prompt, images[:mid])
            rt, rc = generate_oom_safe(prompt, images[mid:])
            return lt + rt, lc + rc

    def coerce_points(raw):
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 4:
            x1, y1, x2, y2 = [float(v) for v in arr]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if arr.size >= 8 and arr.size % 2 == 0:
            return arr.reshape(-1, 2).tolist()
        return None

    def parse(decoded, tsk, size):
        try:
            parsed = processor.post_process_generation(decoded, task=tsk, image_size=size)
        except Exception:
            return []
        payload = parsed.get(tsk) if isinstance(parsed, dict) else None
        if payload is None:
            return []
        if isinstance(payload, str):
            return [(None, payload)]
        out = []
        if isinstance(payload, dict):
            labels = payload.get("labels") or []
            quads = payload.get("quad_boxes")
            if quads is None:
                quads = payload.get("bboxes")
            for i, lab in enumerate(labels):
                pts = coerce_points(quads[i]) if (quads is not None and i < len(quads)) else None
                out.append((pts, str(lab)))
        return out

    warned_no_conf = [False]

    def run_ocr(paths):
        grays = [np.load(pth) for pth in paths]
        prepared = [to_pil(g) for g in grays]
        images = [p[0] for p in prepared]
        scales = [p[1] for p in prepared]
        sizes = [im.size for im in images]

        decoded, confs = generate_oom_safe(task, images)
        if len(confs) != len(images):
            confs = [None] * len(images)

        if not warned_no_conf[0] and all(c is None for c in confs):
            warned_no_conf[0] = True
            log("WARNING: no generation scores available from this model/"
                "transformers version. Confidence will be reported as null and "
                "FLORENCE_MIN_CONFIDENCE cannot filter anything.")

        results = []
        need_fallback = []
        for i, txt in enumerate(decoded):
            hits = parse(txt, task, sizes[i])
            if not hits and fallback_plain and task != "<OCR>":
                need_fallback.append(i)
            c_i = None if confs[i] is None else float(confs[i])
            results.append([(p, t, c_i) for p, t in hits])

        if need_fallback:
            try:
                sub = [images[i] for i in need_fallback]
                dec2, cf2 = generate_oom_safe("<OCR>", sub)
                for j, i in enumerate(need_fallback):
                    if j < len(dec2):
                        c = float(cf2[j]) if (j < len(cf2) and cf2[j] is not None) else None
                        results[i] = [(p, t, c) for p, t in parse(dec2[j], "<OCR>", sizes[i])]
            except Exception as exc:
                log("fallback <OCR> failed: %s" % exc)

        final = []
        for hits, scale in zip(results, scales):
            rows = []
            for pts, txt, conf in hits:
                if pts is not None and abs(scale - 1.0) > 1e-6:
                    pts = (np.asarray(pts, dtype=np.float32) / scale).tolist()
                rows.append({"points": pts, "text": txt,
                             "conf": (None if conf is None else float(conf))})
            final.append(rows)

        calls[0] += 1
        if empty_cache_every and device == "cuda" and calls[0] % empty_cache_every == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return final

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            emit({"ok": False, "error": "bad request: %s" % exc})
            continue

        cmd = req.get("cmd", "ocr")
        if cmd == "quit":
            emit({"ok": True, "bye": True})
            break
        if cmd == "ping":
            emit({"ok": True, "pong": True})
            continue

        try:
            emit({"ok": True, "results": run_ocr(req.get("paths") or [])})
        except Exception as exc:
            emit({"ok": False, "error": str(exc)})
            log(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


def _florence_worker_python():
    """Locate a real Python interpreter for the worker.

    sys.executable can point at the Slicer application binary rather than an
    interpreter, so PythonSlicer is preferred when it can be found.
    """
    exe = str(FLORENCE_WORKER_PYTHON or "").strip()
    if exe and os.path.exists(exe):
        return exe

    candidates = []
    try:
        home = slicer.app.slicerHome
        if home:
            candidates += [os.path.join(home, "bin", "PythonSlicer"),
                           os.path.join(home, "bin", "PythonSlicer.exe")]
    except Exception:
        pass
    bindir = os.path.dirname(sys.executable or "")
    if bindir:
        candidates += [os.path.join(bindir, "PythonSlicer"),
                       os.path.join(bindir, "PythonSlicer.exe")]
    try:
        found = shutil.which("PythonSlicer")
        if found:
            candidates.append(found)
    except Exception:
        pass

    for c in candidates:
        if c and os.path.exists(c):
            return c
    return sys.executable


def _florence_worker_env(minimal=False):
    """Environment for the worker.

    Slicer prepends its own library directories to LD_LIBRARY_PATH, which can
    shadow the system NVIDIA libraries and make NVML fail inside child
    processes. slicer.util.startupEnvironment() returns the environment as it
    was before Slicer modified it, which is what external processes want.

    minimal=True drops every CUDA-related variable this module would otherwise
    set, so a GPU failure can be attributed to the machine rather than to us.
    """
    env = None
    if FLORENCE_USE_SLICER_STARTUP_ENV:
        try:
            env = {str(k): str(v) for k, v in dict(slicer.util.startupEnvironment()).items()}
        except Exception:
            env = None
    if not env:
        env = dict(os.environ)

    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if minimal:
        for key in ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES"):
            env.pop(key, None)
        return env

    if str(FLORENCE_CUDA_VISIBLE_DEVICES) != "":
        env["CUDA_VISIBLE_DEVICES"] = str(FLORENCE_CUDA_VISIBLE_DEVICES)

    if str(FLORENCE_PYTORCH_CUDA_ALLOC_CONF or "").strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(FLORENCE_PYTORCH_CUDA_ALLOC_CONF).strip()
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    try:
        for k, v in dict(FLORENCE_EXTRA_WORKER_ENV or {}).items():
            env[str(k)] = str(v)
    except Exception:
        pass

    return env


FLORENCE_GPU_PROBE_SOURCE = r"""
import os, sys, json

info = {
    "python": sys.executable,
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "LD_LIBRARY_PATH": (os.environ.get("LD_LIBRARY_PATH") or "")[:400],
}

# Where does the loader find NVML? A stub or a version-mismatched copy here is
# a classic cause of nvmlInit_v2() failing while the GPU itself is fine.
try:
    import ctypes.util
    info["find_library_nvidia-ml"] = ctypes.util.find_library("nvidia-ml")
except Exception as exc:
    info["find_library_error"] = str(exc)

try:
    import ctypes
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    rc = lib.nvmlInit_v2()
    info["direct_nvmlInit_v2"] = ("ok" if rc == 0 else "FAILED rc=%d" % rc)
except Exception as exc:
    info["direct_nvmlInit_v2"] = "could not load libnvidia-ml.so.1: %s" % exc

try:
    import torch
    info["torch"] = torch.__version__
    # None here means a CPU-ONLY wheel: no CUDA support was compiled in.
    info["torch_built_with_cuda"] = torch.version.cuda
    try:
        info["is_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["is_available"] = False
        info["is_available_error"] = str(exc)

    if info.get("is_available"):
        try:
            info["device_count"] = torch.cuda.device_count()
            info["device_names"] = [torch.cuda.get_device_name(i)
                                    for i in range(torch.cuda.device_count())]
        except Exception as exc:
            info["device_query_error"] = str(exc)

        # Stage 1: plain allocation, no allocator features.
        try:
            t = torch.zeros(8, device="cuda")
            t = t + 1
            torch.cuda.synchronize()
            info["allocation_probe"] = "ok"
            del t
        except Exception as exc:
            info["allocation_probe"] = "FAILED: %s" % exc

        # Stage 2: does expandable_segments specifically break it? That option
        # routes the allocator through DriverAPI -> nvmlInit_v2().
        try:
            import subprocess as _sp
            code = ("import torch; torch.zeros(8, device='cuda'); "
                    "torch.cuda.synchronize(); print('ok')")
            _env = dict(os.environ)
            _env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            pr = _sp.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=180, env=_env)
            info["expandable_segments_probe"] = (
                "ok" if pr.returncode == 0 else
                "FAILED: %s" % ((pr.stderr or pr.stdout).strip().splitlines() or [""])[-1])
        except Exception as exc:
            info["expandable_segments_probe"] = "could not test: %s" % exc
except Exception as exc:
    info["torch_import_error"] = str(exc)

print(json.dumps(info, indent=2))
"""


def florence_gpu_report():
    '''Diagnose GPU availability in the worker's own context.

    Run from Slicer's Python console:

        from HeadCTDeid import florence_gpu_report
        print(florence_gpu_report())

    The probe runs in the same interpreter and environment as the Florence-2
    worker, so its answer reflects what the worker will actually see - which can
    differ from what Slicer's own Python reports.
    '''
    lines = []
    exe = _florence_worker_python()
    env = _florence_worker_env()

    lines.append("worker interpreter : %s" % exe)
    lines.append("FLORENCE_DEVICE    : %s" % FLORENCE_DEVICE)
    lines.append("CUDA_VISIBLE_DEVICES set for worker: %s"
                 % env.get("CUDA_VISIBLE_DEVICES"))
    lines.append("PYTORCH_CUDA_ALLOC_CONF   : %s"
                 % (env.get("PYTORCH_CUDA_ALLOC_CONF") or "<unset>"))
    lines.append("")

    try:
        pr = subprocess.run(["nvidia-smi",
                             "--query-gpu=index,name,driver_version,memory.total,memory.used",
                             "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=30, env=env)
        if pr.returncode == 0 and pr.stdout.strip():
            lines.append("nvidia-smi:")
            for row in pr.stdout.strip().splitlines():
                lines.append("    " + row.strip())
        else:
            lines.append("nvidia-smi FAILED (rc=%s): %s"
                         % (pr.returncode, (pr.stderr or pr.stdout).strip()[:300]))
    except FileNotFoundError:
        lines.append("nvidia-smi: not found on PATH")
    except Exception as exc:
        lines.append("nvidia-smi error: %s" % exc)
    lines.append("")

    info = {}
    try:
        tmpdir = tempfile.mkdtemp(prefix="headctdeid_gpuprobe_")
        script = os.path.join(tmpdir, "gpu_probe.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(FLORENCE_GPU_PROBE_SOURCE)
        pr = subprocess.run([exe, script], capture_output=True, text=True,
                            timeout=300, env=env, cwd=tmpdir)
        lines.append("torch probe (in worker environment):")
        out = (pr.stdout or "").strip()
        lines.append(out if out else "    <no output>")
        if pr.returncode != 0:
            lines.append("    probe exited rc=%s: %s" % (pr.returncode, (pr.stderr or "").strip()[:500]))
        try:
            info = json.loads(out)
        except Exception:
            info = {}
    except Exception as exc:
        lines.append("torch probe failed to run: %s" % exc)
    lines.append("")

    lines.append("diagnosis:")
    if info.get("torch_import_error"):
        lines.append("    torch cannot be imported by the worker interpreter: %s"
                     % info["torch_import_error"])
        lines.append("    -> install torch into that interpreter, or point "
                     "FLORENCE_WORKER_PYTHON at one that has it.")
    elif info.get("torch_built_with_cuda") in (None, "None", ""):
        lines.append("    torch %s is a CPU-ONLY build (torch.version.cuda is None)."
                     % info.get("torch", "?"))
        lines.append("    No GPU is possible with this wheel, regardless of driver.")
        lines.append("    -> reinstall a CUDA build, e.g.")
        lines.append("       pip install --force-reinstall torch "
                     "--index-url https://download.pytorch.org/whl/cu124")
    elif not info.get("is_available"):
        lines.append("    torch was built with CUDA %s but reports no usable device."
                     % info.get("torch_built_with_cuda"))
        if info.get("is_available_error"):
            lines.append("    error: %s" % info["is_available_error"])
        lines.append("    -> if nvidia-smi above also failed, this is a driver problem "
                     "(a driver update without a reboot is the usual cause).")
        lines.append("    -> if nvidia-smi works, CUDA_VISIBLE_DEVICES may be excluding "
                     "the GPU: try FLORENCE_CUDA_VISIBLE_DEVICES = \"\" to leave it unset.")
    elif str(info.get("allocation_probe", "")).startswith("FAILED"):
        lines.append("    the device is visible but plain allocation fails: %s"
                     % info.get("allocation_probe"))
        if str(info.get("direct_nvmlInit_v2", "")).lower().find("ok") < 0:
            lines.append("    NVML itself does not initialise (%s)."
                         % info.get("direct_nvmlInit_v2"))
            lines.append("    -> driver/NVML mismatch on the machine. A driver updated "
                         "without a reboot is the usual cause; reboot and retest.")
        else:
            lines.append("    -> driver/library mismatch, or another process holds all VRAM.")
    elif str(info.get("expandable_segments_probe", "")).startswith("FAILED"):
        lines.append("    plain CUDA allocation WORKS, but expandable_segments fails:")
        lines.append("      %s" % info.get("expandable_segments_probe"))
        lines.append("    -> keep FLORENCE_PYTORCH_CUDA_ALLOC_CONF = \"\" (the default). "
                     "That option routes the allocator through NVML, which is broken here.")
    else:
        lines.append("    GPU looks usable from the worker: %s"
                     % ", ".join(info.get("device_names") or ["?"]))
        lines.append("    -> if Florence-2 still runs on CPU, check the worker stderr log "
                     "for a fallback message, and confirm FLORENCE_DEVICE is 'auto' or 'cuda'.")

    report = "\n".join(lines)
    try:
        print(report)
    except Exception:
        pass
    return report


class Florence2WorkerClient:
    """Parent-side handle on the Florence-2 worker process.

    Exposes the same readtext()/readtext_batch() surface as the in-process
    Florence2Engine, so DicomProcessor does not care which one it is holding.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.proc = None
        self.device = "?"
        self.dtype = "?"
        self._script_path = None
        self._stderr_path = None
        self._stderr_handle = None
        self._tmpdir = None
        self._out_q = None
        self._reader = None
        self._restarts = 0
        self._minimal_env = False
        self.cuda_error = None
        self._launch()

    def _log(self, msg, error=False):
        try:
            if self.logger:
                (self.logger.error if error else self.logger.info)(msg)
        except Exception:
            pass

    def _worker_python(self):
        return _florence_worker_python()

    def _worker_env(self, minimal=False):
        return _florence_worker_env(minimal=minimal)

    def _launch(self, minimal_env=False):
        self._minimal_env = bool(minimal_env)
        self._tmpdir = tempfile.mkdtemp(prefix="headctdeid_florence_")

        self._script_path = os.path.join(self._tmpdir, "florence_worker.py")
        with open(self._script_path, "w", encoding="utf-8") as f:
            f.write(FLORENCE_WORKER_SOURCE)

        cfg = {
            "model_id": FLORENCE_MODEL_ID,
            "task": FLORENCE_TASK,
            "fallback_plain_ocr": bool(FLORENCE_FALLBACK_PLAIN_OCR),
            "max_new_tokens": int(FLORENCE_MAX_NEW_TOKENS),
            "num_beams": int(FLORENCE_NUM_BEAMS),
            "dtype": str(FLORENCE_DTYPE),
            "device": str(FLORENCE_DEVICE),
            "attn_impl": str(FLORENCE_ATTN_IMPL),
            "local_files_only": bool(FLORENCE_LOCAL_FILES_ONLY),
            "prefer_native": bool(FLORENCE_PREFER_NATIVE),
            "cpu_fallback": bool(FLORENCE_FALLBACK_TO_CPU_ON_CUDA_ERROR),
            "native_equivalent": dict(FLORENCE_NATIVE_EQUIVALENT),
            "upscale": float(FLORENCE_UPSCALE),
            "empty_cache_every": int(FLORENCE_EMPTY_CACHE_EVERY_N_CALLS),
            "hf_cache_dir": str(FLORENCE_HF_CACHE_DIR or ""),
        }

        self._stderr_path = os.path.join(self._tmpdir, "florence_worker_stderr.log")
        self._stderr_handle = open(self._stderr_path, "w", encoding="utf-8")

        popen_kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            bufsize=1,
            cwd=self._tmpdir,
            env=self._worker_env(minimal=minimal_env),
        )
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        cmd = [self._worker_python(), self._script_path, json.dumps(cfg)]
        self.proc = subprocess.Popen(cmd, **popen_kwargs)

        self._out_q = queue.Queue()
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()

        _safe_show_status(
            f"Loading {FLORENCE_MODEL_ID} in a worker process "
            f"(first run downloads the weights, ~1.5 GB)...", 10000)

        line = self._read_json(timeout_sec=float(FLORENCE_WORKER_LOAD_TIMEOUT_SEC))
        if not line or not line.get("ready"):
            err = (line or {}).get("error", "worker did not start")
            self._log(f"Florence-2 worker failed to start: {err}\n{self._stderr_tail()}", error=True)
            self.close()
            raise RuntimeError(err)

        self.device = line.get("device", "?")
        self.dtype = line.get("dtype", "?")
        self.impl = line.get("impl", "?")
        self.model_id = line.get("model_id", FLORENCE_MODEL_ID)
        self.transformers_version = line.get("transformers", "?")
        self.cuda_error = line.get("cuda_error")
        self._log(f"Florence-2 worker ready: model={self.model_id} impl={self.impl} "
                  f"device={self.device} dtype={self.dtype} "
                  f"transformers={self.transformers_version}"
                  f"{' [minimal env]' if minimal_env else ''}")

        if self.cuda_error:
            nvml_related = "nvml" in str(self.cuda_error).lower()
            if (nvml_related and not minimal_env
                    and FLORENCE_RETRY_CUDA_WITH_MINIMAL_ENV):
                self._log("Florence-2: NVML failure on first launch; retrying with a "
                          "minimal environment (no PYTORCH_CUDA_ALLOC_CONF, no "
                          "CUDA_VISIBLE_DEVICES override).", error=True)
                _safe_show_status("Retrying Florence-2 GPU init with a clean environment...", 6000)
                try:
                    self.close()
                except Exception:
                    pass
                self._launch(minimal_env=True)
                return

            self._log(f"Florence-2 fell back to {self.device}: {self.cuda_error}", error=True)
            if minimal_env:
                self._log("The GPU also failed with a minimal environment, so this is a "
                          "driver/NVML problem on the machine rather than a setting in "
                          "this module. Run florence_gpu_report() for details.", error=True)
            _safe_show_status(f"Florence-2 running on {self.device} (GPU unavailable).", 8000)

    def _pump_stdout(self):
        try:
            for line in self.proc.stdout:
                self._out_q.put(line)
        except Exception:
            pass
        finally:
            self._out_q.put(None)

    def _read_json(self, timeout_sec):
        """Block for one JSON line, keeping the Slicer UI responsive."""
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            try:
                line = self._out_q.get(timeout=0.2)
            except queue.Empty:
                try:
                    slicer.app.processEvents()
                except Exception:
                    pass
                if self.proc is not None and self.proc.poll() is not None:
                    try:
                        line = self._out_q.get_nowait()
                    except Exception:
                        return None
                    if line is None:
                        return None
                else:
                    continue
            if line is None:
                return None
            line = str(line).strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except Exception:
                continue
        return None

    def _stderr_tail(self, n_lines=25):
        try:
            with open(self._stderr_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:])
        except Exception:
            return ""

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _restart(self):
        if self._restarts >= int(FLORENCE_WORKER_MAX_RESTARTS):
            return False
        self._restarts += 1
        self._log(f"Restarting Florence-2 worker (attempt {self._restarts}) after:"
                  f"\n{self._stderr_tail()}", error=True)
        try:
            self.close()
        except Exception:
            pass
        try:
            self._launch(minimal_env=getattr(self, "_minimal_env", False))
            return True
        except Exception as e:
            self._log(f"Florence-2 worker restart failed: {e}", error=True)
            return False

    def readtext_batch(self, grays):
        if not grays:
            return []

        if not self._alive() and not self._restart():
            raise RuntimeError("Florence-2 worker is not running")

        paths = []
        try:
            for i, g in enumerate(grays):
                pth = os.path.join(self._tmpdir, f"req_{os.getpid()}_{i}.npy")
                np.save(pth, np.ascontiguousarray(g))
                paths.append(pth)

            req = json.dumps({"cmd": "ocr", "paths": paths}) + "\n"
            try:
                self.proc.stdin.write(req)
                self.proc.stdin.flush()
            except Exception as e:
                if not self._restart():
                    raise RuntimeError(f"Florence-2 worker write failed: {e}")
                self.proc.stdin.write(req)
                self.proc.stdin.flush()

            resp = self._read_json(timeout_sec=float(FLORENCE_WORKER_CALL_TIMEOUT_SEC))

            if resp is None:
                self._log(f"Florence-2 worker did not respond.\n{self._stderr_tail()}", error=True)
                self._restart()
                return [[] for _ in grays]

            if not resp.get("ok"):
                self._log(f"Florence-2 worker error: {resp.get('error')}", error=True)
                return [[] for _ in grays]

            out = []
            for rows in resp.get("results", []):
                hits = []
                for r in rows:
                    pts = r.get("points")
                    pts = None if pts is None else np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                    conf = r.get("conf", None)
                    hits.append((pts, _strip_special(r.get("text", "")),
                                 None if conf is None else float(conf)))
                out.append(hits)

            while len(out) < len(grays):
                out.append([])
            return out

        finally:
            for pth in paths:
                try:
                    os.remove(pth)
                except Exception:
                    pass

    def readtext(self, gray8):
        out = self.readtext_batch([gray8])
        return out[0] if out else []

    def close(self):
        try:
            if self._alive():
                try:
                    self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self.proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=10)
                except Exception:
                    try:
                        self.proc.terminate()
                        self.proc.wait(timeout=5)
                    except Exception:
                        try:
                            self.proc.kill()
                        except Exception:
                            pass
        finally:
            for closer in (self.proc.stdin if self.proc else None,
                           self.proc.stdout if self.proc else None,
                           self._stderr_handle):
                try:
                    if closer:
                        closer.close()
                except Exception:
                    pass
            self.proc = None


_FLORENCE_SHARED_ENGINE = None
_FLORENCE_LOAD_FAILED = False


def get_shared_florence_engine(logger=None):
    """Return the shared engine (worker process by default), or None."""
    global _FLORENCE_SHARED_ENGINE, _FLORENCE_LOAD_FAILED

    if _FLORENCE_SHARED_ENGINE is not None:
        return _FLORENCE_SHARED_ENGINE
    if _FLORENCE_LOAD_FAILED:
        return None

    try:
        if FLORENCE_RUN_IN_SUBPROCESS:
            engine = Florence2WorkerClient(logger=logger)
            where = "worker process"
        else:
            _safe_show_status(
                f"Loading {FLORENCE_MODEL_ID} in-process "
                f"(first run downloads the weights, ~1.5 GB)...", 10000)
            engine = Florence2Engine()
            where = "in-process"

        msg = (f"Florence-2 ready ({where}): model={FLORENCE_MODEL_ID} "
               f"device={engine.device} dtype={engine.dtype} task={FLORENCE_TASK}")
        if logger:
            try:
                logger.info(msg)
            except Exception:
                pass
        _safe_show_status(msg, 3000)

        _FLORENCE_SHARED_ENGINE = engine
        return engine

    except Exception as e:
        _FLORENCE_LOAD_FAILED = True
        if logger:
            try:
                logger.error(f"Failed to initialize Florence-2: {e}")
            except Exception:
                pass
        _safe_show_status(f"Florence-2 init failed; text detection skipped. ({e})", 6000)
        return None


def shutdown_shared_florence_engine():
    """Release the model and its memory. Safe to call more than once."""
    global _FLORENCE_SHARED_ENGINE, _FLORENCE_LOAD_FAILED

    engine = _FLORENCE_SHARED_ENGINE
    _FLORENCE_SHARED_ENGINE = None
    _FLORENCE_LOAD_FAILED = False

    if engine is None:
        return
    try:
        engine.close()
    except Exception:
        pass


def prefetch_florence_weights(model_id=None):
    """Download the weights ahead of time.

    Handy on a PHI machine: run this once from Slicer's Python console while
    the machine has internet, then set FLORENCE_LOCAL_FILES_ONLY = True.

        from HeadCTDeid import prefetch_florence_weights
        prefetch_florence_weights()
    """
    mid = model_id or FLORENCE_MODEL_ID

    if FLORENCE_PREFER_NATIVE and mid in FLORENCE_NATIVE_EQUIVALENT:
        try:
            from transformers import Florence2ForConditionalGeneration
            mid = FLORENCE_NATIVE_EQUIVALENT[mid]
        except Exception:
            pass

    if FLORENCE_HF_CACHE_DIR:
        os.environ["HF_HOME"] = str(FLORENCE_HF_CACHE_DIR)
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(str(FLORENCE_HF_CACHE_DIR), "hub")

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=mid)
    _safe_show_status(f"Florence-2 weights cached at: {path}", 8000)
    return path


class HeadCTDeid(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Head CT De-identification"
        self.parent.categories = ["Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = ["Anh Tuan Tran, Sam Payabvash"]
        self.parent.helpText = "This module de-identifies DICOM files by removing patient information based on a given mapping table."
        self.parent.acknowledgementText = "This file was developed by Anh Tuan Tran, Sam Payabvash (Columbia University)."


class HeadCTDeidWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/HeadCTDeid.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = HeadCTDeidLogic()

        self.ui.inputFolderButton.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.outputFolderButton.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.applyButton.connect("clicked()", self.onApplyButton)
        self.ui.excelFileButton.connect("clicked()", self.onBrowseExcelFile)
        self.ui.deidentifyCheckbox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.deidentifyCTACheckbox.connect("toggled(bool)", self.updateParameterNodeFromGUI)

        self.initializeParameterNode()

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode):
        import vtk
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        self._updatingGUIFromParameterNode = True

        self.ui.inputFolderButton.directory = self._parameterNode.GetParameter("InputFolder")
        excelFile = self._parameterNode.GetParameter("ExcelFile")
        if excelFile:
            self.ui.excelFileButton.text = excelFile
        self.ui.outputFolderButton.directory = self._parameterNode.GetParameter("OutputFolder")

        self.ui.deidentifyCheckbox.setChecked(self._parameterNode.GetParameter("Deidentify") == "true")
        self.ui.deidentifyCTACheckbox.setChecked(self._parameterNode.GetParameter("DeidentifyCTA") == "true")

        if (
            len(self._parameterNode.GetParameter("InputFolder")) > 1
            and len(self._parameterNode.GetParameter("ExcelFile")) > 4
            and len(self._parameterNode.GetParameter("OutputFolder")) > 1
            and self._parameterNode.GetParameter("ExcelFile") != "Browse"
        ):
            self.ui.applyButton.setEnabled(True)
        else:
            self.ui.applyButton.setEnabled(False)

        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()
        self._parameterNode.SetParameter("InputFolder", self.ui.inputFolderButton.directory)
        self._parameterNode.SetParameter("ExcelFile", self.ui.excelFileButton.text)
        self._parameterNode.SetParameter("OutputFolder", self.ui.outputFolderButton.directory)
        self._parameterNode.SetParameter("Deidentify", str(self.ui.deidentifyCheckbox.isChecked()).lower())
        self._parameterNode.SetParameter("DeidentifyCTA", str(self.ui.deidentifyCTACheckbox.isChecked()).lower())
        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        try:
            import gdcm
            slicer.util.infoDisplay(
                "This tool is a work-in-progress being validated in project. "
                "Contact sp4479@columbia.edu for details. Use at your own risk.",
                windowTitle="Warning",
            )

            force_ocr_all = self.ui.deidentifyCheckbox.isChecked()
            remove_CTA = self.ui.deidentifyCTACheckbox.isChecked()

            if self.ui.progressBar:
                self.ui.progressBar.setValue(0)

            self.logic.process(
                self.ui.inputFolderButton.directory,
                self.ui.excelFileButton.text,
                self.ui.outputFolderButton.directory,
                force_ocr_all=force_ocr_all,
                remove_CTA=remove_CTA,
                progressBar=self.ui.progressBar,
            )
        except Exception:
            slicer.util.pip_install("python-gdcm==3.0.25")
            slicer.util.pip_uninstall("torch")
            slicer.util.pip_install([ "torch", "--extra-index-url", "https://download.pytorch.org/whl/cu121"])
            slicer.util.pip_install("pandas==2.2.3")
            slicer.util.pip_install("openpyxl")
            slicer.util.pip_install("pydicom")
            slicer.util.pip_install("pylibjpeg")
            slicer.util.pip_install("pylibjpeg-libjpeg")
            slicer.util.pip_install("pylibjpeg-openjpeg")
            slicer.util.pip_install("scikit-image")
            slicer.util.pip_uninstall("opencv-python")
            slicer.util.pip_uninstall("opencv-python-headless")
            slicer.util.pip_install("opencv-python-headless")
            slicer.util.pip_install("pillow")
            slicer.util.pip_install("transformers>=4.56")
            slicer.util.pip_install("timm")
            slicer.util.pip_install("einops")
            slicer.util.pip_install("accelerate")
            slicer.util.pip_install("huggingface_hub")
            import torch
            from packaging import version
            if version.parse(torch.__version__) < version.parse("2.3"):
                slicer.util.pip_uninstall("numpy")
                slicer.util.pip_install("numpy<2")
            slicer.util.infoDisplay(
                "To support full encoding DICOM.\nPlease restart Slicer to complete the setup.",
                windowTitle="Warning",
            )

    def onBrowseExcelFile(self):
        from ctk import ctkFileDialog
        fileDialog = ctkFileDialog()
        fileDialog.setWindowTitle("Select Excel/CSV File")
        fileDialog.setNameFilters(["Excel Files (*.xlsx)", "CSV Files (*.csv)", "All Files (*)"])
        fileDialog.setFileMode(ctkFileDialog.ExistingFile)
        fileDialog.setOption(ctkFileDialog.DontUseNativeDialog, False)
        if fileDialog.exec_():
            selectedFile = fileDialog.selectedFiles()[0]
            self.ui.excelFileButton.text = selectedFile
            self._parameterNode.SetParameter("ExcelFile", selectedFile)
            self.updateGUIFromParameterNode()


class HeadCTDeidLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.logger = logging.getLogger("PatientProcessor")

    def setDefaultParameters(self, parameterNode):
        if not parameterNode.GetParameter("InputFolder"):
            parameterNode.SetParameter("InputFolder", "")
        if not parameterNode.GetParameter("ExcelFile"):
            parameterNode.SetParameter("ExcelFile", "")
        if not parameterNode.GetParameter("OutputFolder"):
            parameterNode.SetParameter("OutputFolder", "")
        if not parameterNode.GetParameter("Deidentify"):
            parameterNode.SetParameter("Deidentify", "false")
        if not parameterNode.GetParameter("DeidentifyCTA"):
            parameterNode.SetParameter("DeidentifyCTA", "false")

    def _ensure_logger(self, outputFolder):
        try:
            os.makedirs(outputFolder, exist_ok=True)
            log_file = os.path.join(outputFolder, "patient_processing.log")
            already = any(
                isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_file
                for h in self.logger.handlers
            )
            if not already:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
            self.logger.info(f"Initialized patient processing module {log_file}")
        except Exception:
            pass

    def _init_global_drop_csv(self, csv_path):
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=[
                        "timestamp",
                        "patient_old_id",
                        "patient_new_id",
                        "series_folder",
                        "source_dir",
                        "source_filename",
                        "instance_number",
                        "series_instance_uid",
                        "study_instance_uid",
                        "sop_instance_uid",
                        "burned_in_annotation",
                        "decision",
                        "reason",
                        "hit_text",
                        "hit_conf",
                        "hit_bbox",
                    ])
                    w.writeheader()
        except Exception as e:
            self.logger.error(f"Failed to initialize global drop csv: {e}")

    def process(
        self,
        inputFolder,
        excelFile,
        outputFolder,
        force_ocr_all,
        remove_CTA,
        progressBar,
    ):
        import pandas

        if not os.path.exists(inputFolder):
            raise ValueError(f"Input folder does not exist: {inputFolder}")
        if not os.path.exists(excelFile):
            raise ValueError(f"Excel/CSV file does not exist: {excelFile}")

        os.makedirs(outputFolder, exist_ok=True)
        self._ensure_logger(outputFolder)

        columns_as_text = ["original_folder_name", "new_folder_name"]
        ext = os.path.splitext(excelFile)[1].lower()
        if ext == ".csv":
            df = pandas.read_csv(excelFile, dtype={col: str for col in columns_as_text})
        elif ext in [".xlsx", ".xls"]:
            df = pandas.read_excel(excelFile, dtype={col: str for col in columns_as_text})
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if ("original_folder_name" not in df.columns) or ("new_folder_name" not in df.columns):
            raise ValueError("Excel file must contain 'original_folder_name' and 'new_folder_name' columns")

        df["original_folder_name"] = df["original_folder_name"].astype(str).str.strip()
        df["new_folder_name"] = df["new_folder_name"].astype(str).str.strip()
        id_mapping = dict(zip(df["original_folder_name"], df["new_folder_name"]))

        dicom_folders = [d for d in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, d))]
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(outputFolder, f"Processed for Anonymization_{current_time}")
        os.makedirs(out_path, exist_ok=True)

        original_face_render_dir = os.path.join(out_path, "original_face_render")
        after_deidentification_render_dir = os.path.join(out_path, "after_deidentification_render")
        os.makedirs(original_face_render_dir, exist_ok=True)
        os.makedirs(after_deidentification_render_dir, exist_ok=True)

        global_drop_csv_path = os.path.join(out_path, GLOBAL_DROPPED_CSV_NAME)
        self._init_global_drop_csv(global_drop_csv_path)

        ocr_debug_root = os.path.join(out_path, OCR_DEBUG_ROOT_DIRNAME)
        ocr_detected_dir = os.path.join(ocr_debug_root, OCR_DEBUG_DETECTED_DIRNAME)
        ocr_no_text_dir = os.path.join(ocr_debug_root, OCR_DEBUG_NO_TEXT_DIRNAME)
        ocr_redacted_dir = os.path.join(ocr_debug_root, OCR_DEBUG_REDACTED_DIRNAME)
        ocr_verify_fail_dir = os.path.join(ocr_debug_root, OCR_DEBUG_VERIFY_FAIL_DIRNAME)
        ocr_prescreen_dir = os.path.join(ocr_debug_root, OCR_DEBUG_PRESCREEN_DIRNAME)
        ocr_not_examined_dir = os.path.join(ocr_debug_root, OCR_DEBUG_NOT_EXAMINED_DIRNAME)
        os.makedirs(ocr_detected_dir, exist_ok=True)
        os.makedirs(ocr_no_text_dir, exist_ok=True)
        os.makedirs(ocr_redacted_dir, exist_ok=True)
        if SAVE_PRESCREEN_SKIPPED_DEBUG_PNG:
            os.makedirs(ocr_prescreen_dir, exist_ok=True)
        if SAVE_NOT_EXAMINED_DEBUG_PNG:
            os.makedirs(ocr_not_examined_dir, exist_ok=True)
        if REDACT_VERIFY_WITH_SECOND_PASS:
            os.makedirs(ocr_verify_fail_dir, exist_ok=True)

        folders_to_process = [f for f in sorted(dicom_folders) if f in id_mapping]
        total = max(1, len(folders_to_process))
        done = 0

        processors = []

        if progressBar:
            progressBar.setValue(0)

        for foldername in folders_to_process:
            dst_folder = ""
            try:
                dst_folder = os.path.join(out_path, id_mapping[foldername])

                processor = DicomProcessor(force_ocr_all=bool(force_ocr_all))
                processors.append(processor)

                src_folder = os.path.join(inputFolder, foldername)

                _safe_show_status(f"Processing patient folder: {foldername} → {id_mapping[foldername]}", 4000)
                self.logger.info(f"Processing patient folder: {foldername} → {id_mapping[foldername]}")

                _ = processor.drown_volume(
                    in_path=src_folder,
                    out_path=dst_folder,
                    replacer="face",
                    id=id_mapping[foldername],
                    patient_old_id=foldername,
                    patient_id="0",
                    name=f"Processed for Anonymization {id_mapping[foldername]}",
                    remove_CTA=remove_CTA,
                    global_drop_csv_path=global_drop_csv_path,
                    global_detected_png_dir=ocr_detected_dir,
                    global_no_text_png_dir=ocr_no_text_dir,
                    global_redacted_png_dir=ocr_redacted_dir,
                    global_verify_fail_png_dir=ocr_verify_fail_dir,
                    global_prescreen_png_dir=ocr_prescreen_dir,
                    global_not_examined_png_dir=ocr_not_examined_dir,
                    patient_input_root=src_folder,
                    original_face_render_dir=original_face_render_dir,
                    after_deidentification_render_dir=after_deidentification_render_dir,
                )

                processor.wait_for_all_subprocesses(timeout_total_sec=7200)

                done += 1
                if progressBar:
                    progressBar.setValue(int(done * 99 / total) if done < total else 99)

                _safe_show_status(f"Finished: {foldername}", 3000)
                self.logger.info(f"Finished processing folder: {foldername}")

            except Exception as e:
                self.logger.error(f"Error processing folder {foldername}: {str(e)}")
                if dst_folder and os.path.exists(dst_folder):
                    shutil.rmtree(dst_folder)

        for p in processors:
            try:
                p.wait_for_all_subprocesses(timeout_total_sec=7200)
            except Exception as e:
                self.logger.error(f"Final wait_for_all_subprocesses error: {e}")

        if FLORENCE_SHUTDOWN_WORKER_AFTER_RUN:
            try:
                shutdown_shared_florence_engine()
                self.logger.info("Florence-2 worker shut down; model memory released.")
            except Exception as e:
                self.logger.error(f"Florence-2 shutdown error: {e}")

        if progressBar:
            progressBar.setValue(100)

        _safe_show_status("All processing finished.", 5000)
        self.logger.info("All processing finished.")


class DicomProcessor:
    """
    Pipeline: de-identification + face/air replacement + Florence-2 detect->drop.

    Detection run decision per slice:
      if force_ocr_all == True:
          run Florence-2 detection on every slice
      else:
          run Florence-2 detection only when BurnedInAnnotation==YES

    A cheap OpenCV top-hat pre-screen can skip obviously empty slices first
    (PRESCREEN_MODE), because Florence-2 is much slower than EasyOCR was.
    """

    def __init__(self, force_ocr_all=False):
        self.study_uid_map = defaultdict(str)
        self.series_uid_map = defaultdict(str)
        self.sop_uid_map = defaultdict(str)
        self.uid_map_general = defaultdict(str)

        self.logger = logging.getLogger("PatientProcessor")
        self._force_ocr_all = bool(force_ocr_all)

        self._running_subprocesses = []

        self._ocr = None
        self._prescreen_skipped = 0
        self._detect_calls = 0
        self._warned_unknown_conf = False
        try:
            if ENABLE_TEXT_DETECTION:
                logging.getLogger(__name__).info(
                    "Burned-in text mode: action=%s, never_drop=%s -> slices with text "
                    "are blacked out and KEPT." % (TEXT_ACTION, NEVER_DROP_SLICES))
        except Exception:
            pass
        self._unknown_conf_count = 0
        self._conf_samples = []

    def _popen_and_wait(self, cmd, timeout_sec):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as e:
            return -1, "", str(e)

        self._running_subprocesses.append(proc)

        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except Exception:
                    stdout, stderr = "", ""
            rc = proc.returncode if proc.returncode is not None else -1
        except Exception as e:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            rc = proc.returncode if proc.returncode is not None else -1
            stderr = (stderr or "") + f"\ncommunicate_error: {e}"
        finally:
            try:
                self._running_subprocesses = [p for p in self._running_subprocesses if p is not proc]
            except Exception:
                pass

        return rc, (stdout or ""), (stderr or "")

    def wait_for_all_subprocesses(self, timeout_total_sec=7200):
        start = time.time()
        procs = list(self._running_subprocesses)
        for proc in procs:
            try:
                remaining = max(0.1, timeout_total_sec - (time.time() - start))
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._running_subprocesses = [p for p in self._running_subprocesses if p.poll() is None]
        return len(self._running_subprocesses) == 0

    def _detect_gpu_available(self):
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                try:
                    return True, torch.cuda.get_device_name(0)
                except Exception:
                    return True, "CUDA GPU"
            return False, "CPU"
        except Exception:
            return False, "CPU"

    def _ensure_ocr(self):
        """Lazily obtain the shared Florence-2 engine."""
        if self._ocr is not None:
            return True

        engine = get_shared_florence_engine(logger=self.logger)
        if engine is None:
            self._ocr = None
            return False

        self._ocr = engine
        return True

    def _alnum_count(self, s: str) -> int:
        return _alnum_count(s)

    def _text_plausible(self, txt: str) -> bool:
        return _text_plausible(txt)

    def _dicom_pixels_to_gray8_for_ocr(self, ds):
        """
        DICOM -> 8-bit grayscale for the OCR model:
        - pixels = ds.pixel_array (take first frame if multi-frame)
        - apply RescaleSlope / RescaleIntercept
        - min-max normalize to [0, 255] uint8
        - invert MONOCHROME1 so text is bright on dark (RESPECT_MONOCHROME1)
        """
        import cv2

        pixels = ds.pixel_array
        samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)

        if pixels.ndim == 4:
            pixels = pixels[0]

        is_color = False
        if pixels.ndim == 3:
            if samples >= 3 and pixels.shape[-1] in (3, 4):
                is_color = True
            else:
                pixels = pixels[0]

        if is_color:
            rgb = pixels[..., :3].astype(np.float32)
            mx = float(np.max(rgb)) if rgb.size else 0.0
            if mx > 255.0:
                rgb = rgb / mx * 255.0
            return cv2.cvtColor(rgb.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        pixels = pixels.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        pixels_hu = pixels * slope + intercept

        mn = float(np.min(pixels_hu))
        mx = float(np.max(pixels_hu))
        if mx <= mn:
            mx = mn + 1.0

        gray8 = ((pixels_hu - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

        if RESPECT_MONOCHROME1:
            try:
                pi = str(getattr(ds, "PhotometricInterpretation", "") or "").upper().strip()
                if pi == "MONOCHROME1":
                    gray8 = 255 - gray8
            except Exception:
                pass

        return gray8

    def _run_florence(self, gray8):
        """Return raw hits [(points|None, text, confidence), ...]."""
        try:
            return self._ocr.readtext(gray8)
        except Exception as e:
            try:
                self.logger.error(f"Florence-2 inference failed: {e}")
            except Exception:
                pass
            return []

    def _filter_florence_hits(self, hits, gray8):
        """Apply confidence, plausibility, box-size and hallucination filters."""
        kept = []
        any_flagged = False
        dropped_low_conf = 0

        for points, txt, conf in hits:
            try:
                if conf is None:
                    if not self._warned_unknown_conf:
                        self._warned_unknown_conf = True
                        try:
                            self.logger.warning(
                                "Florence-2 returned no confidence score; "
                                "FLORENCE_MIN_CONFIDENCE=%s cannot filter these hits "
                                "(FLORENCE_KEEP_WHEN_CONFIDENCE_UNKNOWN=%s)."
                                % (FLORENCE_MIN_CONFIDENCE,
                                   FLORENCE_KEEP_WHEN_CONFIDENCE_UNKNOWN))
                        except Exception:
                            pass
                    self._unknown_conf_count += 1
                    if not FLORENCE_KEEP_WHEN_CONFIDENCE_UNKNOWN:
                        dropped_low_conf += 1
                        continue
                elif float(conf) < float(FLORENCE_MIN_CONFIDENCE):
                    dropped_low_conf += 1
                    continue

                if not _text_plausible(txt):
                    continue
                if not _box_big_enough(points):
                    continue
                if FLORENCE_RESTRICT_TO_BORDER_BAND and not _in_border_band(points, gray8.shape[:2]):
                    continue

                flags = _hallucination_flags(txt, gray8)
                if flags:
                    any_flagged = True
                    if FLORENCE_DROP_SUSPECTED_HALLUCINATIONS:
                        continue

                kept.append((points, txt, (None if conf is None else float(conf)), flags))
            except Exception:
                continue

        return kept, any_flagged, dropped_low_conf

    def _draw_ocr_results(self, gray8, items):
        """Draw Florence-2 boxes/labels onto a BGR debug image."""
        import cv2

        det_img = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        h, w = gray8.shape[:2]

        for item in items:
            try:
                points, txt, sc = item[0], item[1], item[2]
                flags = item[3] if len(item) > 3 else []

                colour = OCR_DEBUG_COLOR_FLAGGED if flags else OCR_DEBUG_COLOR_OK
                label = f"{txt} ({float(sc):.2f})" if sc is not None else f"{txt} (conf n/a)"
                if flags:
                    label += f" [{','.join(flags)}]"
                label = label.replace("\n", " ")[:60]

                if points is None:
                    if OCR_DEBUG_DRAW_LABELS:
                        cv2.putText(
                            det_img,
                            "(no box) " + label,
                            (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            OCR_DEBUG_FONT_SCALE,
                            colour,
                            OCR_DEBUG_FONT_THICKNESS,
                            cv2.LINE_AA,
                        )
                    continue

                pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                if pts.shape[0] < 2:
                    continue

                if OCR_DEBUG_DRAW_BOXES:
                    cv2.polylines(det_img, [pts], True, colour, OCR_DEBUG_BOX_THICKNESS)

                if OCR_DEBUG_DRAW_LABELS:
                    x = int(np.min(pts[:, 0]))
                    y = int(np.min(pts[:, 1])) - 5
                    if y < 10:
                        y = int(np.max(pts[:, 1])) + 15
                    cv2.putText(
                        det_img,
                        label,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        OCR_DEBUG_FONT_SCALE,
                        colour,
                        OCR_DEBUG_FONT_THICKNESS,
                        cv2.LINE_AA,
                    )
            except Exception:
                continue

        return det_img

    def detect_text_debug(self, ds, burned_flag=False):
        """
        Run Florence-2 burned-in text detection on one DICOM slice.

        Returns:
          has_text: bool
          hit_text: str            (all kept strings joined with " | ")
          hit_conf: float|None     (sequence-level generation probability)
          hit_bbox: list|None      (points of the first kept hit, None for <OCR>)
          gray8: uint8 image
          detection_img: BGR image with boxes drawn
          boxes: list of (N,2) float arrays, one per localised hit
          boxless: int, number of hits with no coordinates (plain <OCR>)
        """
        import cv2

        if not self._ensure_ocr():
            return False, "", None, None, None, None, [], 0

        try:
            gray8 = self._dicom_pixels_to_gray8_for_ocr(ds)
        except Exception:
            return False, "", None, None, None, None, [], 0

        if str(PRESCREEN_MODE).lower() == "on":
            run_anyway = bool(burned_flag) and PRESCREEN_ALWAYS_RUN_ON_GROUND_TRUTH
            if not run_anyway:
                maybe_text, _n = _prescreen_says_maybe_text(gray8)
                if not maybe_text:
                    self._prescreen_skipped += 1
                    return (False, "", None, None, gray8,
                            cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR), [], 0)

        self._detect_calls += 1
        raw_hits = self._run_florence(gray8)
        if not raw_hits:
            return (False, "", None, None, gray8,
                    cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR), [], 0)

        kept, _flagged, _low_conf = self._filter_florence_hits(raw_hits, gray8)
        detection_img = self._draw_ocr_results(gray8, kept if kept else [])

        if kept:
            texts = " | ".join(str(t).replace("\n", " ").strip() for _, t, _, _ in kept if str(t).strip())

            known = [float(c) for _, _, c, _ in kept if c is not None]
            best_conf = max(known) if known else None
            if best_conf is not None:
                self._conf_samples.append(best_conf)

            boxes = []
            boxless = 0
            for points, _t, _c, _f in kept:
                if points is None:
                    boxless += 1
                else:
                    boxes.append(np.asarray(points, np.float32).reshape(-1, 2))

            bbox_list = boxes[0].tolist() if boxes else None

            return (True, texts, (None if best_conf is None else float(best_conf)),
                    bbox_list, gray8, detection_img, boxes, boxless)

        return False, "", None, None, gray8, detection_img, [], 0

    def _textlike_component_boxes(self, gray8):
        """Every character-sized bright component in the slice, as (x,y,w,h)."""
        try:
            import cv2

            k = int(PRESCREEN_TOPHAT_KERNEL)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            tophat = cv2.morphologyEx(gray8, cv2.MORPH_TOPHAT, kernel)
            if float(np.max(tophat)) < 1.0:
                return []
            _, bw = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            num, _lab, stats, _cent = cv2.connectedComponentsWithStats(bw, 8)

            out = []
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if (PRESCREEN_MIN_CHAR_H <= h <= PRESCREEN_MAX_CHAR_H
                        and PRESCREEN_MIN_CHAR_W <= w <= PRESCREEN_MAX_CHAR_W
                        and area >= PRESCREEN_MIN_CHAR_AREA):
                    out.append((int(x), int(y), int(w), int(h)))
            return out
        except Exception:
            return []

    def _group_components_into_lines(self, comps):
        """Cluster character boxes into text lines by vertical then horizontal run."""
        if not comps:
            return []

        rows = []
        for (x, y, w, h) in sorted(comps, key=lambda c: (c[1], c[0])):
            cy = y + h / 2.0
            placed = False
            for row in rows:
                if abs(cy - row["cy"]) <= float(REDACT_SWEEP_LINE_GAP_Y) + row["h"] / 2.0:
                    row["items"].append((x, y, w, h))
                    n = len(row["items"])
                    row["cy"] = ((row["cy"] * (n - 1)) + cy) / n
                    row["h"] = max(row["h"], h)
                    placed = True
                    break
            if not placed:
                rows.append({"cy": cy, "h": h, "items": [(x, y, w, h)]})

        lines = []
        for row in rows:
            items = sorted(row["items"], key=lambda c: c[0])
            run = [items[0]]
            for cur in items[1:]:
                prev = run[-1]
                if cur[0] - (prev[0] + prev[2]) <= float(REDACT_SWEEP_LINE_GAP_X):
                    run.append(cur)
                else:
                    lines.append(run)
                    run = [cur]
            lines.append(run)

        out = []
        for run in lines:
            if len(run) < int(REDACT_SWEEP_MIN_CHARS_PER_LINE):
                continue
            x0 = min(c[0] for c in run)
            y0 = min(c[1] for c in run)
            x1 = max(c[0] + c[2] for c in run)
            y1 = max(c[1] + c[3] for c in run)
            out.append((x0, y0, x1, y1))
        return out

    def _line_is_relevant(self, line, seed_rects, shape_hw):
        """Keep a swept line only near the edge or near a model-flagged region."""
        h, w = int(shape_hw[0]), int(shape_hw[1])
        x0, y0, x1, y1 = line
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        fx = float(REDACT_SWEEP_BORDER_FRAC) * w
        fy = float(REDACT_SWEEP_BORDER_FRAC) * h
        if cx <= fx or cx >= w - fx or cy <= fy or cy >= h - fy:
            return True

        pad = float(REDACT_SWEEP_NEAR_HIT_PX)
        for (sx0, sy0, sx1, sy1) in seed_rects:
            if (x0 < sx1 + pad and x1 > sx0 - pad
                    and y0 < sy1 + pad and y1 > sy0 - pad):
                return True
        return False

    @staticmethod
    def _merge_rects(rects):
        """Union overlapping rectangles so masks read as solid blocks."""
        items = list(rects)
        changed = True
        while changed:
            changed = False
            merged = []
            while items:
                a = items.pop()
                hit = True
                while hit:
                    hit = False
                    rest = []
                    for b in items:
                        if (a[0] <= b[2] and b[0] <= a[2]
                                and a[1] <= b[3] and b[1] <= a[3]):
                            a = (min(a[0], b[0]), min(a[1], b[1]),
                                 max(a[2], b[2]), max(a[3], b[3]))
                            hit = True
                            changed = True
                        else:
                            rest.append(b)
                    items = rest
                merged.append(a)
            items = merged
        return items

    def _redaction_rects(self, boxes, boxless, shape_hw, gray_shape_hw=None,
                         boxless_mode=None, gray8=None):
        """Turn detected boxes into padded axis-aligned rectangles to blank out.

        Masks are grown deliberately: Florence-2's boxes hug the glyphs, and a
        tight mask can leave readable fragments at the edges.
        """
        h, w = int(shape_hw[0]), int(shape_hw[1])

        sx = sy = 1.0
        if gray_shape_hw is not None:
            gh, gw = int(gray_shape_hw[0]), int(gray_shape_hw[1])
            if gh > 0 and gw > 0 and (gh != h or gw != w):
                sy = float(h) / float(gh)
                sx = float(w) / float(gw)

        exact = str(REDACT_GEOMETRY).lower() == "exact"

        rects = []
        for pts in (boxes or []):
            try:
                arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                if arr.size == 0:
                    continue
                x0 = float(np.min(arr[:, 0])) * sx
                x1 = float(np.max(arr[:, 0])) * sx
                y0 = float(np.min(arr[:, 1])) * sy
                y1 = float(np.max(arr[:, 1])) * sy

                if exact:
                    pad = float(REDACT_EXACT_PAD_PX)
                else:
                    bh = max(1.0, y1 - y0)
                    pad = max(float(REDACT_PAD_PX), float(REDACT_PAD_FRAC) * bh)

                rx0 = int(max(0, np.floor(x0 - pad)))
                ry0 = int(max(0, np.floor(y0 - pad)))
                rx1 = int(min(w, np.ceil(x1 + pad)))
                ry1 = int(min(h, np.ceil(y1 + pad)))
                if rx1 > rx0 and ry1 > ry0:
                    rects.append((rx0, ry0, rx1, ry1))
            except Exception:
                continue

        seed_rects = list(rects)

        if not exact and REDACT_SWEEP_TEXTLIKE_COMPONENTS and gray8 is not None:
            try:
                gh, gw = gray8.shape[:2]
                comps = self._textlike_component_boxes(gray8)
                lines = self._group_components_into_lines(comps)

                seed_in_gray = []
                for (rx0, ry0, rx1, ry1) in seed_rects:
                    seed_in_gray.append((rx0 / sx, ry0 / sy, rx1 / sx, ry1 / sy))

                for line in lines:
                    if not self._line_is_relevant(line, seed_in_gray, (gh, gw)):
                        continue
                    lx0, ly0, lx1, ly1 = line
                    bh_ = max(1.0, ly1 - ly0)
                    pad = max(float(REDACT_PAD_PX), float(REDACT_PAD_FRAC) * bh_)
                    rects.append((
                        int(max(0, np.floor((lx0 - pad) * sx))),
                        int(max(0, np.floor((ly0 - pad) * sy))),
                        int(min(w, np.ceil((lx1 + pad) * sx))),
                        int(min(h, np.ceil((ly1 + pad) * sy))),
                    ))
            except Exception:
                pass

        if not exact and REDACT_EXPAND_TO_LINE and gray8 is not None and rects:
            try:
                gh, gw = gray8.shape[:2]
                comps = self._textlike_component_boxes(gray8)
                extra = float(REDACT_LINE_EXTRA_PX)
                grown = []
                gap = float(REDACT_SWEEP_LINE_GAP_X)
                for (rx0, ry0, rx1, ry1) in rects:
                    gx0, gy0 = rx0 / sx, ry0 / sy
                    gx1, gy1 = rx1 / sx, ry1 / sy

                    row = [(float(cx), float(cx + cw)) for (cx, cy, cw, ch) in comps
                           if gy0 <= (cy + ch / 2.0) <= gy1]

                    nx0, nx1 = gx0, gx1
                    changed = True
                    while changed:
                        changed = False
                        for (cx0, cx1) in row:
                            if cx1 >= nx0 - gap and cx0 <= nx1 + gap:
                                if cx0 < nx0:
                                    nx0 = cx0
                                    changed = True
                                if cx1 > nx1:
                                    nx1 = cx1
                                    changed = True

                    grown.append((
                        int(max(0, np.floor((nx0 - extra) * sx))),
                        ry0,
                        int(min(w, np.ceil((nx1 + extra) * sx))),
                        ry1,
                    ))
                rects = grown
            except Exception:
                pass

        if not exact:
            rects = self._merge_rects(rects)

        mode = str(boxless_mode if boxless_mode is not None
                   else REDACT_BOXLESS_STRATEGY).lower()
        if boxless and mode == "border_band":
            bx = int(round(float(REDACT_BORDER_BAND_FRAC) * w))
            by = int(round(float(REDACT_BORDER_BAND_FRAC) * h))
            bx = max(1, min(bx, w // 2))
            by = max(1, min(by, h // 2))
            rects.extend([
                (0, 0, w, by),
                (0, h - by, w, h),
                (0, 0, bx, h),
                (w - bx, 0, w, h),
            ])

        return rects

    def _apply_redaction(self, hu_slice, rects):
        """Blank the given rectangles in a HU array. Returns regions written."""
        if not rects:
            return 0

        fill = str(REDACT_FILL).lower()
        if fill == "air":
            value = float(REDACT_AIR_HU)
        elif fill == "min":
            try:
                value = float(np.min(hu_slice))
            except Exception:
                value = float(REDACT_AIR_HU)
        else:
            try:
                value = float(REDACT_FILL)
            except Exception:
                value = float(REDACT_AIR_HU)

        n = 0
        for (x0, y0, x1, y1) in rects:
            try:
                hu_slice[y0:y1, x0:x1] = value
                n += 1
            except Exception:
                continue
        return n

    def _draw_mask_rects(self, det_img, rects, gray_shape_hw=None, hu_shape_hw=None):
        """Outline the rectangles that will actually be blacked out.

        Mask rectangles are computed in pixel-array coordinates, while the debug
        image is the OCR grayscale, so they are mapped back if the two differ.
        """
        if det_img is None or not rects or not OCR_DEBUG_DRAW_MASK_RECTS:
            return det_img
        try:
            import cv2

            out = det_img.copy()
            h, w = out.shape[:2]

            sx = sy = 1.0
            if gray_shape_hw is not None and hu_shape_hw is not None:
                gh, gw = int(gray_shape_hw[0]), int(gray_shape_hw[1])
                hh, hw_ = int(hu_shape_hw[0]), int(hu_shape_hw[1])
                if hh > 0 and hw_ > 0 and (gh != hh or gw != hw_):
                    sy = float(gh) / float(hh)
                    sx = float(gw) / float(hw_)

            for (x0, y0, x1, y1) in rects:
                ax0 = int(max(0, min(w - 1, round(x0 * sx))))
                ay0 = int(max(0, min(h - 1, round(y0 * sy))))
                ax1 = int(max(0, min(w, round(x1 * sx))))
                ay1 = int(max(0, min(h, round(y1 * sy))))
                if ax1 > ax0 and ay1 > ay0:
                    cv2.rectangle(out, (ax0, ay0), (ax1, ay1), (0, 165, 255), 1)
            return out
        except Exception:
            return det_img

    def _gray_to_bgr(self, gray8):
        import cv2

        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    def _render_redacted_png(self, gray8, rects):
        """Render the slice as it will look after masking: solid black boxes.

        This is a picture of the result, not an annotation of the detection, so
        it can be checked directly for surviving text.
        """
        if gray8 is None:
            return None
        try:
            import cv2

            out = gray8.copy()
            for (x0, y0, x1, y1) in (rects or []):
                out[y0:y1, x0:x1] = 0
            return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None

    def _hu_to_gray8(self, hu, ds=None):
        """Same normalisation as the OCR input, but from a HU array."""
        arr = np.asarray(hu, dtype=np.float32)
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if mx <= mn:
            mx = mn + 1.0
        gray8 = ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

        if RESPECT_MONOCHROME1 and ds is not None:
            try:
                pi = str(getattr(ds, "PhotometricInterpretation", "") or "").upper().strip()
                if pi == "MONOCHROME1":
                    gray8 = 255 - gray8
            except Exception:
                pass
        return gray8

    def _verify_redaction(self, hu_slice, ds):
        """Re-run detection on the masked slice. Returns (still_has_text, text)."""
        try:
            gray8 = self._hu_to_gray8(hu_slice, ds)
        except Exception:
            return False, ""

        raw = self._run_florence(gray8)
        if not raw:
            return False, ""

        kept, _flagged, _low = self._filter_florence_hits(raw, gray8)
        if not kept:
            return False, ""

        texts = " | ".join(str(t).replace("\n", " ").strip()
                           for _, t, _, _ in kept if str(t).strip())
        return True, texts

    def _save_debug_png(self, out_dir, patient_new_id, series_folder, source_filename, img):
        """Write a debug PNG into one of the global only_for_debug folders."""
        if not out_dir or img is None:
            return None
        try:
            import cv2

            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(str(source_filename))[0]
            name = _safe_filename(f"{patient_new_id}_{series_folder}_{stem}") + ".png"
            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, img)
            return out_path
        except Exception as e:
            try:
                self.logger.error(f"Failed to write debug PNG: {e}")
            except Exception:
                pass
            return None

    def is_dicom(self, file_path, remove_CTA=False):
        import pydicom
        try:
            ds = pydicom.dcmread(file_path, force=True)
            try:
                ds.decompress()
            except Exception:
                pass
            return self.checkCTmeta(ds, remove_CTA) == 1
        except Exception:
            return False

    def load_scan(self, path):
        import pydicom
        p = Path(path)
        if p.is_file():
            return pydicom.dcmread(str(p), force=True)
        raise FileNotFoundError(f"Not a file: {path}")

    def get_pixels_hu(self, ds):
        image = ds.pixel_array.astype(np.int16)
        image[image <= -2000] = 0
        intercept = getattr(ds, "RescaleIntercept", 0)
        slope = getattr(ds, "RescaleSlope", 1)
        if slope != 1:
            image = (image.astype(np.float64) * slope).astype(np.int16)
        image += np.int16(intercept)
        return image

    def binarize_volume(self, volume, air_hu=AIR_THRESHOLD):
        out = np.zeros_like(volume, dtype=np.uint8)
        out[volume <= air_hu] = 1
        return out

    def largest_connected_component(self, binary_image):
        import cv2
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(binary_image, dtype=np.uint8)
        largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        lcc = np.zeros(labels.shape, dtype=np.uint8)
        lcc[labels == largest_idx] = 1
        return lcc

    def get_largest_component_volume(self, volume):
        return self.largest_connected_component(volume)

    def _kernel_from_pixel_spacing(self, ds):
        try:
            ps = ds.get((0x0028, 0x0030), None)
            if ps is None:
                raise ValueError("No PixelSpacing")
            v = ps.value

            if isinstance(v, str):
                parts = v.replace(",", "\\").split("\\")
                pixel = float(parts[0])
            elif hasattr(v, "__len__"):
                pixel = float(v[0])
            else:
                pixel = float(v)

            if not (pixel > 0):
                raise ValueError("PixelSpacing <= 0")

            lo = int(ceil(FACE_KERNEL_MIN_MM / pixel))
            hi = int(ceil(FACE_KERNEL_MAX_MM / pixel))
            if hi < lo:
                hi = lo

            lo = max(1, min(lo, 999))
            hi = max(1, min(hi, 999))
            return random.randint(lo, hi)
        except Exception:
            return random.randint(int(FACE_KERNEL_MIN_MM), int(FACE_KERNEL_MAX_MM))

    def apply_mask_and_get_values(self, image_volume, mask_volume):
        masked = image_volume * mask_volume
        vals = np.unique(masked)
        vals = vals[(vals > FACE_MIN_VALUE) & (vals < FACE_MAX_VALUE)]
        return vals.tolist()

    def is_substring_in_list(self, substring, string_list):
        return any(substring in str(s) for s in string_list)

    def _is_secondary_capture_sop_class(self, ds):
        """Return True for Secondary Capture SOP Class UIDs in (0008,0016)."""
        try:
            sop_class = ds.get((0x0008, 0x0016), "")
            sop_class = sop_class.value if hasattr(sop_class, "value") else sop_class
            sop_class = str(sop_class).strip()

            secondary_capture_uids = {
                "1.2.840.10008.5.1.4.1.1.7",
                "1.2.840.10008.5.1.4.1.1.7.1",
                "1.2.840.10008.5.1.4.1.1.7.2",
                "1.2.840.10008.5.1.4.1.1.7.3",
                "1.2.840.10008.5.1.4.1.1.7.4",
            }

            return sop_class in secondary_capture_uids
        except Exception:
            return False

    def checkCTmeta(self, ds, remove_CTA=False):
        """
        Accept only CT head (original/primary/axial), and exclude Secondary Capture.
        By default, exclude CTA/perfusion.
        If remove_CTA=True -> do not exclude CTA (i.e., include such series as well).
        """
        try:
            modality = ds.get((0x08, 0x60), "")
            modality = [modality.value] if hasattr(modality, "value") else [modality]
            modality = [str(x).lower().replace(" ", "") for x in modality]
            status1 = any(self.is_substring_in_list(c, modality) for c in ["ct", "computedtomography", "ctprotocal"])

            imageType = ds.get((0x08, 0x08), "")
            imageType = [imageType.value] if hasattr(imageType, "value") else [imageType]
            imageType = [str(x).lower().replace(" ", "") for x in imageType]
            status2 = all(self.is_substring_in_list(c, imageType) for c in ["original", "primary", "axial"])

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

            status5 = not self._is_secondary_capture_sop_class(ds)

            return int(status1 and status2 and status3 and status4 and status5)
        except Exception as e:
            self.error = str(e)
            return 0

    def _keep_only_components_touching_seed(self, mask_uint8, seed_uint8):
        import cv2
        m = (mask_uint8 > 0).astype(np.uint8)
        s = (seed_uint8 > 0).astype(np.uint8)
        if m.max() == 0:
            return m

        nlab, labels = cv2.connectedComponents(m, connectivity=8)
        if nlab <= 1:
            return m

        overlap_labels = np.unique(labels[s > 0])
        keep = np.zeros_like(m, dtype=np.uint8)
        for lab in overlap_labels:
            if lab == 0:
                continue
            keep[labels == lab] = 1
        return keep

    def _anterior_axis_and_sign(self, ds):
        try:
            iop = ds.get((0x0020, 0x0037), None)
            if iop is None:
                return 0, +1

            v = np.array(iop.value, dtype=float).reshape(2, 3)
            row_cos = v[0]
            col_cos = v[1]
            anterior_LPS = np.array([0.0, -1.0, 0.0])

            dr = float(np.dot(row_cos, anterior_LPS))
            dc = float(np.dot(col_cos, anterior_LPS))

            if abs(dr) >= abs(dc):
                axis = 0
                sign = +1 if dr > 0 else -1
            else:
                axis = 1
                sign = +1 if dc > 0 else -1
            return axis, sign
        except Exception:
            return 0, +1

    def _anterior_region_mask(self, shape_hw, ds, front_fraction=0.55):
        H, W = shape_hw
        axis, sign = self._anterior_axis_and_sign(ds)
        Y, X = np.ogrid[:H, :W]
        cy, cx = H // 2, W // 2

        if axis == 0:
            if sign > 0:
                cutoff = int(cy + (1.0 - front_fraction) * (H - 1 - cy))
                m = (Y >= cutoff)
            else:
                cutoff = int(cy - (1.0 - front_fraction) * (cy))
                m = (Y <= cutoff)
        else:
            if sign > 0:
                cutoff = int(cx + (1.0 - front_fraction) * (W - 1 - cx))
                m = (X >= cutoff)
            else:
                cutoff = int(cx - (1.0 - front_fraction) * (cx))
                m = (X <= cutoff)

        return m.astype(np.uint8)

    def bounded_dilate_with_front_boost(
        self,
        lcc_air_seed,
        pixels_hu,
        ds,
        k_max,
        bone_stop_hu=BONE_STOP_HU,
        front_fraction=0.55,
    ):
        import cv2

        seed = (lcc_air_seed > 0).astype(np.uint8)
        H, W = seed.shape

        allowed = (pixels_hu < int(bone_stop_hu)).astype(np.uint8)

        k_max = int(max(1, k_max))
        kmax = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_max, k_max))
        max_once = cv2.dilate(seed, kmax)

        max_once = (max_once & allowed).astype(np.uint8)
        max_once = self._keep_only_components_touching_seed(max_once, seed)

        k33 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FRONT_BOOST_KERNEL)
        anterior_region = self._anterior_region_mask((H, W), ds, front_fraction=front_fraction)

        boosted = cv2.dilate(max_once, k33)
        boosted = (boosted & anterior_region & max_once).astype(np.uint8)

        combined = (max_once | boosted).astype(np.uint8)
        combined = (combined & max_once).astype(np.uint8)
        combined = (combined & allowed).astype(np.uint8)
        combined = self._keep_only_components_touching_seed(combined, seed)

        return combined

    def apply_random_values_optimized(
        self,
        pixels_hu,
        dilated_mask,
        unique_values_list,
        bone_stop_hu=BONE_STOP_HU,
        fill_mode="air",
    ):
        new_vol = np.array(pixels_hu, copy=True)
        mask = (dilated_mask == 1) & (pixels_hu < int(bone_stop_hu))

        if fill_mode == "sample" and unique_values_list:
            repl = np.random.choice(unique_values_list, size=int(mask.sum()))
            new_vol[mask] = repl.astype(new_vol.dtype)
        else:
            new_vol[mask] = -1000

        return new_vol

    def curves_callback(self, ds, elem):
        if elem.tag.group & 0xFF00 == 0x5000:
            del ds[elem.tag]

    def _remap_uid(self, uid_value, uid_dict, generate_uid_fn):
        s = str(uid_value).strip()
        if not s:
            return s
        if s not in uid_dict:
            uid_dict[s] = generate_uid_fn()
        return uid_dict[s]

    def _current_date_da(self):
        return datetime.now().strftime("%Y%m%d")

    def _replace_dt_date_preserve_time(self, original_value):
        """Return DICOM DT with today's date and the original time component.

        Examples:
          20130605142311.123 -> <today>142311.123
          20130605          -> <today>
        """
        today = self._current_date_da()
        try:
            s = str(original_value).strip()
        except Exception:
            s = ""
        if len(s) > 8:
            return today + s[8:]
        return today

    def _set_safe_value_by_vr(self, ds, tag, vr, generate_uid_fn, patient_id_value):
        if tag == (0x0010, 0x0020):
            ds[tag].value = patient_id_value
            return
        if tag == (0x0010, 0x0010):
            ds[tag].value = "Processed for anonymization"
            return
        if tag == (0x0008, 0x0050):
            ds[tag].value = patient_id_value
            return

        if vr == "DA":
            ds[tag].value = self._current_date_da()
            return
        if vr == "TM":
            return
        if vr == "DT":
            ds[tag].value = self._replace_dt_date_preserve_time(ds[tag].value)
            return

        if vr == "UI":
            if tag == (0x0020, 0x000D):
                ds[tag].value = self._remap_uid(ds[tag].value, self.study_uid_map, generate_uid_fn)
                return
            if tag == (0x0020, 0x000E):
                ds[tag].value = self._remap_uid(ds[tag].value, self.series_uid_map, generate_uid_fn)
                return
            if tag == (0x0008, 0x0018):
                ds[tag].value = self._remap_uid(ds[tag].value, self.sop_uid_map, generate_uid_fn)
                return
            if tag == (0x0020, 0x0052):
                ds[tag].value = self._remap_uid(ds[tag].value, self.uid_map_general, generate_uid_fn)
                return
            ds[tag].value = self._remap_uid(ds[tag].value, self.uid_map_general, generate_uid_fn)
            return

        if vr == "PN":
            ds[tag].value = "anonymous"
            return

        if vr in {"LO", "SH", "ST", "LT", "UT", "CS", "AE"}:
            ds[tag].value = "anonymous"
            return

        if vr in {"IS", "DS", "US", "UL", "SS", "SL", "FL", "FD"}:
            try:
                ds[tag].value = 0
            except Exception:
                ds[tag].value = "0"
            return

        if vr == "AS":
            ds[tag].value = "000Y"
            return

        try:
            ds[tag].value = "anonymous"
        except Exception:
            pass

    def _fix_invalid_uids(self, ds):
        """Replace syntactically invalid UIDs, including those in File Meta.

        The same map used for de-identification is reused, so a given original
        UID always maps to the same replacement and cross-references between
        instances survive. Standard concept UIDs (SOP Class, Transfer Syntax)
        are left alone.
        """
        from pydicom.uid import generate_uid

        fixed = [0]

        def fix_one(value):
            if _uid_is_valid(value):
                return value, False
            return self._remap_uid(value, self.uid_map_general, generate_uid), True

        def walk(dataset):
            for elem in list(dataset):
                try:
                    if elem.VR == "SQ":
                        for item in elem.value:
                            walk(item)
                        continue
                    if elem.VR != "UI" or elem.value is None:
                        continue
                    if (elem.tag.group, elem.tag.element) in UID_TAGS_NEVER_REMAPPED:
                        continue

                    val = elem.value
                    if isinstance(val, (list, tuple)) or (
                            hasattr(val, "__iter__") and not isinstance(val, (str, bytes))):
                        out = []
                        changed = False
                        for v in list(val):
                            nv, ch = fix_one(v)
                            out.append(nv)
                            changed = changed or ch
                            if ch:
                                fixed[0] += 1
                        if changed:
                            elem.value = out
                    else:
                        nv, ch = fix_one(val)
                        if ch:
                            elem.value = nv
                            fixed[0] += 1
                except Exception:
                    continue

        walk(ds)
        fm = getattr(ds, "file_meta", None)
        if fm is not None:
            walk(fm)

        return fixed[0]

    def _sync_file_meta(self, ds):
        """Bring group 0002 into line with the de-identified dataset.

        Dataset iteration does not include file_meta, so the anonymisation pass
        never sees it and the original instance UID would otherwise be written
        into the output file.
        """
        fm = getattr(ds, "file_meta", None)
        if fm is None:
            return

        try:
            sop_instance = getattr(ds, "SOPInstanceUID", None)
            if sop_instance:
                fm.MediaStorageSOPInstanceUID = sop_instance
        except Exception:
            pass

        try:
            sop_class = getattr(ds, "SOPClassUID", None)
            if sop_class:
                fm.MediaStorageSOPClassUID = sop_class
        except Exception:
            pass

        if DEID_CLEAR_SOURCE_AE_TITLE:
            try:
                tag = (0x0002, 0x0016)
                if tag in fm:
                    del fm[tag]
            except Exception:
                pass

    def _anonymize_dataset_recursive(self, ds, patient_id_value):
        from pydicom.uid import generate_uid

        def recurse(dataset):
            for elem in list(dataset):
                try:
                    if elem.VR == "SQ":
                        tag_sq = (elem.tag.group, elem.tag.element)
                        if tag_sq in PDF_TAGS_TO_DEID:
                            try:
                                del dataset[elem.tag]
                            except Exception:
                                pass
                            continue
                        for item in elem.value:
                            recurse(item)
                        continue

                    tag = (elem.tag.group, elem.tag.element)

                    if tag in PDF_TAGS_TO_DEID and tag in dataset:
                        self._set_safe_value_by_vr(dataset, tag, elem.VR, generate_uid, patient_id_value)

                except Exception:
                    continue

        recurse(ds)

    def _append_global_drop_rows(self, csv_path, rows):
        if not csv_path or not rows:
            return
        try:
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "timestamp",
                    "patient_old_id",
                    "patient_new_id",
                    "series_folder",
                    "source_dir",
                    "source_filename",
                    "instance_number",
                    "series_instance_uid",
                    "study_instance_uid",
                    "sop_instance_uid",
                    "burned_in_annotation",
                    "decision",
                    "reason",
                    "hit_text",
                    "hit_conf",
                    "hit_bbox",
                    "n_redacted_regions",
                    "boxless_hits",
                    "forced_keep",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    w.writeheader()
                for row in rows:
                    w.writerow(row)
        except Exception as e:
            try:
                self.logger.error(f"Failed writing global dropped rows: {e}")
            except Exception:
                pass

    def save_new_dicom_files(
        self,
        original_dir,
        out_dir,
        replacer="face",
        id="new_folder_name",
        patient_old_id="",
        patient_id="0",
        new_patient_id="Processed for anonymization",
        remove_CTA=False,
        global_drop_csv_path=None,
        global_detected_png_dir=None,
        global_no_text_png_dir=None,
        global_redacted_png_dir=None,
        global_verify_fail_png_dir=None,
        global_prescreen_png_dir=None,
        global_not_examined_png_dir=None,
        patient_input_root=None,
        original_face_render_dir=None,
        after_deidentification_render_dir=None,
    ):
        import pydicom

        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(original_dir) if self.is_dicom(os.path.join(original_dir, f), remove_CTA)]
        errors = []

        dropped_rows = []

        def _instnum(path):
            try:
                ds_ = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                return int(getattr(ds_, "InstanceNumber", 1))
            except Exception:
                return sys.maxsize

        files.sort(key=lambda fn: (_instnum(os.path.join(original_dir, fn)), fn))

        tmp_root = None
        prepared = []

        kept_count = 0
        drop_count = 0
        redact_count = 0
        unmasked_count = 0
        verify_fail_count = 0
        png_detected = 0
        png_no_text = 0
        png_prescreen = 0
        png_not_examined = 0
        png_none = 0
        err_count = 0
        uid_fix_count = 0

        progress_every = 50

        try:
            tmp_root = tempfile.mkdtemp(prefix="headctdeid_tmpdicom_")

            if files:
                _safe_show_status(f"[{id}] Series: {os.path.basename(original_dir)} | slices={len(files)}", 2500)

            for i, fname in enumerate(files, start=1):
                src_path = os.path.join(original_dir, fname)
                try:
                    ds = self.load_scan(src_path)
                    try:
                        ds.decompress()
                    except Exception:
                        pass

                    inst = None
                    try:
                        inst = int(getattr(ds, "InstanceNumber", 1))
                    except Exception:
                        inst = None

                    burned_flag = dicom_has_burned_in(ds)

                    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
                    study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
                    sop_uid = str(getattr(ds, "SOPInstanceUID", "") or "")

                    ds.remove_private_tags()
                    ds.walk(self.curves_callback)

                    if (0x0010, 0x0020) not in ds:
                        ds.add_new((0x0010, 0x0020), "LO", id)
                    else:
                        ds[(0x0010, 0x0020)].value = id

                    if (0x0010, 0x0010) not in ds:
                        ds.add_new((0x0010, 0x0010), "PN", "Processed for anonymization")
                    else:
                        ds[(0x0010, 0x0010)].value = "Processed for anonymization"

                    if (0x0008, 0x0050) not in ds:
                        ds.add_new((0x0008, 0x0050), "SH", id)
                    else:
                        ds[(0x0008, 0x0050)].value = id

                    self._anonymize_dataset_recursive(ds, patient_id_value=id)

                    if DEID_FIX_INVALID_UIDS:
                        n_fixed = self._fix_invalid_uids(ds)
                        if n_fixed:
                            uid_fix_count += n_fixed

                    if DEID_SYNC_FILE_META:
                        self._sync_file_meta(ds)

                    pixels_hu = self.get_pixels_hu(ds)

                    redact_rects = []
                    forced_keep = False
                    was_prescreen_skipped = False

                    want_detect = ENABLE_TEXT_DETECTION and (self._force_ocr_all or burned_flag)
                    if want_detect:
                        _skipped_before = self._prescreen_skipped
                        (has_text, hit_txt, hit_conf, hit_bbox, gray8,
                         detection_img, boxes, boxless) = self.detect_text_debug(
                            ds, burned_flag=burned_flag)
                        was_prescreen_skipped = (self._prescreen_skipped > _skipped_before)

                        if has_text:
                            png_detected += 1

                            action = str(TEXT_ACTION).lower()

                            if (action == "redact" and boxless and not boxes
                                    and str(REDACT_BOXLESS_STRATEGY).lower() == "drop"):
                                action = "drop"

                            if NEVER_DROP_SLICES and action == "drop":
                                action = "redact"
                                forced_keep = True

                            if action == "redact":
                                boxless_mode = str(REDACT_BOXLESS_STRATEGY).lower()
                                if NEVER_DROP_SLICES and boxless and not boxes \
                                        and boxless_mode != "border_band":
                                    boxless_mode = "border_band"

                                redact_rects = self._redaction_rects(
                                    boxes,
                                    boxless,
                                    pixels_hu.shape[:2],
                                    gray_shape_hw=(None if gray8 is None else gray8.shape[:2]),
                                    boxless_mode=boxless_mode,
                                    gray8=gray8,
                                )

                            if SAVE_DETECTED_DEBUG_PNG:
                                self._save_debug_png(
                                    global_detected_png_dir,
                                    id,
                                    os.path.basename(original_dir),
                                    fname,
                                    self._draw_mask_rects(
                                        detection_img, redact_rects,
                                        gray_shape_hw=(None if gray8 is None
                                                       else gray8.shape[:2]),
                                        hu_shape_hw=pixels_hu.shape[:2]),
                                )

                            if action == "redact" and redact_rects:
                                redact_count += 1

                                dropped_rows.append({
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "patient_old_id": patient_old_id,
                                    "patient_new_id": id,
                                    "series_folder": os.path.basename(original_dir),
                                    "source_dir": original_dir,
                                    "source_filename": fname,
                                    "instance_number": inst,
                                    "series_instance_uid": series_uid,
                                    "study_instance_uid": study_uid,
                                    "sop_instance_uid": sop_uid,
                                    "burned_in_annotation": bool(burned_flag),
                                    "decision": "REDACTED",
                                    "reason": "florence2_redacted_text",
                                    "hit_text": hit_txt,
                                    "hit_conf": hit_conf,
                                    "hit_bbox": hit_bbox,
                                    "n_redacted_regions": len(redact_rects),
                                    "boxless_hits": int(boxless),
                                    "forced_keep": bool(forced_keep),
                                })

                            elif action == "drop" and not NEVER_DROP_SLICES:
                                drop_count += 1
                                dropped_rows.append({
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "patient_old_id": patient_old_id,
                                    "patient_new_id": id,
                                    "series_folder": os.path.basename(original_dir),
                                    "source_dir": original_dir,
                                    "source_filename": fname,
                                    "instance_number": inst,
                                    "series_instance_uid": series_uid,
                                    "study_instance_uid": study_uid,
                                    "sop_instance_uid": sop_uid,
                                    "burned_in_annotation": bool(burned_flag),
                                    "decision": "DROPPED",
                                    "reason": "florence2_detected_text",
                                    "hit_text": hit_txt,
                                    "hit_conf": hit_conf,
                                    "hit_bbox": hit_bbox,
                                    "n_redacted_regions": 0,
                                    "boxless_hits": int(boxless),
                                    "forced_keep": False,
                                })
                                del ds, pixels_hu
                                continue

                            else:
                                unmasked_count += 1
                                self.logger.warning(
                                    "[%s] %s: text detected but NOT masked "
                                    "(boxless=%d, strategy=%s): %r"
                                    % (id, fname, boxless, REDACT_BOXLESS_STRATEGY, hit_txt))

                        elif detection_img is not None:
                            if was_prescreen_skipped:
                                png_prescreen += 1
                                if SAVE_PRESCREEN_SKIPPED_DEBUG_PNG:
                                    self._save_debug_png(
                                        global_prescreen_png_dir,
                                        id,
                                        os.path.basename(original_dir),
                                        fname,
                                        detection_img,
                                    )
                            else:
                                png_no_text += 1
                                if SAVE_NO_TEXT_DEBUG_PNG:
                                    self._save_debug_png(
                                        global_no_text_png_dir,
                                        id,
                                        os.path.basename(original_dir),
                                        fname,
                                        detection_img,
                                    )
                        else:
                            png_none += 1

                    if not want_detect:
                        png_not_examined += 1
                        if SAVE_NOT_EXAMINED_DEBUG_PNG:
                            try:
                                self._save_debug_png(
                                    global_not_examined_png_dir,
                                    id,
                                    os.path.basename(original_dir),
                                    fname,
                                    self._gray_to_bgr(
                                        self._dicom_pixels_to_gray8_for_ocr(ds)),
                                )
                            except Exception:
                                png_none += 1

                    bin_mask = self.binarize_volume(pixels_hu)
                    lcc = self.largest_connected_component(bin_mask)

                    k_max = int(self._kernel_from_pixel_spacing(ds))
                    dilated = self.bounded_dilate_with_front_boost(
                        lcc_air_seed=lcc,
                        pixels_hu=pixels_hu,
                        ds=ds,
                        k_max=k_max,
                        bone_stop_hu=BONE_STOP_HU,
                        front_fraction=0.55,
                    )

                    ring = ((dilated > 0) & (lcc == 0)).astype(np.uint8)

                    if replacer == "face":
                        vals = self.apply_mask_and_get_values(pixels_hu, ring)
                    elif replacer == "air":
                        vals = [0]
                    else:
                        try:
                            vals = [int(replacer)]
                        except Exception:
                            vals = self.apply_mask_and_get_values(pixels_hu, ring)

                    new_volume = self.apply_random_values_optimized(
                        pixels_hu,
                        dilated,
                        vals,
                        bone_stop_hu=BONE_STOP_HU,
                        fill_mode="air",
                    )

                    if redact_rects:
                        self._apply_redaction(new_volume, redact_rects)

                        if SAVE_REDACTED_DEBUG_PNG:
                            self._save_debug_png(
                                global_redacted_png_dir,
                                id,
                                os.path.basename(original_dir),
                                fname,
                                self._render_redacted_png(
                                    self._hu_to_gray8(new_volume, ds), []),
                            )

                        if REDACT_VERIFY_WITH_SECOND_PASS:
                            still_text, still_txt = self._verify_redaction(new_volume, ds)
                            if still_text:
                                verify_fail_count += 1
                                self.logger.warning(
                                    "[%s] %s: text STILL detected after masking: %r"
                                    % (id, fname, still_txt))
                                self._save_debug_png(
                                    global_verify_fail_png_dir,
                                    id,
                                    os.path.basename(original_dir),
                                    fname,
                                    self._render_redacted_png(
                                        self._hu_to_gray8(new_volume, ds), []),
                                )
                                if (str(REDACT_VERIFY_ON_FAILURE).lower() == "drop"
                                        and not NEVER_DROP_SLICES):
                                    drop_count += 1
                                    del ds, pixels_hu, new_volume
                                    continue

                    slope = float(getattr(ds, "RescaleSlope", 1)) or 1.0
                    intercept = float(getattr(ds, "RescaleIntercept", 0))
                    new_slice = (new_volume - intercept) / slope

                    ds.PixelData = new_slice.astype(np.int16).tobytes()
                    ds.BitsAllocated = 16
                    ds.BitsStored = 16
                    ds.HighBit = 15
                    ds.PixelRepresentation = 1

                    out_name = f"{id}_{i:05d}.dcm"
                    tmp_path = os.path.join(tmp_root, out_name)
                    final_path = os.path.join(out_dir, out_name)

                    _dcm_save_as(ds, tmp_path, enforce_file_format=True)
                    prepared.append((tmp_path, final_path))
                    kept_count += 1

                    del ds, pixels_hu, new_volume

                except Exception as e:
                    err_count += 1
                    errors.append((fname, str(e)))

                if (i % progress_every == 0) or (i == len(files)):
                    _safe_show_status(
                        f"[{id}] slices {i}/{len(files)} | kept={kept_count} "
                        f"redacted={redact_count} dropped={drop_count} errors={err_count}",
                        1500,
                    )

            n_in = len(files)
            n_out = len(prepared)
            if n_in:
                if n_out == n_in:
                    self.logger.info(
                        "[%s] %s: %d/%d slices written (none removed); %d had text "
                        "blacked out." % (id, os.path.basename(original_dir),
                                          n_out, n_in, redact_count))
                else:
                    self.logger.warning(
                        "[%s] %s: %d/%d slices written - %d MISSING "
                        "(dropped=%d, errors=%d). With NEVER_DROP_SLICES=%s, any "
                        "shortfall is a read/processing error, not text removal."
                        % (id, os.path.basename(original_dir), n_out, n_in,
                           n_in - n_out, drop_count, err_count, NEVER_DROP_SLICES))

            if n_in:
                accounted = (png_detected + png_no_text + png_prescreen
                             + png_not_examined + png_none)
                self.logger.info(
                    "[%s] %s: debug PNG accounting over %d slice(s): "
                    "text_found=%d, examined_clean=%d, prescreen_skipped=%d, "
                    "not_examined=%d, no_image=%d (total %d)"
                    % (id, os.path.basename(original_dir), n_in,
                       png_detected, png_no_text, png_prescreen,
                       png_not_examined, png_none, accounted))
                if png_detected != redact_count + drop_count + unmasked_count:
                    self.logger.warning(
                        "[%s] PNG/action mismatch: %d slice(s) with text but "
                        "%d redacted + %d dropped + %d unmasked."
                        % (id, png_detected, redact_count, drop_count, unmasked_count))
                if png_not_examined:
                    self.logger.warning(
                        "[%s] %d slice(s) were NOT examined for burned-in text "
                        "(detection runs only when the de-identify option is on or "
                        "BurnedInAnnotation=YES). See only_for_debug/%s."
                        % (id, png_not_examined, OCR_DEBUG_NOT_EXAMINED_DIRNAME))
                if png_prescreen:
                    self.logger.info(
                        "[%s] %d slice(s) were skipped by the OpenCV pre-screen and "
                        "never reached Florence-2. Set PRESCREEN_MODE = \"off\" to "
                        "examine every slice. See only_for_debug/%s."
                        % (id, png_prescreen, OCR_DEBUG_PRESCREEN_DIRNAME))
                if png_none:
                    self.logger.warning(
                        "[%s] %d slice(s) produced NO debug image (pixel conversion or "
                        "model init failed). These were not checked for text."
                        % (id, png_none))

            if redact_count or unmasked_count:
                self.logger.info(
                    "[%s] burned-in text: %d slice(s) redacted and KEPT, %d dropped, "
                    "%d detected but NOT masked (action=%s, boxless strategy=%s, "
                    "never_drop=%s)."
                    % (id, redact_count, drop_count, unmasked_count,
                       TEXT_ACTION, REDACT_BOXLESS_STRATEGY, NEVER_DROP_SLICES))
                if verify_fail_count:
                    self.logger.warning(
                        "[%s] %d slice(s) still showed text on the SECOND pass after "
                        "masking. Review only_for_debug/%s and consider increasing "
                        "REDACT_PAD_FRAC." % (id, verify_fail_count,
                                              OCR_DEBUG_VERIFY_FAIL_DIRNAME))
                if unmasked_count:
                    self.logger.warning(
                        "[%s] %d slice(s) still contain detected text. Review "
                        "only_for_debug/%s before release."
                        % (id, unmasked_count, OCR_DEBUG_DETECTED_DIRNAME))

            if uid_fix_count:
                self.logger.info(
                    "[%s] repaired %d non-conformant UID value(s) (PS3.5 section 9.1); "
                    "replacements are consistent across the output set." % (id, uid_fix_count))

            if self._conf_samples or self._unknown_conf_count:
                try:
                    arr = np.asarray(self._conf_samples, dtype=float)
                    if arr.size:
                        self.logger.info(
                            "[%s] detection confidence over %d detected slices: "
                            "min=%.3f p10=%.3f median=%.3f max=%.3f | unknown=%d "
                            "(threshold=%.2f)"
                            % (id, arr.size, float(arr.min()),
                               float(np.percentile(arr, 10)), float(np.median(arr)),
                               float(arr.max()), self._unknown_conf_count,
                               float(FLORENCE_MIN_CONFIDENCE)))
                    else:
                        self.logger.info(
                            "[%s] no usable confidence scores (%d hits with unknown "
                            "confidence); FLORENCE_MIN_CONFIDENCE is not filtering."
                            % (id, self._unknown_conf_count))
                except Exception:
                    pass

            if files and not prepared:
                reason = ("all_slices_failed_to_process" if NEVER_DROP_SLICES
                          else "all_slices_removed_due_to_detected_text")
                errors.append((os.path.basename(original_dir), reason))

            for tmp_path, final_path in prepared:
                try:
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    shutil.copy2(tmp_path, final_path)
                except Exception as e:
                    errors.append((os.path.basename(tmp_path), f"finalize_copy_failed: {e}"))

            if prepared:
                render_label = _safe_filename(f"{id}_{os.path.basename(original_dir)}")

                if original_face_render_dir:
                    try:
                        original_png = os.path.join(original_face_render_dir, f"{render_label}_original.png")
                        self._render_one_anterior_vtk_folder_subprocess(
                            dicom_dir=original_dir,
                            out_png=original_png,
                            image_size=1024,
                            zoom_out=4.0,
                            rotate_180=True,
                            view_angle_deg=12.0,
                            min_slices=8,
                            timeout_sec=90,
                        )
                    except Exception as e:
                        try:
                            with open(os.path.join(original_face_render_dir, "render_log.txt"), "a") as f:
                                f.write(f"[{datetime.now()}] Original render failed for {original_dir}: {e}\n")
                        except Exception:
                            pass

                if after_deidentification_render_dir:
                    try:
                        after_png = os.path.join(after_deidentification_render_dir, f"{render_label}_after.png")
                        self._render_one_anterior_vtk_folder_subprocess(
                            dicom_dir=out_dir,
                            out_png=after_png,
                            image_size=1024,
                            zoom_out=4.0,
                            rotate_180=True,
                            view_angle_deg=12.0,
                            min_slices=8,
                            timeout_sec=90,
                        )
                    except Exception as e:
                        try:
                            with open(os.path.join(after_deidentification_render_dir, "render_log.txt"), "a") as f:
                                f.write(f"[{datetime.now()}] After-deidentification render failed for {out_dir}: {e}\n")
                        except Exception:
                            pass

        finally:
            if errors:
                try:
                    with open(os.path.join(out_dir, "log.txt"), "a") as error_file:
                        for dicom_file, err in errors:
                            error_file.write(f"File: {dicom_file}, Error: {err}\n")
                except Exception:
                    pass

            self._append_global_drop_rows(global_drop_csv_path, dropped_rows)

            if tmp_root and os.path.isdir(tmp_root):
                try:
                    shutil.rmtree(tmp_root)
                except Exception:
                    pass

        return errors

    def _render_fallback_middle_slice(self, dicom_dir: str, out_png: str):
        import pydicom
        import cv2

        paths = []
        for fn in os.listdir(dicom_dir):
            fp = os.path.join(dicom_dir, fn)
            if os.path.isfile(fp):
                try:
                    _ = pydicom.dcmread(fp, force=True, stop_before_pixels=True)
                    paths.append(fp)
                except Exception:
                    pass
        if not paths:
            raise RuntimeError("Fallback render: no dicoms found")

        def instnum(p):
            try:
                ds = pydicom.dcmread(p, force=True, stop_before_pixels=True)
                return int(getattr(ds, "InstanceNumber", 1))
            except Exception:
                return sys.maxsize

        paths.sort(key=instnum)
        mid = paths[len(paths) // 2]
        ds = pydicom.dcmread(mid, force=True)
        try:
            ds.decompress()
        except Exception:
            pass
        img = ds.pixel_array.astype(np.float32)

        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slope = float(getattr(ds, "RescaleSlope", 1) or 1.0)
        hu = img * slope + intercept

        w_center = -100.0
        w_width = 350.0
        lo = w_center - (w_width / 2.0)
        hi = w_center + (w_width / 2.0)
        hu = np.clip(hu, lo, hi)
        out8 = ((hu - lo) / max(1e-6, (hi - lo)) * 255.0).astype(np.uint8)

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        cv2.imwrite(out_png, out8)
        return out_png

    def _render_one_anterior_vtk_folder_subprocess(
        self,
        dicom_dir: str,
        out_png: str,
        image_size: int = 1024,
        zoom_out: float = 4.0,
        rotate_180: bool = True,
        view_angle_deg: float = 12.0,
        min_slices: int = 16,
        timeout_sec: int = 60,
    ):
        script = f"""
import os
import vtk

dicom_dir = r\"\"\"{dicom_dir}\"\"\"
out_png  = r\"\"\"{out_png}\"\"\"

reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(dicom_dir)
reader.Update()

img = reader.GetOutput()
if img is None:
    raise RuntimeError("No image output from vtkDICOMImageReader")
dims = img.GetDimensions()
if (not dims) or (dims[0] <= 1) or (dims[1] <= 1) or (dims[2] < int({min_slices})):
    raise RuntimeError(f"Bad/too-thin volume dims: {{dims}} (min_slices={min_slices})")
# Smooth CT volume first
smooth = vtk.vtkImageGaussianSmooth()
smooth.SetInputConnection(reader.GetOutputPort())
smooth.SetStandardDeviations(1.2, 1.2, 1.0)
smooth.SetRadiusFactors(2.0, 2.0, 1.5)
smooth.Update()

# Extract outer skin surface: air/skin boundary
skin_value = -250.0   # try -450 to -250 if needed

try:
    contour = vtk.vtkFlyingEdges3D()
except Exception:
    contour = vtk.vtkMarchingCubes()

contour.SetInputConnection(smooth.GetOutputPort())
contour.SetValue(0, skin_value)
contour.Update()

# Keep only largest connected surface = head/face
connect = vtk.vtkPolyDataConnectivityFilter()
connect.SetInputConnection(contour.GetOutputPort())
connect.SetExtractionModeToLargestRegion()
connect.Update()

# Smooth surface to remove stair-step CT rings
surf_smooth = vtk.vtkSmoothPolyDataFilter()
surf_smooth.SetInputConnection(connect.GetOutputPort())
surf_smooth.SetNumberOfIterations(35)
surf_smooth.SetRelaxationFactor(0.12)
surf_smooth.FeatureEdgeSmoothingOff()
surf_smooth.BoundarySmoothingOn()
surf_smooth.Update()

normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(surf_smooth.GetOutputPort())
normals.SetFeatureAngle(60)
normals.ConsistencyOn()
normals.AutoOrientNormalsOn()
normals.SplittingOff()
normals.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(normals.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.95, 0.62, 0.42)
actor.GetProperty().SetAmbient(0.25)
actor.GetProperty().SetDiffuse(0.75)
actor.GetProperty().SetSpecular(0.18)
actor.GetProperty().SetSpecularPower(18)

ren = vtk.vtkRenderer()
ren.SetBackground(0.62, 0.65, 0.90)
ren.AddActor(actor)
ren.ResetCamera()

renwin = vtk.vtkRenderWindow()
renwin.SetOffScreenRendering(1)
renwin.AddRenderer(ren)
renwin.SetSize(int({image_size}), int({image_size}))
renwin.SetMultiSamples(0)

bounds = actor.GetBounds()
cx = 0.5 * (bounds[0] + bounds[1])
cy = 0.5 * (bounds[2] + bounds[3])
cz = 0.5 * (bounds[4] + bounds[5])

dx = bounds[1] - bounds[0]
dy = bounds[3] - bounds[2]
dz = bounds[5] - bounds[4]
diag = max(1e-6, (dx*dx + dy*dy + dz*dz) ** 0.5)
dist = diag * float({zoom_out})

cam = ren.GetActiveCamera()
cam.SetFocalPoint(cx, cy, cz)
cam.SetViewUp(0, 0, 1)
cam.SetPosition(cx, cy + dist, cz)

try:
    cam.SetViewAngle(float({view_angle_deg}))
except Exception:
    pass

if {str(bool(rotate_180))}:
    try:
        cam.Roll(180)
    except Exception:
        cam.Azimuth(180)

ren.ResetCameraClippingRange()
renwin.Render()

w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(renwin)
w2i.SetReadFrontBuffer(False)
w2i.SetInputBufferTypeToRGB()
w2i.Update()

os.makedirs(os.path.dirname(out_png), exist_ok=True)
writer = vtk.vtkPNGWriter()
writer.SetFileName(out_png)
writer.SetInputConnection(w2i.GetOutputPort())
writer.Write()

ren.RemoveAllViewProps()
renwin.Finalize()

print(out_png)
"""
        with tempfile.NamedTemporaryFile("w", suffix="_vtk_render.py", delete=False) as tf:
            tf.write(script)
            script_path = tf.name

        try:
            rc, stdout, stderr = self._popen_and_wait([sys.executable, script_path], timeout_sec=timeout_sec)
            if rc != 0:
                raise RuntimeError(f"VTK render subprocess failed: {stderr or stdout}")
            if not os.path.exists(out_png):
                raise RuntimeError("VTK render subprocess did not produce output PNG")
            return out_png
        finally:
            try:
                os.remove(script_path)
            except Exception:
                pass

    def _render_one_dicom_folder(self, dicomDir, out_prefix="view"):
        if not os.path.isdir(dicomDir):
            raise RuntimeError(f"Not a folder: {dicomDir}")

        out_path = os.path.join(dicomDir, f"{out_prefix}_anterior.png")
        try:
            self._render_one_anterior_vtk_folder_subprocess(
                dicom_dir=dicomDir,
                out_png=out_path,
                image_size=1024,
                zoom_out=4.0,
                rotate_180=True,
                view_angle_deg=12.0,
                min_slices=16,
                timeout_sec=60,
            )
            return [out_path]
        except Exception as e:
            try:
                with open(os.path.join(dicomDir, "render_log.txt"), "a") as f:
                    f.write(f"[{datetime.now()}] VTK render failed; fallback to middle-slice. Reason: {e}\n")
            except Exception:
                pass
            self._render_fallback_middle_slice(dicomDir, out_path)
            return [out_path]

    def _find_all_dicom_dirs(self, rootFolder):
        import pydicom

        def _has_any_dicom(d):
            try:
                for fn in os.listdir(d):
                    fp = os.path.join(d, fn)
                    if not os.path.isfile(fp):
                        continue
                    try:
                        _ = pydicom.dcmread(fp, force=True, stop_before_pixels=True)
                        return True
                    except Exception:
                        continue
            except Exception:
                return False
            return False

        dicom_dirs = []
        for curr, subdirs, files in os.walk(rootFolder):
            if _has_any_dicom(curr):
                dicom_dirs.append(curr)

        return sorted(set(dicom_dirs))

    def _create_and_save_multi_view_snapshots(self, patientFolder, out_prefix="view"):
        dicom_dirs = self._find_all_dicom_dirs(patientFolder)
        if not dicom_dirs:
            raise RuntimeError("No snapshots produced (no DICOM-containing subfolders found).")

        all_outputs = []
        for d in dicom_dirs:
            try:
                outs = self._render_one_dicom_folder(d, out_prefix=out_prefix)
                all_outputs.extend(outs)
            except Exception as e:
                try:
                    with open(os.path.join(d, "render_log.txt"), "a") as f:
                        f.write(f"[{datetime.now()}] Render failed: {e}\n")
                except Exception:
                    pass
                all_outputs.append(f"[FAILED] {d} :: {e}")

        rendered = [p for p in all_outputs if isinstance(p, str) and p.endswith(".png") and os.path.exists(p)]
        if not rendered:
            raise RuntimeError("No snapshots produced (all DICOM folders failed to render).")

        return all_outputs

    def drown_volume(
        self,
        in_path,
        out_path,
        replacer="face",
        id="new_folder_name",
        patient_old_id="",
        patient_id="0",
        name="",
        remove_CTA=False,
        global_drop_csv_path=None,
        global_detected_png_dir=None,
        global_no_text_png_dir=None,
        global_redacted_png_dir=None,
        global_verify_fail_png_dir=None,
        global_prescreen_png_dir=None,
        global_not_examined_png_dir=None,
        patient_input_root=None,
        original_face_render_dir=None,
        after_deidentification_render_dir=None,
    ):
        try:
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
                        patient_old_id=patient_old_id,
                        patient_id=patient_id,
                        new_patient_id="Processed for anonymization",
                        remove_CTA=remove_CTA,
                        global_drop_csv_path=global_drop_csv_path,
                        global_detected_png_dir=global_detected_png_dir,
                        global_redacted_png_dir=global_redacted_png_dir,
                        global_verify_fail_png_dir=global_verify_fail_png_dir,
                        global_prescreen_png_dir=global_prescreen_png_dir,
                        global_not_examined_png_dir=global_not_examined_png_dir,
                        global_no_text_png_dir=global_no_text_png_dir,
                        patient_input_root=patient_input_root or in_path,
                        original_face_render_dir=original_face_render_dir,
                        after_deidentification_render_dir=after_deidentification_render_dir,
                    )

            for curr, subdirs, files in os.walk(out_path, topdown=True):
                if not subdirs:
                    continue

                subdirs_sorted = sorted(subdirs)
                tmp_map = []
                for i, d in enumerate(subdirs_sorted, start=1):
                    src = os.path.join(curr, d)
                    tmp = os.path.join(curr, f"__TMP__RENAME__{i:04d}__")
                    if os.path.exists(src):
                        os.rename(src, tmp)
                        tmp_map.append(tmp)

                new_names = []
                for i, tmp in enumerate(tmp_map, start=1):
                    dst_name = f"{id}_{i}"
                    dst = os.path.join(curr, dst_name)
                    os.rename(tmp, dst)
                    new_names.append(dst_name)

                subdirs[:] = new_names

            try:
                self._create_and_save_multi_view_snapshots(out_path, out_prefix="view")
            except Exception as e:
                try:
                    with open(os.path.join(out_path, "render_summary.txt"), "a") as f:
                        f.write(f"[{datetime.now()}] Snapshot phase failed: {e}\n")
                except Exception:
                    pass

            self.wait_for_all_subprocesses(timeout_total_sec=7200)

        except Exception as e:
            try:
                os.makedirs(out_path, exist_ok=True)
                with open(os.path.join(out_path, "log.txt"), "a") as f:
                    f.write(f"Error: {e}\n")
            except Exception:
                pass
            return 0

        return 1


class HeadCTDeidTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_HeadCTDeid1()

    def test_HeadCTDeid1(self):
        self.assertTrue(True)
