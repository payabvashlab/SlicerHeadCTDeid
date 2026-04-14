# -*- coding: utf-8 -*-
"""
HeadCTDeid (3D Slicer scripted module)
"""

import csv
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
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

# =============================================================================
# EasyOCR (Burned-in text detection)
# =============================================================================

EASYOCR_LANG = "en"

# Recognition filters (match standalone)
EASYOCR_MIN_ALNUM = 2
EASYOCR_CONF_THRESH = 0.50

# Geometry filters (not used; kept only for compatibility/readability)
EASYOCR_MIN_BOX_AREA = 0
EASYOCR_MAX_BOX_AREA_FRAC = 1.0
EASYOCR_MAX_BOX_H_FRAC = 1.0
EASYOCR_MAX_BOX_W_FRAC = 1.0
EASYOCR_MIN_ASPECT = 0.0

# Preprocess: standalone behavior
EASYOCR_USE_CLAHE = False
EASYOCR_USE_INVERTED_PASS = False
EASYOCR_USE_RAW_PASS = True
EASYOCR_USE_ROTATED_180_PASS = False

# Optional border restriction: disabled to match standalone
EASYOCR_RESTRICT_TO_BORDER_BAND = False
EASYOCR_BORDER_FRAC = 0.30

# Brightness heuristic: disabled to match standalone
EASYOCR_BRIGHT_PIXEL_THR = 0
EASYOCR_BRIGHT_FRAC_THRESH = 0.0

# CT windowing for OCR: disabled to match standalone
EASYOCR_USE_CT_WINDOWING = False
EASYOCR_DEFAULT_WC = 40.0
EASYOCR_DEFAULT_WW = 80.0

# EasyOCR inference knobs — exact standalone call settings
EASYOCR_DECODER = "greedy"
EASYOCR_MIN_SIZE = 15
EASYOCR_TEXT_THRESHOLD = 0.85
EASYOCR_LOW_TEXT = 0.50
EASYOCR_LINK_THRESHOLD = 0.60
EASYOCR_MAG_RATIO = 1.0
EASYOCR_CONTRAST_THS = 0.10
EASYOCR_ADJUST_CONTRAST = 0.5
EASYOCR_ADD_MARGIN = 0.0
EASYOCR_WIDTH_THS = 0.5
EASYOCR_SLOPE_THS = 0.5
EASYOCR_HEIGHT_THS = 0.5
EASYOCR_YCENTER_THS = 0.5

EASYOCR_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:-_/(). "

# Plausibility
EASYOCR_ACCEPT_DIGITS_ONLY = True
EASYOCR_ACCEPT_SINGLE_TOKEN = True

# Debug PNG drawing
OCR_DEBUG_DRAW_BOXES = True
OCR_DEBUG_DRAW_LABELS = True
OCR_DEBUG_BOX_THICKNESS = 2
OCR_DEBUG_FONT_SCALE = 0.5
OCR_DEBUG_FONT_THICKNESS = 1

# Global PNG debug folders
OCR_DEBUG_ROOT_DIRNAME = "only_for_debug"
OCR_DEBUG_DETECTED_DIRNAME = "detected_text"
OCR_DEBUG_NO_TEXT_DIRNAME = "no_text_detected"

# =============================================================================
# Face/air drowning parameters (unchanged)
# =============================================================================
FACE_MAX_VALUE = 50
FACE_MIN_VALUE = -125
AIR_THRESHOLD = -800
BONE_STOP_HU = 250
FRONT_BOOST_KERNEL = (3, 3)

# ---------------------------------------------------------------------------
# DICOM tags to de-id (unchanged)
# ---------------------------------------------------------------------------
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

GLOBAL_DROPPED_CSV_NAME = "global_removed_slices_detected_text.csv"

# =============================================================================
# Small helpers
# =============================================================================
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


def _safe_filename(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.replace("\\", "__").replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "unknown"


# =============================================================================
# Module
# =============================================================================
class HeadCTDeid(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Head CT De-identification"
        self.parent.categories = ["Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = ["Anh Tuan Tran, Sam Payabvash"]
        self.parent.helpText = "This module de-identifies DICOM files by removing patient information based on a given mapping table."
        self.parent.acknowledgementText = "This file was developed by Anh Tuan Tran, Sam Payabvash (Columbia University)."


# =============================================================================
# Widget
# =============================================================================
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
            slicer.util.pip_install("opencv-python")
            slicer.util.pip_install("opencv-python-headless")
            slicer.util.pip_install("easyocr")
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


# =============================================================================
# Logic
# =============================================================================
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

        global_drop_csv_path = os.path.join(out_path, GLOBAL_DROPPED_CSV_NAME)
        self._init_global_drop_csv(global_drop_csv_path)

        ocr_debug_root = os.path.join(out_path, OCR_DEBUG_ROOT_DIRNAME)
        ocr_detected_dir = os.path.join(ocr_debug_root, OCR_DEBUG_DETECTED_DIRNAME)
        ocr_no_text_dir = os.path.join(ocr_debug_root, OCR_DEBUG_NO_TEXT_DIRNAME)
        os.makedirs(ocr_detected_dir, exist_ok=True)
        os.makedirs(ocr_no_text_dir, exist_ok=True)

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
                    patient_input_root=src_folder,
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

        if progressBar:
            progressBar.setValue(100)

        _safe_show_status("All processing finished.", 5000)
        self.logger.info("All processing finished.")


# =============================================================================
# DICOM Processor
# =============================================================================
class DicomProcessor:
    """
    Pipeline: de-identification + face/air replacement + EasyOCR detect->drop.

    Detection run decision per slice:
      if force_ocr_all == True:
          run OCR detect on every slice
      else:
          run OCR detect only when BurnedInAnnotation==YES
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
        self._ocr_langs = [EASYOCR_LANG] if EASYOCR_LANG else ["en"]

    # -------------------------------------------------------------------------
    # Subprocess utilities (render only)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # EasyOCR detection
    # -------------------------------------------------------------------------
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
        if self._ocr is not None:
            return True

        try:
            import easyocr

            use_gpu, device_name = self._detect_gpu_available()

            try:
                self._ocr = easyocr.Reader(self._ocr_langs, gpu=use_gpu)
            except TypeError:
                self._ocr = easyocr.Reader(language_list=self._ocr_langs, gpu=use_gpu)

            if use_gpu:
                msg = f"EasyOCR initialized (GPU: {device_name})"
            else:
                msg = "EasyOCR initialized (CPU)"

            try:
                self.logger.info(msg)
            except Exception:
                pass
            _safe_show_status(msg, 2500)
            return True

        except Exception as e:
            try:
                self.logger.error(f"Failed to init EasyOCR in-process: {e}")
            except Exception:
                pass
            _safe_show_status(f"EasyOCR init failed; OCR will be skipped. ({e})", 5000)
            self._ocr = None
            return False

    def _alnum_count(self, s: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]", str(s) if s is not None else ""))

    def _text_plausible(self, txt: str) -> bool:
        s = str(txt).strip() if txt is not None else ""
        if not s:
            return False
        return self._alnum_count(s) >= int(EASYOCR_MIN_ALNUM)

    def _parse_easyocr_output(self, res):
        out = []
        if res is None:
            return out
        for it in res:
            try:
                bbox = it[0]
                txt = str(it[1]).strip()
                sc = None
                try:
                    sc = float(it[2])
                except Exception:
                    sc = None
                quad = np.asarray(bbox, np.float32).reshape(4, 2)
                out.append((quad, txt, sc))
            except Exception:
                continue
        return out

    def _dicom_pixels_to_gray8_for_ocr(self, ds):
        """
        Match standalone dicom_to_gray8(ds):
        - pixels = ds.pixel_array
        - if pixels.ndim == 3: pixels = pixels[0]
        - apply slope/intercept
        - min-max normalize to [0,255] uint8
        """
        pixels = ds.pixel_array

        if pixels.ndim == 3:
            pixels = pixels[0]

        pixels = pixels.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixels_hu = pixels * slope + intercept

        mn = float(np.min(pixels_hu))
        mx = float(np.max(pixels_hu))
        if mx <= mn:
            mx = mn + 1.0

        gray8 = ((pixels_hu - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
        return gray8

    def _run_easyocr_variant(self, bgr):
        try:
            return self._ocr.readtext(
                bgr,
                text_threshold=EASYOCR_TEXT_THRESHOLD,
                low_text=EASYOCR_LOW_TEXT,
                link_threshold=EASYOCR_LINK_THRESHOLD,
                min_size=EASYOCR_MIN_SIZE,
                allowlist=EASYOCR_ALLOWLIST,
            )
        except TypeError:
            try:
                return self._ocr.readtext(bgr)
            except Exception:
                return None
        except Exception:
            return None

    def _draw_ocr_results(self, gray8, items):
        import cv2

        det_img = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

        for quad, txt, sc in items:
            try:
                pts = np.asarray(quad, dtype=np.int32).reshape(4, 2)

                if OCR_DEBUG_DRAW_BOXES:
                    cv2.polylines(
                        det_img,
                        [pts],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=OCR_DEBUG_BOX_THICKNESS,
                    )

                if OCR_DEBUG_DRAW_LABELS:
                    x = int(np.min(pts[:, 0]))
                    y = int(np.min(pts[:, 1])) - 5
                    if y < 10:
                        y = int(np.max(pts[:, 1])) + 15

                    label = f"{txt} ({float(sc):.2f})" if sc is not None else str(txt)
                    cv2.putText(
                        det_img,
                        label,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        OCR_DEBUG_FONT_SCALE,
                        (0, 255, 0),
                        OCR_DEBUG_FONT_THICKNESS,
                        cv2.LINE_AA,
                    )
            except Exception:
                continue

        return det_img

    def detect_text_debug(self, ds):
        """
        Returns:
          has_text: bool
          hit_text: str
          hit_conf: float|None
          hit_bbox: list|None
          gray8: uint8 image
          detection_img: image with boxes
        """
        import cv2

        if not self._ensure_ocr():
            return False, "", None, None, None, None

        try:
            gray8 = self._dicom_pixels_to_gray8_for_ocr(ds)
        except Exception:
            return False, "", None, None, None, None

        try:
            bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        except Exception:
            return False, "", None, None, gray8, None

        res = self._run_easyocr_variant(bgr)
        items = self._parse_easyocr_output(res)
        if not items:
            return False, "", None, None, gray8, cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

        kept = []
        for quad, txt, sc in items:
            if sc is None:
                continue
            if float(sc) < float(EASYOCR_CONF_THRESH):
                continue
            if not self._text_plausible(txt):
                continue
            kept.append((quad, txt, sc))

        detection_img = self._draw_ocr_results(gray8, kept if kept else [])

        if kept:
            quad, txt, sc = kept[0]
            bbox_list = np.asarray(quad, np.float32).reshape(4, 2).tolist()
            return True, txt, float(sc), bbox_list, gray8, detection_img

        return False, "", None, None, gray8, detection_img

    # -------------------------------------------------------------------------
    # CT helpers
    # -------------------------------------------------------------------------
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

            lo = int(ceil(10.0 / pixel))
            hi = int(ceil(15.0 / pixel))
            if hi < lo:
                hi = lo

            lo = max(1, min(lo, 999))
            hi = max(1, min(hi, 999))
            return random.randint(lo, hi)
        except Exception:
            return random.randint(30, 40)

    def apply_mask_and_get_values(self, image_volume, mask_volume):
        masked = image_volume * mask_volume
        vals = np.unique(masked)
        vals = vals[(vals > FACE_MIN_VALUE) & (vals < FACE_MAX_VALUE)]
        return vals.tolist()

    # -------------------------------------------------------------------------
    # CT meta filter
    # -------------------------------------------------------------------------
    def is_substring_in_list(self, substring, string_list):
        return any(substring in str(s) for s in string_list)

    def checkCTmeta(self, ds, remove_CTA=False):
        """
        Accept only CT head (original/primary/axial). By default, exclude CTA/perfusion.
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

            return int(status1 and status2 and status3 and status4)
        except Exception as e:
            self.error = str(e)
            return 0

    # -------------------------------------------------------------------------
    # Dilation with barriers
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # DICOM anonymization helpers
    # -------------------------------------------------------------------------
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

    def _current_da_tm_dt(self):
        now = datetime.now()
        da = now.strftime("%Y%m%d")
        tm = now.strftime("%H%M%S")
        dt = now.strftime("%Y%m%d%H%M%S") + "." + f"{now.microsecond:06d}"
        return da, tm, dt

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

        da_now, tm_now, dt_now = self._current_da_tm_dt()
        if vr == "DA":
            ds[tag].value = da_now
            return
        if vr == "TM":
            ds[tag].value = tm_now
            return
        if vr == "DT":
            ds[tag].value = dt_now
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

    # -------------------------------------------------------------------------
    # Global CSV helpers
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Save anonymized dicoms + OCR detect -> DROP slice
    # Enforces: OCR-all-first, save-after via temp staging
    # Writes only global CSV, and also global debug PNG folders
    # -------------------------------------------------------------------------
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
        patient_input_root=None,
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
        err_count = 0

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

                    pixels_hu = self.get_pixels_hu(ds)

                    want_detect = self._force_ocr_all or burned_flag
                    if want_detect:
                        has_text, hit_txt, hit_conf, hit_bbox, gray8, detection_img = self.detect_text_debug(ds)

                        if has_text:
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
                                "reason": "easyocr_detected_text",
                                "hit_text": hit_txt,
                                "hit_conf": hit_conf,
                                "hit_bbox": hit_bbox,
                            })
                            del ds, pixels_hu
                            continue

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

                    ds.save_as(tmp_path, write_like_original=False)
                    prepared.append((tmp_path, final_path))
                    kept_count += 1

                    del ds, pixels_hu, new_volume

                except Exception as e:
                    err_count += 1
                    errors.append((fname, str(e)))

                if (i % progress_every == 0) or (i == len(files)):
                    _safe_show_status(
                        f"[{id}] slices {i}/{len(files)} | kept={kept_count} dropped={drop_count} errors={err_count}",
                        1500,
                    )

            if files and not prepared:
                errors.append((os.path.basename(original_dir), "all_slices_removed_due_to_detected_text"))

            for tmp_path, final_path in prepared:
                try:
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    shutil.copy2(tmp_path, final_path)
                except Exception as e:
                    errors.append((os.path.basename(tmp_path), f"finalize_copy_failed: {e}"))

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

    # -----------------------------------------------------------------------
    # Snapshot rendering (subprocess) 
    # -----------------------------------------------------------------------
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

mapper = vtk.vtkFixedPointVolumeRayCastMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.SetImageSampleDistance(1.0)
mapper.SetSampleDistance(0.5)

ctf = vtk.vtkColorTransferFunction()
ctf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
ctf.AddRGBPoint(-600,  0.0, 0.0, 0.0)
ctf.AddRGBPoint(-200,  0.15, 0.12, 0.10)
ctf.AddRGBPoint(-100,  0.65, 0.55, 0.48)
ctf.AddRGBPoint(0,     0.85, 0.78, 0.70)
ctf.AddRGBPoint(50,    0.92, 0.87, 0.80)
ctf.AddRGBPoint(150,   0.98, 0.96, 0.92)
ctf.AddRGBPoint(300,   1.0, 1.0, 1.0)
ctf.AddRGBPoint(1000,  1.0, 1.0, 1.0)

otf = vtk.vtkPiecewiseFunction()
otf.AddPoint(-1000, 0.00)
otf.AddPoint(-700,  0.00)
otf.AddPoint(-200,  0.00)
otf.AddPoint(-80,   0.02)
otf.AddPoint(-40,   0.03)
otf.AddPoint(0,     0.06)
otf.AddPoint(50,    0.14)
otf.AddPoint(120,   0.26)
otf.AddPoint(250,   0.40)
otf.AddPoint(700,   0.85)

prop = vtk.vtkVolumeProperty()
prop.SetColor(ctf)
prop.SetScalarOpacity(otf)
prop.SetInterpolationTypeToLinear()
prop.ShadeOn()
prop.SetAmbient(0.25)
prop.SetDiffuse(0.9)
prop.SetSpecular(0.12)
prop.SetSpecularPower(10.0)

volume = vtk.vtkVolume()
volume.SetMapper(mapper)
volume.SetProperty(prop)

ren = vtk.vtkRenderer()
ren.SetBackground(0, 0, 0)
ren.AddVolume(volume)
ren.ResetCamera()

renwin = vtk.vtkRenderWindow()
renwin.SetOffScreenRendering(1)
renwin.AddRenderer(ren)
renwin.SetSize(int({image_size}), int({image_size}))
renwin.SetMultiSamples(0)

bounds = volume.GetBounds()
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

    # -----------------------------------------------------------------------
    # MAIN PIPELINE
    # -----------------------------------------------------------------------
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
        patient_input_root=None,
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
                        global_no_text_png_dir=global_no_text_png_dir,
                        patient_input_root=patient_input_root or in_path,
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


# =============================================================================
# Tests
# =============================================================================
class HeadCTDeidTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_HeadCTDeid1()

    def test_HeadCTDeid1(self):
        self.assertTrue(True)
