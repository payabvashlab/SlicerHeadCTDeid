# -*- coding: utf-8 -*-
"""
HeadCTDeid (3D Slicer scripted module)

"""

import glob
import logging
import os
import random
import shutil
import sys
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

# 1) Where to pip-install SAM2 from (must be pip-installable)
SAM2_PIP_SOURCE = "git+https://github.com/facebookresearch/segment-anything-2.git"

# 2) Direct checkpoint URL (.pt) (must be a direct file download, not a web page)
SAM2_CKPT_URL = "https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_small.pt?download=true"

# 3) Direct config URL (.yaml)
SAM2_CFG_URL = "https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_s.yaml?download=true"

# Filenames saved under Resources/models/
SAM2_CKPT_FILENAME = "sam2_hiera_small.pt"
SAM2_CFG_FILENAME = "sam2_hiera_s.yaml"

# =============================================================================

FACE_MAX_VALUE = 50
FACE_MIN_VALUE = -125
AIR_THRESHOLD = -800
BONE_STOP_HU = 250  # tune 250-350 if needed
FRONT_BOOST_KERNEL = (3, 3)

# ---------------------------------------------------------------------------
# DICOM tags to de-id (your list)
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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _to_str(x):
    """Convert pydicom values / DataElement / raw types to lowercase stripped string."""
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
    """
    Return True if BurnedInAnnotation indicates YES.
    Common DICOM CS values are "YES"/"NO". We accept several truthy variants.
    """
    try:
        bia_val = getattr(ds, "BurnedInAnnotation", None)
        if bia_val is None:
            bia_val = ds.get((0x0028, 0x0301), None)
        v = _to_str(bia_val)
        return v in {"yes", "y", "true", "1"}
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class HeadCTDeid(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Head CT De-identification"
        self.parent.categories = ["Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = ["Anh Tuan Tran, Sam Payabvash"]
        self.parent.helpText = """
This module de-identifies DICOM files by removing patient information based on a given mapping table.
"""
        self.parent.acknowledgementText = """
This file was developed by Anh Tuan Tran, Sam Payabvash (Columbia University).
"""


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

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
            try:
                slicer.util.infoDisplay(
                    "This tool is a work-in-progress being validated in project. "
                    "Contact sp4479@columbia.edu for details. Use at your own risk.",
                    windowTitle="Warning",
                )

                # NEW meaning of checkbox:
                #   checked   => SAM for all slices (force)
                #   unchecked => SAM only when BurnedInAnnotation==YES
                force_sam_all = self.ui.deidentifyCheckbox.isChecked()
                remove_cta = self.ui.deidentifyCTACheckbox.isChecked()

                # base deps always
                self.logic.setupPythonRequirements()

                pn = self.logic.getParameterNode()

                # OPTION 1: ALWAYS ensure SAM2 install + download (no dialogs)
                ckpt_path, cfg_path = self.logic.ensureSAM2AutoDownload(parameterNode=pn)
                if pn is not None:
                    pn.SetParameter("SAM2_CKPT", os.path.abspath(ckpt_path))
                    pn.SetParameter("SAM2_CONFIG", os.path.abspath(cfg_path))

                if self.ui.progressBar:
                    self.ui.progressBar.setValue(0)

                ckpt = pn.GetParameter("SAM2_CKPT") if pn else ""
                cfg = pn.GetParameter("SAM2_CONFIG") if pn else ""

                self.logic.process(
                    self.ui.inputFolderButton.directory,
                    self.ui.excelFileButton.text,
                    self.ui.outputFolderButton.directory,
                    force_sam_all=force_sam_all,
                    remove_CTA=remove_cta,
                    progressBar=self.ui.progressBar,
                    sam2_ckpt=ckpt,
                    sam2_cfg=cfg,
                )
            except Exception as e:
                slicer.util.errorDisplay(f"Error: {str(e)}")
        except Exception:
            slicer.util.pip_install("python-gdcm==3.0.25")
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


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

class HeadCTDeidLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.logger = logging.getLogger("PatientProcessor")

    def _checkModuleInstalled(self, moduleName):
        try:
            __import__(moduleName)
            return True
        except ModuleNotFoundError:
            return False

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
        if not parameterNode.GetParameter("SAM2_CKPT"):
            parameterNode.SetParameter("SAM2_CKPT", "")
        if not parameterNode.GetParameter("SAM2_CONFIG"):
            parameterNode.SetParameter("SAM2_CONFIG", "")

    # ---------------------------------------------------------------------
    # SAM2 install + auto-download (NO dialogs) - ALWAYS called (Option 1)
    # ---------------------------------------------------------------------
    def ensureSAM2AutoDownload(self, parameterNode=None, force_reinstall=False, force_redownload=False):
        import importlib
        import urllib.request

        if not SAM2_PIP_SOURCE:
            raise RuntimeError("SAM2_PIP_SOURCE is empty. Set it at top of module.")
        if not SAM2_CKPT_URL or not SAM2_CFG_URL:
            raise RuntimeError("SAM2_CKPT_URL / SAM2_CFG_URL must be set at top of module.")

        def pip_install(args):
            slicer.util.pip_install(args)

        # torch
        try:
            import torch  
        except Exception:
            pip_install(["torch", "torchvision"])

        # common deps (best-effort)
        for pkg in ["opencv-python", "hydra-core", "omegaconf", "tqdm", "pyyaml"]:
            try:
                __import__(pkg.split("-")[0])
            except Exception:
                try:
                    pip_install(pkg)
                except Exception:
                    pass

        # sam2 pip install
        need_install = force_reinstall
        try:
            import sam2  
        except Exception:
            need_install = True

        if need_install:
            slicer.util.showStatusMessage("Installing SAM2 package ...")
            pip_install(SAM2_PIP_SOURCE)

        importlib.invalidate_caches()

        try:
            import sam2  
        except Exception as e:
            raise RuntimeError(f"SAM2 install/import failed: {e}")

        # Local model dir (inside module)
        module_dir = os.path.dirname(__file__)
        model_dir = os.path.join(module_dir, "Resources", "models")
        os.makedirs(model_dir, exist_ok=True)

        default_ckpt_path = os.path.abspath(os.path.join(model_dir, SAM2_CKPT_FILENAME))
        default_cfg_path = os.path.abspath(os.path.join(model_dir, SAM2_CFG_FILENAME))

        pn_ckpt = ""
        pn_cfg = ""
        if parameterNode is not None:
            pn_ckpt = (parameterNode.GetParameter("SAM2_CKPT") or "").strip()
            pn_cfg = (parameterNode.GetParameter("SAM2_CONFIG") or "").strip()

        ckpt_path = os.path.abspath(pn_ckpt) if (pn_ckpt and os.path.exists(pn_ckpt)) else default_ckpt_path
        cfg_path = os.path.abspath(pn_cfg) if (pn_cfg and os.path.exists(pn_cfg)) else default_cfg_path

        def _download(url, out_path):
            tmp = out_path + ".part"
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

            slicer.util.showStatusMessage(f"Downloading {os.path.basename(out_path)} ...")
            urllib.request.urlretrieve(url, tmp)

            if (not os.path.exists(tmp)) or (os.path.getsize(tmp) < 1024):
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                raise RuntimeError(f"Downloaded file looks invalid/too small: {url}")

            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            os.replace(tmp, out_path)

        if force_redownload or (not os.path.exists(cfg_path)):
            _download(SAM2_CFG_URL, cfg_path)

        if force_redownload or (not os.path.exists(ckpt_path)):
            _download(SAM2_CKPT_URL, ckpt_path)

        if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
            raise RuntimeError("SAM2 config/checkpoint missing after download (check URLs and network).")

        if parameterNode is not None:
            parameterNode.SetParameter("SAM2_CKPT", os.path.abspath(ckpt_path))
            parameterNode.SetParameter("SAM2_CONFIG", os.path.abspath(cfg_path))

        return os.path.abspath(ckpt_path), os.path.abspath(cfg_path)

    def setupPythonRequirements(self, upgrade=False):
        def install(package):
            slicer.util.pip_install(package)

        try:
            import pandas  # noqa
        except ModuleNotFoundError:
            install("pandas==2.2.3")

        try:
            import openpyxl  # noqa
        except ModuleNotFoundError:
            install("openpyxl")

        try:
            import pydicom  # noqa
        except ModuleNotFoundError:
            install("pydicom")
            install("pylibjpeg")
            install("pylibjpeg-libjpeg")
            install("pylibjpeg-openjpeg")

        try:
            import cv2  # noqa
        except ModuleNotFoundError:
            install("opencv-python")

        if not self._checkModuleInstalled("scikit-image"):
            install("scikit-image")

        self.dependenciesInstalled = True

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

    def process(
        self,
        inputFolder,
        excelFile,
        outputFolder,
        force_sam_all,
        remove_CTA,
        progressBar,
        sam2_ckpt="",
        sam2_cfg="",
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
        total_rows = max(1, df.shape[0])
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(outputFolder, f"Processed for Anonymization_{current_time}")
        os.makedirs(out_path, exist_ok=True)

        total_time = 0.0
        successful = 0

        for i, foldername in enumerate(sorted(dicom_folders)):
            if foldername in id_mapping:
                dst_folder = ""
                try:
                    start_time = time.time()
                    dst_folder = os.path.join(out_path, id_mapping[foldername])

                    sam2_ckpt_abs = os.path.abspath(sam2_ckpt) if sam2_ckpt else ""
                    sam2_cfg_abs = os.path.abspath(sam2_cfg) if sam2_cfg else ""

                    processor = DicomProcessor(
                        sam2_ckpt=sam2_ckpt_abs,
                        sam2_cfg=sam2_cfg_abs,
                        force_sam_all=bool(force_sam_all),
                    )

                    src_folder = os.path.join(inputFolder, foldername)

                    _ = processor.drown_volume(
                        in_path=src_folder,
                        out_path=dst_folder,
                        replacer="face",
                        id=id_mapping[foldername],
                        patient_id="0",
                        name=f"Processed for Anonymization {id_mapping[foldername]}",
                        remove_CTA=remove_CTA,
                    )

                    if progressBar:
                        progressBar.setValue(int((i + 1) * 100 / total_rows))
                    slicer.util.showStatusMessage(f"Finished processing folder {foldername}")
                    self.logger.info(f"Finished processing folder: {foldername}")

                    elapsed = time.time() - start_time
                    total_time += elapsed
                    successful += 1
                except Exception as e:
                    self.logger.error(f"Error processing folder {foldername}: {str(e)}")
                    if dst_folder and os.path.exists(dst_folder):
                        shutil.rmtree(dst_folder)

        if successful > 0:
            avg = total_time / successful
            self.logger.info(f"Average time per folder: {avg:.2f}s")
        else:
            self.logger.info("No folders were processed successfully.")

        try:
            requested = df["original_folder_name"].tolist()
            actual = set(dicom_folders)
            missing = [f for f in requested if f not in actual]
            if missing:
                self.logger.error(f"Missing Folders {missing}")
                slicer.util.showStatusMessage(f"Missing Folders {missing}")
        except Exception as e:
            self.logger.error(f"Post-check error: {str(e)}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class HeadCTDeidTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_HeadCTDeid1()

    def test_HeadCTDeid1(self):
        self.assertTrue(True)


# ---------------------------------------------------------------------------
# DICOM Processor
# ---------------------------------------------------------------------------

class DicomProcessor:
    """
    Pipeline: de-identification + face/air replacement + optional SAM2 burned-in text removal.

    SAM2 run decision per slice:
      if force_sam_all == True:
          run SAM on every slice
      else:
          run SAM only when dicom_has_burned_in(ds) == True
    """

    def __init__(self, sam2_ckpt="", sam2_cfg="", force_sam_all=False):
        self.error = ""
        self.net = ""
        self.study_uid_map = defaultdict(str)
        self.series_uid_map = defaultdict(str)
        self.sop_uid_map = defaultdict(str)
        self.uid_map_general = defaultdict(str)

        self.logger = logging.getLogger("PatientProcessor")

        self._force_sam_all = bool(force_sam_all)

        # SAM2 state (lazy init: build only when a slice actually needs SAM)
        self._sam2_ready = False
        self._sam2_ckpt = os.path.abspath(sam2_ckpt) if sam2_ckpt else ""
        self._sam2_cfg = os.path.abspath(sam2_cfg) if sam2_cfg else ""
        self._sam2_device = "cpu"
        self._sam2_mask_generator = None

    # ---------------------------------------------------------
    # SAM2 init (lazy)
    # ---------------------------------------------------------
    def _try_init_sam2(self):
        """
        Hydra fix: build_sam2 expects a Hydra *primary config name* (like "sam2_hiera_s.yaml"),
        not a filesystem path. Therefore we copy the yaml into the installed sam2 package dir,
        and pass ONLY the basename to build_sam2.
        """
        try:
            import shutil
            import torch
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except Exception as e:
            self._sam2_ready = False
            self._sam2_mask_generator = None
            try:
                self.logger.error(f"SAM2 import failed: {e}")
            except Exception:
                pass
            return

        ckpt_path = os.path.abspath(self._sam2_ckpt) if self._sam2_ckpt else ""
        cfg_path = os.path.abspath(self._sam2_cfg) if self._sam2_cfg else ""

        if not ckpt_path or not os.path.exists(ckpt_path):
            self._sam2_ready = False
            self._sam2_mask_generator = None
            try:
                self.logger.error(f"SAM2 checkpoint missing: {ckpt_path}")
            except Exception:
                pass
            return

        if not cfg_path or not os.path.exists(cfg_path):
            self._sam2_ready = False
            self._sam2_mask_generator = None
            try:
                self.logger.error(f"SAM2 config missing: {cfg_path}")
            except Exception:
                pass
            return

        cfg_name = os.path.basename(cfg_path)

        # Copy YAML into sam2 package dir so Hydra provider pkg://sam2 can find it
        try:
            sam2_pkg_dir = os.path.dirname(sam2.__file__)
            target_cfg_path = os.path.join(sam2_pkg_dir, cfg_name)

            need_copy = True
            if os.path.exists(target_cfg_path):
                try:
                    if os.path.getsize(target_cfg_path) == os.path.getsize(cfg_path):
                        need_copy = False
                except Exception:
                    need_copy = True

            if need_copy:
                try:
                    shutil.copy2(cfg_path, target_cfg_path)
                except Exception:
                    with open(cfg_path, "rb") as rf, open(target_cfg_path, "wb") as wf:
                        wf.write(rf.read())

            if not os.path.exists(target_cfg_path):
                raise RuntimeError(f"Failed to place config into sam2 package dir: {target_cfg_path}")

        except Exception as e:
            self._sam2_ready = False
            self._sam2_mask_generator = None
            try:
                self.logger.error(
                    "SAM2 init failed: cannot place yaml into sam2 package dir for Hydra to find. "
                    f"Error: {e}"
                )
            except Exception:
                pass
            return

        # GPU detection
        device = "cpu"
        try:
            if torch.cuda.is_available():
                torch.cuda.current_device()
                device = "cuda"
        except Exception:
            device = "cpu"
        self._sam2_device = device

        try:
            self.logger.info(f"SAM2 build with cfg_name={cfg_name}, ckpt={ckpt_path}, device={device}")
        except Exception:
            pass

        try:
            model = build_sam2(cfg_name, ckpt_path, device=device)

            if device == "cuda":
                try:
                    model.to("cuda")
                except Exception:
                    try:
                        self.logger.warning("CUDA detected but model.to(cuda) failed. Falling back to CPU.")
                    except Exception:
                        pass
                    device = "cpu"
                    self._sam2_device = "cpu"
                    model = build_sam2(cfg_name, ckpt_path, device="cpu")

            self._sam2_mask_generator = SAM2AutomaticMaskGenerator(
                model,
                points_per_side=24,
                pred_iou_thresh=0.85,
                stability_score_thresh=0.90,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=128,
            )
            self._sam2_ready = True
            try:
                self.logger.info(f"SAM2 initialized successfully on {self._sam2_device}.")
            except Exception:
                pass

        except Exception as e:
            self._sam2_ready = False
            self._sam2_mask_generator = None
            try:
                self.logger.error(f"SAM2 init failed: {e}")
            except Exception:
                pass

    def _get_sam2_generator(self):
        if self._sam2_mask_generator is None and not self._sam2_ready:
            self._try_init_sam2()
        return self._sam2_mask_generator

    # ---------------------------------------------------------
    # SAM2-based "text overlay" removal
    # ---------------------------------------------------------
    def _sam2_text_mask(self, rgb_uint8):
        gen = self._get_sam2_generator()
        if gen is None:
            return None
        try:
            H, W = rgb_uint8.shape[:2]
            masks = gen.generate(rgb_uint8)
            if not masks:
                return None

            out = np.zeros((H, W), dtype=np.uint8)
            border = int(max(8, round(min(H, W) * 0.08)))  # ~8% border band

            for m in masks:
                seg = m.get("segmentation", None)
                bbox = m.get("bbox", None)
                if seg is None or bbox is None:
                    continue

                x, y, bw, bh = bbox
                x = int(x); y = int(y); bw = int(bw); bh = int(bh)

                area = int(np.sum(seg))
                if area <= 0:
                    continue
                if area > 0.02 * (H * W):
                    continue

                near_border = (x <= border) or (y <= border) or ((x + bw) >= (W - border)) or ((y + bh) >= (H - border))
                if not near_border:
                    continue

                if bw > 0.9 * W and bh < 0.02 * H:
                    continue
                if bh > 0.9 * H and bw < 0.02 * W:
                    continue

                out[seg.astype(bool)] = 1

            if out.max() == 0:
                return None
            return out
        except Exception:
            return None

    def _remove_text_with_sam2(self, new_volume_hu, pixels_hu):
        # If SAM isn't ready, do nothing (lazy init will have been attempted in _sam2_text_mask)
        try:
            import cv2

            mn, mx = int(np.min(pixels_hu)), int(np.max(pixels_hu))
            rng = max(1, mx - mn)
            gray8 = np.uint8(((pixels_hu - mn) / rng) * 255.0)
            rgb = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

            m = self._sam2_text_mask(rgb)
            if m is None:
                return new_volume_hu

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            m2 = cv2.dilate(m.astype(np.uint8), k, iterations=1).astype(bool)

            out = np.array(new_volume_hu, copy=True)
            out[m2] = -1000
            return out
        except Exception as e:
            try:
                self.logger.error(f"_remove_text_with_sam2 {e}")
            except Exception:
                pass
            slicer.util.showStatusMessage(f"_remove_text_with_sam2 {e}")
            return new_volume_hu

    # ---------------------------------------------------------
    # CT helpers
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # CT meta filter
    # ---------------------------------------------------------
    def is_substring_in_list(self, substring, string_list):
        return any(substring in str(s) for s in string_list)

    def checkCTmeta(self, ds, remove_CTA=False):
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
        except Exception:
            self.error = "CT meta check failed"
            return 0

    # ---------------------------------------------------------
    # Anterior region helper
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Dilation with barriers
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # DICOM anonymization helpers
    # ---------------------------------------------------------
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
        if tag == (0x0010, 0x0020):  # PatientID
            ds[tag].value = patient_id_value
            return
        if tag == (0x0010, 0x0010):  # PatientName
            ds[tag].value = "Processed for anonymization"
            return
        if tag == (0x0008, 0x0050):  # AccessionNumber
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
            if tag == (0x0020, 0x000D):  # StudyInstanceUID
                ds[tag].value = self._remap_uid(ds[tag].value, self.study_uid_map, generate_uid_fn)
                return
            if tag == (0x0020, 0x000E):  # SeriesInstanceUID
                ds[tag].value = self._remap_uid(ds[tag].value, self.series_uid_map, generate_uid_fn)
                return
            if tag == (0x0008, 0x0018):  # SOPInstanceUID
                ds[tag].value = self._remap_uid(ds[tag].value, self.sop_uid_map, generate_uid_fn)
                return
            if tag == (0x0020, 0x0052):  # FrameOfReferenceUID
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

                    if elem.VR in {"DA", "DT", "TM"} and tag in dataset:
                        self._set_safe_value_by_vr(dataset, tag, elem.VR, generate_uid, patient_id_value)
                        continue

                    if tag in PDF_TAGS_TO_DEID and tag in dataset:
                        self._set_safe_value_by_vr(dataset, tag, elem.VR, generate_uid, patient_id_value)

                except Exception:
                    continue

        recurse(ds)

    # ---------------------------------------------------------
    # Save anonymized dicoms + conditional SAM2 text removal
    # ---------------------------------------------------------
    def save_new_dicom_files(
        self,
        original_dir,
        out_dir,
        replacer="face",
        id="new_folder_name",
        patient_id="0",
        new_patient_id="Processed for anonymization",
        remove_CTA=False,
    ):
        """
        SAM decision:
          - if self._force_sam_all: run SAM on ALL slices
          - else: run SAM only if dicom_has_burned_in(ds) is True
        """
        import pydicom

        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(original_dir) if self.is_dicom(os.path.join(original_dir, f), remove_CTA)]
        errors = []

        def _instnum(path):
            try:
                ds_ = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                return int(getattr(ds_, "InstanceNumber", 1))
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

                bin_mask = self.binarize_volume(pixels_hu)
                lcc = self.get_largest_component_volume(bin_mask)

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

                # --------------------------
                # NEW SAM rule
                # --------------------------
                want_sam = self._force_sam_all or dicom_has_burned_in(ds)
                if want_sam:
                    new_volume = self._remove_text_with_sam2(new_volume, pixels_hu)

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

    # -----------------------------------------------------------------------
    # Snapshot rendering
    # -----------------------------------------------------------------------
    def _capture_threeDView_png(self, threeDView, out_png_path):
        import vtk
        try:
            threeDView.forceRender()
            slicer.app.processEvents()
            rw = threeDView.renderWindow()
            rw.Render()
        except Exception:
            pass

        rw = threeDView.renderWindow()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.SetReadFrontBuffer(False)
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(out_png_path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()

    def _set_view_and_zoom(self, threeDView, axisIndex, dollyFactor=2.0, viewAngleDeg=12.0):
        threeDView.rotateToViewAxis(int(axisIndex))
        threeDView.resetFocalPoint()
        threeDView.resetCamera()
        threeDView.forceRender()
        slicer.app.processEvents()

        camNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(threeDView.mrmlViewNode())
        cam = camNode.GetCamera()

        try:
            cam.SetViewAngle(float(viewAngleDeg))
        except Exception:
            pass

        try:
            cam.Dolly(float(dollyFactor))
        except Exception:
            fp = cam.GetFocalPoint()
            pos = cam.GetPosition()
            df = float(dollyFactor) if float(dollyFactor) > 0 else 1.0
            newPos = (
                fp[0] + (pos[0] - fp[0]) / df,
                fp[1] + (pos[1] - fp[1]) / df,
                fp[2] + (pos[2] - fp[2]) / df,
            )
            cam.SetPosition(*newPos)

        try:
            camNode.ResetClippingRange()
        except Exception:
            pass

        threeDView.forceRender()
        slicer.app.processEvents()

    def _harden_parent_transform_if_any(self, volumeNode):
        try:
            tnode = volumeNode.GetParentTransformNode()
            if tnode is None:
                return
            try:
                slicer.modules.transforms.logic().hardenTransform(volumeNode)
            except Exception:
                try:
                    logic = slicer.vtkSlicerTransformLogic()
                    logic.hardenTransform(volumeNode)
                except Exception:
                    pass
            try:
                volumeNode.SetAndObserveTransformNodeID(None)
            except Exception:
                pass
        except Exception:
            pass

    def _choose_best_volume_node(self, loadedNodeIDs):
        candidates = []
        for nid in loadedNodeIDs:
            n = slicer.mrmlScene.GetNodeByID(nid)
            if n and n.IsA("vtkMRMLScalarVolumeNode"):
                img = n.GetImageData()
                dims = img.GetDimensions() if img else (0, 0, 0)
                vox = int(dims[0] * dims[1] * max(1, dims[2]))
                name = (n.GetName() or "").upper()
                penalty = 0
                if "BONE" in name:
                    penalty += 1000000000
                candidates.append((penalty, -vox, n))
        if not candidates:
            vols = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            return vols[0] if vols else None
        candidates.sort()
        return candidates[0][2]

    def _apply_new_soft_tissue_opacity(self, displayNode):
        try:
            propNode = displayNode.GetVolumePropertyNode()
            if not propNode:
                return
            vp = propNode.GetVolumeProperty()
            if not vp:
                return

            sof = vp.GetScalarOpacity()
            sof.RemoveAllPoints()

            sof.AddPoint(-200, 0.0)
            sof.AddPoint(-40,  0.0)
            sof.AddPoint(0,    0.05)
            sof.AddPoint(50,   0.15)
            sof.AddPoint(150,  0.35)
            sof.AddPoint(300,  0.6)
            sof.AddPoint(700,  1.0)

            try:
                vp.SetShade(True)
                vp.SetAmbient(0.2)
                vp.SetDiffuse(0.9)
                vp.SetSpecular(0.1)
            except Exception:
                pass
        except Exception:
            pass

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

    def _render_one_dicom_folder(self, dicomDir, out_prefix="view"):
        import tempfile
        import shutil as _shutil
        from DICOMLib import DICOMUtils

        dicom_files = []
        for fn in os.listdir(dicomDir):
            fp = os.path.join(dicomDir, fn)
            if not os.path.isfile(fp):
                continue
            if self.is_dicom(fp, remove_CTA=False):
                dicom_files.append(fp)

        if not dicom_files:
            raise RuntimeError(f"No DICOM files found in: {dicomDir}")

        lm = slicer.app.layoutManager()
        if lm is None:
            raise RuntimeError("No layoutManager available. Run with Slicer GUI (3D view required).")

        slicer.mrmlScene.Clear(False)
        slicer.app.processEvents()

        tmp_root = tempfile.mkdtemp(prefix="slicer_dicoms_")
        try:
            for i, src in enumerate(sorted(dicom_files), start=1):
                dst = os.path.join(tmp_root, f"slice_{i:06d}.dcm")
                try:
                    _shutil.copy2(src, dst)
                except Exception:
                    _shutil.copy(src, dst)

            with DICOMUtils.TemporaryDICOMDatabase() as db:
                DICOMUtils.importDicom(tmp_root, db)
                patientUIDs = db.patients()
                if not patientUIDs:
                    raise RuntimeError(f"No DICOM patients found in: {dicomDir}")
                loadedNodeIDs = DICOMUtils.loadPatientByUID(patientUIDs[0])
        finally:
            try:
                _shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass

        volumeNode = self._choose_best_volume_node(loadedNodeIDs)
        if volumeNode is None:
            raise RuntimeError(f"No scalar volume loaded from: {dicomDir}")

        self._harden_parent_transform_if_any(volumeNode)

        vrLogic = slicer.modules.volumerendering.logic()
        if vrLogic is None:
            raise RuntimeError("VolumeRendering logic not available.")

        displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
        if displayNode is None:
            vrLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
            displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
        if displayNode is None:
            raise RuntimeError("Failed to create Volume Rendering display node.")

        preset_names_to_try = ["CT-Soft-Tissue", "CT Abdomen", "CT-AAA", "CT Air", "CT Bone"]
        presetNode = None
        for pname in preset_names_to_try:
            p = vrLogic.GetPresetByName(pname)
            if p:
                presetNode = p
                break

        if presetNode:
            propNode = displayNode.GetVolumePropertyNode()
            if propNode:
                try:
                    propNode.Copy(presetNode)
                except Exception:
                    pass

        self._apply_new_soft_tissue_opacity(displayNode)

        displayNode.SetVisibility(1)
        try:
            vrLogic.FitROIToVolume(displayNode)
        except Exception:
            pass

        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()

        self._set_view_and_zoom(
            threeDView,
            axisIndex=3,       # anterior
            dollyFactor=1.4,
            viewAngleDeg=12.0,
        )

        out_path = os.path.join(dicomDir, f"{out_prefix}_anterior.png")
        self._capture_threeDView_png(threeDView, out_path)
        return [out_path]

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
        patient_id="0",
        name="",
        remove_CTA=False,
    ):
        try:
            # Phase 1: anonymize + drown + conditional SAM
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
                        remove_CTA=remove_CTA,
                    )

            # Phase 2: rename subfolders
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

            # Phase 3: snapshots
            try:
                self._create_and_save_multi_view_snapshots(out_path, out_prefix="view")
            except Exception as e:
                try:
                    with open(os.path.join(out_path, "render_summary.txt"), "a") as f:
                        f.write(f"[{datetime.now()}] Snapshot phase failed: {e}\n")
                except Exception:
                    pass

        except Exception as e:
            try:
                os.makedirs(out_path, exist_ok=True)
                with open(os.path.join(out_path, "log.txt"), "a") as f:
                    f.write(f"Error: {e}\n")
            except Exception:
                pass
            return 0

        return 1
