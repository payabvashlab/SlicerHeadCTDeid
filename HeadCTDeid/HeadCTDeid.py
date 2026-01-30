# -*- coding: utf-8 -*-
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

# HU ranges used for “face/soft-tissue” sampling
FACE_MAX_VALUE = 50
FACE_MIN_VALUE = -125
AIR_THRESHOLD = -800


# ---------------------------------------------------------------------------
# DICOM tags to de-id 101 tags
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
# Module
# ---------------------------------------------------------------------------

class HeadCTDeid(ScriptedLoadableModule):
    """ScriptedLoadableModule base (3D Slicer)."""

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
            import gdcm  # noqa
        except Exception:
            slicer.util.pip_install("python-gdcm==3.0.25")
            slicer.util.infoDisplay(
                "To support full encoding DICOM.\nPlease restart Slicer to complete the setup.",
                windowTitle="Warning"
            )
        try:
            slicer.util.infoDisplay(
                "This tool is a work-in-progress being validated in project. Contact sp4479@columbia.edu for details. Use at your own risk.",
                windowTitle="Warning"
            )
            self.logic.setupPythonRequirements()
            if self.ui.progressBar:
                self.ui.progressBar.setValue(0)

            self.logic.process(
                self.ui.inputFolderButton.directory,
                self.ui.excelFileButton.text,
                self.ui.outputFolderButton.directory,
                self.ui.deidentifyCheckbox.isChecked(),
                self.ui.deidentifyCTACheckbox.isChecked(),
                self.ui.progressBar,
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Error: {str(e)}")

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

    def setupPythonRequirements(self, upgrade=False):
        def install(package):
            slicer.util.pip_install(package)

        try:
            import pandas  # noqa
        except ModuleNotFoundError:
            slicer.util.pip_install("pandas==2.2.3")
        try:
            import openpyxl  # noqa
        except ModuleNotFoundError:
            slicer.util.pip_install("openpyxl")
        try:
            import pydicom  # noqa
        except ModuleNotFoundError:
            slicer.util.pip_install("pydicom")
            slicer.util.pip_install("pylibjpeg")
            slicer.util.pip_install("pylibjpeg-libjpeg")
            slicer.util.pip_install("pylibjpeg-openjpeg")
        try:
            import cv2  # noqa
        except ModuleNotFoundError:
            slicer.util.pip_install("opencv-python")

        if not self._checkModuleInstalled("scikit-image"):
            install("scikit-image")

        if not self._checkModuleInstalled("easyocr"):
            slicer.util.pip_install(
                ["torch", "easyocr", "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
            )

        self.dependenciesInstalled = True
        logging.debug("Dependencies are set up successfully.")

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

    def process(self, inputFolder, excelFile, outputFolder, remove_text, remove_CTA, progressBar):
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
                    processor = DicomProcessor()
                    src_folder = os.path.join(inputFolder, foldername)

                    _ = processor.drown_volume(
                        in_path=src_folder,
                        out_path=dst_folder,
                        replacer="face",
                        id=id_mapping[foldername],
                        patient_id="0",
                        name=f"Processed for Anonymization {id_mapping[foldername]}",
                        remove_text=remove_text,
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
    Phase 1: Walk input tree, mirror structure to out_path and write anonymized DICOMs.
    Phase 2: For every directory level under out_path, rename its immediate subfolders
             in sorted order to <id>_<1..N>, preserving the nesting depth.
    Phase 3 (NEW): snapshot generation (VR 3D) for every subfolder containing DICOMs.
    """

    def __init__(self):
        self.error = ""
        self.net = ""
        self.study_uid_map = defaultdict(str)
        self.series_uid_map = defaultdict(str)
        self.sop_uid_map = defaultdict(str)
        self.uid_map_general = defaultdict(str)
        self._ocr_reader = None

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"])
            except Exception:
                self._ocr_reader = None
        return self._ocr_reader

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
        """
        Kernel range based on PixelSpacing (0028,0030), first component:
          lo = ceil(10/pixel)
          hi = ceil(15/pixel)
        fallback: random 30..40 if missing/invalid.
        """
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

    def dilate_volume(self, volume, kernel_size):
        import cv2
        k = int(kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(volume.astype(np.uint8), kernel)

    def apply_mask_and_get_values(self, image_volume, mask_volume):
        masked = image_volume * mask_volume
        vals = np.unique(masked)
        vals = vals[(vals > FACE_MIN_VALUE) & (vals < FACE_MAX_VALUE)]
        return vals.tolist()

    def apply_random_values_optimized(self, pixels_hu, dilated_volume, unique_values_list):
        new_vol = np.copy(pixels_hu)
        new_vol[dilated_volume == 1] = -1000
        return new_vol

    def curves_callback(self, ds, elem):
        if elem.tag.group & 0xFF00 == 0x5000:
            del ds[elem.tag]

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
        """
        Overwrite element value using VR-aware rules.
        """
        # enforce required patient identity
        if tag == (0x0010, 0x0020):  # PatientID
            ds[tag].value = patient_id_value
            return
        if tag == (0x0010, 0x0010):  # PatientName
            ds[tag].value = "Processed for anonymization"
            return
        if tag == (0x0008, 0x0050):  # AccessionNumber
            ds[tag].value = patient_id_value
            return

        # current date/time
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

        # UID remap
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

                ksize = self._kernel_from_pixel_spacing(ds)
                dilated = self.dilate_volume(lcc, kernel_size=ksize)

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

                if remove_text:
                    try:
                        import cv2
                        reader = self._get_ocr_reader()
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
                                    cv2.rectangle(new_volume, tl, br, (0, 0, 0), thickness=cv2.FILLED)
                    except Exception as e:
                        errors.append((fname, str(e)))

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

    # ---------------- snapshot helpers ----------------

    def _capture_threeDView_png(self, threeDView, out_png_path):
        """
        Capture current 3D view render window to PNG.
        """
        import vtk

        # ensure up-to-date render
        threeDView.forceRender()
        slicer.app.processEvents()

        rw = threeDView.renderWindow()
        try:
            rw.Render()
        except Exception:
            pass

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        # back buffer is typically correct in Slicer
        w2i.SetReadFrontBuffer(False)
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(out_png_path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()

    def _set_view_and_zoom(self, threeDView, axisIndex, dollyFactor=2.0, viewAngleDeg=12.0):
        """
        axisIndex: 0=Right, 1=Left, 2=Posterior, 3=Anterior, 4=Superior, 5=Inferior
        viewAngleDeg: camera view angle (degrees). SMALLER => zoom-in (more magnification).
        dollyFactor: camera dolly. >1 => closer.
        """
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

    def _apply_soft_tissue_transfer_function(self, displayNode):
        """
        Apply your provided soft-tissue oriented transfer functions.
        """
        try:
            propNode = displayNode.GetVolumePropertyNode()
            if not propNode:
                return
            vp = propNode.GetVolumeProperty()
            if not vp:
                return

            so = vp.GetScalarOpacity()
            so.RemoveAllPoints()
            so.AddPoint(-1000, 0.00)
            so.AddPoint(-700, 0.00)
            so.AddPoint(-200, 0.02)
            so.AddPoint(-100, 0.05)
            so.AddPoint(-50, 0.10)
            so.AddPoint(0, 0.18)
            so.AddPoint(40, 0.28)
            so.AddPoint(80, 0.35)
            so.AddPoint(300, 0.20)
            so.AddPoint(700, 0.25)
            so.AddPoint(1200, 0.30)

            ct = vp.GetRGBTransferFunction()
            ct.RemoveAllPoints()
            ct.AddRGBPoint(-1000, 0.00, 0.00, 0.00)
            ct.AddRGBPoint(-200, 0.25, 0.25, 0.25)
            ct.AddRGBPoint(0, 0.55, 0.55, 0.55)
            ct.AddRGBPoint(80, 0.70, 0.70, 0.70)
            ct.AddRGBPoint(300, 0.85, 0.85, 0.85)
            ct.AddRGBPoint(1200, 1.00, 1.00, 1.00)

            go = vp.GetGradientOpacity()
            go.RemoveAllPoints()
            go.AddPoint(0, 0.00)
            go.AddPoint(30, 0.20)
            go.AddPoint(80, 0.60)
            go.AddPoint(120, 0.90)

            try:
                vp.SetShade(True)
                vp.SetAmbient(0.25)
                vp.SetDiffuse(0.75)
                vp.SetSpecular(0.15)
            except Exception:
                pass
        except Exception:
            pass

    def _find_all_dicom_dirs(self, rootFolder):
        """
        Return ALL directories (under rootFolder) that contain at least one DICOM file.
        This matches your requirement: "every subfolder that contains DICOMs".
        """
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

    def _choose_best_volume_node(self, loadedNodeIDs):
        """Pick the most suitable scalar volume for rendering (avoid BONE if possible)."""
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

    def _render_one_dicom_folder(self, dicomDir, out_prefix="3d"):
        """
        Load DICOMs from dicomDir and create 4 snapshots inside that same folder.
        Returns list of output png paths.
        """
        from DICOMLib import DICOMUtils

        lm = slicer.app.layoutManager()
        if lm is None:
            raise RuntimeError("No layoutManager available. Run with Slicer GUI (3D view required).")

        slicer.mrmlScene.Clear(False)
        slicer.app.processEvents()

        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomDir, db)
            patientUIDs = db.patients()
            if not patientUIDs:
                raise RuntimeError(f"No DICOM patients found in: {dicomDir}")
            loadedNodeIDs = DICOMUtils.loadPatientByUID(patientUIDs[0])

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

        preset_names_to_try = ["CT-Soft-Tissue", "CT Abdomen", "CT-AAA", "CT Bone", "CT Air"]
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

        self._apply_soft_tissue_transfer_function(displayNode)

        displayNode.SetVisibility(1)
        try:
            vrLogic.FitROIToVolume(displayNode)
        except Exception:
            pass

        threeDView = lm.threeDWidget(0).threeDView()

        # Only 4 views (NO superior)
        views = [
            ("anterior", 3, 1.8, 12.0),
            ("left",     1, 2.2, 12.0),
            ("right",    0, 2.2, 12.0),
            ("posterior", 2, 2.2, 12.0),
        ]

        outputs = []
        for name, axisIndex, dolly, angle in views:
            self._set_view_and_zoom(threeDView, axisIndex=axisIndex, dollyFactor=dolly, viewAngleDeg=angle)
            out_path = os.path.join(dicomDir, f"{out_prefix}_{name}.png")
            self._capture_threeDView_png(threeDView, out_path)
            outputs.append(out_path)

        return outputs

    def _create_and_save_multi_view_snapshots(self, patientFolder, out_prefix="3d"):
        """
        NEW BEHAVIOR (as requested):
        - patientFolder is the patient-level output folder (new_folder_name).
        - Find ALL subfolders/sub-subfolders under patientFolder that contain DICOM slices.
        - For EACH DICOM-containing folder => generate snapshots into that folder.
        """
        dicom_dirs = self._find_all_dicom_dirs(patientFolder)
        if not dicom_dirs:
            raise RuntimeError("No snapshots produced (no DICOM-containing subfolders found).")

        all_outputs = []
        for d in dicom_dirs:
            try:
                outs = self._render_one_dicom_folder(d, out_prefix=out_prefix)
                all_outputs.extend(outs)
            except Exception as e:
                # keep going; write a per-folder render log
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
        patient_id="0",
        name="",
        remove_text=False,
        remove_CTA=False,
    ):
        """
        Phase 1: process while mirroring input structure.
        Phase 2: rename, at each level, immediate subdirectories to <id>_<n> (1-based),
                 preserving nesting; uses only os.rename (safe two-step).
        Phase 3: snapshot generation (VR 3D): every DICOM-containing subfolder under out_path
                 gets 4 views: anterior, left, right, posterior.
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

            # Phase 3: snapshots (use your renderer)
            # out_path is the patient-level folder in this function.
            try:
                self._create_and_save_multi_view_snapshots(out_path, out_prefix="3d")
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
