import logging
import os
import time
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from ctk import ctkFileDialog
from datetime import datetime
import shutil
from pathlib import Path
import sys
import importlib
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

FACE_MAX_VALUE = 50
FACE_MIN_VALUE = -125
AIR_THRESHOLD = -800
import random
KERNEL_SIZE = random.randint(30, 40)

# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class HeadCTDeid(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Head CT De-identification for Anonymization"
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

        uiWidget = slicer.util.loadUI(self.resourcePath('UI/HeadCTDeid.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = HeadCTDeidLogic()

        self.ui.inputFolderButton.connect('directoryChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.outputFolderButton.connect('directoryChanged(QString)', self.updateParameterNodeFromGUI)

        self.ui.applyButton.connect('clicked()', self.onApplyButton)
        self.ui.excelFileButton.connect('clicked()', self.onBrowseExcelFile)
        self.ui.deidentifyCheckbox.connect('toggled(bool)', self.updateParameterNodeFromGUI)
        self.ui.deidentifyCTACheckbox.connect('toggled(bool)', self.updateParameterNodeFromGUI)

        self.initializeParameterNode()

        from HeadCTDeidLib.dependency_handler import NonSlicerPythonDependencies
        dependencies = NonSlicerPythonDependencies()
        dependencies.setupPythonRequirements(upgrade=True)

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode):
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

        # reflect checkbox states from node
        self.ui.deidentifyCheckbox.setChecked(self._parameterNode.GetParameter("Deidentify") == "true")
        self.ui.deidentifyCTACheckbox.setChecked(self._parameterNode.GetParameter("DeidentifyCTA") == "true")

        # enable Apply when all required fields present
        if (len(self._parameterNode.GetParameter("InputFolder")) > 1 and
            len(self._parameterNode.GetParameter("ExcelFile")) > 4 and
            len(self._parameterNode.GetParameter("OutputFolder")) > 1 and
            self._parameterNode.GetParameter("ExcelFile") != "Browse"):
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
            slicer.util.infoDisplay(
                "This tool is a work-in-progress being validated in project. Contact sp4479@columbia.edu for details. Use at your own risk.",
                windowTitle="Warning")
            import qt
            try:
                slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
                self.logic.setupPythonRequirements()
                slicer.app.restoreOverrideCursor()
            except Exception as e:
                slicer.app.restoreOverrideCursor()
                slicer.util.errorDisplay(f"Failed to install required packages.\n\n{e}")
                return

            self.ui.progressBar.setValue(0)
            self.logic.process(
                self.ui.inputFolderButton.directory,
                self.ui.excelFileButton.text,
                self.ui.outputFolderButton.directory,
                self.ui.deidentifyCheckbox.isChecked(),
                self.ui.deidentifyCTACheckbox.isChecked(),
                self.ui.progressBar
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Error: {str(e)}")

    def onBrowseExcelFile(self):
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
            importlib.import_module(moduleName)
            return True
        except ModuleNotFoundError:
            return False

    def setupPythonRequirements(self, upgrade=False):
        def install(package):
            slicer.util.pip_install(package)

        try:
            import pandas
        except ModuleNotFoundError:
            slicer.util.pip_install("pandas==2.2.3")

        try:
            import openpyxl
        except ModuleNotFoundError:
            slicer.util.pip_install("openpyxl")

        try:
            import pydicom
        except ModuleNotFoundError:
            slicer.util.pip_install("pydicom")
            slicer.util.pip_install("pylibjpeg")
            slicer.util.pip_install("pylibjpeg-libjpeg")
            slicer.util.pip_install("pylibjpeg-openjpeg")

        try:
            import cv2
        except ModuleNotFoundError:
            slicer.util.pip_install("opencv-python")

        if not self._checkModuleInstalled("scikit-image"):
            install("scikit-image")

        if not self._checkModuleInstalled("easyocr"):
            slicer.util.pip_install(["torch", "easyocr", "--extra-index-url", "https://download.pytorch.org/whl/cpu"])

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

    def process(self, inputFolder, excelFile, outputFolder, remove_text, remove_CTA, progressBar):
        if not os.path.exists(inputFolder):
            raise ValueError(f"Input folder does not exist: {inputFolder}")
        if not os.path.exists(excelFile):
            raise ValueError(f"Excel/CSV file does not exist: {excelFile}")
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        import pandas
        columns_as_text = ['original_folder_name', 'new_folder_name']

        ext = os.path.splitext(excelFile)[1].lower()
        if ext == '.csv':
            df = pandas.read_csv(excelFile, dtype={col: str for col in columns_as_text})
        elif ext in ['.xlsx', '.xls']:
            df = pandas.read_excel(excelFile, dtype={col: str for col in columns_as_text})
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if ("original_folder_name" not in df.columns) or ("new_folder_name" not in df.columns):
            raise ValueError("Excel file must contain 'original_folder_name' and 'new_folder_name' columns")

        try:
            log_file = os.path.join(outputFolder, "patient_processing.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename for h in self.logger.handlers):
                self.logger.addHandler(file_handler)
            self.logger.info(f"Initialized patient processing module {log_file}")
        except Exception as e:
            self.logger.info(e)

        df['original_folder_name'] = df['original_folder_name'].astype(str).str.strip()
        df['new_folder_name'] = df['new_folder_name'].astype(str).str.strip()
        id_mapping = dict(zip(df['original_folder_name'], df['new_folder_name']))

        dicom_folders = [d for d in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, d))]
        total_rows = max(1, df.shape[0])  # avoid div by zero in progress
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(outputFolder, f'Processed for AHA_{current_time}')
        os.makedirs(out_path, exist_ok=True)

        total_time = 0.0
        successful = 0

        for i, foldername in enumerate(dicom_folders):
            if foldername in id_mapping:
                dst_folder = ""
                try:
                    start_time = time.time()
                    dst_folder = os.path.join(out_path, id_mapping[foldername])
                    processor = DicomProcessor()
                    src_folder = os.path.join(inputFolder, foldername)
                    result = processor.drown_volume(
                        src_folder, dst_folder, 'face', id_mapping[foldername],
                        patient_id='0', name=f"Processed for AHA {id_mapping[foldername]}",
                        remove_text=remove_text, remove_CTA=remove_CTA
                    )
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
            average_time = total_time / successful
            self.logger.info(f"Average time per folder: {average_time:.2f}s")
        else:
            self.logger.info("No folders were processed successfully.")

        try:
            folder_list_2 = df['original_folder_name'].tolist()
            actual_folders = dicom_folders
            missing_folders = [folder for folder in folder_list_2 if folder not in actual_folders]
            if len(missing_folders) > 0:
                self.logger.error(f"Missing Folders {missing_folders}")
                slicer.util.showStatusMessage(f"Missing Folders {missing_folders}")
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
        self.delayDisplay("Do not take the test")

# ---------------------------------------------------------------------------
# DICOM Processor
# ---------------------------------------------------------------------------

class DicomProcessor:
    def __init__(self):
        self.error = ""
        self.net = ""
        self.study_uid_map = defaultdict(str)
        self.series_uid_map = defaultdict(str)
        self.sop_uid_map = defaultdict(str)

    def is_dicom(self, file_path):
        import pydicom
        try:
            ds = pydicom.dcmread(file_path, force=True)
            try:
                ds.decompress()
                if self.checkCTmeta(ds) == 0:
                    return False
            except Exception:
                if self.checkCTmeta(ds) == 0:
                    return False
            return True
        except Exception as e:
            try:
                with open('log.txt', 'a') as error_file:
                    error_file.write(f"Error: {e}\n")
            except Exception:
                pass
            return False

    def is_dicom_nometa(self, file_path):
        import pydicom
        try:
            ds = pydicom.dcmread(file_path, force=True)
            ds.decompress()
            return True
        except Exception:
            return False

    def list_dicom_directories(self, root_dir):
        dicom_dirs = set()
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if self.is_dicom(file_path):
                    dicom_dirs.add(root)
                    break
        return list(dicom_dirs)

    def load_scan(self, path):
        import pydicom
        p = Path(path)
        if p.is_file():
            slices = pydicom.dcmread(str(p), force=True)
            return slices
        raise FileNotFoundError(f"Not a file: {path}")

    def get_pixels_hu(self, slices):
        image = slices.pixel_array.astype(np.int16)
        image[image <= -2000] = 0
        intercept = getattr(slices, "RescaleIntercept", 0)
        slope = getattr(slices, "RescaleSlope", 1)
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    def binarize_volume(self, volume, air_hu=-800):
        binary_volume = np.zeros_like(volume, dtype=np.uint8)
        binary_volume[volume <= air_hu] = 1
        return binary_volume

    def largest_connected_component(self, binary_image):
        import cv2
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(binary_image, dtype=np.uint8)
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_component_image = np.zeros(labels.shape, dtype=np.uint8)
        largest_component_image[labels == largest_component_index] = 1
        return largest_component_image

    def get_largest_component_volume(self, volume):
        return self.largest_connected_component(volume)

    def dilate_volume(self, volume, kernel_size=KERNEL_SIZE):
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(volume.astype(np.uint8), kernel)

    def apply_mask_and_get_values(self, image_volume, mask_volume):
        masked_volume = image_volume * mask_volume
        unique_values = np.unique(masked_volume)
        unique_values = unique_values[(unique_values > -125) & (unique_values < 50)]
        return unique_values.tolist()

    def apply_random_values_optimized(self, pixels_hu, dilated_volume, unique_values_list):
        new_volume = np.copy(pixels_hu)
        new_volume[dilated_volume == 1] = -1000
        return new_volume

    def person_names_callback(self, ds, elem):
        if elem.VR == "PN":
            elem.value = "anonymous"

    def curves_callback(self, ds, elem):
        if elem.tag.group & 0xFF00 == 0x5000:
            del ds[elem.tag]

    def is_substring_in_list(self, substring, string_list):
        return any(substring in string for string in string_list)

    def checkCTmeta(self, ds):
        try:
            modality = ""
            if (0x08, 0x60) in ds:
                modality = ds[0x08, 0x60].value
            modality = [modality] if isinstance(modality, str) else modality
            modality = list(map(lambda x: str(x).lower().replace(' ', ''), modality))
            status1 = any(self.is_substring_in_list(c, modality) for c in ["ct", "computedtomography", "ctprotocal"])

            imageType = ""
            if (0x08, 0x08) in ds:
                imageType = ds[0x08, 0x08].value
            imageType = [imageType] if isinstance(imageType, str) else imageType
            imageType = list(map(lambda x: str(x).lower().replace(' ', ''), imageType))
            status2 = all(self.is_substring_in_list(c, imageType) for c in ["original", "primary", "axial"])

            studyDes = ""
            if (0x08, 0x1030) in ds:
                studyDes = ds[0x08, 0x1030].value
            elif (0x08, 0x103e) in ds:
                studyDes = ds[0x08, 0x103e].value
            elif (0x18, 0x15) in ds:
                studyDes = ds[0x18, 0x15].value
            elif (0x18, 0x1160) in ds:
                studyDes = ds[0x18, 0x1160].value
            studyDes = [studyDes] if isinstance(studyDes, str) else studyDes
            studyDes = list(map(lambda x: str(x).lower().replace(' ', ''), studyDes))
            include = ["head", "brain", "skull"]
            exclude = ["angio", "cta", "perfusion"]
            status3 = any(self.is_substring_in_list(c, studyDes) for c in include)
            if any(self.is_substring_in_list(e, studyDes) for e in exclude):
                status3 = False

            return int(status1 and status2 and status3)
        except Exception as e:
            self.error = str(e)
        return 0

    def save_new_dicom_files(self, original_dir, out_dir, replacer='face', id='new_folder_name',
                             patient_id='0', new_patient_id='Processed for anonymization', remove_text=False):
        import cv2
        import pydicom
        from pydicom.uid import generate_uid
        from pydicom.datadict import keyword_for_tag

        dicom_files = [f for f in os.listdir(original_dir) if self.is_dicom(os.path.join(original_dir, f))]
        errors = []
        try:
            dicom_files.sort(
                key=lambda x: int(pydicom.dcmread(os.path.join(original_dir, x), force=True).get("InstanceNumber", 1))
            )
        except Exception as e:
            self.error = e

        for i, dicom_file in enumerate(dicom_files, start=1):
            try:
                ds = self.load_scan(os.path.join(original_dir, dicom_file))
                try:
                    ds.decompress()
                except Exception as e:
                    self.error = e

                ds.remove_private_tags()
                if "OtherPatientIDs" in ds:
                    delattr(ds, "OtherPatientIDs")
                if "OtherPatientIDsSequence" in ds:
                    del ds.OtherPatientIDsSequence
                ds.walk(self.person_names_callback)
                ds.walk(self.curves_callback)

                ANONYMOUS = "anonymous"
                today = time.strftime("%Y%m%d")

                # Required identifiers
                ds.add_new((0x08, 0x0050), 'SH', id) if (0x08, 0x0050) not in ds else setattr(ds[0x08, 0x0050], "value", id)
                ds.add_new((0x10, 0x0020), 'LO', ANONYMOUS) if (0x10, 0x0020) not in ds else setattr(ds[0x10, 0x0020], "value", ANONYMOUS)
                ds.add_new((0x10, 0x0010), 'PN', 'Processed for anonymization') if (0x10, 0x0010) not in ds else setattr(ds[0x10, 0x0010], "value", 'Processed for anonymization')

                # Scrub a list of tags
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
                    (0x4008, 0x010A), (0x4008, 0x010C), (0x4008, 0x0114), (0x0032, 0x1033)
                ]
                from pydicom.datadict import keyword_for_tag
                for tag in requirement_tags:
                    if tag in ds:
                        tag_name = keyword_for_tag(tag)
                        tag_vr = ds[tag].VR
                        if "ID" in tag_name:
                            ds[tag].value = "0"
                        elif tag_vr == "DA":
                            ds[tag].value = "00010101"
                        else:
                            ds[tag].value = "anonymous"

                # Regenerate key UIDs
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

                # Normalize a few common tags
                DICOM_TAGS = {
                    "TimezoneOffset": (0x0008, 0x0201),
                    "Country": (0x0010, 0x2150),
                    "Region": (0x0010, 0x2152),
                    "CurrentLocation": (0x0038, 0x0300),
                    "InstitutionName": (0x0008, 0x0080),
                    "InstitutionAddress": (0x0008, 0x0081),
                }
                for tag in DICOM_TAGS.values():
                    if tag in ds:
                        ds[tag].value = "anonymous"

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
                    "NATIVE INDIAN": "Asian"
                }
                if RACE_TAG in ds:
                    race_value = str(ds[RACE_TAG].value).strip().upper() if ds[RACE_TAG].value else "Other"
                    ds[RACE_TAG].value = RACE_MAPPING.get(race_value, "Other")

                # Pixel ops
                pixels_hu = self.get_pixels_hu(ds)
                binarized_volume = self.binarize_volume(pixels_hu)
                processed_volume = self.get_largest_component_volume(binarized_volume)
                dilated_volume = self.dilate_volume(processed_volume, random.randint(30, 40))

                if replacer == 'face':
                    unique_values_list = self.apply_mask_and_get_values(pixels_hu, dilated_volume - processed_volume)
                elif replacer == 'air':
                    unique_values_list = [0]
                else:
                    try:
                        replacer = int(replacer)
                        unique_values_list = [replacer]
                    except Exception:
                        unique_values_list = self.apply_mask_and_get_values(pixels_hu, dilated_volume - processed_volume)

                new_volume = self.apply_random_values_optimized(pixels_hu, dilated_volume, unique_values_list)

                # OCR-based blackout (optional)
                if remove_text:
                    try:
                        import cv2
                        import easyocr
                        min_val = np.min(pixels_hu)
                        max_val = np.max(pixels_hu)
                        pixels_hu_255 = np.uint8(((pixels_hu - min_val) / max(1e-6, (max_val - min_val))) * 255.0)
                        image = Image.fromarray(pixels_hu_255)
                        image = np.array(image)
                        if len(pixels_hu_255.shape) == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        reader = easyocr.Reader(['en'])
                        results = reader.readtext(image)
                        for (bbox, text, prob) in results:
                            if prob > 0.8:
                                top_left = tuple(map(int, bbox[0]))
                                bottom_right = tuple(map(int, bbox[2]))
                                cv2.rectangle(new_volume, top_left, bottom_right, (0, 0, 0), thickness=cv2.FILLED)
                        del reader, results
                    except Exception as e:
                        errors.append((dicom_file, str(e)))

                # Save
                slope = getattr(ds, "RescaleSlope", 1)
                intercept = getattr(ds, "RescaleIntercept", 0)
                new_slice = (new_volume - intercept) / max(1e-6, slope)
                ds.PixelData = new_slice.astype(np.int16).tobytes()
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1
                new_file_name = f"{id}_{i:05d}.dcm"
                final_file_path = os.path.join(out_dir, new_file_name)
                ds.save_as(final_file_path, write_like_original=False)
                del ds, pixels_hu, new_volume
            except Exception as e:
                errors.append((dicom_file, str(e)))

        if errors:
            try:
                with open(os.path.join(out_dir, 'log.txt'), 'a') as error_file:
                    for dicom_file, error in errors:
                        error_file.write(f"File: {dicom_file}, Error: {error}\n")
            except Exception:
                pass

        return errors

    def drown_volume(self, in_path, out_path, replacer='face', id='new_folder_name',
                     patient_id='0', name="", remove_text=False, remove_CTA=False):
        """
        Phase 1: process while mirroring input structure.
        Phase 2: rename subdirectories at each level to <id>_<n> (keep nesting), using os.rename only.
        """
        import os

        try:
            # Phase 1: processing (mirror structure as-is)
            for root, dirs, files in os.walk(in_path):
                rel = os.path.relpath(root, in_path)
                out_dir = os.path.join(out_path, rel)
                dicom_files = [f for f in files if self.is_dicom(os.path.join(root, f))]
                if dicom_files:
                    os.makedirs(out_dir, exist_ok=True)
                    _ = self.save_new_dicom_files(
                        original_dir=root,
                        out_dir=out_dir,
                        replacer=replacer,
                        id=id,
                        patient_id=patient_id,
                        new_patient_id='Processed for anonymization',
                        remove_text=remove_text
                    )
                    gc.collect()

            # Phase 2: rename folders to <id>_<n> (keep nesting)
            for curr, subdirs, files in os.walk(out_path, topdown=True):
                if not subdirs:
                    continue
                subdirs_sorted = sorted(subdirs)

                # temp rename to avoid clashes
                for i, d in enumerate(subdirs_sorted, start=1):
                    src = os.path.join(curr, d)
                    tmp = os.path.join(curr, f"__TMP__{i}__")
                    if os.path.exists(src):
                        os.rename(src, tmp)

                # final rename to <id>_<i>
                new_names = []
                for i, _ in enumerate(subdirs_sorted, start=1):
                    tmp = os.path.join(curr, f"__TMP__{i}__")
                    dst_name = f"{id}_{i}"
                    dst = os.path.join(curr, dst_name)
                    os.rename(tmp, dst)
                    new_names.append(dst_name)

                # ensure walker descends into the renamed dirs
                subdirs[:] = new_names

        except Exception as e:
            try:
                os.makedirs(out_path, exist_ok=True)
                with open(os.path.join(out_path, 'log.txt'), 'a') as f:
                    f.write(f"Error: {e}\n")
            except Exception:
                pass
            return 0

        return 1
