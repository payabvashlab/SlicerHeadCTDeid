import logging
import os
import time
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from ctk import ctkFileDialog
import site
site.main()  # Refresh sys.path and .pth files
from datetime import datetime
import time
import shutil
from pathlib import Path
from datetime import datetime
import sys
import importlib
from PIL import Image
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')
FACE_MAX_VALUE = 50
FACE_MIN_VALUE = -125

AIR_THRESHOLD = -800
import random

KERNEL_SIZE = random.randint(30, 40)
ERROR = ""

class HeadCTDeid(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Head CT de-identification"  # Human-readable title
        self.parent.categories = ["Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = ["Anh Tuan Tran, Sam Payabvash"]
        self.parent.helpText = """
This module de-identifies DICOM files by removing patient information based on a given list of patients.
"""
        self.parent.acknowledgementText = """
This file was developed by Anh Tuan Tran, Sam Payabvash (Columbia University).
"""


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
        self.ui.deidentifyCheckbox.connect('toggled(bool)', self.updateParameterNodeFromGUI)  # Handle checkbox state

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

        # Check if all required fields have values and enable the "Ally" button
        if len(self._parameterNode.GetParameter("InputFolder")) > 1 and len(
                self._parameterNode.GetParameter("ExcelFile")) > 4 and len(
            self._parameterNode.GetParameter("OutputFolder")) > 1 and self._parameterNode.GetParameter(
            "ExcelFile") != "Browse":
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
        self._parameterNode.SetParameter("Deidentify",
                                         str(self.ui.deidentifyCheckbox.isChecked()).lower())  # Set checkbox state

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        try:
            slicer.util.infoDisplay(
                "This tools is work in progress being validated in project. Contact sp4479@columbia.edu for more details. Use at your own risk.",
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
                self.ui.progressBar
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Error: {str(e)}")

    def onBrowseExcelFile(self):
        fileDialog = ctkFileDialog()
        fileDialog.setWindowTitle("Select Excel/CSV File")
        fileDialog.setNameFilters(["Excel Files (*.xlsx)", "CSV Files (*.csv)", "All Files (*)"])
        fileDialog.setFileMode(ctkFileDialog.ExistingFile)  # Ensure only existing files can be selected
        fileDialog.setOption(ctkFileDialog.DontUseNativeDialog, False)

        # Execute the dialog and get the selected file
        if fileDialog.exec_():  # If the user clicks 'OK'
            selectedFile = fileDialog.selectedFiles()[0]  # Get the first selected file
            self.ui.excelFileButton.text = selectedFile
            # Set full path in parameter node
            self._parameterNode.SetParameter("ExcelFile", selectedFile)

            # Trigger GUI update
            self.updateGUIFromParameterNode()


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
          #subprocess.check_call([sys.executable, "-m", "pip", "install", package])
          slicer.util.pip_install(package)
        
        try:
            import pandas
        except ModuleNotFoundError as e:
            slicer.util.pip_install("pandas==2.2.3")
        try:
            import openpyxl
        except ModuleNotFoundError as e:
            slicer.util.pip_install("openpyxl")
        try:
            import pydicom
        except ModuleNotFoundError as e:
            slicer.util.pip_install("pydicom")
            slicer.util.pip_install("pylibjpeg")
            slicer.util.pip_install("pylibjpeg-libjpeg")
            slicer.util.pip_install("pylibjpeg-openjpeg")
        
        try:
            import cv2
        except ModuleNotFoundError as e:
            slicer.util.pip_install("opencv-python")
          
        packageName = "scikit-image"
        if not self._checkModuleInstalled(packageName):
          install(packageName)

        packageName = "easyocr"
        if not self._checkModuleInstalled(packageName):
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

    def process(self, inputFolder, excelFile, outputFolder, remove_text, progressBar):

        if not os.path.exists(inputFolder):
            raise ValueError(f"Input folder does not exist: {inputFolder}")
        if not os.path.exists(excelFile):
            raise ValueError(f"Excel/CSV file does not exist: {excelFile}")
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        columns_as_text = ['Accession_number', 'New_ID'] 
        import pandas
        # Get file extension (lowercase)
        ext = os.path.splitext(excelFile)[1].lower()
        # Read the appropriate file type
        if ext == '.csv':
            df = pandas.read_csv(excelFile, dtype={col: str for col in columns_as_text})
        elif ext in ['.xlsx', '.xls']:
            df = pandas.read_excel(excelFile, dtype={col: str for col in columns_as_text})
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        if ("Accession_number" not in df.columns) or ("New_ID" not in df.columns):
            raise ValueError("Excel file must contain a 'Accession_number' and 'New_ID' column")
            return 0
        else:
            try:
                log_file = os.path.join(outputFolder, "patient_processing.log")
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"Initialized patient processing module {log_file}")
            except Exception as e:
                self.logger.info(e)
            df['Accession_number'] = df['Accession_number'].astype(str).str.strip()
            df['New_ID'] = df['New_ID'].astype(str).str.strip()
            id_mapping = dict(zip(df['Accession_number'], df['New_ID']))
            dicom_folders = [d for d in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, d))]
            total_rows = df.shape[0]
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(outputFolder, f'Processed for anonymization_{current_time}')
            os.makedirs(out_path, exist_ok=True)
            total_time = 0
            number_successul = 0
            for i, foldername in enumerate(dicom_folders):
                if (foldername in id_mapping):
                    dst_folder = ""
                    try:
                        start_time = time.time()
                        dst_folder = os.path.join(out_path, id_mapping[foldername])
                        processor = DicomProcessor()
                        src_folder = os.path.join(inputFolder, foldername)
                        result = processor.drown_volume(src_folder, dst_folder, 'face', id_mapping[foldername], f"Processed for anonymization {id_mapping[foldername]}", remove_text)
                        progressBar.setValue(int((i + 1)* 100/ total_rows))
                        slicer.util.showStatusMessage(f"Finished processing foldername {foldername}")
                        self.logger.info(f"Finished processing folder: {foldername}")
                        elapsed = time.time() - start_time
                        total_time += elapsed
                        number_successul = number_successul + 1
                    except Exception as e:
                        self.logger.error(f"Error processing folder {foldername}: {str(e)}")
                        if os.path.exists(dst_folder):
                            shutil.rmtree(dst_folder)
            average_time = total_time*1.0 /number_successul
            self.logger.info(f"Time processing each folder: {average_time}")
            try:
                folder_list_2 = df['Accession_number'].tolist()  # Convert to string (in case of numbers)
                actual_folders = dicom_folders  # Get folder names in directory
                missing_folders = [folder for folder in folder_list_2 if folder not in actual_folders]
                if len(missing_folders) > 0:
                    self.logger.error(f"Missing Folders {missing_folders}")
                    slicer.util.showStatusMessage(f"Missing Folders {missing_folders}")
            except Exception as e:
                self.logger.error(f"Error processing folder {foldername}: {str(e)}")

class HeadCTDeidTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_HeadCTDeid1()

    def test_HeadCTDeid1(self):
        self.delayDisplay("Do not take the test")


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
                with open('log.txt', 'a') as error_file:
                    error_file.write(f"Error: {e}\n")
                return 0
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
                else:
                    break
        return list(dicom_dirs)

    def load_scan(self, path):
        import pydicom
        p = Path(path)
        if p.is_file():
            slices = pydicom.dcmread(str(p), force=True)
        return slices

    def get_pixels_hu(self, slices):
        image = slices.pixel_array.astype(np.int16)
        image[image <= -2000] = 0
        intercept = slices.RescaleIntercept
        slope = slices.RescaleSlope
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
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_component_image = np.zeros(labels.shape, dtype=np.uint8)
        largest_component_image[labels == largest_component_index] = 1
        return largest_component_image

    def get_largest_component_volume(self, volume):
        processed_volume = self.largest_connected_component(volume)
        return processed_volume

    def dilate_volume(self, volume, kernel_size=KERNEL_SIZE):
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_volume = cv2.dilate(volume.astype(np.uint8), kernel)
        return dilated_volume

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
            modality = list(map(lambda x: x.lower().replace(' ', ''), modality))
            check = ["ct", "computedtomography", "ctprotocal"]
            status1 = any(self.is_substring_in_list(c, modality) for c in check)
            imageType = ""
            if (0x08, 0x08) in ds:
                imageType = ds[0x08, 0x08].value
            imageType = [imageType] if isinstance(imageType, str) else imageType
            imageType = list(map(lambda x: x.lower().replace(' ', ''), imageType))
            check = ["original", "primary", "axial"]
            status2 = all(self.is_substring_in_list(c, imageType) for c in check)

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
            studyDes = list(map(lambda x: x.lower().replace(' ', ''), studyDes))
            check = ["head", "brain", "skull"]
            status3 = 1#any(self.is_substring_in_list(c, studyDes) for c in check)

            return int(status1 and status2 and status3)
        except Exception as e:
            self.error = str(e)
        return 0

    def save_new_dicom_files(self, original_dir, out_dir, replacer='face', id='New_ID', patient_id='0', new_patient_id='Processed for anonymization', remove_text=False):
        import cv2
        import pydicom
        from pydicom.uid import generate_uid
        from pydicom.datadict import keyword_for_tag

        dicom_files = [f for f in os.listdir(original_dir) if self.is_dicom(os.path.join(original_dir, f))]
        errors = []
        try:
            dicom_files.sort(
                key=lambda x: int(pydicom.dcmread(os.path.join(original_dir, x), force=True).InstanceNumber))
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
                # requirement tag
                #Accession Number
                if (0x08, 0x50) not in ds:
                    ds.add_new((0x08, 0x50), 'SH', id)
                else:
                    ds[0x08, 0x50].value = id
                # Patient's ID     
                if (0x10, 0x20) not in ds:
                    ds.add_new((0x10, 0x20), 'LO', ANONYMOUS)
                else:
                    ds[0x10, 0x20].value = ANONYMOUS    
                # Patient's Name           
                if (0x10, 0x10) not in ds:
                    ds.add_new((0x10, 0x10), 'PN', "Processed for anonymization")
                else:
                    ds[0x10, 0x10].value = "Processed for anonymization"
                # requirement tag
                requirement_tags = [(0x10, 0x1000),  # Other Patient IDs
                                 (0x10, 0x1001),  # Other Patient Names
                                 (0x10, 0x1005),  # Patient's Birth Name
                                 (0x10, 0x1040),  # Patient's Address
                                 (0x10, 0x2154),  # Patient's Telephone Numbers
                                 (0x10, 0x2295),  # Breed Registration Number
                                 (0x12, 0x20),  # Clinical Trial Protocol ID
                                 (0x12, 0x30),  # Clinical Trial Site ID
                                 (0x12, 0x40),  # Clinical Trial Subject ID
                                 (0x12, 0x42),  # Clinical Trial Subject Reading ID
                                 (0x12, 0x71),  # Clinical Trial Series ID
                                 (0x18, 0x9445),  # (no description - NEMA placeholder)
                                 (0x20, 0x0010),  # Study ID
                                 (0x20, 0x9056),  # Stack ID
                                 (0x32, 0x0A),  # Study Status ID
                                 (0x32, 0x0C),  # Study Priority ID
                                 (0x32, 0x12),  # Study ID Issuer
                                 (0x38, 0x08),  # Visit Status ID
                                 (0x38, 0x10),  # Admission ID
                                 (0x38, 0x0400),  # Patient's Institution Residence
                                 (0x40, 0x31),  # Local Namespace Entity ID
                                 (0x40, 0x32),  # Universal Entity ID
                                 (0x40, 0x33),  # Universal Entity ID Type
                                 (0x40, 0x2016),  # Placer Order Number
                                 (0x40, 0x2017),  # Filler Order Number
                                 (0x40, 0xA123),  # Person Name
                                 (0x70, 0x80),  # Content Label
                                 (0x0400, 0x0005),  # MAC ID Number
                                 (0x0400, 0x0020),  # Data Elements Signed
                                 (0x0400, 0x0564),  # Source of Previous Values
                                 (0x300A, 0x0182),  # Patient Setup Number
                                 (0x4008, 0x0040),  # Results ID
                                 (0x4008, 0x0119),  # Distribution Name
                                 (0x4008, 0x011A),  # Distribution Address
                                 (0x4008, 0x0210),  # Interpretation ID
                                 (0x4008, 0x0212),  # Interpretation Status ID
                                 (0x10, 0x30),  # Patient's Birth Date
                                 (0x10, 0x2298),  # Responsible Person Role
                                 (0x10, 0x0201),  # Timezone Offset From UTC
                                 (0x0010, 0x2298),  # Responsible Person Role
                                 (0x0012, 0x0060),  # Clinical Trial Coordinating Center Name
                                 (0x0038, 0x0011),  # Issuer of Admission ID
                                 (0x0040, 0x0001),  # Scheduled Station AE Title
                                 (0x0040, 0x0010),  # Scheduled Station Name
                                 (0x0040, 0x0035),  # Identifier Type Code
                                 (0x0040, 0x0241),  # Performed Station AE Title
                                 (0x0040, 0x0242),  # Performed Station Name
                                 (0x0040, 0x1010),  # Names of Intended Recipients of Results
                                 (0x0040, 0x2008),  # Order Entered By
                                 (0x0040, 0x2009),  # Order Enterer's Location
                                 (0x0040, 0x2010),  # Order Callback Phone Number
                                 (0x0040, 0xA075),  # Verifying Observer Name
                                 (0x0070, 0x0084),  # Content Creator's Name
                                 (0x0088, 0x0130),  # Storage Media File-set ID
                                 (0x0400, 0x0115),  # Certificate of Signer
                                 (0x0400, 0x0120),  # Signature
                                 (0x3006, 0x00A6),  # ROI Interpreter
                                 (0x4008, 0x010A),  # Interpretation Transcriber
                                 (0x4008, 0x010C),  # Interpretation Author
                                 (0x4008, 0x0114),  # Physician Approving
                                 (0x0032, 0x1033)   # RequestingService
                                ]
                for tag in requirement_tags:
                    if tag in ds:
                        tag_name = keyword_for_tag(tag)
                        tag_vr = ds[tag].VR  # Check VR type
                        # Check tag name and VR type
                        if "ID" in tag_name:
                            ds[tag].value = "0"
                        elif tag_vr == "DA":  # If VR is Date
                            ds[tag].value = "00010101"  # Valid DA value
                        else:
                            ds[tag].value = ANONYMOUS
                            

                # requirement tag
                uid_tags = [(0x0008, 0x0014),  # Instance Creator UID
                                 #(0x0008, 0x0018),  # SOP Instance UID
                                 (0x0008, 0x010C),  # Coding Scheme UID
                                 (0x0008, 0x010D),  # Context Group Extension Creator UID
                                 (0x0008, 0x1150),  # Referenced SOP Class UID
                                 (0x0008, 0x1155),  # Referenced SOP Instance UID
                                 (0x0008, 0x3010),  # Irradiation Event UID
                                 (0x0008, 0x9123),  # Creator-Version UID
                                 #(0x0020, 0x000D),  # Study Instance UID
                                 #(0x0020, 0x000E),  # Series Instance UID
                                 (0x0020, 0x0052),  # Frame of Reference UID
                                 (0x0020, 0x0200),  # Synchronization Frame of Reference UID
                                 (0x0020, 0x9164),  # Dimension Organization UID
                                 (0x0040, 0xA124),  # UID
                                 (0x0088, 0x0140),  # Storage Media File-set UID
                                 (0x0400, 0x0010),  # MAC Calculation Transfer Syntax UID
                                 (0x0400, 0x0100),  # Digital Signature UID
                                 (0x3006, 0x0024),  # Referenced Frame of Reference UID
                                 (0x3006, 0x00C2),  # Related Frame of Reference UID
                                ]
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
                    sop_uid = str (ds[0x0008, 0x0018].value)
                    if sop_uid and sop_uid not in self.sop_uid_map:
                        self.sop_uid_map[sop_uid] = generate_uid()
                    ds[0x0008, 0x0018].value = self.sop_uid_map.get(sop_uid, generate_uid())


                for tag in uid_tags:
                    if tag in ds:
                        original_value = str(ds[tag].value)
                        # Validate original UID format
                        try:
                            if len(original_value) <= 64 and all(part.isdigit() for part in original_value.split('.')):
                                # Generate new UID
                                ds[tag].value = generate_uid()
                            else:
                                ds[tag].value = '0.0.0'
                        except Exception as e:
                             ds[tag].value = '0.0.0'

                # Patient's Birth Date
                if (0x10, 0x30) in ds:
                    ds[0x10, 0x30].value = today
                # Patient's Sex Neutered
                if (0x0010, 0x2203) in ds and ds[(0x0010, 0x2203)].value.lower() == "unknown":
                    ds[(0x0010, 0x2203)].value = ""

                # DICOM Tags to check
                DICOM_TAGS = {
                    "TimezoneOffset": (0x0008, 0x0201),  # Timezone Offset From UTC
                    "Country": (0x0010, 0x2150),  # Country of Residence
                    "city_tag": (0x0010, 0x2152),  # Region of Residence
                    "state": (0x0038, 0x0300),  # Current Patient Location
                }
                for tag in DICOM_TAGS.values():
                    if tag in ds:
                        ds[tag].value = ANONYMOUS
                # DICOM Tags to Read and Modify
                DICOM_TAGS = {
                    "RetrieveAETitle": (0x0008, 0x0054),  # Retrieve AE Title
                    "ReferringPhysicianName": (0x0008, 0x0090),  # Referring Physician's Name
                    "ReferringPhysicianAddress": (0x0008, 0x0092),  # Referring Physician's Address
                    "ReferringPhysicianPhone": (0x0008, 0x0094),  # Referring Physician's Telephone Numbers
                    "StationName": (0x0008, 0x1010),  # Station Name
                    "PhysiciansOfRecord": (0x0008, 0x1048),  # Physician(s) of Record
                    "PerformingPhysicianName": (0x0008, 0x1050),  # Performing Physician's Name
                    "ReadingPhysicians": (0x0008, 0x1060),  # Name of Physician(s) Reading Study
                    "OperatorsName": (0x0008, 0x1070),  # Operators' Name
                    "IssuerOfPatientID": (0x0010, 0x0021),  # Issuer of Patient ID
                    "ResponsibleOrganization": (0x0010, 0x2299),  # Responsible Organization
                    "ClinicalTrialSponsor": (0x0012, 0x0010),  # Clinical Trial Sponsor Name
                    "ClinicalTrialSiteName": (0x0012, 0x0031),  # Clinical Trial Site Name
                    "city": (0x0008, 0x0080),  # Institution Name
                    "state": (0x0008, 0x0081),  # Institution Address
                }
                for tag in DICOM_TAGS.values():
                    if tag in ds:
                        ds[tag].value = ANONYMOUS
                # Ethnic Group
                if (0x0010, 0x2160) in ds and ds[(0x0010, 0x2160)].value.lower() == "unknown":
                    ds[(0x0010, 0x2160)].value = ""
                """consolidates Race (0010,2201), and saves the updated file."""
                RACE_TAG = (0x0010, 0x2201)  # Patient's Race
                # Race mapping to consolidated categories
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
                    race_value = ds[RACE_TAG].value.strip().upper() if ds[RACE_TAG].value else "Other"
                    ds[RACE_TAG].value = RACE_MAPPING.get(race_value, "Other")
                                
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
                    except:
                        unique_values_list = self.apply_mask_and_get_values(pixels_hu,
                                                                            dilated_volume - processed_volume)
                new_volume = self.apply_random_values_optimized(pixels_hu, dilated_volume, unique_values_list)
                if remove_text == True:
                    #draw text
                    try:
                        min_val = np.min(pixels_hu)
                        max_val = np.max(pixels_hu)
                        pixels_hu_255 = np.uint8(((pixels_hu - min_val) / (max_val - min_val)) * 255.0)

                        image = Image.fromarray(pixels_hu_255)
                        """draw = ImageDraw.Draw(image)
                        try:
                            font = ImageFont.truetype("arial.ttf", 20)
                        except IOError:
                            font = ImageFont.load_default()
                        draw.text((100, 100), "Patient: Nguyen Van A", fill="white", font=font)
                        draw.text((150, 200), "DB:01/01/2000", fill="white", font=font)
                        draw.text((200, 300), "Address= USA", fill="white", font=font)"""
                        image = np.array(image)
                        
                        if len(pixels_hu_255.shape) == 2:  # Grayscale
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                        # Perform OCR on the image
                        import easyocr
                        reader = easyocr.Reader(['en'])
                        results = reader.readtext(image)

                        for (bbox, text, prob) in results:
                            if prob > 0.8:  # Confidence threshold
                                (top_left, bottom_right) = (tuple(map(int, bbox[0])), tuple(map(int, bbox[2])))
                                cv2.rectangle(new_volume, top_left, bottom_right, (0, 0, 0), thickness=cv2.FILLED)  # Black out
                        del reader, results
                    except Exception as e:
                        errors.append((dicom_file, str(e)))
                new_slice = (new_volume - ds.RescaleIntercept) / ds.RescaleSlope
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
            with open(os.path.join(out_dir, 'log.txt'), 'a') as error_file:
                for dicom_file, error in errors:
                    error_file.write(f"File: {dicom_file}, Error: {error}\n")

        return errors

    def drown_volume(self, in_path, out_path, replacer='face', id='New_ID', patient_id='0', name='Processed for anonymization',
                     remove_text=False):
        try:
            error=""
            for root, dirs, files in os.walk(in_path):
                relative_path = os.path.relpath(root, in_path)
                out_dir = os.path.join(out_path, relative_path)
                dicom_files = [f for f in files if self.is_dicom(os.path.join(root, f))]
                if dicom_files:
                    os.makedirs(out_dir, exist_ok=True)
                    error = self.save_new_dicom_files(root, out_dir, replacer, id, patient_id, 'Processed for anonymization', remove_text)
                    gc.collect()
        except Exception as e:
            with open(os.path.join(out_dir, 'log.txt'), 'a') as error_file:
                error_file.write(f"Error: {e}\n")
            return 0
        return 1
