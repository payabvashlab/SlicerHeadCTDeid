# Head CT de-identification tool
<h2>Summary</h2>

Approximately 15% to 30% of CT scans performed annually in the United States are head CTs [1, 2]. As a rapid and widely accessible modality, head CT is the first line of imaging to evaluate acute brain injury, cerebrovascular accidents, altered mental status, and post-procedural monitoring. Sharing head CT scans across institutions can facilitate the creation of large datasets for training deep learning models to guide treatment decisions in acute clinical settings. 

A critical step for medical image sharing is removal of Protected Health Information (PHI) and Personally Identifiable Information (PII) to safeguard patient privacy and comply with HIPAA regulations. In head CT scans, personal and medical information are included in the DICOM file metadata [3]. Additionally, some scans may contain burned-in text displaying PHI/PII directly on the image. Three-dimensional reconstructions of volumetric brain CTs can also reveal facial features that may compromise patient privacy [4]. 

This 3D Slicer extension is designed to remove PHI from head CT DICOM metadata [5], detect and eliminate DICOM images with burned-in text, and strip superficial facial tissue at the air–skin interface to prevent facial feature recognition in 3D reconstructed head CTs. This project was in part supported by the American Heart Association (AHA) Stroke Image Sharing Consortium:
https://professional.heart.org/en/research-programs/aha-funding-opportunities/data-grant-stroke-images
https://newsroom.heart.org/news/sharing-brain-images-can-foster-new-neuroscience-discoveries

*Warning: This tool is a work in progress and is currently being validated as part of an AHA-funded research project. For more information, contact at4049@cumc.columbia.edu. Use at your own risk.*

References: 

1.	Cauley, K.A., Y. Hu, and S.W. Fielden, Head CT: Toward Making Full Use of the Information the X-Rays Give. AJNR Am J Neuroradiol, 2021. 42(8): p. 1362-1369.
2.	Sheppard, J.P., et al., Risk of Brain Tumor Induction from Pediatric Head CT Procedures: A Systematic Literature Review. Brain Tumor Res Treat, 2018. 6(1): p. 1-7.
3.	Clunie, D.A., et al., Report of the Medical Image De-Identification (MIDI) Task Group -- Best Practices and Recommendations. ArXiv, 2025.
4.	Collins, S.A., J. Wu, and H.X. Bai, Facial De-identification of Head CT Scans. Radiology, 2020. 296(1): p. 22.
5.	https://dicom.nema.org/medical/dicom/current/output/html/part15.html#chapter_E

<img width="1720" alt="face" src="https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/images/face.png" />


<h2>Axial head CT detection and de-identification algorithm:</h2>

In addition to removing PHI and PII, the head CT de-identification tool detects and excludes DICOM images from other imaging modalities or body regions based on the information in file meta-data, restricting the output to axial head CT series only. This reduces the risk of inadvertently transferring unrelated medical images and minimizes the computational resources required for data transfer and storage. 

Using the following steps, the application ensures that only axial head CT DICOM images are processed and saved; and any DICOM files from other modalities (e.g., MRI, PET, X-ray) or body parts (e.g., neck, abdomen) are excluded.

- Step 1: Check the DICOM file header meta-data to ensure that (1) modality is "ct" or "computedtomography" or "ctprotocal" AND (2) the ImageType is "original" and "primary" and "axial"; AND (3) the StudyDescription or SeriesDescription or BodyPartExamined or FilterType is "head" or "brain" or "skull”.

- Step 2: Remove PHI/PII from the DICOM file metadata by identifying the tags listed in the *DICOM header removal* PDF file (<a href="https://github.com/payabvashlab/SlicerDeid/blob/main/documents/dicomTags.pdf"> DICOM header removal.pdf </a>) and replacing them with the string “anonymous.” The patient name is replaced with “Processed for anonymization”.

- Step 3: Blurring of facial features using morphology-based image processing [4]. We will identify the skin–air interface based on air-level (-1000) Hounsfield Unit attenuation in CT scan. Superficial subcutaneous fat tissue is then removed using a kernel size of 30 to 40 voxels to prevent facial feature recognition in 3D reconstructions of the scan. Of note, the kernel size randomly varies between 30 and 40 for each slice to minimize the risk of facial reconstruction by reversing the steps of our pipeline code.

<h2>Capabilities and constrains:</h2>

•	This tool allows automatic batch de-identification of head CTs. However, the DICOM files of individual patients should be saved in separate folders/directories.

•	The list of DICOM tags containing PHI or PII that are removed by the tool are provided in dicomTags PDF in documents section. Please be aware that the patient’s sex, age (not DOB), race and ethnicity tags are retained. This was intentional to allow any future analysis of sex, age, race and ethnicity of de-identified scans ensuring diversity of subjects in future studies.

•	This application will replace patient identifier (typically scan accession numbers) with new set of IDs that are provided in an excel sheet or csv file as an input.

•	The program identifies, anonymize, and stores “axial” head CT DICOMs - removing any reconstructed series or additional scout or report files. This will reduce the need for storage of de-identified CTs and minimize the risk of including any patient identifier in accompanying files.

•	The pipeline relies on accurate labeling of “modality” (0008,0060), “image type” (0008,0008), and “Study description” (0008,1030) in meta-data of DICOM files. If these tags are mislabeled during Head CT acquisition or removed during retrieval, the DICOM files will be excluded in de-identification process.

•	The de-identification tool removes approximately 1 cm of superficial soft tissue from the skin–air interface. In rare cases of craniectomy without cranioplasty, where brain tissue lies less than one cm from the skin–air interface, a portion of the outer brain may be removed.

•	The application also applies an OCR (Optical Character Recognition)–based text detection and removes any DICOM with imprinted text character.

<h2>Installing the Slicer module</h2>

1.	Drag and drop a folder "deidXXX" to the Slicer application window.
2.	Select "Add Python scripted modules to the application" in the popup window, and click OK.
3.	Select which modules to add to load immediately and click Yes.
4.	The selected modules will be immediately loaded, installed in all libraries, and made available under: Modules/Utilities/Head CT Deidentification.


<h2>Uninstalling the Slicer module</h2>

1.	Under the Edit menu, select the Application Setting.
2.	In Modules, Select Module Path and Arrow on the right to remove.
3.	Select Remove
4.	Click Ok and Restart the Slicer

<h2>Running the application</h2>

The application requires three inputs: the address of folder that contains the DICOM files; the list folder names containing the head CT of each patient; direction of folder to save the de-identified files.

<img width="1181" alt="application" src="https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/images/application.png" />

1.	<b>Input folder</b>: The input folder should directly contain individual patient folders that include corresponding DICOM files. The application treats each folder within the input folder address as one patient, using the folder name as the patient identifier, and processes and saves the corresponding DICOM files accordingly. Therefore, DICOM files from different patients must not be stored in the same folder. Each patient folder may contain subfolders or non-DICOM files; the application will preserve the subfolder structure and save the de-identified DICOM files using the same organizational hierarchy as the input.

<img width="999" alt="input" src="https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/images/input.png" />

2.	<b>Excel File</b>: The Excel file should contain two columns with the following headers in the first row: <b>Accession_number</b> and <b>New_ID</b>. Each "accession number" must match a patient folder name in the input directory. The application will treat each "accession number" as a unique patient identifier, use it to locate and process the corresponding folder, and then rename the folder using the associated "New_ID" from the same row. Both "accession numbers" and "New_IDs" can be any combination of alphanumeric characters.

<img width="455" alt="list" src="https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/images/list.png" />

3.	<b>Output folde</b>r: The output folder specifies the directory where de-identified DICOM files will be saved. After de-identification, axial head CT DICOM files will be stored in a new set of folders, each renamed using the corresponding "New_ID" from the Excel file, replacing the original patient folder names. Additionally, the DICOM tag *Accession Number (0008,0050)* will be replaced by the "New_ID".

<img width="772" alt="folder" src="https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/images/folder.png" />


<b>Remove text inside dicom</b>: This feature examines for any burned-in text within the images and removes the corresponding DICOM files. While enabling this option will increase processing time, it is recommended for thorough de-identification of scans. 
