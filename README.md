# Head CT de-identification tool
<h2>Summary</h2>

Approximately 15% to 30% of CT scans performed annually in the United States are head CTs [1, 2]. As a rapid and widely accessible modality, head CT is the first line of imaging to evaluate acute brain injury, cerebrovascular accidents, altered mental status, and post-procedural monitoring. Sharing head CT scans across institutions can facilitate the creation of large datasets for training deep learning models to guide treatment decisions in acute clinical settings.

A critical step for medical image sharing is removal of Protected Health Information (PHI) and Personally Identifiable Information (PII) to safeguard patient privacy and comply with HIPAA regulations. In head CT scans, personal and medical information are included in the DICOM file metadata [3]. Additionally, some scans may contain burned-in text displaying PHI/PII directly on the image. Three-dimensional reconstructions of volumetric brain CTs can also reveal facial features that may compromise patient privacy [4].

This 3D Slicer extension removes PHI from head CT DICOM metadata following the DICOM PS3.15 Attribute Confidentiality Profiles [5], detects burned-in text and masks the affected image regions, and strips superficial facial tissue at the air–skin interface to prevent facial feature recognition in 3D reconstructed head CTs. This project was in part supported by the American Heart Association (AHA) Stroke Image Sharing Consortium:
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

- Step 2: De-identify DICOM metadata. Attributes are handled in three different ways following the PS3.15 Basic Application Level Confidentiality Profile. The three actions produce three different results in the output file, and the complete tag lists are given in the documents folder

- Step 3: Replace dates. Every attribute with a Value Representation of 'DA' or 'DT' is replaced with the anonymization date, wherever it occurs, including private vendor attributes and attributes nested inside sequences.

- Step 4: Remove facial features. Morphology-based image processing [4] identifies the skin–air interface using air-level attenuation, and superficial soft tissue is removed by dilation with a kernel of 50–60 mm diameter, corresponding to approximately 25–30 mm of tissue removed from the surface. The kernel size is randomly varied per slice to impede reconstruction of the facial surface by reversing the pipeline. Dilation stops at bone-level attenuation so that the skull is preserved.

- Step 5: Mask burned-in text. Images are screened with Florence-2, a vision-language model, and any detected text region is masked to air attenuation (−1000 HU). The slice is retained; no image is discarded because of detected text. Masking is applied after facial tissue removal.

<h2>Capabilities and constrains:</h2>


•	This tool allows automatic batch de-identification of head CTs. However, the DICOM files of individual patients should be saved in separate folders/directories.

•	The complete lists of DICOM tags, separated by action, are provided in the *documents* section. Please be aware that the patient's sex, age, weight, race and ethnicity tags are retained. This is intentional, to allow future analysis of the demographic composition of de-identified datasets. Ethnic Group (0010,2160) may be quasi-identifying in small cohorts and can be removed by setting PS315_REMOVE_ETHNIC_GROUP = True.

•	This application will replace the patient identifier (typically scan accession numbers) with a new set of IDs provided in an Excel sheet or CSV file as an input.

•	The program identifies, anonymizes, and stores "axial" head CT DICOMs — removing any reconstructed series or additional scout or report files.

•	The pipeline relies on accurate labeling of "modality" (0008,0060), "image type" (0008,0008), and "Study description" (0008,1030) in meta-data of DICOM files. If these tags are mislabeled during head CT acquisition or removed during retrieval, the DICOM files will be excluded from the de-identification process.

•	The tool removes approximately 25–30 mm of superficial soft tissue from the skin–air interface. In rare cases of craniectomy without cranioplasty, where brain tissue lies close to the skin–air interface, a portion of the outer brain may be removed.

•	Burned-in text is masked in place, so a false-positive detection costs a small masked region rather than the loss of an image. This differs from earlier versions of this tool, which removed the entire DICOM file.

•	Because Florence-2 generates text rather than transcribing it, detections are filtered on generation confidence, minimum alphanumeric content, box size, and heuristics for hallucinated output.

•	The de-identification performed is declared in the output file in Patient Identity Removed (0012,0062), De-identification Method (0012,0063) and De-identification Method Code Sequence (0012,0064), the last listing only the profile options actually applied.

•	This tool implements a de-identification policy; it is not a formal PS3.15 conformance checker. Its built-in quality-control pass shares the assumptions of the implementation it validates. Independent validation of the output is recommended before external data sharing.


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

2.	<b>Excel File</b>: The Excel file should contain two columns with the following headers in the first row: <b>original_folder_name</b> and <b>new_folder_name</b>. Each "original_folder_name" must match a patient folder name in the input directory. The application will treat each "original_folder_name" as a unique patient identifier, use it to locate and process the corresponding folder, and then rename the folder using the associated "new_folder_name" from the same row. Both "original_folder_name" and "new_folder_name" can be any combination of alphanumeric characters.

<img width="455" alt="list" src="https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/images/Fig S6.png" />

3.	<b>Output folde</b>r: The output folder specifies the directory where de-identified DICOM files will be saved. After de-identification, axial head CT DICOM files will be stored in a new set of folders, each renamed using the corresponding "new_folder_name" from the Excel file, replacing the original patient folder names. Additionally, the DICOM tag *Accession Number (0008,0050)* will be replaced by the "new_folder_name".

<img width="772" alt="folder" src="https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/images/folder.png" />


<b>Remove text inside dicom</b>: This feature examines for any burned-in text within the images and removes the corresponding DICOM files. While enabling this option will increase processing time, it is recommended for thorough de-identification of scans. 
