cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(HeadCTDeid)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/payabvashlab/SlicerHeadCTDeid")
set(EXTENSION_CATEGORY "Utilities")
set(EXTENSION_CONTRIBUTORS "Anh Tran (Sam Lab)")
set(EXTENSION_DESCRIPTION "Head CT De identification CT dicom")
set(EXTENSION_ICONURL "https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/HeadCTDeid/Resources/Icons/HeadCTDeid.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/payabvashlab/SlicerHeadCTDeid/blob/main/screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(deid)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
