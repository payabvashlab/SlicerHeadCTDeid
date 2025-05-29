cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerHeadCTDeid)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/payabvashlab/SlicerHeadCTDeid")
set(EXTENSION_CONTRIBUTORS "Sam Payabvash (Columbia University)")
set(EXTENSION_DESCRIPTION "Head CT DICOM De identification ")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/payabvashlab/SlicerHeadCTDeid/refs/heads/main/HeadCTDeid/Resources/Icons/HeadCTDeid.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/payabvashlab/SlicerHeadCTDeid/refs/heads/main/screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(HeadCTDeid)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
