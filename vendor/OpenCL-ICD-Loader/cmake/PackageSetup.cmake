set(CPACK_PACKAGE_VENDOR "khronos")

set(CPACK_DEBIAN_RUNTIME_DESCRIPTION "Generic OpenCL ICD Loader
OpenCL (Open Computing Language) is a multivendor open standard for
general-purpose parallel programming of heterogeneous systems that include
CPUs, GPUs and other processors.
.
This package contains an installable client driver loader (ICD Loader)
library that can be used to load any (free or non-free) installable client
driver (ICD) for OpenCL. It acts as a demultiplexer so several ICD can
be installed and used together.")

set(CPACK_DEBIAN_DEV_DESCRIPTION "OpenCL development files
OpenCL (Open Computing Language) is a multivendor open standard for
general-purpose parallel programming of heterogeneous systems that include
CPUs, GPUs and other processors.
.
This package provides the development files: headers and libraries.
.
It also ensures that the ocl-icd ICD loader is installed so its additional
features (compared to the OpenCL norm) can be used: .pc file, ability to
select an ICD without root privilege, etc.")

set(CPACK_DEBIAN_CLLAYERINFO_DESCRIPTION "Query OpenCL Layer system information
OpenCL (Open Computing Language) is a multivendor open standard for
general-purpose parallel programming of heterogeneous systems that include
CPUs, GPUs and other processors. It supports system and user configured layers
to intercept OpenCL API calls.
.
This package contains a tool that lists the layers loaded by the the ocl-icd
OpenCL ICD Loader.")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

# DEB packaging configuration
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_VENDOR})

set(CPACK_DEBIAN_PACKAGE_HOMEPAGE
    "https://github.com/KhronosGroup/OpenCL-ICD-Loader")

# Version number [epoch:]upstream_version[-debian_revision]
set(CPACK_DEBIAN_PACKAGE_VERSION "${PROJECT_VERSION}") # upstream_version
if(DEFINED LATEST_RELEASE_VERSION)
  # Remove leading "v", if exists
  string(LENGTH "${LATEST_RELEASE_VERSION}" LATEST_RELEASE_VERSION_LENGTH)
  string(SUBSTRING "${LATEST_RELEASE_VERSION}" 0 1 LATEST_RELEASE_VERSION_FRONT)
  if(LATEST_RELEASE_VERSION_FRONT STREQUAL "v")
    string(SUBSTRING "${LATEST_RELEASE_VERSION}" 1 ${LATEST_RELEASE_VERSION_LENGTH} LATEST_RELEASE_VERSION)
  endif()

  string(APPEND CPACK_DEBIAN_PACKAGE_VERSION "~${LATEST_RELEASE_VERSION}")
endif()
set(CPACK_DEBIAN_PACKAGE_RELEASE "1") # debian_revision (because this is a
                                      # non-native pkg)
set(PACKAGE_VERSION_REVISION "${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}${DEBIAN_VERSION_SUFFIX}")

# Get architecture
execute_process(COMMAND dpkg "--print-architecture" OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE)
string(STRIP "${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}" CPACK_DEBIAN_PACKAGE_ARCHITECTURE)

##########################################################
#                       Components                       #
##########################################################

set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS OFF) # Component dependencies are NOT reflected in package relationships
set(CPACK_COMPONENTS_ALL runtime dev cllayerinfo)

set(PACKAGE_NAME_PREFIX "khronos-opencl-loader")

## Package runtime component
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PACKAGE_NAME_PREFIX}-libopencl1")

# Package file name in deb format:
# <PackageName>_<VersionNumber>-<DebianRevisionNumber>_<DebianArchitecture>.deb
set(CPACK_DEBIAN_RUNTIME_FILE_NAME "${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME}_${PACKAGE_VERSION_REVISION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_SECTION "libs")
# Dependencies
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "libc6")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_SUGGESTS "opencl-icd")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_CONFLICTS "amd-app, libopencl1, nvidia-libopencl1-dev")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_REPLACES "amd-app, libopencl1, nvidia-libopencl1-dev")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_PROVIDES "libopencl-1.1-1, libopencl-1.2-1, libopencl-2.0-1, libopencl-2.1-1, libopencl-2.2-1, libopencl-3.0-1, libopencl1")

## Package dev component
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PACKAGE_NAME_PREFIX}-opencl-dev")

# Package file name in deb format:
# <PackageName>_<VersionNumber>-<DebianRevisionNumber>_<DebianArchitecture>.deb
set(CPACK_DEBIAN_DEV_FILE_NAME "${CPACK_DEBIAN_DEV_PACKAGE_NAME}_${PACKAGE_VERSION_REVISION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb")
set(CPACK_DEBIAN_DEV_PACKAGE_SECTION "libdevel")

# Dependencies
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "opencl-c-headers (>= ${CPACK_DEBIAN_PACKAGE_VERSION}) | opencl-headers (>= ${CPACK_DEBIAN_PACKAGE_VERSION}), ${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME} (>= ${CPACK_DEBIAN_PACKAGE_VERSION}) | libopencl1")
set(CPACK_DEBIAN_DEV_PACKAGE_RECOMMENDS "libgl1-mesa-dev | libgl-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_CONFLICTS "opencl-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_BREAKS "amd-libopencl1, nvidia-libopencl1")
set(CPACK_DEBIAN_DEV_PACKAGE_REPLACES "amd-libopencl1, nvidia-libopencl1, opencl-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_PROVIDES "opencl-dev")

## Package cllayerinfo component
set(CPACK_DEBIAN_CLLAYERINFO_PACKAGE_NAME "${PACKAGE_NAME_PREFIX}-cllayerinfo")
set(CPACK_DEBIAN_CLLAYERINFO_FILE_NAME "${CPACK_DEBIAN_CLLAYERINFO_PACKAGE_NAME}_${PACKAGE_VERSION_REVISION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb")
# Dependencies
set(CPACK_DEBIAN_CLLAYERINFO_PACKAGE_DEPENDS "libc6")
set(CPACK_DEBIAN_CLLAYERINFO_PACKAGE_SECTION "admin")
