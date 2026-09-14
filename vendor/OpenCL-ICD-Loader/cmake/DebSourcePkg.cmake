# This script produces the changelog, control and rules file in the debian
# directory. These files are needed to build a Debian source package from the repository.
# Run this in CMake script mode, e.g.
# $ cd OpenCL-ICD-Loader
# $ cmake -S . -B ../build -D BUILD_TESTING=OFF
# $ cmake
#    -DCMAKE_CACHE_PATH=../build/CMakeCache.txt
#    -DCPACK_DEBIAN_PACKAGE_MAINTAINER="Example Name <example@example.com>"
#    -DDEBIAN_DISTROSERIES=jammy
#    -DORIG_ARCHIVE=../OpenCL-ICD-Loader.tar.gz
#    -DLATEST_RELEASE_VERSION=v2023.08.29
#    -P cmake/DebSourcePkg.cmake
# $ debuild -S -sa

cmake_minimum_required(VERSION 3.21) # file(COPY_FILE) is added in CMake 3.21

if(NOT EXISTS "${CMAKE_CACHE_PATH}")
    message(FATAL_ERROR "CMAKE_CACHE_PATH is not set or does not exist")
endif()
if(NOT DEFINED DEBIAN_PACKAGE_MAINTAINER)
    message(FATAL_ERROR "DEBIAN_PACKAGE_MAINTAINER is not set")
endif()
if(NOT DEFINED DEBIAN_DISTROSERIES)
    message(FATAL_ERROR "DEBIAN_DISTROSERIES is not set")
endif()
if(NOT DEFINED ORIG_ARCHIVE)
    message(WARNING "ORIG_ARCHIVE is not set")
elseif(NOT EXISTS "${ORIG_ARCHIVE}")
    message(FATAL_ERROR "ORIG_ARCHIVE is defined, but the file does not exist at \"${ORIG_ARCHIVE}\"")
endif()
if(NOT DEFINED LATEST_RELEASE_VERSION)
    message(WARNING "LATEST_RELEASE_VERSION is not set")
endif()
if(NOT DEFINED DEBIAN_VERSION_SUFFIX)
    message(WARNING "DEBIAN_VERSION_SUFFIX is not set")
endif()

# Extracting the project version from the main CMakeLists.txt via regex
file(READ "${CMAKE_CACHE_PATH}" CMAKE_CACHE)
string(REGEX MATCH "CMAKE_PROJECT_VERSION[^=]*=([^\n]*)" REGEX_MATCH "${CMAKE_CACHE}")
if(NOT REGEX_MATCH)
    message(FATAL_ERROR "Could not extract project version from CMakeLists.txt")
endif()
set(PROJECT_VERSION "${CMAKE_MATCH_1}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
# Package.cmake contains all details for packaging
include(PackageSetup)

# Append a space after every newline in the description. This format is required
# in the control file.
string(REPLACE "\n" "\n " CPACK_DEBIAN_DEV_DESCRIPTION "${CPACK_DEBIAN_DEV_DESCRIPTION}")
string(REPLACE "\n" "\n " CPACK_DEBIAN_RUNTIME_DESCRIPTION "${CPACK_DEBIAN_RUNTIME_DESCRIPTION}")
string(REPLACE "\n" "\n " CPACK_DEBIAN_CLLAYERINFO_DESCRIPTION "${CPACK_DEBIAN_CLLAYERINFO_DESCRIPTION}")

set(DEB_SOURCE_PKG_DIR "${CMAKE_CURRENT_LIST_DIR}/../debian")
# Write debian/control
file(WRITE "${DEB_SOURCE_PKG_DIR}/control"
"Source: ${PACKAGE_NAME_PREFIX}
Section: ${CPACK_DEBIAN_DEV_PACKAGE_SECTION}
Priority: optional
Maintainer: ${DEBIAN_PACKAGE_MAINTAINER}
Build-Depends: cmake, debhelper-compat (=13), opencl-c-headers
Rules-Requires-Root: no
Homepage: ${CPACK_DEBIAN_PACKAGE_HOMEPAGE}
Standards-Version: 4.6.2

Package: ${CPACK_DEBIAN_DEV_PACKAGE_NAME}
Architecture: any
Multi-Arch: same
Depends: ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}, ${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME} (=${PACKAGE_VERSION_REVISION})
Recommends: ${CPACK_DEBIAN_DEV_PACKAGE_RECOMMENDS}
Conflicts: ${CPACK_DEBIAN_DEV_PACKAGE_CONFLICTS}
Breaks: ${CPACK_DEBIAN_DEV_PACKAGE_BREAKS}
Replaces: ${CPACK_DEBIAN_DEV_PACKAGE_REPLACES}
Provides: ${CPACK_DEBIAN_DEV_PACKAGE_PROVIDES}
Description: ${CPACK_DEBIAN_DEV_DESCRIPTION}

Package: ${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME}
Section: ${CPACK_DEBIAN_RUNTIME_PACKAGE_SECTION}
Architecture: any
Multi-Arch: same
Depends: ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}
# Conflicts and replaces deliberately not added
# The runtime package provides libOpenCL.so.1.0.0 and libOpenCL.so.1 via update-alternatives
# Conflicts: ${CPACK_DEBIAN_RUNTIME_PACKAGE_CONFLICTS}
# Replaces: ${CPACK_DEBIAN_RUNTIME_PACKAGE_REPLACES}
Provides: ${CPACK_DEBIAN_RUNTIME_PACKAGE_PROVIDES}
Description: ${CPACK_DEBIAN_RUNTIME_DESCRIPTION}

Package: ${CPACK_DEBIAN_CLLAYERINFO_PACKAGE_NAME}
Section: ${CPACK_DEBIAN_CLLAYERINFO_PACKAGE_SECTION}
Architecture: any
Depends: ${CPACK_DEBIAN_CLLAYERINFO_PACKAGE_DEPENDS}, ${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME} (=${PACKAGE_VERSION_REVISION})
Conflicts: ${CPACK_DEBIAN_CLLAYERINFO_PACKAGE_CONFLICTS}
Replaces: ${CPACK_DEBIAN_CLLAYERINFO_PACKAGE_REPLACES}
Provides: ${CPACK_DEBIAN_CLLAYERINFO_PACKAGE_PROVIDES}
Description: ${CPACK_DEBIAN_CLLAYERINFO_DESCRIPTION}
"
)
# Write debian/changelog
string(TIMESTAMP CURRENT_TIMESTAMP "%a, %d %b %Y %H:%M:%S +0000" UTC)
file(WRITE "${DEB_SOURCE_PKG_DIR}/changelog"
"${PACKAGE_NAME_PREFIX} (${PACKAGE_VERSION_REVISION}) ${DEBIAN_DISTROSERIES}; urgency=medium

  * Released version ${PACKAGE_VERSION_REVISION}

 -- ${DEBIAN_PACKAGE_MAINTAINER}  ${CURRENT_TIMESTAMP}
")
# Write debian/rules
file(WRITE "${DEB_SOURCE_PKG_DIR}/rules"
"#!/usr/bin/make -f
%:
\tdh $@

override_dh_auto_configure:
\tdh_auto_configure -- -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

GENERATED_MAINTAINER_SCRIPTS := $(patsubst %.in,%,$(wildcard debian/*.alternatives.in))

$(GENERATED_MAINTAINER_SCRIPTS): %: %.in
\tsed \"s%@DEB_HOST_MULTIARCH@%$(DEB_HOST_MULTIARCH)%g\" < $< > $@

execute_before_dh_install: $(GENERATED_MAINTAINER_SCRIPTS)
\ttrue # An empty rule would confuse dh
")
file(WRITE "${DEB_SOURCE_PKG_DIR}/${CPACK_DEBIAN_DEV_PACKAGE_NAME}.install"
"usr/lib/*/pkgconfig
usr/lib/*/lib*.so
usr/share
")
# The .so files are installed to a different directory, and then linked back
# the the original location via update-alternatives.
file(WRITE "${DEB_SOURCE_PKG_DIR}/${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME}.install"
"usr/lib/*/lib*.so.* usr/lib/\${DEB_HOST_MULTIARCH}/KhronosOpenCLICDLoader
")
file(WRITE "${DEB_SOURCE_PKG_DIR}/${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME}.alternatives.in"
"Name: libOpenCL.so.1.0.0-@DEB_HOST_MULTIARCH@
Link: /usr/lib/@DEB_HOST_MULTIARCH@/libOpenCL.so.1.0.0
Alternative: /usr/lib/@DEB_HOST_MULTIARCH@/KhronosOpenCLICDLoader/libOpenCL.so.1.0.0
Dependents:
  /usr/lib/@DEB_HOST_MULTIARCH@/libOpenCL.so.1 libOpenCL.so.1-@DEB_HOST_MULTIARCH@ /usr/lib/@DEB_HOST_MULTIARCH@/KhronosOpenCLICDLoader/libOpenCL.so.1
Priority: 100
")
file(WRITE "${DEB_SOURCE_PKG_DIR}/${CPACK_DEBIAN_CLLAYERINFO_PACKAGE_NAME}.install"
"usr/bin
")

if(DEFINED ORIG_ARCHIVE)
    # Copy the passed orig.tar.gz file. The target filename is deduced from the version number, as expected by debuild
    cmake_path(IS_ABSOLUTE ORIG_ARCHIVE IS_ORIG_ARCHIVE_ABSOLUTE)
    if (NOT IS_ORIG_ARCHIVE_ABSOLUTE)
        message(FATAL_ERROR "ORIG_ARCHIVE must be an absolute path (passed: \"${ORIG_ARCHIVE}\")")
    endif()
    cmake_path(GET ORIG_ARCHIVE EXTENSION ORIG_ARCHIVE_EXT)
    cmake_path(GET ORIG_ARCHIVE PARENT_PATH ORIG_ARCHIVE_PARENT)
    set(TARGET_PATH "${ORIG_ARCHIVE_PARENT}/${PACKAGE_NAME_PREFIX}_${CPACK_DEBIAN_PACKAGE_VERSION}${ORIG_ARCHIVE_EXT}")
    message(STATUS "Copying \"${ORIG_ARCHIVE}\" to \"${TARGET_PATH}\"")
    file(COPY_FILE "${ORIG_ARCHIVE}" "${TARGET_PATH}")
endif()
