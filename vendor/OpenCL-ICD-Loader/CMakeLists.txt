cmake_minimum_required (VERSION 3.16)

# Include guard for including this project multiple times
if(TARGET OpenCL)
  return()
endif()

project (OpenCL-ICD-Loader
    VERSION 3.0
    LANGUAGES C)

find_package (Threads REQUIRED)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
# The option below allows building the ICD Loader library as a shared library
# (ON, default) or a static library (OFF).
#
# Khronos OpenCL Working Group strongly recommends building and using the ICD
# loader as a shared library due to the following benefits:
#
# 1. The shared library can be updated independent of the application. This
#    allows releasing new fixes and features in the ICD loader without updating
#    the application.
#
#    In rare cases when there are backward-incompatible changes to the ICD
#    loader (due to platform requirements, for instance), using a shared
#    library allows updating the library to make the transition seamless to
#    installed applications.
#
# 2. On platforms that require the ICD mechanism there are multiple vendors
#    shipping their OpenCL implementations. The vendor installers collaborate
#    to make sure that the installed ICD shared library version is suitable for
#    working with all vendor implementations installed on the system.
#
#    If applications statically link to ICD Loader then that version of the ICD
#    loader may not work with one or more installed vendor implementations.
#
# Using the OpenCL ICD loader as a static library is NOT recommended for
# end-user installations in general. However in some controlled environments it
# may be useful to simplify the build and distribution of the application. E.g.
# in test farms, or in cases where the end-user system configs are known in
# advance. Use it with discretion.
if(DEFINED BUILD_SHARED_LIBS)
  set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS_DEFAULT ${BUILD_SHARED_LIBS})
else()
  set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS_DEFAULT ON)
endif()
option(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS "Build OpenCL ICD Loader as shared library" ${OPENCL_ICD_LOADER_BUILD_SHARED_LIBS_DEFAULT})

# This option enables/disables support for OpenCL layers in the ICD loader.
# It is currently needed default while the specification is being formalized,
# and to study the performance impact.
option (ENABLE_OPENCL_LAYERS "Enable OpenCL Layers" ON)
option (ENABLE_OPENCL_LOADER_MANAGED_DISPATCH "Enable OpenCL Loader managed dispatch" ON)
include(CMakeDependentOption)
cmake_dependent_option(ENABLE_OPENCL_LAYERINFO "Enable building cllayerinfo tool" ON ENABLE_OPENCL_LAYERS OFF)

include(GNUInstallDirs)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CheckFunctionExists)
include(JoinPaths)
include(Package)

check_function_exists(secure_getenv HAVE_SECURE_GETENV)
check_function_exists(__secure_getenv HAVE___SECURE_GETENV)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/loader/icd_cmake_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/icd_cmake_config.h)

set (OPENCL_ICD_LOADER_SOURCES
    loader/icd.c
    loader/icd.h
    loader/icd_version.h
    loader/icd_dispatch.c
    loader/icd_dispatch.h
    loader/icd_dispatch_generated.c
    loader/icd_envvars.h
    loader/icd_platform.h)
include_directories (include)

if (WIN32)
    list (APPEND OPENCL_ICD_LOADER_SOURCES
        loader/windows/adapter.h
        loader/windows/icd_windows.c
        loader/windows/icd_windows.h
        loader/windows/icd_windows_dxgk.c
        loader/windows/icd_windows_dxgk.h
        loader/windows/icd_windows_envvars.c
        loader/windows/icd_windows_hkr.c
        loader/windows/icd_windows_hkr.h
        loader/windows/icd_windows_apppackage.c
        loader/windows/icd_windows_apppackage.h
        loader/windows/OpenCL.rc)
    # Only add the DXSDK include directory if the environment variable is
    # defined.  Since the DXSDK has merged into the Windows SDK, this is
    # only required in rare cases.
    if (DEFINED ENV{DXSDK_DIR} AND NOT (MINGW OR MSYS OR CYGWIN))
        include_directories ($ENV{DXSDK_DIR}/Include)
    endif ()

    # For mingw-i686 builds only we need a special .def file with stdcall
    # exports.  In all other cases we can use a standard .def file.
    if ((CMAKE_SIZEOF_VOID_P EQUAL 4) AND (MINGW OR MSYS OR CYGWIN))
        list (APPEND OPENCL_ICD_LOADER_SOURCES loader/windows/OpenCL-mingw-i686.def)
    else ()
        list (APPEND OPENCL_ICD_LOADER_SOURCES loader/windows/OpenCL.def)
    endif ()
else ()
    list (APPEND OPENCL_ICD_LOADER_SOURCES
        loader/linux/icd_linux.c
        loader/linux/icd_linux_envvars.c
        loader/linux/icd_exports.map)
endif ()

set (OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/inc" CACHE PATH "Path to OpenCL Headers")

if (${OPENCL_ICD_LOADER_BUILD_SHARED_LIBS})
  add_library (OpenCL SHARED ${OPENCL_ICD_LOADER_SOURCES})
else()
  add_library (OpenCL STATIC ${OPENCL_ICD_LOADER_SOURCES})
endif()

add_library (OpenCL::OpenCL ALIAS OpenCL)

set_target_properties (OpenCL PROPERTIES VERSION 1\.0\.0 SOVERSION "1")

if (WIN32)
    target_link_libraries (OpenCL PRIVATE cfgmgr32.lib runtimeobject.lib)

    # Generate a DLL without a "lib" prefix for mingw.
    if (MINGW OR MSYS OR CYGWIN)
        set_target_properties(OpenCL PROPERTIES PREFIX "")
        set_target_properties(OpenCL PROPERTIES LINK_FLAGS "-Wl,-disable-stdcall-fixup")
    endif()
else()
    target_link_libraries (OpenCL PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    if (NOT APPLE)
        set_target_properties (OpenCL PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/loader/linux/icd_exports.map")
        if (OPENCL_ICD_LOADER_PIC)
            set_target_properties(OpenCL PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif ()
    endif ()
endif ()

if (EXISTS ${OPENCL_ICD_LOADER_HEADERS_DIR}/CL/cl.h)
    message (STATUS "Defining OpenCL::Headers through OPENCL_ICD_LOADER_HEADERS_DIR")
    add_library (OpenCLHeaders INTERFACE)
    add_library (OpenCL::Headers ALIAS OpenCLHeaders)
    target_include_directories (OpenCLHeaders INTERFACE ${OPENCL_ICD_LOADER_HEADERS_DIR})
    target_include_directories (OpenCL PUBLIC $<BUILD_INTERFACE:${OPENCL_ICD_LOADER_HEADERS_DIR}>)
else ()
    if (NOT TARGET OpenCL::Headers)
        find_package (OpenCLHeaders REQUIRED)
    endif ()
    target_link_libraries (OpenCL PUBLIC OpenCL::Headers)
endif ()

set (OPENCL_COMPILE_DEFINITIONS
    CL_TARGET_OPENCL_VERSION=300
    CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES
    OPENCL_ICD_LOADER_VERSION_MAJOR=3
    OPENCL_ICD_LOADER_VERSION_MINOR=0
    OPENCL_ICD_LOADER_VERSION_REV=7
    $<$<BOOL:${ENABLE_OPENCL_LAYERS}>:CL_ENABLE_LAYERS>
    $<$<BOOL:${ENABLE_OPENCL_LOADER_MANAGED_DISPATCH}>:CL_ENABLE_LOADER_MANAGED_DISPATCH>
)

target_compile_definitions (OpenCL
  PRIVATE
    ${OPENCL_COMPILE_DEFINITIONS}
)

target_include_directories (OpenCL
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    loader
)
target_link_libraries (OpenCL PUBLIC ${CMAKE_DL_LIBS})

if (ENABLE_OPENCL_LAYERINFO)

  set (OPENCL_LAYER_INFO_SOURCES
      loader/cllayerinfo.c
      ${OPENCL_ICD_LOADER_SOURCES}
  )

  add_executable(cllayerinfo ${OPENCL_LAYER_INFO_SOURCES})

  add_executable(OpenCL::cllayerinfo ALIAS cllayerinfo)

  target_compile_definitions (cllayerinfo
    PRIVATE
      CL_LAYER_INFO
      ${OPENCL_COMPILE_DEFINITIONS}
  )

  if (EXISTS ${OPENCL_ICD_LOADER_HEADERS_DIR}/CL/cl.h)
      target_include_directories (cllayerinfo PUBLIC $<BUILD_INTERFACE:${OPENCL_ICD_LOADER_HEADERS_DIR}>)
  else ()
      target_link_libraries (cllayerinfo PUBLIC OpenCL::Headers)
  endif ()

  if (WIN32)
    target_link_libraries (cllayerinfo PRIVATE cfgmgr32.lib runtimeobject.lib)
  else ()
    target_link_libraries (cllayerinfo PRIVATE ${CMAKE_THREAD_LIBS_INIT})
  endif ()

  target_link_libraries (cllayerinfo PUBLIC ${CMAKE_DL_LIBS})

  target_include_directories (cllayerinfo
    PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      loader
  )
endif ()

option (OPENCL_ICD_LOADER_BUILD_TESTING "Enable support for OpenCL ICD Loader testing." OFF)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_ICD_LOADER_BUILD_TESTING)
    include(CTest)
endif()
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR OPENCL_ICD_LOADER_BUILD_TESTING) AND BUILD_TESTING)
    add_subdirectory (test)
endif()

install(
  TARGETS OpenCL
  EXPORT OpenCLICDLoaderTargets
  LIBRARY
  DESTINATION ${CMAKE_INSTALL_LIBDIR} # obtained from GNUInstallDirs
)
install(
# FILES $<TARGET_PDB_FILE:OpenCL> is cleanest, but is MSVC link.exe specific. LLVM's lld.exe and lld-link.exe don't support it (configure-time error)
# FILES $<TARGET_PROPERTY:OpenCL,COMPILE_PDB_OUTPUT_DIRECTORY>/OpenCL.pdb looks OK, but even though there's a PDB, this prop is empty on non-MSVC toolchains
  FILES $<TARGET_FILE_DIR:OpenCL>/OpenCL.pdb # is the most implicit (expect PDB be next to the library), yet the only one that universally works
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  OPTIONAL
)

if (ENABLE_OPENCL_LAYERINFO)
  install(
    TARGETS cllayerinfo
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT cllayerinfo
  )
endif()

export(
  EXPORT OpenCLICDLoaderTargets
  FILE ${PROJECT_BINARY_DIR}/OpenCLICDLoader/OpenCLICDLoaderTargets.cmake
  NAMESPACE OpenCL::
)
file(
  WRITE ${PROJECT_BINARY_DIR}/OpenCLICDLoader/OpenCLICDLoaderConfig.cmake
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/OpenCLICDLoaderTargets.cmake\")"
)

set(config_package_location ${CMAKE_INSTALL_DATADIR}/cmake/OpenCLICDLoader)
install(
  EXPORT OpenCLICDLoaderTargets
  FILE OpenCLICDLoaderTargets.cmake
  NAMESPACE OpenCL::
  DESTINATION ${config_package_location}
  COMPONENT dev
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenCLICDLoader/OpenCLICDLoaderConfig.cmake
  DESTINATION ${config_package_location}
  COMPONENT dev
)

unset(CMAKE_SIZEOF_VOID_P)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/OpenCLICDLoader/OpenCLICDLoaderConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenCLICDLoader/OpenCLICDLoaderConfigVersion.cmake
  DESTINATION ${config_package_location}
  COMPONENT dev
)

# Separate namelink from shared library and symlink for DEB packaging
install (TARGETS OpenCL
    LIBRARY
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT runtime
    NAMELINK_SKIP)

install (TARGETS OpenCL
    LIBRARY
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT dev
    NAMELINK_ONLY)
