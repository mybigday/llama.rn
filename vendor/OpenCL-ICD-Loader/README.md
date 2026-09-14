# OpenCL<sup>TM</sup> ICD Loader

This repo contains the source code and tests for the Khronos official OpenCL ICD Loader.

## CI Build Status

[![Build Status](https://github.com/KhronosGroup/OpenCL-ICD-Loader/actions/workflows/presubmit.yml/badge.svg?branch=main)](https://github.com/KhronosGroup/OpenCL-ICD-Loader/actions/workflows/presubmit.yml)

## Introduction

OpenCL defines an *Installable Client Driver* (ICD) mechanism to allow developers to build applications against an *Installable Client Driver* loader (ICD loader) rather than linking their applications against a specific OpenCL implementation.
The ICD Loader is responsible for:

* Exporting OpenCL API entry points
* Enumerating OpenCL implementations
* Forwarding OpenCL API calls to the correct implementation

This repo contains the source code and tests for the Khronos official OpenCL ICD Loader.

Note that this repo does not contain an OpenCL implementation (ICD).
You will need to obtain and install an OpenCL implementation for your OpenCL device that supports the OpenCL ICD extension `cl_khr_icd` to run an application using the OpenCL ICD Loader.

The OpenCL *Installable Client Driver* extension (`cl_khr_icd`) is described in the OpenCL extensions specification, which may be found on the [Khronos OpenCL Registry](https://www.khronos.org/registry/OpenCL/).

## Build Instructions

> While the ICD Loader can be built and installed in isolation, it is part of the [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK). If looking for streamlined build experience and a complete development package, refer to the SDK build instructions instead of the following guide.

### Dependencies

The OpenCL ICD Loader requires:
- the [OpenCL Headers](https://github.com/KhronosGroup/OpenCL-Headers/).
  - It is recommended to install the headers via CMake, however a convenience shorthand is provided. Providing `OPENCL_ICD_LOADER_HEADERS_DIR` to CMake, one may specify the location of OpenCL Headers. By default, the OpenCL ICD Loader will look for OpenCL Headers in the inc directory.
- The OpenCL ICD Loader uses CMake for its build system.
If CMake is not provided by your build system or OS package manager, please consult the [CMake website](https://cmake.org).

### Example Build

For most Windows and Linux usages, the following steps are sufficient to build the OpenCL ICD Loader:

1. Clone this repo and the OpenCL Headers:

        git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
        git clone https://github.com/KhronosGroup/OpenCL-Headers

1. Install OpenCL Headers CMake package

        cmake -D CMAKE_INSTALL_PREFIX=./OpenCL-Headers/install -S ./OpenCL-Headers -B ./OpenCL-Headers/build 
        cmake --build ./OpenCL-Headers/build --target install

1. Build and install OpenCL ICD Loader CMake package. _(Note that `CMAKE_PREFIX_PATH` need to be an absolute path. Update as needed.)_

        cmake -D CMAKE_PREFIX_PATH=/absolute/path/to/OpenCL-Headers/install -D CMAKE_INSTALL_PREFIX=./OpenCL-ICD-Loader/install -S ./OpenCL-ICD-Loader -B ./OpenCL-ICD-Loader/build 
        cmake --build ./OpenCL-ICD-Loader/build --target install

Notes:

* For x64 Windows builds, you need to instruct the default Visual Studio generator by adding `-A x64` to all your command-lines.

* Some users may prefer to use a CMake GUI frontend, such as `cmake-gui` or `ccmake`, vs. the command-line CMake.

### Example Use

Example CMake invocation

```bash
cmake -D CMAKE_PREFIX_PATH="/chosen/install/prefix/of/headers;/chosen/install/prefix/of/loader" /path/to/opencl/app
```

and sample `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.0...3.18.4)
project(proj)
add_executable(app main.cpp)
find_package(OpenCLHeaders REQUIRED)
find_package(OpenCLICDLoader REQUIRED)
target_link_libraries(app PRIVATE OpenCL::Headers OpenCL::OpenCL)
```

## OpenCL ICD Loader Tests

OpenCL ICD Loader Tests can be run using `ctest` from the `build` directory. CTest which is a companion to CMake. The OpenCL ICD Loader Tests can also be run directly by executing `icd_loader_test[.exe]` executable from the bin folder.

_(Note that running the tests manually requires setting up it's env manually, by setting `OCL_ICD_FILENAMES` to the full path of `libOpenCLDriverStub.so`/`OpenCLDriverStub.dll`, something otherwise done by CTest.)_

## Registering ICDs

The method to installing an ICD is operating system dependent.

### Registering an ICD on Linux

Install your ICD by creating a file with the full path to the library of your implementation in `/etc/OpenCL/vendors` for eg.:

    echo full/path/to/libOpenCLDriverStub.so > /etc/OpenCL/vendors/test.icd

### Registering an ICD on Windows

Install your ICD by adding a `REG_DWORD` value to the registry keys:

    // For 32-bit operating systems, or 64-bit tests on a 64-bit operating system:
    HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors
    
    // For 32-bit tests on a 64-bit operating system:
    HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Khronos\OpenCL\Vendors

    // The name of the REG_DWORD value should be the full path to the library of your implementation, for eg.
    // OpenCLDriverStub.dll, and the data for this value should be 0.

## About Layers

Layers have been added as an experimental feature in the OpenCL ICD Loader. We do not
expect the API or ABI to change significantly, but the OpenCL Working Group reserves
the right to do so. The layer support can also be completely deactivated during
configuration by using the `ENABLE_OPENCL_LAYERS` (`ON` by default) cmake variable:

```bash
cmake -DENABLE_OPENCL_LAYERS=OFF
```

For now, runtime configuration of layers is done using the `OPENCL_LAYERS` environment
variable. A colon (Linux) or semicolon (Windows) list of layers to use can be provided
through this environment variable.

We are looking for feedback.

## Support

Please create a GitHub issue to report an issue or ask questions.

## Contributing

Contributions to the OpenCL ICD Loader are welcomed and encouraged.
You will be prompted with a one-time "click-through" CLA dialog as part of submitting your pull request or other contribution to GitHub.

## Table of Debug Environment Variables

The following debug environment variables are available for use with the OpenCL ICD loader:

| Environment Variable              | Behavior            |  Example Format      |
|:---------------------------------:|---------------------|----------------------|
| OCL_ICD_FILENAMES                 | Specifies a list of additional ICDs to load.  The ICDs will be enumerated first, before any ICDs discovered via default mechanisms. | `export OCL_ICD_FILENAMES=libVendorA.so:libVendorB.so`<br/><br/>`set OCL_ICD_FILENAMES=vendor_a.dll;vendor_b.dll` |
| OCL_ICD_VENDORS                   | On Linux and Android, specifies a directory to scan for ICDs to enumerate in place of the default `/etc/OpenCL/vendors'. |  `export OCL_ICD_VENDORS=/my/local/icd/search/path` |
| OPENCL_LAYERS                     | Specifies a list of layers to load. |  `export OPENCL_LAYERS=libLayerA.so:libLayerB.so`<br/><br/>`set OPENCL_LAYERS=libLayerA.dll;libLayerB.dll` |
| OPENCL_LAYER_PATH                 | On Linux and Android, specifies a directory to scan for layers to enumerate in place of the default `/etc/OpenCL/layers'. | `export OPENCL_LAYER_PATH=/my/local/layers/search/path` |
| OCL_ICD_ENABLE_TRACE              | Enable the trace mechanism          |  `export OCL_ICD_ENABLE_TRACE=True`<br/><br/>`set OCL_ICD_ENABLE_TRACE=True`<br/>`true, T, 1 can also be used here.`  |
