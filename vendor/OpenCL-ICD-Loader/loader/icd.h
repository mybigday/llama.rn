/*
 * Copyright (c) 2016-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.
 */

#ifndef _ICD_H_
#define _ICD_H_

#include "icd_platform.h"
#include "icd_dispatch.h"

#ifndef CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_2_1_APIS
#define CL_USE_DEPRECATED_OPENCL_2_1_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_2_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_2_APIS
#endif

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_icd.h>
#include <stdio.h>

/*
 * type definitions
 */

typedef cl_int (CL_API_CALL *pfn_clIcdGetPlatformIDs)(
    cl_uint num_entries, 
    cl_platform_id *platforms, 
    cl_uint *num_platforms) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int (CL_API_CALL *pfn_clGetPlatformInfo)(
    cl_platform_id   platform, 
    cl_platform_info param_name,
    size_t           param_value_size, 
    void *           param_value,
    size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

typedef void *(CL_API_CALL *pfn_clGetExtensionFunctionAddress)(
    const char *function_name)  CL_API_SUFFIX__VERSION_1_0;

typedef struct KHRicdVendorRec KHRicdVendor;

/* 
 * KHRicdVendor
 *
 * Data for a single ICD vendor platform.
 */
struct KHRicdVendorRec
{
    // the loaded library object (true type varies on Linux versus Windows)
    void *library;

    // the extension suffix for this platform
    char *suffix;

    // function pointer to the ICD platform IDs extracted from the library
    pfn_clGetExtensionFunctionAddress clGetExtensionFunctionAddress;

    // the platform retrieved from clGetIcdPlatformIDsKHR
    cl_platform_id platform;

#if defined(CL_ENABLE_LOADER_MANAGED_DISPATCH)
    // the loader populated dispatch table for cl_khr_icd2 compliant platforms
    struct KHRDisp dispData;
#endif

    // next vendor in the list vendors
    KHRicdVendor *next;
};

// the global state
extern KHRicdVendor * khrIcdVendors;

extern int khrEnableTrace;

#if defined(CL_ENABLE_LAYERS)
/*
 * KHRLayer
 *
 * Data for a single Layer
 */
struct KHRLayer;
struct KHRLayer
{
    // the loaded library object (true type varies on Linux versus Windows)
    void *library;
    // the dispatch table of the layer
    struct _cl_icd_dispatch dispatch;
    // The next layer in the chain
    struct KHRLayer *next;
#ifdef CL_LAYER_INFO
    // The layer library name
    char *libraryName;
    // the pointer to the clGetLayerInfo funciton
    void *p_clGetLayerInfo;
#endif
};

// the global layer state
extern struct KHRLayer * khrFirstLayer;
extern struct _cl_icd_dispatch khrMasterDispatch;
#endif // defined(CL_ENABLE_LAYERS)

/* 
 * khrIcd interface
 */

// read vendors from system configuration and store the data
// loaded into khrIcdState.  this will call the OS-specific
// function khrIcdEnumerateVendors.  this is called at every
// dispatch function which may be a valid first call into the
// API (e.g, getPlatformIDs, etc).
void khrIcdInitialize(void);

// entrypoint to check and initialize trace.
void khrIcdInitializeTrace(void);

// go through the list of vendors (in /etc/OpenCL.conf or through 
// the registry) and call khrIcdVendorAdd for each vendor encountered
// n.b, this call is OS-specific
void khrIcdOsVendorsEnumerateOnce(void);

// read vendors from environment variables
void khrIcdVendorsEnumerateEnv(void);

// add a vendor's implementation to the list of libraries
void khrIcdVendorAdd(const char *libraryName);

// read layers from environment variables
void khrIcdLayersEnumerateEnv(void);

// add a layer to the layer chain
void khrIcdLayerAdd(const char *libraryName);

// dynamically load a library.  returns NULL on failure
// n.b, this call is OS-specific
void *khrIcdOsLibraryLoad(const char *libraryName);

// get a function pointer from a loaded library.  returns NULL on failure.
// n.b, this call is OS-specific
void *khrIcdOsLibraryGetFunctionAddress(void *library, const char *functionName);

// unload a library.
// n.b, this call is OS-specific
void khrIcdOsLibraryUnload(void *library);

// parse properties and determine the platform to use from them
void khrIcdContextPropertiesGetPlatform(
    const cl_context_properties *properties, 
    cl_platform_id *outPlatform);

// condition anonyous union initialization to usage
#if __CL_HAS_ANON_STRUCT__
#define ICD_ANON_UNION_INIT_MEMBER(a) {a}
#else
#define ICD_ANON_UNION_INIT_MEMBER(a) a
#endif

// internal tracing macros
#define KHR_ICD_TRACE(...) \
do \
{ \
    if (khrEnableTrace) \
    { \
        fprintf(stderr, "KHR ICD trace at %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
    } \
} while (0)

#ifdef _WIN32
#define KHR_ICD_WIDE_TRACE(...) \
do \
{ \
    if (khrEnableTrace) \
    { \
        fwprintf(stderr, L"KHR ICD trace at %hs:%d: ", __FILE__, __LINE__); \
        fwprintf(stderr, __VA_ARGS__); \
    } \
} while (0)

#else
#define KHR_ICD_WIDE_TRACE(...)
#endif

#define KHR_ICD_ERROR_RETURN_ERROR(_error)                          \
do {                                                                \
    return _error;                                                  \
} while(0)

#define KHR_ICD_ERROR_RETURN_HANDLE(_error)                         \
do {                                                                \
    if (errcode_ret) {                                              \
        *errcode_ret = _error;                                      \
    }                                                               \
    return NULL;                                                    \
} while(0)

// Check if the passed-in handle is NULL, and if it is, return the error.
#define KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(_handle, _error)       \
do {                                                                \
    if (!_handle) {                                                 \
        KHR_ICD_ERROR_RETURN_ERROR(_error);                         \
    }                                                               \
} while (0)

// Check if the passed-in handle is NULL, and if it is, first check and set
// errcode_ret to the error, then return NULL (NULL being an invalid handle).
#define KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(_handle, _error)      \
do {                                                                \
    if (!_handle) {                                                 \
        KHR_ICD_ERROR_RETURN_HANDLE(_error);                        \
    }                                                               \
} while (0)

// Check if the passed-in function pointer is NULL, and if it is, return
// CL_INVALID_OPERATION.
#define KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(_pointer)             \
do {                                                                \
    if (!_pointer) {                                                \
        KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);           \
    }                                                               \
} while (0)

// Check if the passed-in function pointer is NULL, and if it is, first
// check and set errcode_ret to CL_INVALID_OPERATION, then return NULL
// (NULL being an invalid handle).
#define KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(_pointer)            \
do {                                                                \
    if (!_pointer) {                                                \
        KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);          \
    }                                                               \
} while (0)

#endif
