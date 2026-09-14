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

#include "icd.h"
#include "icd_dispatch.h"
#include "icd_envvars.h"
#if defined(CL_ENABLE_LAYERS)
#include <CL/cl_layer.h>
#endif // defined(CL_ENABLE_LAYERS)
#include <stdlib.h>
#include <string.h>

KHRicdVendor *khrIcdVendors = NULL;
int khrEnableTrace = 0;

#if defined(CL_ENABLE_LAYERS)
struct KHRLayer *khrFirstLayer = NULL;
#endif // defined(CL_ENABLE_LAYERS)

// entrypoint to check and initialize trace.
void khrIcdInitializeTrace(void)
{
    char *enableTrace = khrIcd_getenv("OCL_ICD_ENABLE_TRACE");
    if (enableTrace && (strcmp(enableTrace, "True") == 0 ||
            strcmp(enableTrace, "true") == 0 ||
            strcmp(enableTrace, "T") == 0 ||
            strcmp(enableTrace, "1") == 0))
    {
        khrEnableTrace = 1;
    }
}

// entrypoint to initialize the ICD and add all vendors
void khrIcdInitialize(void)
{
    // enumerate vendors present on the system
    khrIcdOsVendorsEnumerateOnce();
}

void khrIcdVendorAdd(const char *libraryName)
{
    void *library = NULL;
    cl_int result = CL_SUCCESS;
    pfn_clGetExtensionFunctionAddress p_clGetExtensionFunctionAddress = NULL;
    pfn_clIcdGetPlatformIDs p_clIcdGetPlatformIDs = NULL;
#if defined(CL_ENABLE_LOADER_MANAGED_DISPATCH)
    clIcdGetFunctionAddressForPlatformKHR_fn p_clIcdGetFunctionAddressForPlatform = NULL;
    clIcdSetPlatformDispatchDataKHR_fn p_clIcdSetPlatformDispatchData = NULL;
#endif
    cl_uint i = 0;
    cl_uint platformCount = 0;
    cl_platform_id *platforms = NULL;
    KHRicdVendor *vendorIterator = NULL;

    // require that the library name be valid
    if (!libraryName) 
    {
        goto Done;
    }
    KHR_ICD_TRACE("attempting to add vendor %s...\n", libraryName);

    // load its library and query its function pointers
    library = khrIcdOsLibraryLoad(libraryName);
    if (!library)
    {
        KHR_ICD_TRACE("failed to load library %s\n", libraryName);
        goto Done;
    }

    // ensure that we haven't already loaded this vendor
    for (vendorIterator = khrIcdVendors; vendorIterator; vendorIterator = vendorIterator->next)
    {
        if (vendorIterator->library == library)
        {
            KHR_ICD_TRACE("already loaded vendor %s, nothing to do here\n", libraryName);
            goto Done;
        }
    }

    // get the library's clGetExtensionFunctionAddress pointer
    p_clGetExtensionFunctionAddress = (pfn_clGetExtensionFunctionAddress)(size_t)khrIcdOsLibraryGetFunctionAddress(library, "clGetExtensionFunctionAddress");
    if (!p_clGetExtensionFunctionAddress)
    {
        KHR_ICD_TRACE("failed to get function address clGetExtensionFunctionAddress\n");
        goto Done;
    }

    // use that function to get the clIcdGetPlatformIDsKHR function pointer
    p_clIcdGetPlatformIDs = (pfn_clIcdGetPlatformIDs)(size_t)p_clGetExtensionFunctionAddress("clIcdGetPlatformIDsKHR");
    if (!p_clIcdGetPlatformIDs)
    {
        KHR_ICD_TRACE("failed to get extension function address clIcdGetPlatformIDsKHR\n");
        goto Done;
    }

#if defined(CL_ENABLE_LOADER_MANAGED_DISPATCH)
    // try to get clIcdGetFunctionAddressForPlatformKHR and clIcdSetPlatformDispatchDataKHR to detect cl_khr_icd2 support
    p_clIcdGetFunctionAddressForPlatform = (clIcdGetFunctionAddressForPlatformKHR_fn)(size_t)p_clGetExtensionFunctionAddress("clIcdGetFunctionAddressForPlatformKHR");
    p_clIcdSetPlatformDispatchData = (clIcdSetPlatformDispatchDataKHR_fn)(size_t)p_clGetExtensionFunctionAddress("clIcdSetPlatformDispatchDataKHR");
#endif

    // query the number of platforms available and allocate space to store them
    result = p_clIcdGetPlatformIDs(0, NULL, &platformCount);
    if (CL_SUCCESS != result)
    {
        KHR_ICD_TRACE("failed clIcdGetPlatformIDs\n");
        goto Done;
    }
    platforms = (cl_platform_id *)malloc(platformCount * sizeof(cl_platform_id) );
    if (!platforms)
    {
        KHR_ICD_TRACE("failed to allocate memory\n");
        goto Done;
    }
    memset(platforms, 0, platformCount * sizeof(cl_platform_id) );
    result = p_clIcdGetPlatformIDs(platformCount, platforms, NULL);
    if (CL_SUCCESS != result)
    {
        KHR_ICD_TRACE("failed clIcdGetPlatformIDs\n");
        goto Done;
    }

    // for each platform, add it
    for (i = 0; i < platformCount; ++i)
    {
        KHRicdVendor* vendor = NULL;
        char *suffix;
        size_t suffixSize;

        // skip NULL platforms and non dispatchable platforms
        if (!platforms[i] || !platforms[i]->dispatch)
        {
            continue;
        }

#if defined(CL_ENABLE_LOADER_MANAGED_DISPATCH)
        if (KHR_ICD2_HAS_TAG(platforms[i]) && !p_clIcdGetFunctionAddressForPlatform)
        {
           KHR_ICD_TRACE("found icd 2 object, but platform is missing clIcdGetFunctionAddressForPlatformKHR\n");
           continue;
        }
        if (KHR_ICD2_HAS_TAG(platforms[i]) && !p_clIcdSetPlatformDispatchData)
        {
           KHR_ICD_TRACE("found icd 2 object, but platform is missing clIcdSetPlatformDispatchDataKHR\n");
           continue;
        }
        if (KHR_ICD2_HAS_TAG(platforms[i]) && !((intptr_t)((platforms[i])->dispatch->clUnloadCompiler) == CL_ICD2_TAG_KHR))
        {
           KHR_ICD_TRACE("found icd 2 object, but platform is missing tag in clUnloadCompiler\n");
           continue;
        }
#endif

        // allocate a structure for the vendor
        vendor = (KHRicdVendor*)malloc(sizeof(*vendor) );
        if (!vendor)
        {
            KHR_ICD_TRACE("failed to allocate memory\n");
            continue;
        }
        memset(vendor, 0, sizeof(*vendor));

#if defined(CL_ENABLE_LOADER_MANAGED_DISPATCH)
        // populate cl_khr_icd2 platform's loader managed dispatch tables
        if (KHR_ICD2_HAS_TAG(platforms[i]))
        {
            khrIcd2PopulateDispatchTable(platforms[i], p_clIcdGetFunctionAddressForPlatform, &vendor->dispData.dispatch);
            p_clIcdSetPlatformDispatchData(platforms[i], &vendor->dispData);
            KHR_ICD_TRACE("found icd 2 platform, using loader managed dispatch\n");
        }
#endif

        // call clGetPlatformInfo on the returned platform to get the suffix
        result = KHR_ICD2_DISPATCH(platforms[i])->clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_ICD_SUFFIX_KHR,
            0,
            NULL,
            &suffixSize);
        if (CL_SUCCESS != result)
        {
            free(vendor);
            continue;
        }
        suffix = (char *)malloc(suffixSize);
        if (!suffix)
        {
            free(vendor);
            continue;
        }
        result = KHR_ICD2_DISPATCH(platforms[i])->clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_ICD_SUFFIX_KHR,
            suffixSize,
            suffix,
            NULL);            
        if (CL_SUCCESS != result)
        {
            free(suffix);
            free(vendor);
            continue;
        }

        // populate vendor data
        vendor->library = khrIcdOsLibraryLoad(libraryName);
        if (!vendor->library) 
        {
            free(suffix);
            free(vendor);
            KHR_ICD_TRACE("failed get platform handle to library\n");
            continue;
        }
        vendor->clGetExtensionFunctionAddress = p_clGetExtensionFunctionAddress;
        vendor->platform = platforms[i];
        vendor->suffix = suffix;

        // add this vendor to the list of vendors at the tail
        {
            KHRicdVendor **prevNextPointer = NULL;
            for (prevNextPointer = &khrIcdVendors; *prevNextPointer; prevNextPointer = &( (*prevNextPointer)->next) );
            *prevNextPointer = vendor;
        }

        KHR_ICD_TRACE("successfully added vendor %s with suffix %s\n", libraryName, suffix);

    }

Done:

    if (library)
    {
        khrIcdOsLibraryUnload(library);
    }
    if (platforms)
    {
        free(platforms);
    }
}

#if defined(CL_ENABLE_LAYERS)
void khrIcdLayerAdd(const char *libraryName)
{
    void *library = NULL;
    cl_int result = CL_SUCCESS;
    pfn_clGetLayerInfo p_clGetLayerInfo = NULL;
    pfn_clInitLayer p_clInitLayer = NULL;
    struct KHRLayer *layerIterator = NULL;
    struct KHRLayer *layer = NULL;
    cl_layer_api_version api_version = 0;
    const struct _cl_icd_dispatch *targetDispatch = NULL;
    const struct _cl_icd_dispatch *layerDispatch = NULL;
    cl_uint layerDispatchNumEntries = 0;
    cl_uint loaderDispatchNumEntries = 0;

    // require that the library name be valid
    if (!libraryName)
    {
        goto Done;
    }
    KHR_ICD_TRACE("attempting to add layer %s...\n", libraryName);

    // load its library and query its function pointers
    library = khrIcdOsLibraryLoad(libraryName);
    if (!library)
    {
        KHR_ICD_TRACE("failed to load library %s\n", libraryName);
        goto Done;
    }

    // ensure that we haven't already loaded this layer
    for (layerIterator = khrFirstLayer; layerIterator; layerIterator = layerIterator->next)
    {
        if (layerIterator->library == library)
        {
            KHR_ICD_TRACE("already loaded layer %s, nothing to do here\n", libraryName);
            goto Done;
        }
    }

    // get the library's clGetLayerInfo pointer
    p_clGetLayerInfo = (pfn_clGetLayerInfo)(size_t)khrIcdOsLibraryGetFunctionAddress(library, "clGetLayerInfo");
    if (!p_clGetLayerInfo)
    {
        KHR_ICD_TRACE("failed to get function address clGetLayerInfo\n");
        goto Done;
    }

    // use that function to get the clInitLayer function pointer
    p_clInitLayer = (pfn_clInitLayer)(size_t)khrIcdOsLibraryGetFunctionAddress(library, "clInitLayer");
    if (!p_clInitLayer)
    {
        KHR_ICD_TRACE("failed to get function address clInitLayer\n");
        goto Done;
    }

    result = p_clGetLayerInfo(CL_LAYER_API_VERSION, sizeof(api_version), &api_version, NULL);
    if (CL_SUCCESS != result)
    {
        KHR_ICD_TRACE("failed to query layer version\n");
        goto Done;
    }

    if (CL_LAYER_API_VERSION_100 != api_version)
    {
        KHR_ICD_TRACE("unsupported api version\n");
        goto Done;
    }

    layer = (struct KHRLayer*)calloc(sizeof(struct KHRLayer), 1);
    if (!layer)
    {
        KHR_ICD_TRACE("failed to allocate memory\n");
        goto Done;
    }
#ifdef CL_LAYER_INFO
    {
        // Not using strdup as it is not standard c
        size_t sz_name = strlen(libraryName) + 1;
        layer->libraryName = malloc(sz_name);
        if (!layer->libraryName)
        {
            KHR_ICD_TRACE("failed to allocate memory\n");
            goto Done;
        }
        memcpy(layer->libraryName, libraryName, sz_name);
        layer->p_clGetLayerInfo = (void *)(size_t)p_clGetLayerInfo;
    }
#endif

    if (khrFirstLayer) {
        targetDispatch = &(khrFirstLayer->dispatch);
    } else {
        targetDispatch = &khrMasterDispatch;
    }

    loaderDispatchNumEntries = sizeof(khrMasterDispatch)/sizeof(void*);
    result = p_clInitLayer(
        loaderDispatchNumEntries,
        targetDispatch,
        &layerDispatchNumEntries,
        &layerDispatch);
    if (CL_SUCCESS != result)
    {
        KHR_ICD_TRACE("failed to initialize layer\n");
        goto Done;
    }

    layer->next = khrFirstLayer;
    khrFirstLayer = layer;
    layer->library = library;

    cl_uint limit = layerDispatchNumEntries < loaderDispatchNumEntries ? layerDispatchNumEntries : loaderDispatchNumEntries;

    for (cl_uint i = 0; i < limit; i++) {
        ((void **)&(layer->dispatch))[i] =
            ((void *const*)layerDispatch)[i] ?
                ((void *const*)layerDispatch)[i] : ((void *const*)targetDispatch)[i];
    }
    for (cl_uint i = limit; i < loaderDispatchNumEntries; i++) {
        ((void **)&(layer->dispatch))[i] = ((void *const*)targetDispatch)[i];
    }

    KHR_ICD_TRACE("successfully added layer %s\n", libraryName);
    return;
Done:
    if (library)
    {
        khrIcdOsLibraryUnload(library);
    }
    if (layer)
    {
        free(layer);
    }
}
#endif // defined(CL_ENABLE_LAYERS)

// Get next file or dirname given a string list or registry key path.
// Note: the input string may be modified!
static char *loader_get_next_path(char *path) {
    size_t len;
    char *next;

    if (path == NULL) return NULL;
    next = strchr(path, PATH_SEPARATOR);
    if (next == NULL) {
        len = strlen(path);
        next = path + len;
    } else {
        *next = '\0';
        next++;
    }

    return next;
}

void khrIcdVendorsEnumerateEnv(void)
{
    char* icdFilenames = khrIcd_secure_getenv("OCL_ICD_FILENAMES");
    char* cur_file = NULL;
    char* next_file = NULL;
    if (icdFilenames)
    {
        KHR_ICD_TRACE("Found OCL_ICD_FILENAMES environment variable.\n");

        next_file = icdFilenames;
        while (NULL != next_file && *next_file != '\0') {
            cur_file = next_file;
            next_file = loader_get_next_path(cur_file);

            khrIcdVendorAdd(cur_file);
        }

        khrIcd_free_getenv(icdFilenames);
    }
}

#if defined(CL_ENABLE_LAYERS)
void khrIcdLayersEnumerateEnv(void)
{
    char* layerFilenames = khrIcd_secure_getenv("OPENCL_LAYERS");
    char* cur_file = NULL;
    char* next_file = NULL;
    if (layerFilenames)
    {
        KHR_ICD_TRACE("Found OPENCL_LAYERS environment variable.\n");

        next_file = layerFilenames;
        while (NULL != next_file && *next_file != '\0') {
            cur_file = next_file;
            next_file = loader_get_next_path(cur_file);

            khrIcdLayerAdd(cur_file);
        }

        khrIcd_free_getenv(layerFilenames);
    }
}
#endif // defined(CL_ENABLE_LAYERS)

void khrIcdContextPropertiesGetPlatform(const cl_context_properties *properties, cl_platform_id *outPlatform)
{
    if (properties == NULL && khrIcdVendors != NULL)
    {
        *outPlatform = khrIcdVendors[0].platform;
    }
    else
    {
        const cl_context_properties *property = (cl_context_properties *)NULL;
        *outPlatform = NULL;
        for (property = properties; property && property[0]; property += 2)
        {
            if ((cl_context_properties)CL_CONTEXT_PLATFORM == property[0])
            {
                *outPlatform = (cl_platform_id)property[1];
            }
        }
    }
}

