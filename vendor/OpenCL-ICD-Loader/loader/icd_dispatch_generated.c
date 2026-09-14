/*
 * Copyright (c) 2012-2023 The Khronos Group Inc.
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

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
// Core APIs:
#if defined(CL_ENABLE_LAYERS)
extern cl_int CL_API_CALL clGetPlatformIDs_disp(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms) CL_API_SUFFIX__VERSION_1_0;
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetPlatformInfo(
            platform,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return KHR_ICD2_DISPATCH(platform)->clGetPlatformInfo(
        platform,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetPlatformInfo_disp(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return KHR_ICD2_DISPATCH(platform)->clGetPlatformInfo(
        platform,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetDeviceIDs(
            platform,
            device_type,
            num_entries,
            devices,
            num_devices);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return KHR_ICD2_DISPATCH(platform)->clGetDeviceIDs(
        platform,
        device_type,
        num_entries,
        devices,
        num_devices);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetDeviceIDs_disp(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return KHR_ICD2_DISPATCH(platform)->clGetDeviceIDs(
        platform,
        device_type,
        num_entries,
        devices,
        num_devices);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetDeviceInfo(
            device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clGetDeviceInfo(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetDeviceInfo_disp(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clGetDeviceInfo(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateContext(
            properties,
            num_devices,
            devices,
            pfn_notify,
            user_data,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    if (num_devices == 0 || devices == NULL) {
        KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(NULL, CL_INVALID_VALUE);
    }
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(devices[0], CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(devices[0])->clCreateContext(
        properties,
        num_devices,
        devices,
        pfn_notify,
        user_data,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_context CL_API_CALL clCreateContext_disp(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    if (num_devices == 0 || devices == NULL) {
        KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(NULL, CL_INVALID_VALUE);
    }
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(devices[0], CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(devices[0])->clCreateContext(
        properties,
        num_devices,
        devices,
        pfn_notify,
        user_data,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_context CL_API_CALL clCreateContextFromType(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    khrIcdInitialize();
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateContextFromType(
            properties,
            device_type,
            pfn_notify,
            user_data,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    cl_platform_id platform = NULL;
    khrIcdContextPropertiesGetPlatform(properties, &platform);
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(platform, CL_INVALID_PLATFORM);
    return KHR_ICD2_DISPATCH(platform)->clCreateContextFromType(
        properties,
        device_type,
        pfn_notify,
        user_data,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_context CL_API_CALL clCreateContextFromType_disp(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    khrIcdInitialize();
    cl_platform_id platform = NULL;
    khrIcdContextPropertiesGetPlatform(properties, &platform);
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(platform, CL_INVALID_PLATFORM);
    return KHR_ICD2_DISPATCH(platform)->clCreateContextFromType(
        properties,
        device_type,
        pfn_notify,
        user_data,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clRetainContext(
    cl_context context)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainContext(
            context);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clRetainContext(
        context);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainContext_disp(
    cl_context context)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clRetainContext(
        context);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(
    cl_context context)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseContext(
            context);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clReleaseContext(
        context);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseContext_disp(
    cl_context context)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clReleaseContext(
        context);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetContextInfo(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetContextInfo(
            context,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clGetContextInfo(
        context,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetContextInfo_disp(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clGetContextInfo(
        context,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue(
    cl_command_queue command_queue)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainCommandQueue(
            command_queue);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clRetainCommandQueue(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainCommandQueue_disp(
    cl_command_queue command_queue)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clRetainCommandQueue(
        command_queue);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(
    cl_command_queue command_queue)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseCommandQueue(
            command_queue);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clReleaseCommandQueue(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseCommandQueue_disp(
    cl_command_queue command_queue)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clReleaseCommandQueue(
        command_queue);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetCommandQueueInfo(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetCommandQueueInfo(
            command_queue,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clGetCommandQueueInfo(
        command_queue,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetCommandQueueInfo_disp(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clGetCommandQueueInfo(
        command_queue,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateBuffer(
            context,
            flags,
            size,
            host_ptr,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateBuffer(
        context,
        flags,
        size,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateBuffer_disp(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateBuffer(
        context,
        flags,
        size,
        host_ptr,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject(
    cl_mem memobj)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainMemObject(
            memobj);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(memobj)->clRetainMemObject(
        memobj);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainMemObject_disp(
    cl_mem memobj)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(memobj)->clRetainMemObject(
        memobj);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(
    cl_mem memobj)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseMemObject(
            memobj);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(memobj)->clReleaseMemObject(
        memobj);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseMemObject_disp(
    cl_mem memobj)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(memobj)->clReleaseMemObject(
        memobj);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetSupportedImageFormats(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetSupportedImageFormats(
            context,
            flags,
            image_type,
            num_entries,
            image_formats,
            num_image_formats);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clGetSupportedImageFormats(
        context,
        flags,
        image_type,
        num_entries,
        image_formats,
        num_image_formats);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetSupportedImageFormats_disp(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clGetSupportedImageFormats(
        context,
        flags,
        image_type,
        num_entries,
        image_formats,
        num_image_formats);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetMemObjectInfo(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetMemObjectInfo(
            memobj,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(memobj)->clGetMemObjectInfo(
        memobj,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetMemObjectInfo_disp(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(memobj)->clGetMemObjectInfo(
        memobj,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetImageInfo(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetImageInfo(
            image,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(image, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(image)->clGetImageInfo(
        image,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetImageInfo_disp(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(image, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(image)->clGetImageInfo(
        image,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clRetainSampler(
    cl_sampler sampler)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainSampler(
            sampler);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return KHR_ICD2_DISPATCH(sampler)->clRetainSampler(
        sampler);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainSampler_disp(
    cl_sampler sampler)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return KHR_ICD2_DISPATCH(sampler)->clRetainSampler(
        sampler);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clReleaseSampler(
    cl_sampler sampler)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseSampler(
            sampler);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return KHR_ICD2_DISPATCH(sampler)->clReleaseSampler(
        sampler);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseSampler_disp(
    cl_sampler sampler)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return KHR_ICD2_DISPATCH(sampler)->clReleaseSampler(
        sampler);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetSamplerInfo(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetSamplerInfo(
            sampler,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return KHR_ICD2_DISPATCH(sampler)->clGetSamplerInfo(
        sampler,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetSamplerInfo_disp(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return KHR_ICD2_DISPATCH(sampler)->clGetSamplerInfo(
        sampler,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateProgramWithSource(
            context,
            count,
            strings,
            lengths,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateProgramWithSource(
        context,
        count,
        strings,
        lengths,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_program CL_API_CALL clCreateProgramWithSource_disp(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateProgramWithSource(
        context,
        count,
        strings,
        lengths,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBinary(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateProgramWithBinary(
            context,
            num_devices,
            device_list,
            lengths,
            binaries,
            binary_status,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateProgramWithBinary(
        context,
        num_devices,
        device_list,
        lengths,
        binaries,
        binary_status,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_program CL_API_CALL clCreateProgramWithBinary_disp(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateProgramWithBinary(
        context,
        num_devices,
        device_list,
        lengths,
        binaries,
        binary_status,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clRetainProgram(
    cl_program program)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainProgram(
            program);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clRetainProgram(
        program);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainProgram_disp(
    cl_program program)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clRetainProgram(
        program);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(
    cl_program program)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseProgram(
            program);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clReleaseProgram(
        program);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseProgram_disp(
    cl_program program)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clReleaseProgram(
        program);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clBuildProgram(
            program,
            num_devices,
            device_list,
            options,
            pfn_notify,
            user_data);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clBuildProgram(
        program,
        num_devices,
        device_list,
        options,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clBuildProgram_disp(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clBuildProgram(
        program,
        num_devices,
        device_list,
        options,
        pfn_notify,
        user_data);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetProgramInfo(
            program,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clGetProgramInfo(
        program,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetProgramInfo_disp(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clGetProgramInfo(
        program,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetProgramBuildInfo(
            program,
            device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clGetProgramBuildInfo(
        program,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetProgramBuildInfo_disp(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clGetProgramBuildInfo(
        program,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateKernel(
            program,
            kernel_name,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clCreateKernel(
        program,
        kernel_name,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_kernel CL_API_CALL clCreateKernel_disp(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clCreateKernel(
        program,
        kernel_name,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateKernelsInProgram(
            program,
            num_kernels,
            kernels,
            num_kernels_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clCreateKernelsInProgram(
        program,
        num_kernels,
        kernels,
        num_kernels_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clCreateKernelsInProgram_disp(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clCreateKernelsInProgram(
        program,
        num_kernels,
        kernels,
        num_kernels_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clRetainKernel(
    cl_kernel kernel)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainKernel(
            kernel);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clRetainKernel(
        kernel);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainKernel_disp(
    cl_kernel kernel)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clRetainKernel(
        kernel);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(
    cl_kernel kernel)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseKernel(
            kernel);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clReleaseKernel(
        kernel);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseKernel_disp(
    cl_kernel kernel)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clReleaseKernel(
        kernel);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetKernelArg(
            kernel,
            arg_index,
            arg_size,
            arg_value);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clSetKernelArg(
        kernel,
        arg_index,
        arg_size,
        arg_value);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetKernelArg_disp(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clSetKernelArg(
        kernel,
        arg_index,
        arg_size,
        arg_value);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetKernelInfo(
            kernel,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clGetKernelInfo(
        kernel,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetKernelInfo_disp(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clGetKernelInfo(
        kernel,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetKernelWorkGroupInfo(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetKernelWorkGroupInfo(
            kernel,
            device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clGetKernelWorkGroupInfo(
        kernel,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetKernelWorkGroupInfo_disp(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clGetKernelWorkGroupInfo(
        kernel,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents(
    cl_uint num_events,
    const cl_event* event_list)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clWaitForEvents(
            num_events,
            event_list);
#endif // defined(CL_ENABLE_LAYERS)
    if (num_events == 0 || event_list == NULL) {
        return CL_INVALID_VALUE;
    }
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event_list[0], CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event_list[0])->clWaitForEvents(
        num_events,
        event_list);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clWaitForEvents_disp(
    cl_uint num_events,
    const cl_event* event_list)
{
    if (num_events == 0 || event_list == NULL) {
        return CL_INVALID_VALUE;
    }
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event_list[0], CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event_list[0])->clWaitForEvents(
        num_events,
        event_list);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetEventInfo(
            event,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clGetEventInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetEventInfo_disp(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clGetEventInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clRetainEvent(
    cl_event event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainEvent(
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clRetainEvent(
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainEvent_disp(
    cl_event event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clRetainEvent(
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent(
    cl_event event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseEvent(
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clReleaseEvent(
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseEvent_disp(
    cl_event event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clReleaseEvent(
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetEventProfilingInfo(
            event,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clGetEventProfilingInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetEventProfilingInfo_disp(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clGetEventProfilingInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clFlush(
    cl_command_queue command_queue)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clFlush(
            command_queue);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clFlush(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clFlush_disp(
    cl_command_queue command_queue)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clFlush(
        command_queue);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clFinish(
    cl_command_queue command_queue)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clFinish(
            command_queue);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clFinish(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clFinish_disp(
    cl_command_queue command_queue)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clFinish(
        command_queue);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueReadBuffer(
            command_queue,
            buffer,
            blocking_read,
            offset,
            size,
            ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReadBuffer(
        command_queue,
        buffer,
        blocking_read,
        offset,
        size,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueReadBuffer_disp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReadBuffer(
        command_queue,
        buffer,
        blocking_read,
        offset,
        size,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueWriteBuffer(
            command_queue,
            buffer,
            blocking_write,
            offset,
            size,
            ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueWriteBuffer(
        command_queue,
        buffer,
        blocking_write,
        offset,
        size,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueWriteBuffer_disp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueWriteBuffer(
        command_queue,
        buffer,
        blocking_write,
        offset,
        size,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueCopyBuffer(
            command_queue,
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyBuffer(
        command_queue,
        src_buffer,
        dst_buffer,
        src_offset,
        dst_offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueCopyBuffer_disp(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyBuffer(
        command_queue,
        src_buffer,
        dst_buffer,
        src_offset,
        dst_offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadImage(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_read,
    const size_t* origin,
    const size_t* region,
    size_t row_pitch,
    size_t slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueReadImage(
            command_queue,
            image,
            blocking_read,
            origin,
            region,
            row_pitch,
            slice_pitch,
            ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReadImage(
        command_queue,
        image,
        blocking_read,
        origin,
        region,
        row_pitch,
        slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueReadImage_disp(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_read,
    const size_t* origin,
    const size_t* region,
    size_t row_pitch,
    size_t slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReadImage(
        command_queue,
        image,
        blocking_read,
        origin,
        region,
        row_pitch,
        slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteImage(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_write,
    const size_t* origin,
    const size_t* region,
    size_t input_row_pitch,
    size_t input_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueWriteImage(
            command_queue,
            image,
            blocking_write,
            origin,
            region,
            input_row_pitch,
            input_slice_pitch,
            ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueWriteImage(
        command_queue,
        image,
        blocking_write,
        origin,
        region,
        input_row_pitch,
        input_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueWriteImage_disp(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_write,
    const size_t* origin,
    const size_t* region,
    size_t input_row_pitch,
    size_t input_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueWriteImage(
        command_queue,
        image,
        blocking_write,
        origin,
        region,
        input_row_pitch,
        input_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImage(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueCopyImage(
            command_queue,
            src_image,
            dst_image,
            src_origin,
            dst_origin,
            region,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyImage(
        command_queue,
        src_image,
        dst_image,
        src_origin,
        dst_origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueCopyImage_disp(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyImage(
        command_queue,
        src_image,
        dst_image,
        src_origin,
        dst_origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImageToBuffer(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueCopyImageToBuffer(
            command_queue,
            src_image,
            dst_buffer,
            src_origin,
            region,
            dst_offset,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyImageToBuffer(
        command_queue,
        src_image,
        dst_buffer,
        src_origin,
        region,
        dst_offset,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueCopyImageToBuffer_disp(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyImageToBuffer(
        command_queue,
        src_image,
        dst_buffer,
        src_origin,
        region,
        dst_offset,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferToImage(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueCopyBufferToImage(
            command_queue,
            src_buffer,
            dst_image,
            src_offset,
            dst_origin,
            region,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyBufferToImage(
        command_queue,
        src_buffer,
        dst_image,
        src_offset,
        dst_origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueCopyBufferToImage_disp(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyBufferToImage(
        command_queue,
        src_buffer,
        dst_image,
        src_offset,
        dst_origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY void* CL_API_CALL clEnqueueMapBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueMapBuffer(
            command_queue,
            buffer,
            blocking_map,
            map_flags,
            offset,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMapBuffer(
        command_queue,
        buffer,
        blocking_map,
        map_flags,
        offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static void* CL_API_CALL clEnqueueMapBuffer_disp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMapBuffer(
        command_queue,
        buffer,
        blocking_map,
        map_flags,
        offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY void* CL_API_CALL clEnqueueMapImage(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    const size_t* origin,
    const size_t* region,
    size_t* image_row_pitch,
    size_t* image_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueMapImage(
            command_queue,
            image,
            blocking_map,
            map_flags,
            origin,
            region,
            image_row_pitch,
            image_slice_pitch,
            num_events_in_wait_list,
            event_wait_list,
            event,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMapImage(
        command_queue,
        image,
        blocking_map,
        map_flags,
        origin,
        region,
        image_row_pitch,
        image_slice_pitch,
        num_events_in_wait_list,
        event_wait_list,
        event,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static void* CL_API_CALL clEnqueueMapImage_disp(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    const size_t* origin,
    const size_t* region,
    size_t* image_row_pitch,
    size_t* image_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMapImage(
        command_queue,
        image,
        blocking_map,
        map_flags,
        origin,
        region,
        image_row_pitch,
        image_slice_pitch,
        num_events_in_wait_list,
        event_wait_list,
        event,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueUnmapMemObject(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueUnmapMemObject(
            command_queue,
            memobj,
            mapped_ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueUnmapMemObject(
        command_queue,
        memobj,
        mapped_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueUnmapMemObject_disp(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueUnmapMemObject(
        command_queue,
        memobj,
        mapped_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            work_dim,
            global_work_offset,
            global_work_size,
            local_work_size,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        work_dim,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueNDRangeKernel_disp(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        work_dim,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNativeKernel(
    cl_command_queue command_queue,
    void (CL_CALLBACK* user_func)(void*),
    void* args,
    size_t cb_args,
    cl_uint num_mem_objects,
    const cl_mem* mem_list,
    const void** args_mem_loc,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueNativeKernel(
            command_queue,
            user_func,
            args,
            cb_args,
            num_mem_objects,
            mem_list,
            args_mem_loc,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueNativeKernel(
        command_queue,
        user_func,
        args,
        cb_args,
        num_mem_objects,
        mem_list,
        args_mem_loc,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueNativeKernel_disp(
    cl_command_queue command_queue,
    void (CL_CALLBACK* user_func)(void*),
    void* args,
    size_t cb_args,
    cl_uint num_mem_objects,
    const cl_mem* mem_list,
    const void** args_mem_loc,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueNativeKernel(
        command_queue,
        user_func,
        args,
        cb_args,
        num_mem_objects,
        mem_list,
        args_mem_loc,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetCommandQueueProperty(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetCommandQueueProperty(
            command_queue,
            properties,
            enable,
            old_properties);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clSetCommandQueueProperty(
        command_queue,
        properties,
        enable,
        old_properties);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetCommandQueueProperty_disp(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clSetCommandQueueProperty(
        command_queue,
        properties,
        enable,
        old_properties);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage2D(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateImage2D(
            context,
            flags,
            image_format,
            image_width,
            image_height,
            image_row_pitch,
            host_ptr,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateImage2D(
        context,
        flags,
        image_format,
        image_width,
        image_height,
        image_row_pitch,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateImage2D_disp(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateImage2D(
        context,
        flags,
        image_format,
        image_width,
        image_height,
        image_row_pitch,
        host_ptr,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage3D(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_depth,
    size_t image_row_pitch,
    size_t image_slice_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateImage3D(
            context,
            flags,
            image_format,
            image_width,
            image_height,
            image_depth,
            image_row_pitch,
            image_slice_pitch,
            host_ptr,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateImage3D(
        context,
        flags,
        image_format,
        image_width,
        image_height,
        image_depth,
        image_row_pitch,
        image_slice_pitch,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateImage3D_disp(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_depth,
    size_t image_row_pitch,
    size_t image_slice_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateImage3D(
        context,
        flags,
        image_format,
        image_width,
        image_height,
        image_depth,
        image_row_pitch,
        image_slice_pitch,
        host_ptr,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarker(
    cl_command_queue command_queue,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueMarker(
            command_queue,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMarker(
        command_queue,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueMarker_disp(
    cl_command_queue command_queue,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMarker(
        command_queue,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWaitForEvents(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueWaitForEvents(
            command_queue,
            num_events,
            event_list);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueWaitForEvents(
        command_queue,
        num_events,
        event_list);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueWaitForEvents_disp(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueWaitForEvents(
        command_queue,
        num_events,
        event_list);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrier(
    cl_command_queue command_queue)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueBarrier(
            command_queue);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueBarrier(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueBarrier_disp(
    cl_command_queue command_queue)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueBarrier(
        command_queue);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clUnloadCompiler(
    void )
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clUnloadCompiler(
            );
#endif // defined(CL_ENABLE_LAYERS)
    // Nothing!
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clUnloadCompiler_disp(
    void )
{
    // Nothing!
    return CL_SUCCESS;
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
extern void* CL_API_CALL clGetExtensionFunctionAddress_disp(
    const char* func_name) CL_API_SUFFIX__VERSION_1_1_DEPRECATED;
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateCommandQueue(
            context,
            device,
            properties,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateCommandQueue(
        context,
        device,
        properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_command_queue CL_API_CALL clCreateCommandQueue_disp(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateCommandQueue(
        context,
        device,
        properties,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSampler(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateSampler(
            context,
            normalized_coords,
            addressing_mode,
            filter_mode,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateSampler(
        context,
        normalized_coords,
        addressing_mode,
        filter_mode,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_sampler CL_API_CALL clCreateSampler_disp(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateSampler(
        context,
        normalized_coords,
        addressing_mode,
        filter_mode,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueTask(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueTask(
            command_queue,
            kernel,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueTask(
        command_queue,
        kernel,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueTask_disp(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueTask(
        command_queue,
        kernel,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateSubBuffer(
            buffer,
            flags,
            buffer_create_type,
            buffer_create_info,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(buffer, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(buffer)->clCreateSubBuffer(
        buffer,
        flags,
        buffer_create_type,
        buffer_create_info,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateSubBuffer_disp(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(buffer, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(buffer)->clCreateSubBuffer(
        buffer,
        flags,
        buffer_create_type,
        buffer_create_info,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetMemObjectDestructorCallback(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetMemObjectDestructorCallback(
            memobj,
            pfn_notify,
            user_data);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(memobj)->clSetMemObjectDestructorCallback(
        memobj,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetMemObjectDestructorCallback_disp(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(memobj)->clSetMemObjectDestructorCallback(
        memobj,
        pfn_notify,
        user_data);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_event CL_API_CALL clCreateUserEvent(
    cl_context context,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateUserEvent(
            context,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateUserEvent(
        context,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_event CL_API_CALL clCreateUserEvent_disp(
    cl_context context,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateUserEvent(
        context,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetUserEventStatus(
    cl_event event,
    cl_int execution_status)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetUserEventStatus(
            event,
            execution_status);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clSetUserEventStatus(
        event,
        execution_status);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetUserEventStatus_disp(
    cl_event event,
    cl_int execution_status)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clSetUserEventStatus(
        event,
        execution_status);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetEventCallback(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data),
    void* user_data)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetEventCallback(
            event,
            command_exec_callback_type,
            pfn_notify,
            user_data);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clSetEventCallback(
        event,
        command_exec_callback_type,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetEventCallback_disp(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data),
    void* user_data)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return KHR_ICD2_DISPATCH(event)->clSetEventCallback(
        event,
        command_exec_callback_type,
        pfn_notify,
        user_data);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBufferRect(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueReadBufferRect(
            command_queue,
            buffer,
            blocking_read,
            buffer_origin,
            host_origin,
            region,
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReadBufferRect(
        command_queue,
        buffer,
        blocking_read,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueReadBufferRect_disp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReadBufferRect(
        command_queue,
        buffer,
        blocking_read,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBufferRect(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueWriteBufferRect(
            command_queue,
            buffer,
            blocking_write,
            buffer_origin,
            host_origin,
            region,
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueWriteBufferRect(
        command_queue,
        buffer,
        blocking_write,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueWriteBufferRect_disp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueWriteBufferRect(
        command_queue,
        buffer,
        blocking_write,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferRect(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueCopyBufferRect(
            command_queue,
            src_buffer,
            dst_buffer,
            src_origin,
            dst_origin,
            region,
            src_row_pitch,
            src_slice_pitch,
            dst_row_pitch,
            dst_slice_pitch,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyBufferRect(
        command_queue,
        src_buffer,
        dst_buffer,
        src_origin,
        dst_origin,
        region,
        src_row_pitch,
        src_slice_pitch,
        dst_row_pitch,
        dst_slice_pitch,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueCopyBufferRect_disp(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueCopyBufferRect(
        command_queue,
        src_buffer,
        dst_buffer,
        src_origin,
        dst_origin,
        region,
        src_row_pitch,
        src_slice_pitch,
        dst_row_pitch,
        dst_slice_pitch,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clCreateSubDevices(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateSubDevices(
            in_device,
            properties,
            num_devices,
            out_devices,
            num_devices_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(in_device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(in_device)->clCreateSubDevices(
        in_device,
        properties,
        num_devices,
        out_devices,
        num_devices_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clCreateSubDevices_disp(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(in_device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(in_device)->clCreateSubDevices(
        in_device,
        properties,
        num_devices,
        out_devices,
        num_devices_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clRetainDevice(
    cl_device_id device)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainDevice(
            device);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clRetainDevice(
        device);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainDevice_disp(
    cl_device_id device)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clRetainDevice(
        device);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice(
    cl_device_id device)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseDevice(
            device);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clReleaseDevice(
        device);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseDevice_disp(
    cl_device_id device)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clReleaseDevice(
        device);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateImage(
            context,
            flags,
            image_format,
            image_desc,
            host_ptr,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateImage(
        context,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateImage_disp(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateImage(
        context,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBuiltInKernels(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateProgramWithBuiltInKernels(
            context,
            num_devices,
            device_list,
            kernel_names,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateProgramWithBuiltInKernels(
        context,
        num_devices,
        device_list,
        kernel_names,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_program CL_API_CALL clCreateProgramWithBuiltInKernels_disp(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateProgramWithBuiltInKernels(
        context,
        num_devices,
        device_list,
        kernel_names,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clCompileProgram(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_headers,
    const cl_program* input_headers,
    const char** header_include_names,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCompileProgram(
            program,
            num_devices,
            device_list,
            options,
            num_input_headers,
            input_headers,
            header_include_names,
            pfn_notify,
            user_data);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clCompileProgram(
        program,
        num_devices,
        device_list,
        options,
        num_input_headers,
        input_headers,
        header_include_names,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clCompileProgram_disp(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_headers,
    const cl_program* input_headers,
    const char** header_include_names,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clCompileProgram(
        program,
        num_devices,
        device_list,
        options,
        num_input_headers,
        input_headers,
        header_include_names,
        pfn_notify,
        user_data);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_program CL_API_CALL clLinkProgram(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_programs,
    const cl_program* input_programs,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clLinkProgram(
            context,
            num_devices,
            device_list,
            options,
            num_input_programs,
            input_programs,
            pfn_notify,
            user_data,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clLinkProgram(
        context,
        num_devices,
        device_list,
        options,
        num_input_programs,
        input_programs,
        pfn_notify,
        user_data,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_program CL_API_CALL clLinkProgram_disp(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_programs,
    const cl_program* input_programs,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clLinkProgram(
        context,
        num_devices,
        device_list,
        options,
        num_input_programs,
        input_programs,
        pfn_notify,
        user_data,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clUnloadPlatformCompiler(
    cl_platform_id platform)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clUnloadPlatformCompiler(
            platform);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return KHR_ICD2_DISPATCH(platform)->clUnloadPlatformCompiler(
        platform);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clUnloadPlatformCompiler_disp(
    cl_platform_id platform)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return KHR_ICD2_DISPATCH(platform)->clUnloadPlatformCompiler(
        platform);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetKernelArgInfo(
    cl_kernel kernel,
    cl_uint arg_index,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetKernelArgInfo(
            kernel,
            arg_index,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clGetKernelArgInfo(
        kernel,
        arg_index,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetKernelArgInfo_disp(
    cl_kernel kernel,
    cl_uint arg_index,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clGetKernelArgInfo(
        kernel,
        arg_index,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueFillBuffer(
            command_queue,
            buffer,
            pattern,
            pattern_size,
            offset,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueFillBuffer(
        command_queue,
        buffer,
        pattern,
        pattern_size,
        offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueFillBuffer_disp(
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueFillBuffer(
        command_queue,
        buffer,
        pattern,
        pattern_size,
        offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillImage(
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueFillImage(
            command_queue,
            image,
            fill_color,
            origin,
            region,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueFillImage(
        command_queue,
        image,
        fill_color,
        origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueFillImage_disp(
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueFillImage(
        command_queue,
        image,
        fill_color,
        origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemObjects(
    cl_command_queue command_queue,
    cl_uint num_mem_objects,
    const cl_mem* mem_objects,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueMigrateMemObjects(
            command_queue,
            num_mem_objects,
            mem_objects,
            flags,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMigrateMemObjects(
        command_queue,
        num_mem_objects,
        mem_objects,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueMigrateMemObjects_disp(
    cl_command_queue command_queue,
    cl_uint num_mem_objects,
    const cl_mem* mem_objects,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMigrateMemObjects(
        command_queue,
        num_mem_objects,
        mem_objects,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarkerWithWaitList(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueMarkerWithWaitList(
            command_queue,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMarkerWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueMarkerWithWaitList_disp(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueMarkerWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrierWithWaitList(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueBarrierWithWaitList(
            command_queue,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueBarrierWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueBarrierWithWaitList_disp(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueBarrierWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
extern void* CL_API_CALL clGetExtensionFunctionAddressForPlatform_disp(
    cl_platform_id platform,
    const char* func_name) CL_API_SUFFIX__VERSION_1_2;
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateCommandQueueWithProperties(
            context,
            device,
            properties,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateCommandQueueWithProperties(
        context,
        device,
        properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties_disp(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateCommandQueueWithProperties(
        context,
        device,
        properties,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_mem CL_API_CALL clCreatePipe(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreatePipe(
            context,
            flags,
            pipe_packet_size,
            pipe_max_packets,
            properties,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreatePipe(
        context,
        flags,
        pipe_packet_size,
        pipe_max_packets,
        properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreatePipe_disp(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreatePipe(
        context,
        flags,
        pipe_packet_size,
        pipe_max_packets,
        properties,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetPipeInfo(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetPipeInfo(
            pipe,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(pipe, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(pipe)->clGetPipeInfo(
        pipe,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetPipeInfo_disp(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(pipe, CL_INVALID_MEM_OBJECT);
    return KHR_ICD2_DISPATCH(pipe)->clGetPipeInfo(
        pipe,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY void* CL_API_CALL clSVMAlloc(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSVMAlloc(
            context,
            flags,
            size,
            alignment);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, NULL);
    return KHR_ICD2_DISPATCH(context)->clSVMAlloc(
        context,
        flags,
        size,
        alignment);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static void* CL_API_CALL clSVMAlloc_disp(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, NULL);
    return KHR_ICD2_DISPATCH(context)->clSVMAlloc(
        context,
        flags,
        size,
        alignment);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY void CL_API_CALL clSVMFree(
    cl_context context,
    void* svm_pointer)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
    {
        khrFirstLayer->dispatch.clSVMFree(
            context,
            svm_pointer);
    }
    else
#endif // defined(CL_ENABLE_LAYERS)
    if (context != NULL)
    KHR_ICD2_DISPATCH(context)->clSVMFree(
        context,
        svm_pointer);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static void CL_API_CALL clSVMFree_disp(
    cl_context context,
    void* svm_pointer)
{
    if (context != NULL)
    KHR_ICD2_DISPATCH(context)->clSVMFree(
        context,
        svm_pointer);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSamplerWithProperties(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateSamplerWithProperties(
            context,
            sampler_properties,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateSamplerWithProperties(
        context,
        sampler_properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_sampler CL_API_CALL clCreateSamplerWithProperties_disp(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateSamplerWithProperties(
        context,
        sampler_properties,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgSVMPointer(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetKernelArgSVMPointer(
            kernel,
            arg_index,
            arg_value);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clSetKernelArgSVMPointer(
        kernel,
        arg_index,
        arg_value);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetKernelArgSVMPointer_disp(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clSetKernelArgSVMPointer(
        kernel,
        arg_index,
        arg_value);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetKernelExecInfo(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetKernelExecInfo(
            kernel,
            param_name,
            param_value_size,
            param_value);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clSetKernelExecInfo(
        kernel,
        param_name,
        param_value_size,
        param_value);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetKernelExecInfo_disp(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clSetKernelExecInfo(
        kernel,
        param_name,
        param_value_size,
        param_value);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMFree(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    void* svm_pointers[],
    void (CL_CALLBACK* pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers, void* svm_pointers[], void* user_data),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueSVMFree(
            command_queue,
            num_svm_pointers,
            svm_pointers,
            pfn_free_func,
            user_data,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMFree(
        command_queue,
        num_svm_pointers,
        svm_pointers,
        pfn_free_func,
        user_data,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueSVMFree_disp(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    void* svm_pointers[],
    void (CL_CALLBACK* pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers, void* svm_pointers[], void* user_data),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMFree(
        command_queue,
        num_svm_pointers,
        svm_pointers,
        pfn_free_func,
        user_data,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMemcpy(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueSVMMemcpy(
            command_queue,
            blocking_copy,
            dst_ptr,
            src_ptr,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMMemcpy(
        command_queue,
        blocking_copy,
        dst_ptr,
        src_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueSVMMemcpy_disp(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMMemcpy(
        command_queue,
        blocking_copy,
        dst_ptr,
        src_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMemFill(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueSVMMemFill(
            command_queue,
            svm_ptr,
            pattern,
            pattern_size,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMMemFill(
        command_queue,
        svm_ptr,
        pattern,
        pattern_size,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueSVMMemFill_disp(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMMemFill(
        command_queue,
        svm_ptr,
        pattern,
        pattern_size,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMap(
    cl_command_queue command_queue,
    cl_bool blocking_map,
    cl_map_flags flags,
    void* svm_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueSVMMap(
            command_queue,
            blocking_map,
            flags,
            svm_ptr,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMMap(
        command_queue,
        blocking_map,
        flags,
        svm_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueSVMMap_disp(
    cl_command_queue command_queue,
    cl_bool blocking_map,
    cl_map_flags flags,
    void* svm_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMMap(
        command_queue,
        blocking_map,
        flags,
        svm_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMUnmap(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueSVMUnmap(
            command_queue,
            svm_ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMUnmap(
        command_queue,
        svm_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueSVMUnmap_disp(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMUnmap(
        command_queue,
        svm_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetDefaultDeviceCommandQueue(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetDefaultDeviceCommandQueue(
            context,
            device,
            command_queue);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clSetDefaultDeviceCommandQueue(
        context,
        device,
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetDefaultDeviceCommandQueue_disp(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clSetDefaultDeviceCommandQueue(
        context,
        device,
        command_queue);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceAndHostTimer(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetDeviceAndHostTimer(
            device,
            device_timestamp,
            host_timestamp);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clGetDeviceAndHostTimer(
        device,
        device_timestamp,
        host_timestamp);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetDeviceAndHostTimer_disp(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clGetDeviceAndHostTimer(
        device,
        device_timestamp,
        host_timestamp);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetHostTimer(
    cl_device_id device,
    cl_ulong* host_timestamp)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetHostTimer(
            device,
            host_timestamp);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clGetHostTimer(
        device,
        host_timestamp);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetHostTimer_disp(
    cl_device_id device,
    cl_ulong* host_timestamp)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return KHR_ICD2_DISPATCH(device)->clGetHostTimer(
        device,
        host_timestamp);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateProgramWithIL(
            context,
            il,
            length,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateProgramWithIL(
        context,
        il,
        length,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_program CL_API_CALL clCreateProgramWithIL_disp(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateProgramWithIL(
        context,
        il,
        length,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_kernel CL_API_CALL clCloneKernel(
    cl_kernel source_kernel,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCloneKernel(
            source_kernel,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(source_kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(source_kernel)->clCloneKernel(
        source_kernel,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_kernel CL_API_CALL clCloneKernel_disp(
    cl_kernel source_kernel,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(source_kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(source_kernel)->clCloneKernel(
        source_kernel,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clGetKernelSubGroupInfo(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetKernelSubGroupInfo(
            kernel,
            device,
            param_name,
            input_value_size,
            input_value,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clGetKernelSubGroupInfo(
        kernel,
        device,
        param_name,
        input_value_size,
        input_value,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetKernelSubGroupInfo_disp(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return KHR_ICD2_DISPATCH(kernel)->clGetKernelSubGroupInfo(
        kernel,
        device,
        param_name,
        input_value_size,
        input_value,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMigrateMem(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueSVMMigrateMem(
            command_queue,
            num_svm_pointers,
            svm_pointers,
            sizes,
            flags,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMMigrateMem(
        command_queue,
        num_svm_pointers,
        svm_pointers,
        sizes,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueSVMMigrateMem_disp(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueSVMMigrateMem(
        command_queue,
        num_svm_pointers,
        svm_pointers,
        sizes,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetProgramSpecializationConstant(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetProgramSpecializationConstant(
            program,
            spec_id,
            spec_size,
            spec_value);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clSetProgramSpecializationConstant(
        program,
        spec_id,
        spec_size,
        spec_value);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetProgramSpecializationConstant_disp(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clSetProgramSpecializationConstant(
        program,
        spec_id,
        spec_size,
        spec_value);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetProgramReleaseCallback(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetProgramReleaseCallback(
            program,
            pfn_notify,
            user_data);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clSetProgramReleaseCallback(
        program,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetProgramReleaseCallback_disp(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return KHR_ICD2_DISPATCH(program)->clSetProgramReleaseCallback(
        program,
        pfn_notify,
        user_data);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL clSetContextDestructorCallback(
    cl_context context,
    void (CL_CALLBACK* pfn_notify)(cl_context context, void* user_data),
    void* user_data)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clSetContextDestructorCallback(
            context,
            pfn_notify,
            user_data);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clSetContextDestructorCallback(
        context,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clSetContextDestructorCallback_disp(
    cl_context context,
    void (CL_CALLBACK* pfn_notify)(cl_context context, void* user_data),
    void* user_data)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clSetContextDestructorCallback(
        context,
        pfn_notify,
        user_data);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_mem CL_API_CALL clCreateBufferWithProperties(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateBufferWithProperties(
            context,
            properties,
            flags,
            size,
            host_ptr,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateBufferWithProperties(
        context,
        properties,
        flags,
        size,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateBufferWithProperties_disp(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateBufferWithProperties(
        context,
        properties,
        flags,
        size,
        host_ptr,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_mem CL_API_CALL clCreateImageWithProperties(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateImageWithProperties(
            context,
            properties,
            flags,
            image_format,
            image_desc,
            host_ptr,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateImageWithProperties(
        context,
        properties,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateImageWithProperties_disp(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return KHR_ICD2_DISPATCH(context)->clCreateImageWithProperties(
        context,
        properties,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

// cl_ext_device_fission

CL_API_ENTRY cl_int CL_API_CALL clReleaseDeviceEXT(
    cl_device_id device)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clReleaseDeviceEXT(
            device);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(device)->clReleaseDeviceEXT);
    return KHR_ICD2_DISPATCH(device)->clReleaseDeviceEXT(
        device);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clReleaseDeviceEXT_disp(
    cl_device_id device)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(device)->clReleaseDeviceEXT);
    return KHR_ICD2_DISPATCH(device)->clReleaseDeviceEXT(
        device);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clRetainDeviceEXT(
    cl_device_id device)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clRetainDeviceEXT(
            device);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(device)->clRetainDeviceEXT);
    return KHR_ICD2_DISPATCH(device)->clRetainDeviceEXT(
        device);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clRetainDeviceEXT_disp(
    cl_device_id device)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(device)->clRetainDeviceEXT);
    return KHR_ICD2_DISPATCH(device)->clRetainDeviceEXT(
        device);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clCreateSubDevicesEXT(
    cl_device_id in_device,
    const cl_device_partition_property_ext* properties,
    cl_uint num_entries,
    cl_device_id* out_devices,
    cl_uint* num_devices)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateSubDevicesEXT(
            in_device,
            properties,
            num_entries,
            out_devices,
            num_devices);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(in_device, CL_INVALID_DEVICE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(in_device)->clCreateSubDevicesEXT);
    return KHR_ICD2_DISPATCH(in_device)->clCreateSubDevicesEXT(
        in_device,
        properties,
        num_entries,
        out_devices,
        num_devices);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clCreateSubDevicesEXT_disp(
    cl_device_id in_device,
    const cl_device_partition_property_ext* properties,
    cl_uint num_entries,
    cl_device_id* out_devices,
    cl_uint* num_devices)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(in_device, CL_INVALID_DEVICE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(in_device)->clCreateSubDevicesEXT);
    return KHR_ICD2_DISPATCH(in_device)->clCreateSubDevicesEXT(
        in_device,
        properties,
        num_entries,
        out_devices,
        num_devices);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_d3d10_sharing

#if defined(_WIN32)

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDsFromD3D10KHR(
    cl_platform_id platform,
    cl_d3d10_device_source_khr d3d_device_source,
    void* d3d_object,
    cl_d3d10_device_set_khr d3d_device_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetDeviceIDsFromD3D10KHR(
            platform,
            d3d_device_source,
            d3d_object,
            d3d_device_set,
            num_entries,
            devices,
            num_devices);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromD3D10KHR);
    return KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromD3D10KHR(
        platform,
        d3d_device_source,
        d3d_object,
        d3d_device_set,
        num_entries,
        devices,
        num_devices);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetDeviceIDsFromD3D10KHR_disp(
    cl_platform_id platform,
    cl_d3d10_device_source_khr d3d_device_source,
    void* d3d_object,
    cl_d3d10_device_set_khr d3d_device_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromD3D10KHR);
    return KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromD3D10KHR(
        platform,
        d3d_device_source,
        d3d_object,
        d3d_device_set,
        num_entries,
        devices,
        num_devices);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D10BufferKHR(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Buffer* resource,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromD3D10BufferKHR(
            context,
            flags,
            resource,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D10BufferKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D10BufferKHR(
        context,
        flags,
        resource,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromD3D10BufferKHR_disp(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Buffer* resource,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D10BufferKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D10BufferKHR(
        context,
        flags,
        resource,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D10Texture2DKHR(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromD3D10Texture2DKHR(
            context,
            flags,
            resource,
            subresource,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D10Texture2DKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D10Texture2DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromD3D10Texture2DKHR_disp(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D10Texture2DKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D10Texture2DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D10Texture3DKHR(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromD3D10Texture3DKHR(
            context,
            flags,
            resource,
            subresource,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D10Texture3DKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D10Texture3DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromD3D10Texture3DKHR_disp(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D10Texture3DKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D10Texture3DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireD3D10ObjectsKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueAcquireD3D10ObjectsKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireD3D10ObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireD3D10ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueAcquireD3D10ObjectsKHR_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireD3D10ObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireD3D10ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseD3D10ObjectsKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueReleaseD3D10ObjectsKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseD3D10ObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseD3D10ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueReleaseD3D10ObjectsKHR_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseD3D10ObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseD3D10ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_d3d11_sharing

#if defined(_WIN32)

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDsFromD3D11KHR(
    cl_platform_id platform,
    cl_d3d11_device_source_khr d3d_device_source,
    void* d3d_object,
    cl_d3d11_device_set_khr d3d_device_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetDeviceIDsFromD3D11KHR(
            platform,
            d3d_device_source,
            d3d_object,
            d3d_device_set,
            num_entries,
            devices,
            num_devices);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromD3D11KHR);
    return KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromD3D11KHR(
        platform,
        d3d_device_source,
        d3d_object,
        d3d_device_set,
        num_entries,
        devices,
        num_devices);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetDeviceIDsFromD3D11KHR_disp(
    cl_platform_id platform,
    cl_d3d11_device_source_khr d3d_device_source,
    void* d3d_object,
    cl_d3d11_device_set_khr d3d_device_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromD3D11KHR);
    return KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromD3D11KHR(
        platform,
        d3d_device_source,
        d3d_object,
        d3d_device_set,
        num_entries,
        devices,
        num_devices);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D11BufferKHR(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Buffer* resource,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromD3D11BufferKHR(
            context,
            flags,
            resource,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D11BufferKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D11BufferKHR(
        context,
        flags,
        resource,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromD3D11BufferKHR_disp(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Buffer* resource,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D11BufferKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D11BufferKHR(
        context,
        flags,
        resource,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D11Texture2DKHR(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromD3D11Texture2DKHR(
            context,
            flags,
            resource,
            subresource,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D11Texture2DKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D11Texture2DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromD3D11Texture2DKHR_disp(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D11Texture2DKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D11Texture2DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D11Texture3DKHR(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromD3D11Texture3DKHR(
            context,
            flags,
            resource,
            subresource,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D11Texture3DKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D11Texture3DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromD3D11Texture3DKHR_disp(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromD3D11Texture3DKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromD3D11Texture3DKHR(
        context,
        flags,
        resource,
        subresource,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireD3D11ObjectsKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueAcquireD3D11ObjectsKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireD3D11ObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireD3D11ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueAcquireD3D11ObjectsKHR_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireD3D11ObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireD3D11ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseD3D11ObjectsKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueReleaseD3D11ObjectsKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseD3D11ObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseD3D11ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueReleaseD3D11ObjectsKHR_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseD3D11ObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseD3D11ObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_dx9_media_sharing

#if defined(_WIN32)

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDsFromDX9MediaAdapterKHR(
    cl_platform_id platform,
    cl_uint num_media_adapters,
    cl_dx9_media_adapter_type_khr* media_adapter_type,
    void* media_adapters,
    cl_dx9_media_adapter_set_khr media_adapter_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetDeviceIDsFromDX9MediaAdapterKHR(
            platform,
            num_media_adapters,
            media_adapter_type,
            media_adapters,
            media_adapter_set,
            num_entries,
            devices,
            num_devices);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromDX9MediaAdapterKHR);
    return KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromDX9MediaAdapterKHR(
        platform,
        num_media_adapters,
        media_adapter_type,
        media_adapters,
        media_adapter_set,
        num_entries,
        devices,
        num_devices);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetDeviceIDsFromDX9MediaAdapterKHR_disp(
    cl_platform_id platform,
    cl_uint num_media_adapters,
    cl_dx9_media_adapter_type_khr* media_adapter_type,
    void* media_adapters,
    cl_dx9_media_adapter_set_khr media_adapter_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromDX9MediaAdapterKHR);
    return KHR_ICD2_DISPATCH(platform)->clGetDeviceIDsFromDX9MediaAdapterKHR(
        platform,
        num_media_adapters,
        media_adapter_type,
        media_adapters,
        media_adapter_set,
        num_entries,
        devices,
        num_devices);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromDX9MediaSurfaceKHR(
    cl_context context,
    cl_mem_flags flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    void* surface_info,
    cl_uint plane,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromDX9MediaSurfaceKHR(
            context,
            flags,
            adapter_type,
            surface_info,
            plane,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromDX9MediaSurfaceKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromDX9MediaSurfaceKHR(
        context,
        flags,
        adapter_type,
        surface_info,
        plane,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromDX9MediaSurfaceKHR_disp(
    cl_context context,
    cl_mem_flags flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    void* surface_info,
    cl_uint plane,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromDX9MediaSurfaceKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromDX9MediaSurfaceKHR(
        context,
        flags,
        adapter_type,
        surface_info,
        plane,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireDX9MediaSurfacesKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueAcquireDX9MediaSurfacesKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireDX9MediaSurfacesKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireDX9MediaSurfacesKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueAcquireDX9MediaSurfacesKHR_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireDX9MediaSurfacesKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireDX9MediaSurfacesKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseDX9MediaSurfacesKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueReleaseDX9MediaSurfacesKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseDX9MediaSurfacesKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseDX9MediaSurfacesKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueReleaseDX9MediaSurfacesKHR_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseDX9MediaSurfacesKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseDX9MediaSurfacesKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_egl_event

CL_API_ENTRY cl_event CL_API_CALL clCreateEventFromEGLSyncKHR(
    cl_context context,
    CLeglSyncKHR sync,
    CLeglDisplayKHR display,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateEventFromEGLSyncKHR(
            context,
            sync,
            display,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateEventFromEGLSyncKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateEventFromEGLSyncKHR(
        context,
        sync,
        display,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_event CL_API_CALL clCreateEventFromEGLSyncKHR_disp(
    cl_context context,
    CLeglSyncKHR sync,
    CLeglDisplayKHR display,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateEventFromEGLSyncKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateEventFromEGLSyncKHR(
        context,
        sync,
        display,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_egl_image

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromEGLImageKHR(
    cl_context context,
    CLeglDisplayKHR egldisplay,
    CLeglImageKHR eglimage,
    cl_mem_flags flags,
    const cl_egl_image_properties_khr* properties,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromEGLImageKHR(
            context,
            egldisplay,
            eglimage,
            flags,
            properties,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromEGLImageKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromEGLImageKHR(
        context,
        egldisplay,
        eglimage,
        flags,
        properties,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromEGLImageKHR_disp(
    cl_context context,
    CLeglDisplayKHR egldisplay,
    CLeglImageKHR eglimage,
    cl_mem_flags flags,
    const cl_egl_image_properties_khr* properties,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromEGLImageKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateFromEGLImageKHR(
        context,
        egldisplay,
        eglimage,
        flags,
        properties,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireEGLObjectsKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueAcquireEGLObjectsKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireEGLObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireEGLObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueAcquireEGLObjectsKHR_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireEGLObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireEGLObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseEGLObjectsKHR(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueReleaseEGLObjectsKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseEGLObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseEGLObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueReleaseEGLObjectsKHR_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseEGLObjectsKHR);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseEGLObjectsKHR(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_gl_event

CL_API_ENTRY cl_event CL_API_CALL clCreateEventFromGLsyncKHR(
    cl_context context,
    cl_GLsync sync,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateEventFromGLsyncKHR(
            context,
            sync,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateEventFromGLsyncKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateEventFromGLsyncKHR(
        context,
        sync,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_event CL_API_CALL clCreateEventFromGLsyncKHR_disp(
    cl_context context,
    cl_GLsync sync,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateEventFromGLsyncKHR);
    return KHR_ICD2_DISPATCH(context)->clCreateEventFromGLsyncKHR(
        context,
        sync,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_gl_sharing

CL_API_ENTRY cl_int CL_API_CALL clGetGLContextInfoKHR(
    const cl_context_properties* properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetGLContextInfoKHR(
            properties,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    cl_platform_id platform = NULL;
    khrIcdContextPropertiesGetPlatform(properties, &platform);
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(platform)->clGetGLContextInfoKHR);
    return KHR_ICD2_DISPATCH(platform)->clGetGLContextInfoKHR(
        properties,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetGLContextInfoKHR_disp(
    const cl_context_properties* properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    cl_platform_id platform = NULL;
    khrIcdContextPropertiesGetPlatform(properties, &platform);
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(platform)->clGetGLContextInfoKHR);
    return KHR_ICD2_DISPATCH(platform)->clGetGLContextInfoKHR(
        properties,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLBuffer(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint bufobj,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromGLBuffer(
            context,
            flags,
            bufobj,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLBuffer);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLBuffer(
        context,
        flags,
        bufobj,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromGLBuffer_disp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint bufobj,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLBuffer);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLBuffer(
        context,
        flags,
        bufobj,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromGLTexture(
            context,
            flags,
            target,
            miplevel,
            texture,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromGLTexture_disp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLRenderbuffer(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint renderbuffer,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromGLRenderbuffer(
            context,
            flags,
            renderbuffer,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLRenderbuffer);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLRenderbuffer(
        context,
        flags,
        renderbuffer,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromGLRenderbuffer_disp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint renderbuffer,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLRenderbuffer);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLRenderbuffer(
        context,
        flags,
        renderbuffer,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clGetGLObjectInfo(
    cl_mem memobj,
    cl_gl_object_type* gl_object_type,
    cl_GLuint* gl_object_name)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetGLObjectInfo(
            memobj,
            gl_object_type,
            gl_object_name);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(memobj)->clGetGLObjectInfo);
    return KHR_ICD2_DISPATCH(memobj)->clGetGLObjectInfo(
        memobj,
        gl_object_type,
        gl_object_name);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetGLObjectInfo_disp(
    cl_mem memobj,
    cl_gl_object_type* gl_object_type,
    cl_GLuint* gl_object_name)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(memobj)->clGetGLObjectInfo);
    return KHR_ICD2_DISPATCH(memobj)->clGetGLObjectInfo(
        memobj,
        gl_object_type,
        gl_object_name);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clGetGLTextureInfo(
    cl_mem memobj,
    cl_gl_texture_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetGLTextureInfo(
            memobj,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(memobj)->clGetGLTextureInfo);
    return KHR_ICD2_DISPATCH(memobj)->clGetGLTextureInfo(
        memobj,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetGLTextureInfo_disp(
    cl_mem memobj,
    cl_gl_texture_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(memobj)->clGetGLTextureInfo);
    return KHR_ICD2_DISPATCH(memobj)->clGetGLTextureInfo(
        memobj,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireGLObjects(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueAcquireGLObjects(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireGLObjects);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireGLObjects(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueAcquireGLObjects_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireGLObjects);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueAcquireGLObjects(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseGLObjects(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clEnqueueReleaseGLObjects(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseGLObjects);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseGLObjects(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clEnqueueReleaseGLObjects_disp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseGLObjects);
    return KHR_ICD2_DISPATCH(command_queue)->clEnqueueReleaseGLObjects(
        command_queue,
        num_objects,
        mem_objects,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture2D(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromGLTexture2D(
            context,
            flags,
            target,
            miplevel,
            texture,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture2D);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture2D(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromGLTexture2D_disp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture2D);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture2D(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture3D(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clCreateFromGLTexture3D(
            context,
            flags,
            target,
            miplevel,
            texture,
            errcode_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture3D);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture3D(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_mem CL_API_CALL clCreateFromGLTexture3D_disp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    KHR_ICD_VALIDATE_POINTER_RETURN_HANDLE(KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture3D);
    return KHR_ICD2_DISPATCH(context)->clCreateFromGLTexture3D(
        context,
        flags,
        target,
        miplevel,
        texture,
        errcode_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_subgroups

CL_API_ENTRY cl_int CL_API_CALL clGetKernelSubGroupInfoKHR(
    cl_kernel in_kernel,
    cl_device_id in_device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
#if defined(CL_ENABLE_LAYERS)
    if (khrFirstLayer)
        return khrFirstLayer->dispatch.clGetKernelSubGroupInfoKHR(
            in_kernel,
            in_device,
            param_name,
            input_value_size,
            input_value,
            param_value_size,
            param_value,
            param_value_size_ret);
#endif // defined(CL_ENABLE_LAYERS)
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(in_kernel, CL_INVALID_KERNEL);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(in_kernel)->clGetKernelSubGroupInfoKHR);
    return KHR_ICD2_DISPATCH(in_kernel)->clGetKernelSubGroupInfoKHR(
        in_kernel,
        in_device,
        param_name,
        input_value_size,
        input_value,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#if defined(CL_ENABLE_LAYERS)
static cl_int CL_API_CALL clGetKernelSubGroupInfoKHR_disp(
    cl_kernel in_kernel,
    cl_device_id in_device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(in_kernel, CL_INVALID_KERNEL);
    KHR_ICD_VALIDATE_POINTER_RETURN_ERROR(KHR_ICD2_DISPATCH(in_kernel)->clGetKernelSubGroupInfoKHR);
    return KHR_ICD2_DISPATCH(in_kernel)->clGetKernelSubGroupInfoKHR(
        in_kernel,
        in_device,
        param_name,
        input_value_size,
        input_value,
        param_value_size,
        param_value,
        param_value_size_ret);
}
#endif // defined(CL_ENABLE_LAYERS)

///////////////////////////////////////////////////////////////////////////////

#if defined(CL_ENABLE_LAYERS)
struct _cl_icd_dispatch khrMasterDispatch = {
    ICD_ANON_UNION_INIT_MEMBER(&clGetPlatformIDs_disp),
    &clGetPlatformInfo_disp,
    &clGetDeviceIDs_disp,
    &clGetDeviceInfo_disp,
    &clCreateContext_disp,
    &clCreateContextFromType_disp,
    &clRetainContext_disp,
    &clReleaseContext_disp,
    &clGetContextInfo_disp,
    &clCreateCommandQueue_disp,
    &clRetainCommandQueue_disp,
    &clReleaseCommandQueue_disp,
    &clGetCommandQueueInfo_disp,
    &clSetCommandQueueProperty_disp,
    &clCreateBuffer_disp,
    &clCreateImage2D_disp,
    &clCreateImage3D_disp,
    &clRetainMemObject_disp,
    &clReleaseMemObject_disp,
    &clGetSupportedImageFormats_disp,
    &clGetMemObjectInfo_disp,
    &clGetImageInfo_disp,
    &clCreateSampler_disp,
    &clRetainSampler_disp,
    &clReleaseSampler_disp,
    &clGetSamplerInfo_disp,
    &clCreateProgramWithSource_disp,
    &clCreateProgramWithBinary_disp,
    &clRetainProgram_disp,
    &clReleaseProgram_disp,
    &clBuildProgram_disp,
    ICD_ANON_UNION_INIT_MEMBER(&clUnloadCompiler_disp),
    &clGetProgramInfo_disp,
    &clGetProgramBuildInfo_disp,
    &clCreateKernel_disp,
    &clCreateKernelsInProgram_disp,
    &clRetainKernel_disp,
    &clReleaseKernel_disp,
    &clSetKernelArg_disp,
    &clGetKernelInfo_disp,
    &clGetKernelWorkGroupInfo_disp,
    &clWaitForEvents_disp,
    &clGetEventInfo_disp,
    &clRetainEvent_disp,
    &clReleaseEvent_disp,
    &clGetEventProfilingInfo_disp,
    &clFlush_disp,
    &clFinish_disp,
    &clEnqueueReadBuffer_disp,
    &clEnqueueWriteBuffer_disp,
    &clEnqueueCopyBuffer_disp,
    &clEnqueueReadImage_disp,
    &clEnqueueWriteImage_disp,
    &clEnqueueCopyImage_disp,
    &clEnqueueCopyImageToBuffer_disp,
    &clEnqueueCopyBufferToImage_disp,
    &clEnqueueMapBuffer_disp,
    &clEnqueueMapImage_disp,
    &clEnqueueUnmapMemObject_disp,
    &clEnqueueNDRangeKernel_disp,
    &clEnqueueTask_disp,
    &clEnqueueNativeKernel_disp,
    &clEnqueueMarker_disp,
    &clEnqueueWaitForEvents_disp,
    &clEnqueueBarrier_disp,
    &clGetExtensionFunctionAddress_disp,
    &clCreateFromGLBuffer_disp,
    &clCreateFromGLTexture2D_disp,
    &clCreateFromGLTexture3D_disp,
    &clCreateFromGLRenderbuffer_disp,
    &clGetGLObjectInfo_disp,
    &clGetGLTextureInfo_disp,
    &clEnqueueAcquireGLObjects_disp,
    &clEnqueueReleaseGLObjects_disp,
    &clGetGLContextInfoKHR_disp,

  /* cl_khr_d3d10_sharing */
#if defined(_WIN32)
    &clGetDeviceIDsFromD3D10KHR_disp,
    &clCreateFromD3D10BufferKHR_disp,
    &clCreateFromD3D10Texture2DKHR_disp,
    &clCreateFromD3D10Texture3DKHR_disp,
    &clEnqueueAcquireD3D10ObjectsKHR_disp,
    &clEnqueueReleaseD3D10ObjectsKHR_disp,
#else
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif

  /* OpenCL 1.1 */
    &clSetEventCallback_disp,
    &clCreateSubBuffer_disp,
    &clSetMemObjectDestructorCallback_disp,
    &clCreateUserEvent_disp,
    &clSetUserEventStatus_disp,
    &clEnqueueReadBufferRect_disp,
    &clEnqueueWriteBufferRect_disp,
    &clEnqueueCopyBufferRect_disp,

  /* cl_ext_device_fission */
    &clCreateSubDevicesEXT_disp,
    &clRetainDeviceEXT_disp,
    &clReleaseDeviceEXT_disp,

  /* cl_khr_gl_event */
    &clCreateEventFromGLsyncKHR_disp,

  /* OpenCL 1.2 */
    &clCreateSubDevices_disp,
    &clRetainDevice_disp,
    &clReleaseDevice_disp,
    &clCreateImage_disp,
    &clCreateProgramWithBuiltInKernels_disp,
    &clCompileProgram_disp,
    &clLinkProgram_disp,
    &clUnloadPlatformCompiler_disp,
    &clGetKernelArgInfo_disp,
    &clEnqueueFillBuffer_disp,
    &clEnqueueFillImage_disp,
    &clEnqueueMigrateMemObjects_disp,
    &clEnqueueMarkerWithWaitList_disp,
    &clEnqueueBarrierWithWaitList_disp,
    &clGetExtensionFunctionAddressForPlatform_disp,
    &clCreateFromGLTexture_disp,

  /* cl_khr_d3d11_sharing */
#if defined(_WIN32)
    &clGetDeviceIDsFromD3D11KHR_disp,
    &clCreateFromD3D11BufferKHR_disp,
    &clCreateFromD3D11Texture2DKHR_disp,
    &clCreateFromD3D11Texture3DKHR_disp,
    &clCreateFromDX9MediaSurfaceKHR_disp,
    &clEnqueueAcquireD3D11ObjectsKHR_disp,
    &clEnqueueReleaseD3D11ObjectsKHR_disp,
#else
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif

  /* cl_khr_dx9_media_sharing */
#if defined(_WIN32)
    &clGetDeviceIDsFromDX9MediaAdapterKHR_disp,
    &clEnqueueAcquireDX9MediaSurfacesKHR_disp,
    &clEnqueueReleaseDX9MediaSurfacesKHR_disp,
#else
    NULL,
    NULL,
    NULL,
#endif

  /* cl_khr_egl_image */
    &clCreateFromEGLImageKHR_disp,
    &clEnqueueAcquireEGLObjectsKHR_disp,
    &clEnqueueReleaseEGLObjectsKHR_disp,

  /* cl_khr_egl_event */
    &clCreateEventFromEGLSyncKHR_disp,

  /* OpenCL 2.0 */
    &clCreateCommandQueueWithProperties_disp,
    &clCreatePipe_disp,
    &clGetPipeInfo_disp,
    &clSVMAlloc_disp,
    &clSVMFree_disp,
    &clEnqueueSVMFree_disp,
    &clEnqueueSVMMemcpy_disp,
    &clEnqueueSVMMemFill_disp,
    &clEnqueueSVMMap_disp,
    &clEnqueueSVMUnmap_disp,
    &clCreateSamplerWithProperties_disp,
    &clSetKernelArgSVMPointer_disp,
    &clSetKernelExecInfo_disp,

  /* cl_khr_sub_groups */
    &clGetKernelSubGroupInfoKHR_disp,

  /* OpenCL 2.1 */
    &clCloneKernel_disp,
    &clCreateProgramWithIL_disp,
    &clEnqueueSVMMigrateMem_disp,
    &clGetDeviceAndHostTimer_disp,
    &clGetHostTimer_disp,
    &clGetKernelSubGroupInfo_disp,
    &clSetDefaultDeviceCommandQueue_disp,

  /* OpenCL 2.2 */
    &clSetProgramReleaseCallback_disp,
    &clSetProgramSpecializationConstant_disp,

  /* OpenCL 3.0 */
    &clCreateBufferWithProperties_disp,
    &clCreateImageWithProperties_disp,
    &clSetContextDestructorCallback_disp
};
#endif // defined(CL_ENABLE_LAYERS)

#if defined(CL_ENABLE_LOADER_MANAGED_DISPATCH)
///////////////////////////////////////////////////////////////////////////////
// Core APIs:
static cl_int CL_API_CALL clGetPlatformIDs_unsupp(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms)
{
    (void)num_entries;
    (void)platforms;
    (void)num_platforms;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetPlatformInfo_unsupp(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)platform;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetDeviceIDs_unsupp(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    (void)platform;
    (void)device_type;
    (void)num_entries;
    (void)devices;
    (void)num_devices;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetDeviceInfo_unsupp(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)device;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_context CL_API_CALL clCreateContext_unsupp(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    (void)properties;
    (void)num_devices;
    (void)devices;
    (void)pfn_notify;
    (void)user_data;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_context CL_API_CALL clCreateContextFromType_unsupp(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    (void)properties;
    (void)device_type;
    (void)pfn_notify;
    (void)user_data;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainContext_unsupp(
    cl_context context)
{
    (void)context;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clReleaseContext_unsupp(
    cl_context context)
{
    (void)context;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetContextInfo_unsupp(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)context;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainCommandQueue_unsupp(
    cl_command_queue command_queue)
{
    (void)command_queue;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clReleaseCommandQueue_unsupp(
    cl_command_queue command_queue)
{
    (void)command_queue;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetCommandQueueInfo_unsupp(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)command_queue;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateBuffer_unsupp(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)size;
    (void)host_ptr;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainMemObject_unsupp(
    cl_mem memobj)
{
    (void)memobj;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clReleaseMemObject_unsupp(
    cl_mem memobj)
{
    (void)memobj;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetSupportedImageFormats_unsupp(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats)
{
    (void)context;
    (void)flags;
    (void)image_type;
    (void)num_entries;
    (void)image_formats;
    (void)num_image_formats;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetMemObjectInfo_unsupp(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)memobj;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetImageInfo_unsupp(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)image;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainSampler_unsupp(
    cl_sampler sampler)
{
    (void)sampler;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clReleaseSampler_unsupp(
    cl_sampler sampler)
{
    (void)sampler;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetSamplerInfo_unsupp(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)sampler;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_program CL_API_CALL clCreateProgramWithSource_unsupp(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
    (void)context;
    (void)count;
    (void)strings;
    (void)lengths;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_program CL_API_CALL clCreateProgramWithBinary_unsupp(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret)
{
    (void)context;
    (void)num_devices;
    (void)device_list;
    (void)lengths;
    (void)binaries;
    (void)binary_status;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainProgram_unsupp(
    cl_program program)
{
    (void)program;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clReleaseProgram_unsupp(
    cl_program program)
{
    (void)program;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clBuildProgram_unsupp(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    (void)program;
    (void)num_devices;
    (void)device_list;
    (void)options;
    (void)pfn_notify;
    (void)user_data;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetProgramInfo_unsupp(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)program;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetProgramBuildInfo_unsupp(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)program;
    (void)device;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_kernel CL_API_CALL clCreateKernel_unsupp(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret)
{
    (void)program;
    (void)kernel_name;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clCreateKernelsInProgram_unsupp(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
    (void)program;
    (void)num_kernels;
    (void)kernels;
    (void)num_kernels_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainKernel_unsupp(
    cl_kernel kernel)
{
    (void)kernel;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clReleaseKernel_unsupp(
    cl_kernel kernel)
{
    (void)kernel;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetKernelArg_unsupp(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
    (void)kernel;
    (void)arg_index;
    (void)arg_size;
    (void)arg_value;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetKernelInfo_unsupp(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)kernel;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetKernelWorkGroupInfo_unsupp(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)kernel;
    (void)device;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clWaitForEvents_unsupp(
    cl_uint num_events,
    const cl_event* event_list)
{
    (void)num_events;
    (void)event_list;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetEventInfo_unsupp(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)event;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainEvent_unsupp(
    cl_event event)
{
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clReleaseEvent_unsupp(
    cl_event event)
{
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetEventProfilingInfo_unsupp(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)event;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clFlush_unsupp(
    cl_command_queue command_queue)
{
    (void)command_queue;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clFinish_unsupp(
    cl_command_queue command_queue)
{
    (void)command_queue;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueReadBuffer_unsupp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)buffer;
    (void)blocking_read;
    (void)offset;
    (void)size;
    (void)ptr;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueWriteBuffer_unsupp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)buffer;
    (void)blocking_write;
    (void)offset;
    (void)size;
    (void)ptr;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueCopyBuffer_unsupp(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)src_buffer;
    (void)dst_buffer;
    (void)src_offset;
    (void)dst_offset;
    (void)size;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueReadImage_unsupp(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_read,
    const size_t* origin,
    const size_t* region,
    size_t row_pitch,
    size_t slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)image;
    (void)blocking_read;
    (void)origin;
    (void)region;
    (void)row_pitch;
    (void)slice_pitch;
    (void)ptr;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueWriteImage_unsupp(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_write,
    const size_t* origin,
    const size_t* region,
    size_t input_row_pitch,
    size_t input_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)image;
    (void)blocking_write;
    (void)origin;
    (void)region;
    (void)input_row_pitch;
    (void)input_slice_pitch;
    (void)ptr;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueCopyImage_unsupp(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)src_image;
    (void)dst_image;
    (void)src_origin;
    (void)dst_origin;
    (void)region;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueCopyImageToBuffer_unsupp(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)src_image;
    (void)dst_buffer;
    (void)src_origin;
    (void)region;
    (void)dst_offset;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueCopyBufferToImage_unsupp(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)src_buffer;
    (void)dst_image;
    (void)src_offset;
    (void)dst_origin;
    (void)region;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static void* CL_API_CALL clEnqueueMapBuffer_unsupp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
    (void)command_queue;
    (void)buffer;
    (void)blocking_map;
    (void)map_flags;
    (void)offset;
    (void)size;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static void* CL_API_CALL clEnqueueMapImage_unsupp(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    const size_t* origin,
    const size_t* region,
    size_t* image_row_pitch,
    size_t* image_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
    (void)command_queue;
    (void)image;
    (void)blocking_map;
    (void)map_flags;
    (void)origin;
    (void)region;
    (void)image_row_pitch;
    (void)image_slice_pitch;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueUnmapMemObject_unsupp(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)memobj;
    (void)mapped_ptr;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueNDRangeKernel_unsupp(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)kernel;
    (void)work_dim;
    (void)global_work_offset;
    (void)global_work_size;
    (void)local_work_size;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueNativeKernel_unsupp(
    cl_command_queue command_queue,
    void (CL_CALLBACK* user_func)(void*),
    void* args,
    size_t cb_args,
    cl_uint num_mem_objects,
    const cl_mem* mem_list,
    const void** args_mem_loc,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)user_func;
    (void)args;
    (void)cb_args;
    (void)num_mem_objects;
    (void)mem_list;
    (void)args_mem_loc;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetCommandQueueProperty_unsupp(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties)
{
    (void)command_queue;
    (void)properties;
    (void)enable;
    (void)old_properties;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateImage2D_unsupp(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)image_format;
    (void)image_width;
    (void)image_height;
    (void)image_row_pitch;
    (void)host_ptr;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateImage3D_unsupp(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_depth,
    size_t image_row_pitch,
    size_t image_slice_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)image_format;
    (void)image_width;
    (void)image_height;
    (void)image_depth;
    (void)image_row_pitch;
    (void)image_slice_pitch;
    (void)host_ptr;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueMarker_unsupp(
    cl_command_queue command_queue,
    cl_event* event)
{
    (void)command_queue;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueWaitForEvents_unsupp(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list)
{
    (void)command_queue;
    (void)num_events;
    (void)event_list;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueBarrier_unsupp(
    cl_command_queue command_queue)
{
    (void)command_queue;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clUnloadCompiler_unsupp(
    void )
{
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static void* CL_API_CALL clGetExtensionFunctionAddress_unsupp(
    const char* func_name)
{
    (void)func_name;
    KHR_ICD_ERROR_RETURN_ERROR(NULL);
}
static cl_command_queue CL_API_CALL clCreateCommandQueue_unsupp(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret)
{
    (void)context;
    (void)device;
    (void)properties;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_sampler CL_API_CALL clCreateSampler_unsupp(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret)
{
    (void)context;
    (void)normalized_coords;
    (void)addressing_mode;
    (void)filter_mode;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueTask_unsupp(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)kernel;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateSubBuffer_unsupp(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret)
{
    (void)buffer;
    (void)flags;
    (void)buffer_create_type;
    (void)buffer_create_info;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetMemObjectDestructorCallback_unsupp(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data)
{
    (void)memobj;
    (void)pfn_notify;
    (void)user_data;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_event CL_API_CALL clCreateUserEvent_unsupp(
    cl_context context,
    cl_int* errcode_ret)
{
    (void)context;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetUserEventStatus_unsupp(
    cl_event event,
    cl_int execution_status)
{
    (void)event;
    (void)execution_status;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetEventCallback_unsupp(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data),
    void* user_data)
{
    (void)event;
    (void)command_exec_callback_type;
    (void)pfn_notify;
    (void)user_data;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueReadBufferRect_unsupp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)buffer;
    (void)blocking_read;
    (void)buffer_origin;
    (void)host_origin;
    (void)region;
    (void)buffer_row_pitch;
    (void)buffer_slice_pitch;
    (void)host_row_pitch;
    (void)host_slice_pitch;
    (void)ptr;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueWriteBufferRect_unsupp(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)buffer;
    (void)blocking_write;
    (void)buffer_origin;
    (void)host_origin;
    (void)region;
    (void)buffer_row_pitch;
    (void)buffer_slice_pitch;
    (void)host_row_pitch;
    (void)host_slice_pitch;
    (void)ptr;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueCopyBufferRect_unsupp(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)src_buffer;
    (void)dst_buffer;
    (void)src_origin;
    (void)dst_origin;
    (void)region;
    (void)src_row_pitch;
    (void)src_slice_pitch;
    (void)dst_row_pitch;
    (void)dst_slice_pitch;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clCreateSubDevices_unsupp(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret)
{
    (void)in_device;
    (void)properties;
    (void)num_devices;
    (void)out_devices;
    (void)num_devices_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainDevice_unsupp(
    cl_device_id device)
{
    (void)device;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clReleaseDevice_unsupp(
    cl_device_id device)
{
    (void)device;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateImage_unsupp(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)image_format;
    (void)image_desc;
    (void)host_ptr;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_program CL_API_CALL clCreateProgramWithBuiltInKernels_unsupp(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret)
{
    (void)context;
    (void)num_devices;
    (void)device_list;
    (void)kernel_names;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clCompileProgram_unsupp(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_headers,
    const cl_program* input_headers,
    const char** header_include_names,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    (void)program;
    (void)num_devices;
    (void)device_list;
    (void)options;
    (void)num_input_headers;
    (void)input_headers;
    (void)header_include_names;
    (void)pfn_notify;
    (void)user_data;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_program CL_API_CALL clLinkProgram_unsupp(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_programs,
    const cl_program* input_programs,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    (void)context;
    (void)num_devices;
    (void)device_list;
    (void)options;
    (void)num_input_programs;
    (void)input_programs;
    (void)pfn_notify;
    (void)user_data;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clUnloadPlatformCompiler_unsupp(
    cl_platform_id platform)
{
    (void)platform;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetKernelArgInfo_unsupp(
    cl_kernel kernel,
    cl_uint arg_index,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)kernel;
    (void)arg_index;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueFillBuffer_unsupp(
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)buffer;
    (void)pattern;
    (void)pattern_size;
    (void)offset;
    (void)size;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueFillImage_unsupp(
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)image;
    (void)fill_color;
    (void)origin;
    (void)region;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueMigrateMemObjects_unsupp(
    cl_command_queue command_queue,
    cl_uint num_mem_objects,
    const cl_mem* mem_objects,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_mem_objects;
    (void)mem_objects;
    (void)flags;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueMarkerWithWaitList_unsupp(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueBarrierWithWaitList_unsupp(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static void* CL_API_CALL clGetExtensionFunctionAddressForPlatform_unsupp(
    cl_platform_id platform,
    const char* func_name)
{
    (void)platform;
    (void)func_name;
    KHR_ICD_ERROR_RETURN_ERROR(NULL);
}
static cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties_unsupp(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret)
{
    (void)context;
    (void)device;
    (void)properties;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreatePipe_unsupp(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)pipe_packet_size;
    (void)pipe_max_packets;
    (void)properties;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetPipeInfo_unsupp(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)pipe;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static void* CL_API_CALL clSVMAlloc_unsupp(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment)
{
    (void)context;
    (void)flags;
    (void)size;
    (void)alignment;
    KHR_ICD_ERROR_RETURN_ERROR(NULL);
}
static void CL_API_CALL clSVMFree_unsupp(
    cl_context context,
    void* svm_pointer)
{
    (void)context;
    (void)svm_pointer;
    return;
}
static cl_sampler CL_API_CALL clCreateSamplerWithProperties_unsupp(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret)
{
    (void)context;
    (void)sampler_properties;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetKernelArgSVMPointer_unsupp(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value)
{
    (void)kernel;
    (void)arg_index;
    (void)arg_value;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetKernelExecInfo_unsupp(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value)
{
    (void)kernel;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueSVMFree_unsupp(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    void* svm_pointers[],
    void (CL_CALLBACK* pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers, void* svm_pointers[], void* user_data),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_svm_pointers;
    (void)svm_pointers;
    (void)pfn_free_func;
    (void)user_data;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueSVMMemcpy_unsupp(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)blocking_copy;
    (void)dst_ptr;
    (void)src_ptr;
    (void)size;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueSVMMemFill_unsupp(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)svm_ptr;
    (void)pattern;
    (void)pattern_size;
    (void)size;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueSVMMap_unsupp(
    cl_command_queue command_queue,
    cl_bool blocking_map,
    cl_map_flags flags,
    void* svm_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)blocking_map;
    (void)flags;
    (void)svm_ptr;
    (void)size;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueSVMUnmap_unsupp(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)svm_ptr;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetDefaultDeviceCommandQueue_unsupp(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue)
{
    (void)context;
    (void)device;
    (void)command_queue;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetDeviceAndHostTimer_unsupp(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp)
{
    (void)device;
    (void)device_timestamp;
    (void)host_timestamp;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetHostTimer_unsupp(
    cl_device_id device,
    cl_ulong* host_timestamp)
{
    (void)device;
    (void)host_timestamp;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_program CL_API_CALL clCreateProgramWithIL_unsupp(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret)
{
    (void)context;
    (void)il;
    (void)length;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_kernel CL_API_CALL clCloneKernel_unsupp(
    cl_kernel source_kernel,
    cl_int* errcode_ret)
{
    (void)source_kernel;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetKernelSubGroupInfo_unsupp(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)kernel;
    (void)device;
    (void)param_name;
    (void)input_value_size;
    (void)input_value;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueSVMMigrateMem_unsupp(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_svm_pointers;
    (void)svm_pointers;
    (void)sizes;
    (void)flags;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetProgramSpecializationConstant_unsupp(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value)
{
    (void)program;
    (void)spec_id;
    (void)spec_size;
    (void)spec_value;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetProgramReleaseCallback_unsupp(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    (void)program;
    (void)pfn_notify;
    (void)user_data;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clSetContextDestructorCallback_unsupp(
    cl_context context,
    void (CL_CALLBACK* pfn_notify)(cl_context context, void* user_data),
    void* user_data)
{
    (void)context;
    (void)pfn_notify;
    (void)user_data;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateBufferWithProperties_unsupp(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    (void)context;
    (void)properties;
    (void)flags;
    (void)size;
    (void)host_ptr;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateImageWithProperties_unsupp(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    (void)context;
    (void)properties;
    (void)flags;
    (void)image_format;
    (void)image_desc;
    (void)host_ptr;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}

///////////////////////////////////////////////////////////////////////////////
// cl_ext_device_fission
static cl_int CL_API_CALL clReleaseDeviceEXT_unsupp(
    cl_device_id device)
{
    (void)device;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clRetainDeviceEXT_unsupp(
    cl_device_id device)
{
    (void)device;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clCreateSubDevicesEXT_unsupp(
    cl_device_id in_device,
    const cl_device_partition_property_ext* properties,
    cl_uint num_entries,
    cl_device_id* out_devices,
    cl_uint* num_devices)
{
    (void)in_device;
    (void)properties;
    (void)num_entries;
    (void)out_devices;
    (void)num_devices;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
///////////////////////////////////////////////////////////////////////////////

// cl_khr_d3d10_sharing

#if defined(_WIN32)
static cl_int CL_API_CALL clGetDeviceIDsFromD3D10KHR_unsupp(
    cl_platform_id platform,
    cl_d3d10_device_source_khr d3d_device_source,
    void* d3d_object,
    cl_d3d10_device_set_khr d3d_device_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    (void)platform;
    (void)d3d_device_source;
    (void)d3d_object;
    (void)d3d_device_set;
    (void)num_entries;
    (void)devices;
    (void)num_devices;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromD3D10BufferKHR_unsupp(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Buffer* resource,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)resource;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromD3D10Texture2DKHR_unsupp(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)resource;
    (void)subresource;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromD3D10Texture3DKHR_unsupp(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)resource;
    (void)subresource;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueAcquireD3D10ObjectsKHR_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueReleaseD3D10ObjectsKHR_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_d3d11_sharing

#if defined(_WIN32)
static cl_int CL_API_CALL clGetDeviceIDsFromD3D11KHR_unsupp(
    cl_platform_id platform,
    cl_d3d11_device_source_khr d3d_device_source,
    void* d3d_object,
    cl_d3d11_device_set_khr d3d_device_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    (void)platform;
    (void)d3d_device_source;
    (void)d3d_object;
    (void)d3d_device_set;
    (void)num_entries;
    (void)devices;
    (void)num_devices;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromD3D11BufferKHR_unsupp(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Buffer* resource,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)resource;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromD3D11Texture2DKHR_unsupp(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)resource;
    (void)subresource;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromD3D11Texture3DKHR_unsupp(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)resource;
    (void)subresource;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueAcquireD3D11ObjectsKHR_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueReleaseD3D11ObjectsKHR_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_dx9_media_sharing

#if defined(_WIN32)
static cl_int CL_API_CALL clGetDeviceIDsFromDX9MediaAdapterKHR_unsupp(
    cl_platform_id platform,
    cl_uint num_media_adapters,
    cl_dx9_media_adapter_type_khr* media_adapter_type,
    void* media_adapters,
    cl_dx9_media_adapter_set_khr media_adapter_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    (void)platform;
    (void)num_media_adapters;
    (void)media_adapter_type;
    (void)media_adapters;
    (void)media_adapter_set;
    (void)num_entries;
    (void)devices;
    (void)num_devices;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromDX9MediaSurfaceKHR_unsupp(
    cl_context context,
    cl_mem_flags flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    void* surface_info,
    cl_uint plane,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)adapter_type;
    (void)surface_info;
    (void)plane;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueAcquireDX9MediaSurfacesKHR_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueReleaseDX9MediaSurfacesKHR_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_egl_event
static cl_event CL_API_CALL clCreateEventFromEGLSyncKHR_unsupp(
    cl_context context,
    CLeglSyncKHR sync,
    CLeglDisplayKHR display,
    cl_int* errcode_ret)
{
    (void)context;
    (void)sync;
    (void)display;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
///////////////////////////////////////////////////////////////////////////////

// cl_khr_egl_image
static cl_mem CL_API_CALL clCreateFromEGLImageKHR_unsupp(
    cl_context context,
    CLeglDisplayKHR egldisplay,
    CLeglImageKHR eglimage,
    cl_mem_flags flags,
    const cl_egl_image_properties_khr* properties,
    cl_int* errcode_ret)
{
    (void)context;
    (void)egldisplay;
    (void)eglimage;
    (void)flags;
    (void)properties;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueAcquireEGLObjectsKHR_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueReleaseEGLObjectsKHR_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
///////////////////////////////////////////////////////////////////////////////

// cl_khr_gl_event
static cl_event CL_API_CALL clCreateEventFromGLsyncKHR_unsupp(
    cl_context context,
    cl_GLsync sync,
    cl_int* errcode_ret)
{
    (void)context;
    (void)sync;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
///////////////////////////////////////////////////////////////////////////////

// cl_khr_gl_sharing
static cl_int CL_API_CALL clGetGLContextInfoKHR_unsupp(
    const cl_context_properties* properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)properties;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromGLBuffer_unsupp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint bufobj,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)bufobj;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromGLTexture_unsupp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)target;
    (void)miplevel;
    (void)texture;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromGLRenderbuffer_unsupp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint renderbuffer,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)renderbuffer;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetGLObjectInfo_unsupp(
    cl_mem memobj,
    cl_gl_object_type* gl_object_type,
    cl_GLuint* gl_object_name)
{
    (void)memobj;
    (void)gl_object_type;
    (void)gl_object_name;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clGetGLTextureInfo_unsupp(
    cl_mem memobj,
    cl_gl_texture_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)memobj;
    (void)param_name;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueAcquireGLObjects_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_int CL_API_CALL clEnqueueReleaseGLObjects_unsupp(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    (void)command_queue;
    (void)num_objects;
    (void)mem_objects;
    (void)num_events_in_wait_list;
    (void)event_wait_list;
    (void)event;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromGLTexture2D_unsupp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)target;
    (void)miplevel;
    (void)texture;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
static cl_mem CL_API_CALL clCreateFromGLTexture3D_unsupp(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
    (void)context;
    (void)flags;
    (void)target;
    (void)miplevel;
    (void)texture;
    (void)errcode_ret;
    KHR_ICD_ERROR_RETURN_HANDLE(CL_INVALID_OPERATION);
}
///////////////////////////////////////////////////////////////////////////////

// cl_khr_subgroups
static cl_int CL_API_CALL clGetKernelSubGroupInfoKHR_unsupp(
    cl_kernel in_kernel,
    cl_device_id in_device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    (void)in_kernel;
    (void)in_device;
    (void)param_name;
    (void)input_value_size;
    (void)input_value;
    (void)param_value_size;
    (void)param_value;
    (void)param_value_size_ret;
    KHR_ICD_ERROR_RETURN_ERROR(CL_INVALID_OPERATION);
}
///////////////////////////////////////////////////////////////////////////////


void khrIcd2PopulateDispatchTable(
    cl_platform_id platform,
    clIcdGetFunctionAddressForPlatformKHR_fn p_clIcdGetFunctionAddressForPlatform,
    struct _cl_icd_dispatch* dispatch)
{
///////////////////////////////////////////////////////////////////////////////
// Core APIs:
   dispatch->clGetPlatformIDs = (clGetPlatformIDs_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetPlatformIDs");
   if (!dispatch->clGetPlatformIDs)
       dispatch->clGetPlatformIDs = &clGetPlatformIDs_unsupp;
   dispatch->clGetPlatformInfo = (clGetPlatformInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetPlatformInfo");
   if (!dispatch->clGetPlatformInfo)
       dispatch->clGetPlatformInfo = &clGetPlatformInfo_unsupp;
   dispatch->clGetDeviceIDs = (clGetDeviceIDs_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetDeviceIDs");
   if (!dispatch->clGetDeviceIDs)
       dispatch->clGetDeviceIDs = &clGetDeviceIDs_unsupp;
   dispatch->clGetDeviceInfo = (clGetDeviceInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetDeviceInfo");
   if (!dispatch->clGetDeviceInfo)
       dispatch->clGetDeviceInfo = &clGetDeviceInfo_unsupp;
   dispatch->clCreateContext = (clCreateContext_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateContext");
   if (!dispatch->clCreateContext)
       dispatch->clCreateContext = &clCreateContext_unsupp;
   dispatch->clCreateContextFromType = (clCreateContextFromType_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateContextFromType");
   if (!dispatch->clCreateContextFromType)
       dispatch->clCreateContextFromType = &clCreateContextFromType_unsupp;
   dispatch->clRetainContext = (clRetainContext_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainContext");
   if (!dispatch->clRetainContext)
       dispatch->clRetainContext = &clRetainContext_unsupp;
   dispatch->clReleaseContext = (clReleaseContext_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseContext");
   if (!dispatch->clReleaseContext)
       dispatch->clReleaseContext = &clReleaseContext_unsupp;
   dispatch->clGetContextInfo = (clGetContextInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetContextInfo");
   if (!dispatch->clGetContextInfo)
       dispatch->clGetContextInfo = &clGetContextInfo_unsupp;
   dispatch->clRetainCommandQueue = (clRetainCommandQueue_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainCommandQueue");
   if (!dispatch->clRetainCommandQueue)
       dispatch->clRetainCommandQueue = &clRetainCommandQueue_unsupp;
   dispatch->clReleaseCommandQueue = (clReleaseCommandQueue_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseCommandQueue");
   if (!dispatch->clReleaseCommandQueue)
       dispatch->clReleaseCommandQueue = &clReleaseCommandQueue_unsupp;
   dispatch->clGetCommandQueueInfo = (clGetCommandQueueInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetCommandQueueInfo");
   if (!dispatch->clGetCommandQueueInfo)
       dispatch->clGetCommandQueueInfo = &clGetCommandQueueInfo_unsupp;
   dispatch->clCreateBuffer = (clCreateBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateBuffer");
   if (!dispatch->clCreateBuffer)
       dispatch->clCreateBuffer = &clCreateBuffer_unsupp;
   dispatch->clRetainMemObject = (clRetainMemObject_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainMemObject");
   if (!dispatch->clRetainMemObject)
       dispatch->clRetainMemObject = &clRetainMemObject_unsupp;
   dispatch->clReleaseMemObject = (clReleaseMemObject_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseMemObject");
   if (!dispatch->clReleaseMemObject)
       dispatch->clReleaseMemObject = &clReleaseMemObject_unsupp;
   dispatch->clGetSupportedImageFormats = (clGetSupportedImageFormats_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetSupportedImageFormats");
   if (!dispatch->clGetSupportedImageFormats)
       dispatch->clGetSupportedImageFormats = &clGetSupportedImageFormats_unsupp;
   dispatch->clGetMemObjectInfo = (clGetMemObjectInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetMemObjectInfo");
   if (!dispatch->clGetMemObjectInfo)
       dispatch->clGetMemObjectInfo = &clGetMemObjectInfo_unsupp;
   dispatch->clGetImageInfo = (clGetImageInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetImageInfo");
   if (!dispatch->clGetImageInfo)
       dispatch->clGetImageInfo = &clGetImageInfo_unsupp;
   dispatch->clRetainSampler = (clRetainSampler_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainSampler");
   if (!dispatch->clRetainSampler)
       dispatch->clRetainSampler = &clRetainSampler_unsupp;
   dispatch->clReleaseSampler = (clReleaseSampler_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseSampler");
   if (!dispatch->clReleaseSampler)
       dispatch->clReleaseSampler = &clReleaseSampler_unsupp;
   dispatch->clGetSamplerInfo = (clGetSamplerInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetSamplerInfo");
   if (!dispatch->clGetSamplerInfo)
       dispatch->clGetSamplerInfo = &clGetSamplerInfo_unsupp;
   dispatch->clCreateProgramWithSource = (clCreateProgramWithSource_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateProgramWithSource");
   if (!dispatch->clCreateProgramWithSource)
       dispatch->clCreateProgramWithSource = &clCreateProgramWithSource_unsupp;
   dispatch->clCreateProgramWithBinary = (clCreateProgramWithBinary_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateProgramWithBinary");
   if (!dispatch->clCreateProgramWithBinary)
       dispatch->clCreateProgramWithBinary = &clCreateProgramWithBinary_unsupp;
   dispatch->clRetainProgram = (clRetainProgram_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainProgram");
   if (!dispatch->clRetainProgram)
       dispatch->clRetainProgram = &clRetainProgram_unsupp;
   dispatch->clReleaseProgram = (clReleaseProgram_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseProgram");
   if (!dispatch->clReleaseProgram)
       dispatch->clReleaseProgram = &clReleaseProgram_unsupp;
   dispatch->clBuildProgram = (clBuildProgram_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clBuildProgram");
   if (!dispatch->clBuildProgram)
       dispatch->clBuildProgram = &clBuildProgram_unsupp;
   dispatch->clGetProgramInfo = (clGetProgramInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetProgramInfo");
   if (!dispatch->clGetProgramInfo)
       dispatch->clGetProgramInfo = &clGetProgramInfo_unsupp;
   dispatch->clGetProgramBuildInfo = (clGetProgramBuildInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetProgramBuildInfo");
   if (!dispatch->clGetProgramBuildInfo)
       dispatch->clGetProgramBuildInfo = &clGetProgramBuildInfo_unsupp;
   dispatch->clCreateKernel = (clCreateKernel_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateKernel");
   if (!dispatch->clCreateKernel)
       dispatch->clCreateKernel = &clCreateKernel_unsupp;
   dispatch->clCreateKernelsInProgram = (clCreateKernelsInProgram_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateKernelsInProgram");
   if (!dispatch->clCreateKernelsInProgram)
       dispatch->clCreateKernelsInProgram = &clCreateKernelsInProgram_unsupp;
   dispatch->clRetainKernel = (clRetainKernel_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainKernel");
   if (!dispatch->clRetainKernel)
       dispatch->clRetainKernel = &clRetainKernel_unsupp;
   dispatch->clReleaseKernel = (clReleaseKernel_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseKernel");
   if (!dispatch->clReleaseKernel)
       dispatch->clReleaseKernel = &clReleaseKernel_unsupp;
   dispatch->clSetKernelArg = (clSetKernelArg_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetKernelArg");
   if (!dispatch->clSetKernelArg)
       dispatch->clSetKernelArg = &clSetKernelArg_unsupp;
   dispatch->clGetKernelInfo = (clGetKernelInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetKernelInfo");
   if (!dispatch->clGetKernelInfo)
       dispatch->clGetKernelInfo = &clGetKernelInfo_unsupp;
   dispatch->clGetKernelWorkGroupInfo = (clGetKernelWorkGroupInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetKernelWorkGroupInfo");
   if (!dispatch->clGetKernelWorkGroupInfo)
       dispatch->clGetKernelWorkGroupInfo = &clGetKernelWorkGroupInfo_unsupp;
   dispatch->clWaitForEvents = (clWaitForEvents_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clWaitForEvents");
   if (!dispatch->clWaitForEvents)
       dispatch->clWaitForEvents = &clWaitForEvents_unsupp;
   dispatch->clGetEventInfo = (clGetEventInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetEventInfo");
   if (!dispatch->clGetEventInfo)
       dispatch->clGetEventInfo = &clGetEventInfo_unsupp;
   dispatch->clRetainEvent = (clRetainEvent_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainEvent");
   if (!dispatch->clRetainEvent)
       dispatch->clRetainEvent = &clRetainEvent_unsupp;
   dispatch->clReleaseEvent = (clReleaseEvent_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseEvent");
   if (!dispatch->clReleaseEvent)
       dispatch->clReleaseEvent = &clReleaseEvent_unsupp;
   dispatch->clGetEventProfilingInfo = (clGetEventProfilingInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetEventProfilingInfo");
   if (!dispatch->clGetEventProfilingInfo)
       dispatch->clGetEventProfilingInfo = &clGetEventProfilingInfo_unsupp;
   dispatch->clFlush = (clFlush_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clFlush");
   if (!dispatch->clFlush)
       dispatch->clFlush = &clFlush_unsupp;
   dispatch->clFinish = (clFinish_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clFinish");
   if (!dispatch->clFinish)
       dispatch->clFinish = &clFinish_unsupp;
   dispatch->clEnqueueReadBuffer = (clEnqueueReadBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueReadBuffer");
   if (!dispatch->clEnqueueReadBuffer)
       dispatch->clEnqueueReadBuffer = &clEnqueueReadBuffer_unsupp;
   dispatch->clEnqueueWriteBuffer = (clEnqueueWriteBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueWriteBuffer");
   if (!dispatch->clEnqueueWriteBuffer)
       dispatch->clEnqueueWriteBuffer = &clEnqueueWriteBuffer_unsupp;
   dispatch->clEnqueueCopyBuffer = (clEnqueueCopyBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueCopyBuffer");
   if (!dispatch->clEnqueueCopyBuffer)
       dispatch->clEnqueueCopyBuffer = &clEnqueueCopyBuffer_unsupp;
   dispatch->clEnqueueReadImage = (clEnqueueReadImage_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueReadImage");
   if (!dispatch->clEnqueueReadImage)
       dispatch->clEnqueueReadImage = &clEnqueueReadImage_unsupp;
   dispatch->clEnqueueWriteImage = (clEnqueueWriteImage_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueWriteImage");
   if (!dispatch->clEnqueueWriteImage)
       dispatch->clEnqueueWriteImage = &clEnqueueWriteImage_unsupp;
   dispatch->clEnqueueCopyImage = (clEnqueueCopyImage_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueCopyImage");
   if (!dispatch->clEnqueueCopyImage)
       dispatch->clEnqueueCopyImage = &clEnqueueCopyImage_unsupp;
   dispatch->clEnqueueCopyImageToBuffer = (clEnqueueCopyImageToBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueCopyImageToBuffer");
   if (!dispatch->clEnqueueCopyImageToBuffer)
       dispatch->clEnqueueCopyImageToBuffer = &clEnqueueCopyImageToBuffer_unsupp;
   dispatch->clEnqueueCopyBufferToImage = (clEnqueueCopyBufferToImage_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueCopyBufferToImage");
   if (!dispatch->clEnqueueCopyBufferToImage)
       dispatch->clEnqueueCopyBufferToImage = &clEnqueueCopyBufferToImage_unsupp;
   dispatch->clEnqueueMapBuffer = (clEnqueueMapBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueMapBuffer");
   if (!dispatch->clEnqueueMapBuffer)
       dispatch->clEnqueueMapBuffer = &clEnqueueMapBuffer_unsupp;
   dispatch->clEnqueueMapImage = (clEnqueueMapImage_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueMapImage");
   if (!dispatch->clEnqueueMapImage)
       dispatch->clEnqueueMapImage = &clEnqueueMapImage_unsupp;
   dispatch->clEnqueueUnmapMemObject = (clEnqueueUnmapMemObject_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueUnmapMemObject");
   if (!dispatch->clEnqueueUnmapMemObject)
       dispatch->clEnqueueUnmapMemObject = &clEnqueueUnmapMemObject_unsupp;
   dispatch->clEnqueueNDRangeKernel = (clEnqueueNDRangeKernel_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueNDRangeKernel");
   if (!dispatch->clEnqueueNDRangeKernel)
       dispatch->clEnqueueNDRangeKernel = &clEnqueueNDRangeKernel_unsupp;
   dispatch->clEnqueueNativeKernel = (clEnqueueNativeKernel_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueNativeKernel");
   if (!dispatch->clEnqueueNativeKernel)
       dispatch->clEnqueueNativeKernel = &clEnqueueNativeKernel_unsupp;
   dispatch->clSetCommandQueueProperty = (clSetCommandQueueProperty_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetCommandQueueProperty");
   if (!dispatch->clSetCommandQueueProperty)
       dispatch->clSetCommandQueueProperty = &clSetCommandQueueProperty_unsupp;
   dispatch->clCreateImage2D = (clCreateImage2D_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateImage2D");
   if (!dispatch->clCreateImage2D)
       dispatch->clCreateImage2D = &clCreateImage2D_unsupp;
   dispatch->clCreateImage3D = (clCreateImage3D_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateImage3D");
   if (!dispatch->clCreateImage3D)
       dispatch->clCreateImage3D = &clCreateImage3D_unsupp;
   dispatch->clEnqueueMarker = (clEnqueueMarker_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueMarker");
   if (!dispatch->clEnqueueMarker)
       dispatch->clEnqueueMarker = &clEnqueueMarker_unsupp;
   dispatch->clEnqueueWaitForEvents = (clEnqueueWaitForEvents_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueWaitForEvents");
   if (!dispatch->clEnqueueWaitForEvents)
       dispatch->clEnqueueWaitForEvents = &clEnqueueWaitForEvents_unsupp;
   dispatch->clEnqueueBarrier = (clEnqueueBarrier_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueBarrier");
   if (!dispatch->clEnqueueBarrier)
       dispatch->clEnqueueBarrier = &clEnqueueBarrier_unsupp;
   dispatch->clUnloadCompiler = (clUnloadCompiler_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clUnloadCompiler");
   if (!dispatch->clUnloadCompiler)
       dispatch->clUnloadCompiler = &clUnloadCompiler_unsupp;
   dispatch->clGetExtensionFunctionAddress = (clGetExtensionFunctionAddress_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetExtensionFunctionAddress");
   if (!dispatch->clGetExtensionFunctionAddress)
       dispatch->clGetExtensionFunctionAddress = &clGetExtensionFunctionAddress_unsupp;
   dispatch->clCreateCommandQueue = (clCreateCommandQueue_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateCommandQueue");
   if (!dispatch->clCreateCommandQueue)
       dispatch->clCreateCommandQueue = &clCreateCommandQueue_unsupp;
   dispatch->clCreateSampler = (clCreateSampler_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateSampler");
   if (!dispatch->clCreateSampler)
       dispatch->clCreateSampler = &clCreateSampler_unsupp;
   dispatch->clEnqueueTask = (clEnqueueTask_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueTask");
   if (!dispatch->clEnqueueTask)
       dispatch->clEnqueueTask = &clEnqueueTask_unsupp;
   dispatch->clCreateSubBuffer = (clCreateSubBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateSubBuffer");
   if (!dispatch->clCreateSubBuffer)
       dispatch->clCreateSubBuffer = &clCreateSubBuffer_unsupp;
   dispatch->clSetMemObjectDestructorCallback = (clSetMemObjectDestructorCallback_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetMemObjectDestructorCallback");
   if (!dispatch->clSetMemObjectDestructorCallback)
       dispatch->clSetMemObjectDestructorCallback = &clSetMemObjectDestructorCallback_unsupp;
   dispatch->clCreateUserEvent = (clCreateUserEvent_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateUserEvent");
   if (!dispatch->clCreateUserEvent)
       dispatch->clCreateUserEvent = &clCreateUserEvent_unsupp;
   dispatch->clSetUserEventStatus = (clSetUserEventStatus_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetUserEventStatus");
   if (!dispatch->clSetUserEventStatus)
       dispatch->clSetUserEventStatus = &clSetUserEventStatus_unsupp;
   dispatch->clSetEventCallback = (clSetEventCallback_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetEventCallback");
   if (!dispatch->clSetEventCallback)
       dispatch->clSetEventCallback = &clSetEventCallback_unsupp;
   dispatch->clEnqueueReadBufferRect = (clEnqueueReadBufferRect_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueReadBufferRect");
   if (!dispatch->clEnqueueReadBufferRect)
       dispatch->clEnqueueReadBufferRect = &clEnqueueReadBufferRect_unsupp;
   dispatch->clEnqueueWriteBufferRect = (clEnqueueWriteBufferRect_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueWriteBufferRect");
   if (!dispatch->clEnqueueWriteBufferRect)
       dispatch->clEnqueueWriteBufferRect = &clEnqueueWriteBufferRect_unsupp;
   dispatch->clEnqueueCopyBufferRect = (clEnqueueCopyBufferRect_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueCopyBufferRect");
   if (!dispatch->clEnqueueCopyBufferRect)
       dispatch->clEnqueueCopyBufferRect = &clEnqueueCopyBufferRect_unsupp;
   dispatch->clCreateSubDevices = (clCreateSubDevices_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateSubDevices");
   if (!dispatch->clCreateSubDevices)
       dispatch->clCreateSubDevices = &clCreateSubDevices_unsupp;
   dispatch->clRetainDevice = (clRetainDevice_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainDevice");
   if (!dispatch->clRetainDevice)
       dispatch->clRetainDevice = &clRetainDevice_unsupp;
   dispatch->clReleaseDevice = (clReleaseDevice_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseDevice");
   if (!dispatch->clReleaseDevice)
       dispatch->clReleaseDevice = &clReleaseDevice_unsupp;
   dispatch->clCreateImage = (clCreateImage_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateImage");
   if (!dispatch->clCreateImage)
       dispatch->clCreateImage = &clCreateImage_unsupp;
   dispatch->clCreateProgramWithBuiltInKernels = (clCreateProgramWithBuiltInKernels_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateProgramWithBuiltInKernels");
   if (!dispatch->clCreateProgramWithBuiltInKernels)
       dispatch->clCreateProgramWithBuiltInKernels = &clCreateProgramWithBuiltInKernels_unsupp;
   dispatch->clCompileProgram = (clCompileProgram_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCompileProgram");
   if (!dispatch->clCompileProgram)
       dispatch->clCompileProgram = &clCompileProgram_unsupp;
   dispatch->clLinkProgram = (clLinkProgram_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clLinkProgram");
   if (!dispatch->clLinkProgram)
       dispatch->clLinkProgram = &clLinkProgram_unsupp;
   dispatch->clUnloadPlatformCompiler = (clUnloadPlatformCompiler_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clUnloadPlatformCompiler");
   if (!dispatch->clUnloadPlatformCompiler)
       dispatch->clUnloadPlatformCompiler = &clUnloadPlatformCompiler_unsupp;
   dispatch->clGetKernelArgInfo = (clGetKernelArgInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetKernelArgInfo");
   if (!dispatch->clGetKernelArgInfo)
       dispatch->clGetKernelArgInfo = &clGetKernelArgInfo_unsupp;
   dispatch->clEnqueueFillBuffer = (clEnqueueFillBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueFillBuffer");
   if (!dispatch->clEnqueueFillBuffer)
       dispatch->clEnqueueFillBuffer = &clEnqueueFillBuffer_unsupp;
   dispatch->clEnqueueFillImage = (clEnqueueFillImage_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueFillImage");
   if (!dispatch->clEnqueueFillImage)
       dispatch->clEnqueueFillImage = &clEnqueueFillImage_unsupp;
   dispatch->clEnqueueMigrateMemObjects = (clEnqueueMigrateMemObjects_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueMigrateMemObjects");
   if (!dispatch->clEnqueueMigrateMemObjects)
       dispatch->clEnqueueMigrateMemObjects = &clEnqueueMigrateMemObjects_unsupp;
   dispatch->clEnqueueMarkerWithWaitList = (clEnqueueMarkerWithWaitList_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueMarkerWithWaitList");
   if (!dispatch->clEnqueueMarkerWithWaitList)
       dispatch->clEnqueueMarkerWithWaitList = &clEnqueueMarkerWithWaitList_unsupp;
   dispatch->clEnqueueBarrierWithWaitList = (clEnqueueBarrierWithWaitList_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueBarrierWithWaitList");
   if (!dispatch->clEnqueueBarrierWithWaitList)
       dispatch->clEnqueueBarrierWithWaitList = &clEnqueueBarrierWithWaitList_unsupp;
   dispatch->clGetExtensionFunctionAddressForPlatform = (clGetExtensionFunctionAddressForPlatform_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetExtensionFunctionAddressForPlatform");
   if (!dispatch->clGetExtensionFunctionAddressForPlatform)
       dispatch->clGetExtensionFunctionAddressForPlatform = &clGetExtensionFunctionAddressForPlatform_unsupp;
   dispatch->clCreateCommandQueueWithProperties = (clCreateCommandQueueWithProperties_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateCommandQueueWithProperties");
   if (!dispatch->clCreateCommandQueueWithProperties)
       dispatch->clCreateCommandQueueWithProperties = &clCreateCommandQueueWithProperties_unsupp;
   dispatch->clCreatePipe = (clCreatePipe_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreatePipe");
   if (!dispatch->clCreatePipe)
       dispatch->clCreatePipe = &clCreatePipe_unsupp;
   dispatch->clGetPipeInfo = (clGetPipeInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetPipeInfo");
   if (!dispatch->clGetPipeInfo)
       dispatch->clGetPipeInfo = &clGetPipeInfo_unsupp;
   dispatch->clSVMAlloc = (clSVMAlloc_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSVMAlloc");
   if (!dispatch->clSVMAlloc)
       dispatch->clSVMAlloc = &clSVMAlloc_unsupp;
   dispatch->clSVMFree = (clSVMFree_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSVMFree");
   if (!dispatch->clSVMFree)
       dispatch->clSVMFree = &clSVMFree_unsupp;
   dispatch->clCreateSamplerWithProperties = (clCreateSamplerWithProperties_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateSamplerWithProperties");
   if (!dispatch->clCreateSamplerWithProperties)
       dispatch->clCreateSamplerWithProperties = &clCreateSamplerWithProperties_unsupp;
   dispatch->clSetKernelArgSVMPointer = (clSetKernelArgSVMPointer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetKernelArgSVMPointer");
   if (!dispatch->clSetKernelArgSVMPointer)
       dispatch->clSetKernelArgSVMPointer = &clSetKernelArgSVMPointer_unsupp;
   dispatch->clSetKernelExecInfo = (clSetKernelExecInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetKernelExecInfo");
   if (!dispatch->clSetKernelExecInfo)
       dispatch->clSetKernelExecInfo = &clSetKernelExecInfo_unsupp;
   dispatch->clEnqueueSVMFree = (clEnqueueSVMFree_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueSVMFree");
   if (!dispatch->clEnqueueSVMFree)
       dispatch->clEnqueueSVMFree = &clEnqueueSVMFree_unsupp;
   dispatch->clEnqueueSVMMemcpy = (clEnqueueSVMMemcpy_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueSVMMemcpy");
   if (!dispatch->clEnqueueSVMMemcpy)
       dispatch->clEnqueueSVMMemcpy = &clEnqueueSVMMemcpy_unsupp;
   dispatch->clEnqueueSVMMemFill = (clEnqueueSVMMemFill_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueSVMMemFill");
   if (!dispatch->clEnqueueSVMMemFill)
       dispatch->clEnqueueSVMMemFill = &clEnqueueSVMMemFill_unsupp;
   dispatch->clEnqueueSVMMap = (clEnqueueSVMMap_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueSVMMap");
   if (!dispatch->clEnqueueSVMMap)
       dispatch->clEnqueueSVMMap = &clEnqueueSVMMap_unsupp;
   dispatch->clEnqueueSVMUnmap = (clEnqueueSVMUnmap_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueSVMUnmap");
   if (!dispatch->clEnqueueSVMUnmap)
       dispatch->clEnqueueSVMUnmap = &clEnqueueSVMUnmap_unsupp;
   dispatch->clSetDefaultDeviceCommandQueue = (clSetDefaultDeviceCommandQueue_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetDefaultDeviceCommandQueue");
   if (!dispatch->clSetDefaultDeviceCommandQueue)
       dispatch->clSetDefaultDeviceCommandQueue = &clSetDefaultDeviceCommandQueue_unsupp;
   dispatch->clGetDeviceAndHostTimer = (clGetDeviceAndHostTimer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetDeviceAndHostTimer");
   if (!dispatch->clGetDeviceAndHostTimer)
       dispatch->clGetDeviceAndHostTimer = &clGetDeviceAndHostTimer_unsupp;
   dispatch->clGetHostTimer = (clGetHostTimer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetHostTimer");
   if (!dispatch->clGetHostTimer)
       dispatch->clGetHostTimer = &clGetHostTimer_unsupp;
   dispatch->clCreateProgramWithIL = (clCreateProgramWithIL_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateProgramWithIL");
   if (!dispatch->clCreateProgramWithIL)
       dispatch->clCreateProgramWithIL = &clCreateProgramWithIL_unsupp;
   dispatch->clCloneKernel = (clCloneKernel_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCloneKernel");
   if (!dispatch->clCloneKernel)
       dispatch->clCloneKernel = &clCloneKernel_unsupp;
   dispatch->clGetKernelSubGroupInfo = (clGetKernelSubGroupInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetKernelSubGroupInfo");
   if (!dispatch->clGetKernelSubGroupInfo)
       dispatch->clGetKernelSubGroupInfo = &clGetKernelSubGroupInfo_unsupp;
   dispatch->clEnqueueSVMMigrateMem = (clEnqueueSVMMigrateMem_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueSVMMigrateMem");
   if (!dispatch->clEnqueueSVMMigrateMem)
       dispatch->clEnqueueSVMMigrateMem = &clEnqueueSVMMigrateMem_unsupp;
   dispatch->clSetProgramSpecializationConstant = (clSetProgramSpecializationConstant_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetProgramSpecializationConstant");
   if (!dispatch->clSetProgramSpecializationConstant)
       dispatch->clSetProgramSpecializationConstant = &clSetProgramSpecializationConstant_unsupp;
   dispatch->clSetProgramReleaseCallback = (clSetProgramReleaseCallback_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetProgramReleaseCallback");
   if (!dispatch->clSetProgramReleaseCallback)
       dispatch->clSetProgramReleaseCallback = &clSetProgramReleaseCallback_unsupp;
   dispatch->clSetContextDestructorCallback = (clSetContextDestructorCallback_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clSetContextDestructorCallback");
   if (!dispatch->clSetContextDestructorCallback)
       dispatch->clSetContextDestructorCallback = &clSetContextDestructorCallback_unsupp;
   dispatch->clCreateBufferWithProperties = (clCreateBufferWithProperties_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateBufferWithProperties");
   if (!dispatch->clCreateBufferWithProperties)
       dispatch->clCreateBufferWithProperties = &clCreateBufferWithProperties_unsupp;
   dispatch->clCreateImageWithProperties = (clCreateImageWithProperties_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateImageWithProperties");
   if (!dispatch->clCreateImageWithProperties)
       dispatch->clCreateImageWithProperties = &clCreateImageWithProperties_unsupp;

///////////////////////////////////////////////////////////////////////////////
// cl_ext_device_fission
   dispatch->clReleaseDeviceEXT = (clReleaseDeviceEXT_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clReleaseDeviceEXT");
   if (!dispatch->clReleaseDeviceEXT)
       dispatch->clReleaseDeviceEXT = &clReleaseDeviceEXT_unsupp;
   dispatch->clRetainDeviceEXT = (clRetainDeviceEXT_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clRetainDeviceEXT");
   if (!dispatch->clRetainDeviceEXT)
       dispatch->clRetainDeviceEXT = &clRetainDeviceEXT_unsupp;
   dispatch->clCreateSubDevicesEXT = (clCreateSubDevicesEXT_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateSubDevicesEXT");
   if (!dispatch->clCreateSubDevicesEXT)
       dispatch->clCreateSubDevicesEXT = &clCreateSubDevicesEXT_unsupp;
///////////////////////////////////////////////////////////////////////////////

// cl_khr_d3d10_sharing

#if defined(_WIN32)
   dispatch->clGetDeviceIDsFromD3D10KHR = (clGetDeviceIDsFromD3D10KHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetDeviceIDsFromD3D10KHR");
   if (!dispatch->clGetDeviceIDsFromD3D10KHR)
       dispatch->clGetDeviceIDsFromD3D10KHR = &clGetDeviceIDsFromD3D10KHR_unsupp;
   dispatch->clCreateFromD3D10BufferKHR = (clCreateFromD3D10BufferKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromD3D10BufferKHR");
   if (!dispatch->clCreateFromD3D10BufferKHR)
       dispatch->clCreateFromD3D10BufferKHR = &clCreateFromD3D10BufferKHR_unsupp;
   dispatch->clCreateFromD3D10Texture2DKHR = (clCreateFromD3D10Texture2DKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromD3D10Texture2DKHR");
   if (!dispatch->clCreateFromD3D10Texture2DKHR)
       dispatch->clCreateFromD3D10Texture2DKHR = &clCreateFromD3D10Texture2DKHR_unsupp;
   dispatch->clCreateFromD3D10Texture3DKHR = (clCreateFromD3D10Texture3DKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromD3D10Texture3DKHR");
   if (!dispatch->clCreateFromD3D10Texture3DKHR)
       dispatch->clCreateFromD3D10Texture3DKHR = &clCreateFromD3D10Texture3DKHR_unsupp;
   dispatch->clEnqueueAcquireD3D10ObjectsKHR = (clEnqueueAcquireD3D10ObjectsKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueAcquireD3D10ObjectsKHR");
   if (!dispatch->clEnqueueAcquireD3D10ObjectsKHR)
       dispatch->clEnqueueAcquireD3D10ObjectsKHR = &clEnqueueAcquireD3D10ObjectsKHR_unsupp;
   dispatch->clEnqueueReleaseD3D10ObjectsKHR = (clEnqueueReleaseD3D10ObjectsKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueReleaseD3D10ObjectsKHR");
   if (!dispatch->clEnqueueReleaseD3D10ObjectsKHR)
       dispatch->clEnqueueReleaseD3D10ObjectsKHR = &clEnqueueReleaseD3D10ObjectsKHR_unsupp;
#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_d3d11_sharing

#if defined(_WIN32)
   dispatch->clGetDeviceIDsFromD3D11KHR = (clGetDeviceIDsFromD3D11KHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetDeviceIDsFromD3D11KHR");
   if (!dispatch->clGetDeviceIDsFromD3D11KHR)
       dispatch->clGetDeviceIDsFromD3D11KHR = &clGetDeviceIDsFromD3D11KHR_unsupp;
   dispatch->clCreateFromD3D11BufferKHR = (clCreateFromD3D11BufferKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromD3D11BufferKHR");
   if (!dispatch->clCreateFromD3D11BufferKHR)
       dispatch->clCreateFromD3D11BufferKHR = &clCreateFromD3D11BufferKHR_unsupp;
   dispatch->clCreateFromD3D11Texture2DKHR = (clCreateFromD3D11Texture2DKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromD3D11Texture2DKHR");
   if (!dispatch->clCreateFromD3D11Texture2DKHR)
       dispatch->clCreateFromD3D11Texture2DKHR = &clCreateFromD3D11Texture2DKHR_unsupp;
   dispatch->clCreateFromD3D11Texture3DKHR = (clCreateFromD3D11Texture3DKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromD3D11Texture3DKHR");
   if (!dispatch->clCreateFromD3D11Texture3DKHR)
       dispatch->clCreateFromD3D11Texture3DKHR = &clCreateFromD3D11Texture3DKHR_unsupp;
   dispatch->clEnqueueAcquireD3D11ObjectsKHR = (clEnqueueAcquireD3D11ObjectsKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueAcquireD3D11ObjectsKHR");
   if (!dispatch->clEnqueueAcquireD3D11ObjectsKHR)
       dispatch->clEnqueueAcquireD3D11ObjectsKHR = &clEnqueueAcquireD3D11ObjectsKHR_unsupp;
   dispatch->clEnqueueReleaseD3D11ObjectsKHR = (clEnqueueReleaseD3D11ObjectsKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueReleaseD3D11ObjectsKHR");
   if (!dispatch->clEnqueueReleaseD3D11ObjectsKHR)
       dispatch->clEnqueueReleaseD3D11ObjectsKHR = &clEnqueueReleaseD3D11ObjectsKHR_unsupp;
#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_dx9_media_sharing

#if defined(_WIN32)
   dispatch->clGetDeviceIDsFromDX9MediaAdapterKHR = (clGetDeviceIDsFromDX9MediaAdapterKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetDeviceIDsFromDX9MediaAdapterKHR");
   if (!dispatch->clGetDeviceIDsFromDX9MediaAdapterKHR)
       dispatch->clGetDeviceIDsFromDX9MediaAdapterKHR = &clGetDeviceIDsFromDX9MediaAdapterKHR_unsupp;
   dispatch->clCreateFromDX9MediaSurfaceKHR = (clCreateFromDX9MediaSurfaceKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromDX9MediaSurfaceKHR");
   if (!dispatch->clCreateFromDX9MediaSurfaceKHR)
       dispatch->clCreateFromDX9MediaSurfaceKHR = &clCreateFromDX9MediaSurfaceKHR_unsupp;
   dispatch->clEnqueueAcquireDX9MediaSurfacesKHR = (clEnqueueAcquireDX9MediaSurfacesKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueAcquireDX9MediaSurfacesKHR");
   if (!dispatch->clEnqueueAcquireDX9MediaSurfacesKHR)
       dispatch->clEnqueueAcquireDX9MediaSurfacesKHR = &clEnqueueAcquireDX9MediaSurfacesKHR_unsupp;
   dispatch->clEnqueueReleaseDX9MediaSurfacesKHR = (clEnqueueReleaseDX9MediaSurfacesKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueReleaseDX9MediaSurfacesKHR");
   if (!dispatch->clEnqueueReleaseDX9MediaSurfacesKHR)
       dispatch->clEnqueueReleaseDX9MediaSurfacesKHR = &clEnqueueReleaseDX9MediaSurfacesKHR_unsupp;
#endif // defined(_WIN32)

///////////////////////////////////////////////////////////////////////////////

// cl_khr_egl_event
   dispatch->clCreateEventFromEGLSyncKHR = (clCreateEventFromEGLSyncKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateEventFromEGLSyncKHR");
   if (!dispatch->clCreateEventFromEGLSyncKHR)
       dispatch->clCreateEventFromEGLSyncKHR = &clCreateEventFromEGLSyncKHR_unsupp;
///////////////////////////////////////////////////////////////////////////////

// cl_khr_egl_image
   dispatch->clCreateFromEGLImageKHR = (clCreateFromEGLImageKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromEGLImageKHR");
   if (!dispatch->clCreateFromEGLImageKHR)
       dispatch->clCreateFromEGLImageKHR = &clCreateFromEGLImageKHR_unsupp;
   dispatch->clEnqueueAcquireEGLObjectsKHR = (clEnqueueAcquireEGLObjectsKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueAcquireEGLObjectsKHR");
   if (!dispatch->clEnqueueAcquireEGLObjectsKHR)
       dispatch->clEnqueueAcquireEGLObjectsKHR = &clEnqueueAcquireEGLObjectsKHR_unsupp;
   dispatch->clEnqueueReleaseEGLObjectsKHR = (clEnqueueReleaseEGLObjectsKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueReleaseEGLObjectsKHR");
   if (!dispatch->clEnqueueReleaseEGLObjectsKHR)
       dispatch->clEnqueueReleaseEGLObjectsKHR = &clEnqueueReleaseEGLObjectsKHR_unsupp;
///////////////////////////////////////////////////////////////////////////////

// cl_khr_gl_event
   dispatch->clCreateEventFromGLsyncKHR = (clCreateEventFromGLsyncKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateEventFromGLsyncKHR");
   if (!dispatch->clCreateEventFromGLsyncKHR)
       dispatch->clCreateEventFromGLsyncKHR = &clCreateEventFromGLsyncKHR_unsupp;
///////////////////////////////////////////////////////////////////////////////

// cl_khr_gl_sharing
   dispatch->clGetGLContextInfoKHR = (clGetGLContextInfoKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetGLContextInfoKHR");
   if (!dispatch->clGetGLContextInfoKHR)
       dispatch->clGetGLContextInfoKHR = &clGetGLContextInfoKHR_unsupp;
   dispatch->clCreateFromGLBuffer = (clCreateFromGLBuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromGLBuffer");
   if (!dispatch->clCreateFromGLBuffer)
       dispatch->clCreateFromGLBuffer = &clCreateFromGLBuffer_unsupp;
   dispatch->clCreateFromGLTexture = (clCreateFromGLTexture_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromGLTexture");
   if (!dispatch->clCreateFromGLTexture)
       dispatch->clCreateFromGLTexture = &clCreateFromGLTexture_unsupp;
   dispatch->clCreateFromGLRenderbuffer = (clCreateFromGLRenderbuffer_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromGLRenderbuffer");
   if (!dispatch->clCreateFromGLRenderbuffer)
       dispatch->clCreateFromGLRenderbuffer = &clCreateFromGLRenderbuffer_unsupp;
   dispatch->clGetGLObjectInfo = (clGetGLObjectInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetGLObjectInfo");
   if (!dispatch->clGetGLObjectInfo)
       dispatch->clGetGLObjectInfo = &clGetGLObjectInfo_unsupp;
   dispatch->clGetGLTextureInfo = (clGetGLTextureInfo_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetGLTextureInfo");
   if (!dispatch->clGetGLTextureInfo)
       dispatch->clGetGLTextureInfo = &clGetGLTextureInfo_unsupp;
   dispatch->clEnqueueAcquireGLObjects = (clEnqueueAcquireGLObjects_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueAcquireGLObjects");
   if (!dispatch->clEnqueueAcquireGLObjects)
       dispatch->clEnqueueAcquireGLObjects = &clEnqueueAcquireGLObjects_unsupp;
   dispatch->clEnqueueReleaseGLObjects = (clEnqueueReleaseGLObjects_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clEnqueueReleaseGLObjects");
   if (!dispatch->clEnqueueReleaseGLObjects)
       dispatch->clEnqueueReleaseGLObjects = &clEnqueueReleaseGLObjects_unsupp;
   dispatch->clCreateFromGLTexture2D = (clCreateFromGLTexture2D_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromGLTexture2D");
   if (!dispatch->clCreateFromGLTexture2D)
       dispatch->clCreateFromGLTexture2D = &clCreateFromGLTexture2D_unsupp;
   dispatch->clCreateFromGLTexture3D = (clCreateFromGLTexture3D_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clCreateFromGLTexture3D");
   if (!dispatch->clCreateFromGLTexture3D)
       dispatch->clCreateFromGLTexture3D = &clCreateFromGLTexture3D_unsupp;
///////////////////////////////////////////////////////////////////////////////

// cl_khr_subgroups
   dispatch->clGetKernelSubGroupInfoKHR = (clGetKernelSubGroupInfoKHR_t *)(size_t)p_clIcdGetFunctionAddressForPlatform(platform, "clGetKernelSubGroupInfoKHR");
   if (!dispatch->clGetKernelSubGroupInfoKHR)
       dispatch->clGetKernelSubGroupInfoKHR = &clGetKernelSubGroupInfoKHR_unsupp;
///////////////////////////////////////////////////////////////////////////////

}
#endif // defined(CL_ENABLE_LOADER_MANAGED_DISPATCH)

#ifdef __cplusplus
}
#endif
