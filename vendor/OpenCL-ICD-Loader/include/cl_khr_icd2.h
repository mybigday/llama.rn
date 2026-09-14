/*
 * Copyright (c) 2016-2025 The Khronos Group Inc.
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

#include <CL/cl.h>

#if !defined(CL_ICD2_TAG_KHR)
#if INTPTR_MAX == INT32_MAX
#define CL_ICD2_TAG_KHR ((intptr_t)0x434C3331)
#else
#define CL_ICD2_TAG_KHR ((intptr_t)0x4F50454E434C3331)
#endif

typedef void * CL_API_CALL
clIcdGetFunctionAddressForPlatformKHR_t(
    cl_platform_id platform,
    const char* func_name);

typedef clIcdGetFunctionAddressForPlatformKHR_t *
clIcdGetFunctionAddressForPlatformKHR_fn;

typedef cl_int CL_API_CALL
clIcdSetPlatformDispatchDataKHR_t(
    cl_platform_id platform,
    void *disp_data);

typedef clIcdSetPlatformDispatchDataKHR_t *
clIcdSetPlatformDispatchDataKHR_fn;
#endif // !defined(CL_ICD2_TAG_KHR)
