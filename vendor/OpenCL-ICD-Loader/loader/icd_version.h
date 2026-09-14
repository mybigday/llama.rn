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

#ifndef _ICD_VERSION_H_
#define _ICD_VERSION_H_

#define OPENCL_ICD_LOADER_NAME_STRING   "Khronos OpenCL ICD Loader"
#define OPENCL_ICD_LOADER_VENDOR_STRING "Khronos Group"

#define OPENCL_ICD_LOADER_VAL(_v) #_v
#define OPENCL_ICD_LOADER_TOSTRING(_d) OPENCL_ICD_LOADER_VAL(_d)
#define OPENCL_ICD_LOADER_VERSION_STRING \
    OPENCL_ICD_LOADER_TOSTRING(OPENCL_ICD_LOADER_VERSION_MAJOR) "." \
    OPENCL_ICD_LOADER_TOSTRING(OPENCL_ICD_LOADER_VERSION_MINOR) "." \
    OPENCL_ICD_LOADER_TOSTRING(OPENCL_ICD_LOADER_VERSION_REV)

#if CL_TARGET_OPENCL_VERSION == 100
#define OPENCL_ICD_LOADER_OCL_VERSION_NUMBER "1.0"
#endif
#if CL_TARGET_OPENCL_VERSION == 110
#define OPENCL_ICD_LOADER_OCL_VERSION_NUMBER "1.1"
#endif
#if CL_TARGET_OPENCL_VERSION == 120
#define OPENCL_ICD_LOADER_OCL_VERSION_NUMBER "1.2"
#endif
#if CL_TARGET_OPENCL_VERSION == 200
#define OPENCL_ICD_LOADER_OCL_VERSION_NUMBER "2.0"
#endif
#if CL_TARGET_OPENCL_VERSION == 210
#define OPENCL_ICD_LOADER_OCL_VERSION_NUMBER "2.1"
#endif
#if CL_TARGET_OPENCL_VERSION == 220
#define OPENCL_ICD_LOADER_OCL_VERSION_NUMBER "2.2"
#endif
#if CL_TARGET_OPENCL_VERSION == 300
#define OPENCL_ICD_LOADER_OCL_VERSION_NUMBER "3.0"
#endif

#define OPENCL_ICD_LOADER_OCL_VERSION_STRING \
    "OpenCL " OPENCL_ICD_LOADER_OCL_VERSION_NUMBER

#endif
