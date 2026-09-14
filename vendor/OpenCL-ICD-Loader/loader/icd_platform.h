/*
 * Copyright (c) 2016-2019 The Khronos Group Inc.
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

#ifndef _ICD_PLATFORM_H_
#define _ICD_PLATFORM_H_

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__QNXNTO__)

#define PATH_SEPARATOR  ':'
#define DIRECTORY_SYMBOL '/'
#ifdef __ANDROID__
#ifndef ICD_VENDOR_PATH
#define ICD_VENDOR_PATH "/system/vendor/Khronos/OpenCL/vendors"
#endif // ICD_VENDOR_PATH
#ifndef LAYER_PATH
#define LAYER_PATH "/system/vendor/Khronos/OpenCL/layers"
#endif // LAYER_PATH
#else
#define ICD_VENDOR_PATH "/etc/OpenCL/vendors"
#define LAYER_PATH "/etc/OpenCL/layers"
#endif // ANDROID

#elif defined(_WIN32)

#define PATH_SEPARATOR ';'
#define DIRECTORY_SYMBOL '\\'

#else
#error Unknown OS!
#endif

#ifdef __MINGW32__
#if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0600)
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#endif // __MINGW32__

#endif
