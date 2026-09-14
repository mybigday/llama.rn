/*
 * Copyright (c) 2017-2019 The Khronos Group Inc.
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

#include <stdbool.h>
#include <windows.h>

#ifdef _LP64
#define PRIDW_PREFIX
#define PRIUL_PREFIX
#else
#define PRIDW_PREFIX "l"
#define PRIUL_PREFIX "l"
#endif
#define PRIuDW PRIDW_PREFIX "u"
#define PRIxDW PRIDW_PREFIX "x"
#define PRIuUL PRIUL_PREFIX "u"
#define PRIxUL PRIUL_PREFIX "x"

#ifdef __cplusplus
extern "C" {
#endif
extern const LUID ZeroLuid;

BOOL adapterAdd(const char* szName, LUID luid);

// Do not free the memory returned by this function.
const char* getOpenCLRegKeyName(void);

#ifdef __cplusplus
}
#endif
