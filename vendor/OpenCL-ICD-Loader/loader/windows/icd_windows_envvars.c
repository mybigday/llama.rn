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

#include <icd.h>
#include <stdbool.h>
#include <windows.h>

char *khrIcd_getenv(const char *name) {
    char *retVal;
    DWORD valSize;

    valSize = GetEnvironmentVariableA(name, NULL, 0);

    // valSize DOES include the null terminator, so for any set variable
    // will always be at least 1. If it's 0, the variable wasn't set.
    if (valSize == 0) return NULL;

    // Allocate the space necessary for the registry entry
    retVal = (char *)malloc(valSize);

    if (NULL != retVal) {
        GetEnvironmentVariableA(name, retVal, valSize);
    }

    return retVal;
}

static bool khrIcd_IsHighIntegrityLevel()
{
    bool isHighIntegrityLevel = false;

    HANDLE processToken;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_QUERY_SOURCE, &processToken)) {
        // Maximum possible size of SID_AND_ATTRIBUTES is maximum size of a SID + size of attributes DWORD.
        char mandatoryLabelBuffer[SECURITY_MAX_SID_SIZE + sizeof(DWORD)] = {0};
        DWORD bufferSize;
        if (GetTokenInformation(processToken, TokenIntegrityLevel, mandatoryLabelBuffer, sizeof(mandatoryLabelBuffer),
                                &bufferSize) != 0) {
            const TOKEN_MANDATORY_LABEL* mandatoryLabel = (const TOKEN_MANDATORY_LABEL*)(mandatoryLabelBuffer);
            const DWORD subAuthorityCount = *GetSidSubAuthorityCount(mandatoryLabel->Label.Sid);
            const DWORD integrityLevel = *GetSidSubAuthority(mandatoryLabel->Label.Sid, subAuthorityCount - 1);

            isHighIntegrityLevel = integrityLevel > SECURITY_MANDATORY_MEDIUM_RID;
        }

        CloseHandle(processToken);
    }

    return isHighIntegrityLevel;
}

char *khrIcd_secure_getenv(const char *name) {
    if (khrIcd_IsHighIntegrityLevel()) {
        KHR_ICD_TRACE("Running at a high integrity level, so secure_getenv is returning NULL\n");
        return NULL;
    }

    return khrIcd_getenv(name);
}

void khrIcd_free_getenv(char *val) {
    free((void *)val);
}
