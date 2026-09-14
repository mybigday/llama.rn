/*
 * Copyright (c) 2017-2020 The Khronos Group Inc.
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
#include "icd_windows_dxgk.h"

#include <windows.h>
#include "adapter.h"

#ifndef NTSTATUS
typedef LONG NTSTATUS;
#define STATUS_SUCCESS  ((NTSTATUS)0x00000000L)
#define STATUS_BUFFER_TOO_SMALL ((NTSTATUS)0xC0000023)
#define NT_SUCCESS(status) (((NTSTATUS)(status)) >= 0)
#endif

bool khrIcdOsVendorsEnumerateDXGK(void)
{
    bool ret = false;
    int result = 0;

    // Get handle to GDI Runtime
    HMODULE h = LoadLibraryA("gdi32.dll");
    if (h == NULL)
        return ret;

    if(GetProcAddress(h, "D3DKMTSubmitPresentBltToHwQueue")) // OS Version check
    {
        LoaderEnumAdapters2 EnumAdapters;
        NTSTATUS status = STATUS_SUCCESS;

        EnumAdapters.adapter_count = 0;
        EnumAdapters.adapters = NULL;
        PFN_LoaderEnumAdapters2 pEnumAdapters2 = (PFN_LoaderEnumAdapters2)(void*)GetProcAddress(h, "D3DKMTEnumAdapters2");
        if (!pEnumAdapters2)
        {
            KHR_ICD_TRACE("GetProcAddress failed for D3DKMTEnumAdapters2\n");
            goto out;
        }
        PFN_LoaderQueryAdapterInfo pQueryAdapterInfo = (PFN_LoaderQueryAdapterInfo)(void*)GetProcAddress(h, "D3DKMTQueryAdapterInfo");
        if (!pQueryAdapterInfo)
        {
            KHR_ICD_TRACE("GetProcAddress failed for D3DKMTQueryAdapterInfo\n");
            goto out;
        }
        while (1)
        {
            EnumAdapters.adapter_count = 0;
            EnumAdapters.adapters = NULL;
            status = pEnumAdapters2(&EnumAdapters);
            if (status == STATUS_BUFFER_TOO_SMALL)
            {
                // Number of Adapters increased between calls, retry;
                continue;
            }
            else if (!NT_SUCCESS(status))
            {
                KHR_ICD_TRACE("D3DKMTEnumAdapters2 status != SUCCESS\n");
                goto out;
            }
            break;
        }
        EnumAdapters.adapters = malloc(sizeof(*EnumAdapters.adapters)*(EnumAdapters.adapter_count));
        if (EnumAdapters.adapters == NULL)
        {
            KHR_ICD_TRACE("Allocation failure for adapters buffer\n");
            goto out;
        }
        status = pEnumAdapters2(&EnumAdapters);
        if (!NT_SUCCESS(status))
        {
            KHR_ICD_TRACE("D3DKMTEnumAdapters2 status != SUCCESS\n");
            goto out;
        }
        const char* cszOpenCLRegKeyName = getOpenCLRegKeyName();
        const int szOpenCLRegKeyName = (int)(strlen(cszOpenCLRegKeyName) + 1)*sizeof(cszOpenCLRegKeyName[0]);
        for (UINT AdapterIndex = 0; AdapterIndex < EnumAdapters.adapter_count; AdapterIndex++)
        {
            LoaderQueryRegistryInfo queryArgs = {0};
            LoaderQueryRegistryInfo* pQueryArgs = &queryArgs;
            LoaderQueryRegistryInfo* pQueryBuffer = NULL;
            queryArgs.query_type = LOADER_QUERY_REGISTRY_ADAPTER_KEY;
            queryArgs.query_flags.translate_path = TRUE;
            queryArgs.value_type = REG_SZ;
            result = MultiByteToWideChar(
                CP_UTF8,
                0,
                cszOpenCLRegKeyName,
                szOpenCLRegKeyName,
                queryArgs.value_name,
                ARRAYSIZE(queryArgs.value_name));
            if (!result)
            {
                KHR_ICD_TRACE("MultiByteToWideChar status != SUCCESS\n");
                continue;
            }
            LoaderQueryAdapterInfo queryAdapterInfo = {0};
            queryAdapterInfo.handle = EnumAdapters.adapters[AdapterIndex].handle;
            queryAdapterInfo.type = LOADER_QUERY_TYPE_REGISTRY;
            queryAdapterInfo.private_data = &queryArgs;
            queryAdapterInfo.private_data_size = sizeof(queryArgs);
            status = pQueryAdapterInfo(&queryAdapterInfo);
            if (!NT_SUCCESS(status))
            {
                // Try a different value type.  Some vendors write the key as a multi-string type.
                queryArgs.value_type = REG_MULTI_SZ;
                status = pQueryAdapterInfo(&queryAdapterInfo);
                if (NT_SUCCESS(status))
                {
                    KHR_ICD_TRACE("Accepting multi-string registry key type\n");
                }
                else
                {
                    // Continue trying to get as much info on each adapter as possible.
                    // It's too late to return FALSE and claim WDDM2_4 enumeration is not available here.
                    continue;
                }
            }
            if (NT_SUCCESS(status) && pQueryArgs->status == LOADER_QUERY_REGISTRY_STATUS_BUFFER_OVERFLOW)
            {
                ULONG queryBufferSize = sizeof(LoaderQueryRegistryInfo) + queryArgs.output_value_size;
                pQueryBuffer = malloc(queryBufferSize);
                if (pQueryBuffer == NULL)
                    continue;
                memcpy(pQueryBuffer, &queryArgs, sizeof(LoaderQueryRegistryInfo));
                queryAdapterInfo.private_data = pQueryBuffer;
                queryAdapterInfo.private_data_size = queryBufferSize;
                status = pQueryAdapterInfo(&queryAdapterInfo);
                pQueryArgs = pQueryBuffer;
            }
            if (NT_SUCCESS(status) && pQueryArgs->status == LOADER_QUERY_REGISTRY_STATUS_SUCCESS)
            {
                char cszLibraryName[MAX_PATH];
                result = WideCharToMultiByte(
                    CP_UTF8,
                    0,
                    pQueryArgs->output_string,
                    -1,
                    cszLibraryName,
                    MAX_PATH,
                    NULL,
                    NULL);
                if (!result)
                {
                    KHR_ICD_TRACE("WideCharToMultiByte status != SUCCESS\n");
                }
                else
                {
                    ret |= adapterAdd(cszLibraryName, EnumAdapters.adapters[AdapterIndex].luid);
                }
            }
            else if (status == (NTSTATUS)STATUS_INVALID_PARAMETER)
            {
                free(pQueryBuffer);
                goto out;
            }
            free(pQueryBuffer);
        }
out:
      free(EnumAdapters.adapters);
    }

    FreeLibrary(h);

    return ret;
}
