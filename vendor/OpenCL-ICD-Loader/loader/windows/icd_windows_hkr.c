/*
 * Copyright (c) 2017-2022 The Khronos Group Inc.
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
#include "icd_windows_hkr.h"
#include <windows.h>
#include "icd_windows_dxgk.h"
#include <cfgmgr32.h>
#include <assert.h>
#include <stdbool.h>
#include <initguid.h>
#include <devpkey.h>
#include <devguid.h>

 // This GUID was only added to devguid.h on Windows SDK v10.0.16232 which
 // corresponds to Windows 10 Redstone 3 (Windows 10 Fall Creators Update).
DEFINE_GUID(OCL_GUID_DEVCLASS_SOFTWARECOMPONENT, 0x5c4c3332, 0x344d, 0x483c, 0x87, 0x39, 0x25, 0x9e, 0x93, 0x4c, 0x9c, 0xc8);

typedef enum
{
    ProbeFailure,
    PendingReboot,
    Valid
} DeviceProbeResult;

#define KHR_SAFE_RELEASE(mem)       \
    do                              \
    {                               \
        free(mem);                  \
        mem = NULL;                 \
    } while (0)

static const char OPENCL_REG_SUB_KEY[] = "OpenCLDriverName";

#ifndef _WIN64
static const char OPENCL_REG_SUB_KEY_WOW[] = "OpenCLDriverNameWow";
#endif

// Do not free the memory returned by this function.
const char* getOpenCLRegKeyName(void)
{
#ifdef _WIN64
    return OPENCL_REG_SUB_KEY;
#else
    // The suffix/substring "WoW" is meaningful only when a 32-bit
    // application is running on a 64-bit Windows OS. A 32-bit application
    // running on a 32-bit OS uses non-WoW names.
    BOOL is_wow64;
    if (IsWow64Process(GetCurrentProcess(), &is_wow64) && is_wow64)
    {
        return OPENCL_REG_SUB_KEY_WOW;
    }

    return OPENCL_REG_SUB_KEY;
#endif
}

static bool ReadOpenCLKey(DEVINST dnDevNode)
{
    HKEY hkey = 0;
    CONFIGRET ret;
    bool bRet = false;
    DWORD dwLibraryNameType = 0;
    char *cszOclPath = NULL;
    DWORD dwOclPathSize = 0;
    LSTATUS result;

    ret = CM_Open_DevNode_Key(
        dnDevNode,
        KEY_QUERY_VALUE,
        0,
        RegDisposition_OpenExisting,
        &hkey,
        CM_REGISTRY_SOFTWARE);

    if (CR_SUCCESS != ret)
    {
        KHR_ICD_TRACE("Failed with ret 0x%"PRIxDW"\n", ret);
        goto out;
    }
    else
    {
        result = RegQueryValueExA(
            hkey,
            getOpenCLRegKeyName(),
            NULL,
            &dwLibraryNameType,
            NULL,
            &dwOclPathSize);

        if (ERROR_SUCCESS != result)
        {
            KHR_ICD_TRACE("Failed to open sub key 0x%"PRIxDW"\n", result);
            goto out;
        }

        cszOclPath = malloc(dwOclPathSize);
        if (NULL == cszOclPath)
        {
            KHR_ICD_TRACE("Failed to allocate %"PRIuDW" bytes for registry value\n", dwOclPathSize);
            goto out;
        }

        result = RegQueryValueExA(
            hkey,
            getOpenCLRegKeyName(),
            NULL,
            &dwLibraryNameType,
            (LPBYTE)cszOclPath,
            &dwOclPathSize);
        if (ERROR_SUCCESS != result)
        {
            KHR_ICD_TRACE("Failed to open sub key 0x%"PRIxDW"\n", result);
            goto out;
        }

        if (REG_SZ != dwLibraryNameType)
        {
            if (REG_MULTI_SZ == dwLibraryNameType)
            {
                KHR_ICD_TRACE("Accepting multi-string registry key type\n");
            }
            else
            {
                KHR_ICD_TRACE("Unexpected registry entry 0x%"PRIxDW"! continuing\n", dwLibraryNameType);
                goto out;
            }
        }

        KHR_ICD_TRACE("    Path: %s\n", cszOclPath);

        bRet |= adapterAdd(cszOclPath, ZeroLuid);
    }

out:
    free(cszOclPath);

    if (hkey)
    {
        result = RegCloseKey(hkey);
        if (ERROR_SUCCESS != result)
        {
            KHR_ICD_TRACE("WARNING: failed to close hkey 0x%"PRIxDW"\n", result);
        }
    }

    return bRet;
}

static DeviceProbeResult ProbeDevice(DEVINST devnode)
{
    CONFIGRET ret;
    ULONG ulStatus;
    ULONG ulProblem;

    ret = CM_Get_DevNode_Status(
        &ulStatus,
        &ulProblem,
        devnode,
        0);

    if (CR_SUCCESS != ret)
    {
        KHR_ICD_TRACE("    WARNING: failed to probe the status of the device 0x%"PRIxDW"\n", ret);
        return ProbeFailure;
    }

    //
    // Careful here, we need to check 2 scenarios:
    // 1. DN_NEED_RESTART
    //    status flag indicates that a reboot is needed when an _already started_
    //    device cannot be stopped. This covers devices that are still started with their
    //    old KMD (because they couldn't be stopped/restarted) while the UMD is updated
    //    and possibly out of sync.
    //
    // 2.  Status & DN_HAS_PROBLEM  && Problem == CM_PROB_NEED_RESTART
    //     indicates that a reboot is needed when a _stopped device_ cannot be (re)started.
    //
    if (((ulStatus & DN_HAS_PROBLEM) && ulProblem == CM_PROB_NEED_RESTART) ||
          ulStatus & DN_NEED_RESTART)
    {
        KHR_ICD_TRACE("    WARNING: device is pending reboot (0x%" PRIxUL "), skipping...\n", ulStatus);
        return PendingReboot;
    }

    return Valid;
}

// Tries to look for the OpenCL key under the display devices and
// if not found, falls back to software component devices.
bool khrIcdOsVendorsEnumerateHKR(void)
{
    CONFIGRET ret;
    int iret;
    bool foundOpenCLKey = false;
    DEVINST devinst = 0;
    DEVINST devchild = 0;
    wchar_t *deviceIdList = NULL;
    ULONG szBuffer = 0;

    OLECHAR display_adapter_guid_str[MAX_GUID_STRING_LEN];
#if defined(CM_GETIDLIST_FILTER_CLASS) && defined(CM_GETIDLIST_FILTER_PRESENT)
    ULONG ulFlags = CM_GETIDLIST_FILTER_CLASS |
                    CM_GETIDLIST_FILTER_PRESENT;
#else
    ULONG ulFlags = 0x300;
#endif

    iret = StringFromGUID2(
        &GUID_DEVCLASS_DISPLAY,
        display_adapter_guid_str,
        MAX_GUID_STRING_LEN);

    if (MAX_GUID_STRING_LEN != iret)
    {
        KHR_ICD_TRACE("StringFromGUID2 failed with %d\n", iret);
        goto out;
    }

    // Paranoia: we might have a new device added to the list between the call
    // to CM_Get_Device_ID_List_Size() and the call to CM_Get_Device_ID_List().
    do
    {
        ret = CM_Get_Device_ID_List_SizeW(
            &szBuffer,
            display_adapter_guid_str,
            ulFlags);

        if (CR_SUCCESS != ret)
        {
            KHR_ICD_TRACE("CM_Get_Device_ID_List_size failed with 0x%"PRIxDW"\n", ret);
            break;
        }

        // "pulLen [out] Receives a value representing the required buffer
        //  size, in characters."
        //  So we need to allocate the right size in bytes but we still need
        //  to keep szBuffer as it was returned from CM_Get_Device_ID_List_Size so
        //  the call to CM_Get_Device_ID_List will receive the correct size.
        deviceIdList = malloc(szBuffer * sizeof(wchar_t));
        if (NULL == deviceIdList)
        {
            KHR_ICD_TRACE("Failed to allocate %" PRIuUL " bytes for device ID strings\n", szBuffer);
            break;
        }

        ret = CM_Get_Device_ID_ListW(
            display_adapter_guid_str,
            deviceIdList,
            szBuffer,
            ulFlags);

        if (CR_SUCCESS != ret)
        {
            KHR_ICD_TRACE("CM_Get_Device_ID_List failed with 0x%"PRIxDW"\n", ret);
            KHR_SAFE_RELEASE(deviceIdList);
        }
    } while (CR_BUFFER_SMALL == ret);

    if (NULL == deviceIdList)
    {
        goto out;
    }

    for (PWSTR deviceId = deviceIdList; *deviceId; deviceId += wcslen(deviceId) + 1)
    {
        KHR_ICD_WIDE_TRACE(L"Device ID: %ls\n", deviceId);

        ret = CM_Locate_DevNodeW(&devinst, deviceId, 0);
        if (CR_SUCCESS == ret)
        {
            KHR_ICD_TRACE("    devinst: %lu\n", devinst);
        }
        else
        {
            KHR_ICD_TRACE("CM_Locate_DevNode failed with 0x%"PRIxDW"\n", ret);
            continue;
        }

        if (ProbeDevice(devinst) != Valid)
        {
            continue;
        }

        KHR_ICD_TRACE("    Trying to look for the key in the display adapter HKR...\n");
        if (ReadOpenCLKey(devinst))
        {
            foundOpenCLKey = true;
            continue;
        }

        KHR_ICD_TRACE("    Could not find the key, proceeding to children software components...\n");

        ret = CM_Get_Child(
            &devchild,
            devinst,
            0);

        if (CR_SUCCESS != ret)
        {
            KHR_ICD_TRACE("    CM_Get_Child returned 0x%"PRIxDW", skipping children...\n", ret);
        }
        else
        {
            do
            {
                wchar_t deviceInstanceID[MAX_DEVICE_ID_LEN] = { 0 };
                GUID guid;
                ULONG szGuid = sizeof(guid);

                KHR_ICD_TRACE("    devchild: %lu\n", devchild);
                ret = CM_Get_Device_IDW(
                    devchild,
                    deviceInstanceID,
                    sizeof(deviceInstanceID),
                    0);

                if (CR_SUCCESS != ret)
                {
                    KHR_ICD_TRACE("    CM_Get_Device_ID returned 0x%"PRIxDW", skipping device...\n", ret);
                    continue;
                }
                else
                {
                    KHR_ICD_WIDE_TRACE(L"    deviceInstanceID: %ls\n", deviceInstanceID);
                }

                ret = CM_Get_DevNode_Registry_PropertyW(
                    devchild,
                    CM_DRP_CLASSGUID,
                    NULL,
                    (PBYTE)&guid,
                    &szGuid,
                    0);

                if (CR_SUCCESS != ret)
                {
                    KHR_ICD_TRACE("    CM_Get_DevNode_Registry_PropertyW returned 0x%"PRIxDW", skipping device...\n", ret);
                    continue;
                }
                else if (!IsEqualGUID(&OCL_GUID_DEVCLASS_SOFTWARECOMPONENT, &guid))
                {
                    KHR_ICD_TRACE("    GUID does not match, skipping device...\n");
                    continue;
                }

                if (ProbeDevice(devchild) != Valid)
                {
                    continue;
                }

                if (ReadOpenCLKey(devchild))
                {
                    foundOpenCLKey = true;
                    break;
                }
            } while (CM_Get_Sibling(&devchild, devchild, 0) == CR_SUCCESS);
        }
    }

out:
    free(deviceIdList);
    return foundOpenCLKey;
}
