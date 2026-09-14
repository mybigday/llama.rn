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

#include <initguid.h>

#include "icd.h"
#include "icd_windows.h"
#include "icd_windows_hkr.h"
#include "icd_windows_dxgk.h"
#include "icd_windows_apppackage.h"
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <winreg.h>

#include <dxgi.h>
typedef HRESULT (WINAPI *PFN_CREATE_DXGI_FACTORY)(REFIID, void **);

static INIT_ONCE initialized = INIT_ONCE_STATIC_INIT;

typedef struct WinAdapter
{
    char * szName;
    LUID luid;
} WinAdapter;

const LUID ZeroLuid = { 0, 0 };

static WinAdapter* pWinAdapterBegin = NULL;
static WinAdapter* pWinAdapterEnd = NULL;
static WinAdapter* pWinAdapterCapacity = NULL;

BOOL adapterAdd(const char* szName, LUID luid)
{
    BOOL result = TRUE;
    if (pWinAdapterEnd == pWinAdapterCapacity)
    {
        size_t oldCapacity = pWinAdapterCapacity - pWinAdapterBegin;
        size_t newCapacity = oldCapacity;
        if (0 == newCapacity)
        {
            newCapacity = 1;
        }
        else if(newCapacity < UINT_MAX/2)
        {
            newCapacity *= 2;
        }

        WinAdapter* pNewBegin = malloc(newCapacity * sizeof(*pWinAdapterBegin));
        if (!pNewBegin)
            result = FALSE;
        else
        {
            if (pWinAdapterBegin)
            {
                memcpy(pNewBegin, pWinAdapterBegin, oldCapacity * sizeof(*pWinAdapterBegin));
                free(pWinAdapterBegin);
            }
            pWinAdapterCapacity = pNewBegin + newCapacity;
            pWinAdapterEnd = pNewBegin + oldCapacity;
            pWinAdapterBegin = pNewBegin;
        }
    }
    if (pWinAdapterEnd != pWinAdapterCapacity)
    {
        size_t nameLen = (strlen(szName) + 1)*sizeof(szName[0]);
        pWinAdapterEnd->szName = malloc(nameLen);
        if (!pWinAdapterEnd->szName)
            result = FALSE;
        else
        {
            memcpy(pWinAdapterEnd->szName, szName, nameLen);
            pWinAdapterEnd->luid = luid;
            ++pWinAdapterEnd;
        }
    }
    return result;
}

void adapterFree(WinAdapter *pWinAdapter)
{
    free(pWinAdapter->szName);
    pWinAdapter->szName = NULL;
}

#if defined(CL_ENABLE_LAYERS)
typedef struct WinLayer
{
    char * szName;
    DWORD priority;
} WinLayer;

static WinLayer* pWinLayerBegin;
static WinLayer* pWinLayerEnd;
static WinLayer* pWinLayerCapacity;

static int __cdecl compareLayer(const void *a, const void *b)
{
    return ((const WinLayer *)a)->priority < ((const WinLayer *)b)->priority ? -1 :
           ((const WinLayer *)a)->priority > ((const WinLayer *)b)->priority ? 1 : 0;
}

static BOOL layerAdd(const char* szName, DWORD priority)
{
    BOOL result = TRUE;
    if (pWinLayerEnd == pWinLayerCapacity)
    {
        size_t oldCapacity = pWinLayerCapacity - pWinLayerBegin;
        size_t newCapacity = oldCapacity;
        if (0 == newCapacity)
        {
            newCapacity = 1;
        }
        else if(newCapacity < UINT_MAX/2)
        {
            newCapacity *= 2;
        }

        WinLayer* pNewBegin = malloc(newCapacity * sizeof(*pWinLayerBegin));
        if (!pNewBegin)
        {
            KHR_ICD_TRACE("Failed allocate space for Layers array\n");
            result = FALSE;
        }
        else
        {
            if (pWinLayerBegin)
            {
                memcpy(pNewBegin, pWinLayerBegin, oldCapacity * sizeof(*pWinLayerBegin));
                free(pWinLayerBegin);
            }
            pWinLayerCapacity = pNewBegin + newCapacity;
            pWinLayerEnd = pNewBegin + oldCapacity;
            pWinLayerBegin = pNewBegin;
        }
    }
    if (pWinLayerEnd != pWinLayerCapacity)
    {
        size_t nameLen = (strlen(szName) + 1)*sizeof(szName[0]);
        pWinLayerEnd->szName = malloc(nameLen);
        if (!pWinLayerEnd->szName)
        {
            KHR_ICD_TRACE("Failed allocate space for Layer file path\n");
            result = FALSE;
        }
        else
        {
            memcpy(pWinLayerEnd->szName, szName, nameLen);
            pWinLayerEnd->priority = priority;
            ++pWinLayerEnd;
        }
    }
    return result;
}

void layerFree(WinLayer *pWinLayer)
{
    free(pWinLayer->szName);
    pWinLayer->szName = NULL;
}
#endif // defined(CL_ENABLE_LAYERS)

/*
 *
 * Vendor enumeration functions
 *
 */

// go through the list of vendors in the registry and call khrIcdVendorAdd
// for each vendor encountered
BOOL CALLBACK khrIcdOsVendorsEnumerate(PINIT_ONCE InitOnce, PVOID Parameter, PVOID *lpContext)
{
    (void)InitOnce;
    (void)Parameter;
    (void)lpContext;

    LONG result;
    BOOL status = FALSE, currentStatus = FALSE;
    const char* platformsName = "SOFTWARE\\Khronos\\OpenCL\\Vendors";
    HKEY platformsKey = NULL;
    DWORD dwIndex;

    khrIcdInitializeTrace();
    khrIcdVendorsEnumerateEnv();

    currentStatus = khrIcdOsVendorsEnumerateDXGK();
    status |= currentStatus;
    if (!currentStatus)
    {
        KHR_ICD_TRACE("Failed to load via DXGK interface on RS4, continuing\n");
    }

    currentStatus = khrIcdOsVendorsEnumerateHKR();
    status |= currentStatus;
    if (!currentStatus)
    {
        KHR_ICD_TRACE("Failed to enumerate HKR entries, continuing\n");
    }

    currentStatus = khrIcdOsVendorsEnumerateAppPackage();
    status |= currentStatus;
    if (!currentStatus)
    {
        KHR_ICD_TRACE("Failed to enumerate App package entry, continuing\n");
    }

    KHR_ICD_TRACE("Opening key HKLM\\%s...\n", platformsName);
    result = RegOpenKeyExA(
        HKEY_LOCAL_MACHINE,
        platformsName,
        0,
        KEY_READ,
        &platformsKey);
    if (ERROR_SUCCESS != result)
    {
        KHR_ICD_TRACE("Failed to open platforms key %s, continuing\n", platformsName);
    }
    else
    {
        // for each value
        for (dwIndex = 0;; ++dwIndex)
        {
            char cszLibraryName[1024] = {0};
            DWORD dwLibraryNameSize = sizeof(cszLibraryName);
            DWORD dwLibraryNameType = 0;
            DWORD dwValue = 0;
            DWORD dwValueSize = sizeof(dwValue);

            // read the value name
            KHR_ICD_TRACE("Reading value %"PRIuDW"...\n", dwIndex);
            result = RegEnumValueA(
                  platformsKey,
                  dwIndex,
                  cszLibraryName,
                  &dwLibraryNameSize,
                  NULL,
                  &dwLibraryNameType,
                  (LPBYTE)&dwValue,
                  &dwValueSize);
            // if RegEnumKeyEx fails, we are done with the enumeration
            if (ERROR_SUCCESS != result)
            {
                KHR_ICD_TRACE("Failed to read value %"PRIuDW", done reading key.\n", dwIndex);
                break;
            }
            KHR_ICD_TRACE("Value %s found...\n", cszLibraryName);

            // Require that the value be a DWORD and equal zero
            if (REG_DWORD != dwLibraryNameType)
            {
                KHR_ICD_TRACE("Value not a DWORD, skipping\n");
                continue;
            }
            if (dwValue)
            {
                KHR_ICD_TRACE("Value not zero, skipping\n");
                continue;
            }
            // add the library
            status |= adapterAdd(cszLibraryName, ZeroLuid);
        }
    }

    // Add adapters according to DXGI's preference order
    HMODULE hDXGI = LoadLibraryA("dxgi.dll");
    if (hDXGI)
    {
        IDXGIFactory* pFactory = NULL;
        PFN_CREATE_DXGI_FACTORY pCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY)(void*)GetProcAddress(hDXGI, "CreateDXGIFactory");
        if (pCreateDXGIFactory)
        {
            HRESULT hr = pCreateDXGIFactory(&IID_IDXGIFactory, (void **)&pFactory);
            if (SUCCEEDED(hr))
            {
                UINT i = 0;
                IDXGIAdapter* pAdapter = NULL;
                while (SUCCEEDED(pFactory->lpVtbl->EnumAdapters(pFactory, i++, &pAdapter)))
                {
                    DXGI_ADAPTER_DESC AdapterDesc;
                    if (SUCCEEDED(pAdapter->lpVtbl->GetDesc(pAdapter, &AdapterDesc)))
                    {
                        for (WinAdapter* iterAdapter = pWinAdapterBegin; iterAdapter != pWinAdapterEnd; ++iterAdapter)
                        {
                            if (iterAdapter->luid.LowPart == AdapterDesc.AdapterLuid.LowPart
                                && iterAdapter->luid.HighPart == AdapterDesc.AdapterLuid.HighPart)
                            {
                                khrIcdVendorAdd(iterAdapter->szName);
                                break;
                            }
                        }
                    }

                    pAdapter->lpVtbl->Release(pAdapter);
                }
                pFactory->lpVtbl->Release(pFactory);
            }
        }
        FreeLibrary(hDXGI);
    }

    // Go through the list again, putting any remaining adapters at the end of the list in an undefined order
    for (WinAdapter* iterAdapter = pWinAdapterBegin; iterAdapter != pWinAdapterEnd; ++iterAdapter)
    {
        khrIcdVendorAdd(iterAdapter->szName);
        adapterFree(iterAdapter);
    }

    free(pWinAdapterBegin);
    pWinAdapterBegin = NULL;
    pWinAdapterEnd = NULL;
    pWinAdapterCapacity = NULL;

    result = RegCloseKey(platformsKey);
    if (ERROR_SUCCESS != result)
    {
        KHR_ICD_TRACE("Failed to close platforms key %s, ignoring\n", platformsName);
    }

#if defined(CL_ENABLE_LAYERS)
    const char* layersName = "SOFTWARE\\Khronos\\OpenCL\\Layers";
    HKEY layersKey = NULL;

    KHR_ICD_TRACE("Opening key HKLM\\%s...\n", layersName);
    result = RegOpenKeyExA(
        HKEY_LOCAL_MACHINE,
        layersName,
        0,
        KEY_READ,
        &layersKey);
    if (ERROR_SUCCESS != result)
    {
        KHR_ICD_TRACE("Failed to open layers key %s, continuing\n", layersName);
    }
    else
    {
        // for each value
        for (dwIndex = 0;; ++dwIndex)
        {
            char cszLibraryName[1024] = {0};
            DWORD dwLibraryNameSize = sizeof(cszLibraryName);
            DWORD dwLibraryNameType = 0;
            DWORD dwValue = 0;
            DWORD dwValueSize = sizeof(dwValue);

            // read the value name
            KHR_ICD_TRACE("Reading value %"PRIuDW"...\n", dwIndex);
            result = RegEnumValueA(
                  layersKey,
                  dwIndex,
                  cszLibraryName,
                  &dwLibraryNameSize,
                  NULL,
                  &dwLibraryNameType,
                  (LPBYTE)&dwValue,
                  &dwValueSize);
            // if RegEnumKeyEx fails, we are done with the enumeration
            if (ERROR_SUCCESS != result)
            {
                KHR_ICD_TRACE("Failed to read value %"PRIuDW", done reading key.\n", dwIndex);
                break;
            }
            KHR_ICD_TRACE("Value %s found...\n", cszLibraryName);

            // Require that the value be a DWORD
            if (REG_DWORD != dwLibraryNameType)
            {
                KHR_ICD_TRACE("Value not a DWORD, skipping\n");
                continue;
            }
            // add the library
            status |= layerAdd(cszLibraryName, dwValue);
        }
        qsort(pWinLayerBegin, pWinLayerEnd - pWinLayerBegin, sizeof(WinLayer), compareLayer);
        for (WinLayer* iterLayer = pWinLayerBegin; iterLayer != pWinLayerEnd; ++iterLayer)
        {
            khrIcdLayerAdd(iterLayer->szName);
            layerFree(iterLayer);
        }
    }

    free(pWinLayerBegin);
    pWinLayerBegin = NULL;
    pWinLayerEnd = NULL;
    pWinLayerCapacity = NULL;

    result = RegCloseKey(layersKey);

    khrIcdLayersEnumerateEnv();
#endif // defined(CL_ENABLE_LAYERS)
    return status;
}

// go through the list of vendors only once
void khrIcdOsVendorsEnumerateOnce()
{
    InitOnceExecuteOnce(&initialized, khrIcdOsVendorsEnumerate, NULL, NULL);
}

/*
 *
 * Dynamic library loading functions
 *
 */

// dynamically load a library.  returns NULL on failure
void *khrIcdOsLibraryLoad(const char *libraryName)
{
    HMODULE hTemp = LoadLibraryExA(libraryName, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!hTemp && GetLastError() == ERROR_INVALID_PARAMETER)
    {
        hTemp = LoadLibraryExA(libraryName, NULL, 0);
    }
    if (!hTemp)
    {
        KHR_ICD_TRACE("Failed to load driver. Windows error code is %"PRIuDW".\n", GetLastError());
    }
    return (void*)hTemp;
}

// get a function pointer from a loaded library.  returns NULL on failure.
void *khrIcdOsLibraryGetFunctionAddress(void *library, const char *functionName)
{
    if (!library || !functionName)
    {
        return NULL;
    }
    return GetProcAddress( (HMODULE)library, functionName);
}

// unload a library.
void khrIcdOsLibraryUnload(void *library)
{
    FreeLibrary( (HMODULE)library);
}
