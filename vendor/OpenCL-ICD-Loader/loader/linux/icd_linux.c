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
#include "icd_envvars.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <pthread.h>
#include <limits.h>

static pthread_once_t initialized = PTHREAD_ONCE_INIT;

/*
 *
 * Vendor enumeration functions
 *
 */

typedef void khrIcdFileAdd(const char *);

static inline void khrIcdOsDirEntryValidateAndAdd(const char *d_name, const char *path,
                                                  const char *extension, khrIcdFileAdd addFunc)
{
    struct stat statBuff;
    char* fileName = NULL;

    // make sure the file name ends in `extension` (eg. .icd, or .lay)
    if (strlen(extension) > strlen(d_name))
    {
        return;
    }
    if (strcmp(d_name + strlen(d_name) - strlen(extension), extension))
    {
        return;
    }

    // allocate space for the full path of the vendor library name
    fileName = malloc(strlen(d_name) + strlen(path) + 2);
    if (!fileName)
    {
        KHR_ICD_TRACE("Failed allocate space for ICD file path\n");
        return;
    }
    sprintf(fileName, "%s/%s", path, d_name);

    if (stat(fileName, &statBuff))
    {
        KHR_ICD_TRACE("Failed stat for: %s, continuing\n", fileName);
        free(fileName);
        return;
    }

    if (S_ISREG(statBuff.st_mode) || S_ISLNK(statBuff.st_mode))
    {
        FILE *fin = NULL;
        char* buffer = NULL;
        long bufferSize = 0;

        // open the file and read its contents
        fin = fopen(fileName, "r");
        if (!fin)
        {
            free(fileName);
            return;
        }
        fseek(fin, 0, SEEK_END);
        bufferSize = ftell(fin);

        buffer = malloc(bufferSize+1);
        if (!buffer)
        {
            free(fileName);
            fclose(fin);
            return;
        }
        memset(buffer, 0, bufferSize+1);
        fseek(fin, 0, SEEK_SET);
        if (bufferSize != (long)fread(buffer, 1, bufferSize, fin))
        {
            free(fileName);
            free(buffer);
            fclose(fin);
            return;
        }
        // ignore a newline at the end of the file
        if (buffer[bufferSize-1] == '\n') buffer[bufferSize-1] = '\0';

        // load the string read from the file
        addFunc(buffer);

        free(fileName);
        free(buffer);
        fclose(fin);
     }
     else
     {
         KHR_ICD_TRACE("File %s is not a regular file nor symbolic link, continuing\n", fileName);
         free(fileName);
     }
}

struct dirElem
{
    char *d_name;
    unsigned char d_type;
};

static int compareDirElem(const void *a, const void *b)
{
    // sort files the same way libc alpahnumerically sorts directory entries.
    return strcoll(((const struct dirElem *)a)->d_name, ((const struct dirElem *)b)->d_name);
}

static inline void khrIcdOsDirEnumerate(const char *path, const char *env,
                                        const char *extension,
                                        khrIcdFileAdd addFunc, int bSort)
{
    DIR *dir = NULL;
    char* envPath = NULL;

    envPath = khrIcd_secure_getenv(env);
    if (NULL != envPath)
    {
        path = envPath;
    }

    dir = opendir(path);
    if (NULL == dir) 
    {
        KHR_ICD_TRACE("Failed to open path %s, continuing\n", path);
    }
    else
    {
        struct dirent *dirEntry = NULL;

        // attempt to load all files in the directory
        if (bSort) {
            // store the entries name and type in a buffer for sorting
            size_t sz = 0;
            size_t elemCount = 0;
            size_t elemAlloc = 0;
            struct dirElem *dirElems = NULL;
            struct dirElem *newDirElems = NULL;
            const size_t startupAlloc = 8;

            // start with a small buffer
            dirElems = (struct dirElem *)malloc(startupAlloc*sizeof(struct dirElem));
            if (NULL != dirElems) {
                elemAlloc = startupAlloc;
                for (dirEntry = readdir(dir); dirEntry; dirEntry = readdir(dir) ) {
                    char *nameCopy = NULL;

                    if (elemCount + 1 > elemAlloc) {
                        // double buffer size if necessary and possible
                        if (elemAlloc >= UINT_MAX/2)
                            break;
                        newDirElems = (struct dirElem *)realloc(dirElems, elemAlloc*2*sizeof(struct dirElem));
                        if (NULL == newDirElems)
                            break;
                        dirElems = newDirElems;
                        elemAlloc *= 2;
                    }
                    sz = strlen(dirEntry->d_name) + 1;
                    nameCopy = (char *)malloc(sz);
                    if (NULL == nameCopy)
                         break;
                    memcpy(nameCopy, dirEntry->d_name, sz);
                    dirElems[elemCount].d_name = nameCopy;
#if defined(__QNXNTO__)
                    dirElems[elemCount].d_type = _DEXTRA_FIRST(dirEntry)->d_type;
#else
                    dirElems[elemCount].d_type = dirEntry->d_type;
#endif
                    elemCount++;
                }
                qsort(dirElems, elemCount, sizeof(struct dirElem), compareDirElem);
                for (struct dirElem *elem = dirElems; elem < dirElems + elemCount; ++elem) {
                    khrIcdOsDirEntryValidateAndAdd(elem->d_name, path, extension, addFunc);
                    free(elem->d_name);
                }
                free(dirElems);
            }
        } else
            // use system provided ordering
            for (dirEntry = readdir(dir); dirEntry; dirEntry = readdir(dir) )
                khrIcdOsDirEntryValidateAndAdd(dirEntry->d_name, path, extension, addFunc);

        closedir(dir);
    }

    if (NULL != envPath)
    {
        khrIcd_free_getenv(envPath);
    }
}

// go through the list of vendors in the two configuration files
void khrIcdOsVendorsEnumerate(void)
{
    khrIcdInitializeTrace();
    khrIcdVendorsEnumerateEnv();

    khrIcdOsDirEnumerate(ICD_VENDOR_PATH, "OCL_ICD_VENDORS", ".icd", khrIcdVendorAdd, 0);

#if defined(CL_ENABLE_LAYERS)
    // system layers should be closer to the driver
    khrIcdOsDirEnumerate(LAYER_PATH, "OPENCL_LAYER_PATH", ".lay", khrIcdLayerAdd, 1);

    khrIcdLayersEnumerateEnv();
#endif // defined(CL_ENABLE_LAYERS)
}

// go through the list of vendors only once
void khrIcdOsVendorsEnumerateOnce(void)
{
    pthread_once(&initialized, khrIcdOsVendorsEnumerate);
}

/*
 *
 * Dynamic library loading functions
 *
 */

// dynamically load a library.  returns NULL on failure
void *khrIcdOsLibraryLoad(const char *libraryName)
{
    void* ret = dlopen (libraryName, RTLD_NOW);
    if (NULL == ret)
    {
        KHR_ICD_TRACE("Failed to load driver because %s.\n", dlerror());
    }
    return ret;
}

// get a function pointer from a loaded library.  returns NULL on failure.
void *khrIcdOsLibraryGetFunctionAddress(void *library, const char *functionName)
{
    return dlsym(library, functionName);
}

// unload a library
void khrIcdOsLibraryUnload(void *library)
{
    dlclose(library);
}
