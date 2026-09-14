/*
 * Copyright (c) 2022 The Khronos Group Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl_layer.h>
#if defined(_WIN32)
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif
#include <fcntl.h>

int stdout_bak, stderr_bak;

// Temporarily deactivate stdout:
// https://stackoverflow.com/a/4832902

#if defined(_WIN32)
#define SECURE 1
#define OPEN _open
#define OPEN_FLAGS _O_WRONLY
#define CLOSE _close
#define DUP _dup
#define DUP2 _dup2
#define NULL_STREAM "nul"
#else
#define OPEN open
#define OPEN_FLAGS O_WRONLY
#define CLOSE close
#define DUP dup
#define DUP2 dup2
#define NULL_STREAM "/dev/null"
#endif

static inline int
silence_stream(FILE *file, int fd)
{
    int new_fd, fd_bak;
    fflush(file);
    fd_bak = DUP(fd);
#if defined(_WIN32) && SECURE
    _sopen_s(&new_fd, NULL_STREAM, OPEN_FLAGS, _SH_DENYNO, _S_IWRITE);
#else
    new_fd = OPEN(NULL_STREAM, OPEN_FLAGS);
#endif
    DUP2(new_fd, fd);
    CLOSE(new_fd);
    return fd_bak;
}

static void silence_layers(void)
{
    stdout_bak = silence_stream(stdout, 1);
    stderr_bak = silence_stream(stderr, 2);
}

static inline void
restore_stream(FILE *file, int fd, int fd_bak)
{
    fflush(file);
    DUP2(fd_bak, fd);
    CLOSE(fd_bak);
}

static void restore_outputs(void)
{
    restore_stream(stdout, 1, stdout_bak);
    restore_stream(stderr, 2, stderr_bak);
}

void printLayerInfo(const struct KHRLayer *layer)
{
    cl_layer_api_version api_version = 0;
    pfn_clGetLayerInfo p_clGetLayerInfo = (pfn_clGetLayerInfo)(size_t)layer->p_clGetLayerInfo;
    cl_int result = CL_SUCCESS;
    size_t sz;

    printf("%s:\n", layer->libraryName);
    result = p_clGetLayerInfo(CL_LAYER_API_VERSION, sizeof(api_version), &api_version, NULL);
    if (CL_SUCCESS == result)
        printf("\tCL_LAYER_API_VERSION: %d\n", (int)api_version);

    result = p_clGetLayerInfo(CL_LAYER_NAME, 0, NULL, &sz);
    if (CL_SUCCESS == result)
    {
        char *name = (char *)malloc(sz);
        if (name)
        {
            result = p_clGetLayerInfo(CL_LAYER_NAME, sz, name, NULL);
            if (CL_SUCCESS == result)
                 printf("\tCL_LAYER_NAME: %s\n", name);
            free(name);
        }
    }
}

int main (int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    silence_layers();
    atexit(restore_outputs);
    khrIcdInitialize();
    restore_outputs();
    atexit(silence_layers);
    const struct KHRLayer *layer = khrFirstLayer;
    while (layer)
    {
        printLayerInfo(layer);
        layer = layer->next;
    }
    return 0;
}
