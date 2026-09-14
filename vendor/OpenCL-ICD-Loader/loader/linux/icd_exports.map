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

OPENCL_1.0 {
    global:
        clBuildProgram;
        clCreateBuffer;
        clCreateCommandQueue;
        clCreateContext;
        clCreateContextFromType;
        clCreateFromGLBuffer;
        clCreateFromGLRenderbuffer;
        clCreateFromGLTexture2D;
        clCreateFromGLTexture3D;
        clCreateImage2D;
        clCreateImage3D;
        clCreateKernel;
        clCreateKernelsInProgram;
        clCreateProgramWithBinary;
        clCreateProgramWithSource;
        clCreateSampler;
        clEnqueueAcquireGLObjects;
        clEnqueueBarrier;
        clEnqueueCopyBuffer;
        clEnqueueCopyBufferToImage;
        clEnqueueCopyImage;
        clEnqueueCopyImageToBuffer;
        clEnqueueMapBuffer;
        clEnqueueMapImage;
        clEnqueueMarker;
        clEnqueueNDRangeKernel;
        clEnqueueNativeKernel;
        clEnqueueReadBuffer;
        clEnqueueReadImage;
        clEnqueueReleaseGLObjects;
        clEnqueueTask;
        clEnqueueUnmapMemObject;
        clEnqueueWaitForEvents;
        clEnqueueWriteBuffer;
        clEnqueueWriteImage;
        clFinish;
        clFlush;
        clGetCommandQueueInfo;
        clGetContextInfo;
        clGetDeviceIDs;
        clGetDeviceInfo;
        clGetEventInfo;
        clGetEventProfilingInfo;
        clGetExtensionFunctionAddress;
        clGetGLObjectInfo;
        clGetGLTextureInfo;
        clGetImageInfo;
        clGetKernelInfo;
        clGetKernelWorkGroupInfo;
        clGetMemObjectInfo;
        clGetPlatformIDs;
        clGetPlatformInfo;
        clGetProgramBuildInfo;
        clGetProgramInfo;
        clGetSamplerInfo;
        clGetSupportedImageFormats;
        clReleaseCommandQueue;
        clReleaseContext;
        clReleaseEvent;
        clReleaseKernel;
        clReleaseMemObject;
        clReleaseProgram;
        clReleaseSampler;
        clRetainCommandQueue;
        clRetainContext;
        clRetainEvent;
        clRetainKernel;
        clRetainMemObject;
        clRetainProgram;
        clRetainSampler;
        clSetCommandQueueProperty;
        clSetKernelArg;
        clUnloadCompiler;
        clWaitForEvents;

    local:
        /* Everything else is local to ICD. */
        *;
};

OPENCL_1.1 {
    global:
        clCreateSubBuffer;
        clCreateUserEvent;
        clEnqueueCopyBufferRect;
        clEnqueueReadBufferRect;
        clEnqueueWriteBufferRect;
        clSetEventCallback;
        clSetMemObjectDestructorCallback;
        clSetUserEventStatus;
} OPENCL_1.0;

OPENCL_1.2 {
    global:
        clCompileProgram;
        clCreateFromGLTexture;
        clCreateImage;
        clCreateProgramWithBuiltInKernels;
        clCreateSubDevices;
        clEnqueueBarrierWithWaitList;
        clEnqueueFillBuffer;
        clEnqueueFillImage;
        clEnqueueMarkerWithWaitList;
        clEnqueueMigrateMemObjects;
        clGetExtensionFunctionAddressForPlatform;
        clGetKernelArgInfo;
        clLinkProgram;
        clReleaseDevice;
        clRetainDevice;
        clUnloadPlatformCompiler;
} OPENCL_1.1;

OPENCL_2.0 {
    global:
        clCreateCommandQueueWithProperties;
        clCreatePipe;
        clGetPipeInfo;
        clSVMAlloc;
        clSVMFree;
        clEnqueueSVMFree;
        clEnqueueSVMMemcpy;
        clEnqueueSVMMemFill;
        clEnqueueSVMMap;
        clEnqueueSVMUnmap;
        clCreateSamplerWithProperties;
        clSetKernelArgSVMPointer;
        clSetKernelExecInfo;
} OPENCL_1.2;

OPENCL_2.1 {
    global:
        clCloneKernel;
        clCreateProgramWithIL;
        clEnqueueSVMMigrateMem;
        clGetDeviceAndHostTimer;
        clGetHostTimer;
        clGetKernelSubGroupInfo;
        clSetDefaultDeviceCommandQueue;
} OPENCL_2.0;

OPENCL_2.2 {
    global:
        clSetProgramReleaseCallback;
        clSetProgramSpecializationConstant;
} OPENCL_2.1;

OPENCL_3.0 {
    global:
        clCreateBufferWithProperties;
        clCreateImageWithProperties;
        clSetContextDestructorCallback;
} OPENCL_2.2;
