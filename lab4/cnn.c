#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "cnn.h"
#include <CL/cl.h>
#include "kernel_cl.h"

#define N_THREADS 512
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_SM 16
#define MAX_WARPS_PER_SM 64
#define THREADS_PER_WARP 32

#define GLOBAL_WORK_ITEMS 32


// Sequential CNN implementation
void CONV(float Cout[NUM][IMROW][IMROW], float Cin[NUM][INIMROW][INIMROW],
          float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{
	for(int i=0; i<NUM; i++) {
		for(int h=0; h<IMROW; h++) {
			for(int w=0; w<IMROW; w++)
				Cout[i][h][w] = bias[i];
		}
	}
	for(int i=0; i<NUM; i++) {
		for(int j=0; j<NUM; j++) {
			for(int h=0; h<IMROW; h++) {
				for(int w=0; w<IMROW; w++) {
					for(int p=0; p<KERNEL; p++) {
						for(int q=0; q<KERNEL; q++)
							Cout[i][h][w] += weight[i][j][p][q]*Cin[j][1*h+p][1*w+q];
					}
				}
			}
		}
	}
}


int main()
{
	static float Cout[NUM][IMROW][IMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

	// Use this to check the output of each API call
    cl_int status;

    // For profiling
    cl_event event;
    cl_ulong time_start, time_end;
    double total_time;

    // 1) GET PLATFORMS
    printf("1) Get Platforms \n");
    // Call 1: Retrieve the number of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    // Allocate enough space for numPlatforms platforms
    cl_platform_id *platforms = NULL;
    platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    // Call 2: Get the actual Platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);


    // 2) GET PLATFORM INFO
    printf("2) Get Platform Info\n");
	int platform_index = -1;
	int i;
	for (i = 0; i < numPlatforms; i++)
	{
		char vendor[128];
		clGetPlatformInfo (platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
		char vendorF[7];
		memcpy((void*)vendorF, (void*)vendor, 6);
		vendorF[7] = '\0';
		//Print the patform vender
		fprintf(stderr, "%s\n", vendorF);
		//If it is an Intel platform set this to be the patform indez
		if (strcmp(vendorF, "NVIDIA") == 0)
		{
			platform_index = i;
			break;
		}
	}
	//If you dont find an NVIDIA platform exit with an error
	if (platform_index == -1){
		printf("Didn't find GPU platform!\n");
		exit(1);
	}


	// 3) GET DEVICES
    printf("3) Get Devices\n");
	// Call 1: Retrieve the number of devices
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	printf("#devices: %d, status %d\n", numDevices, status);
    // Allocate enough space for numDevices devices
    cl_device_id *devices;
    devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
    // Call 2: Get the actual devices
	status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL,        
	        numDevices, devices, NULL);
    if (status < 0) printf("ERROR: clGetDeviceIDs: %d\n", status);
    else printf("Success: clGetDeviceIDs\n");


	// 4) CREATE CONTEXT, and associate it with the devices
    printf("4) Create Context\n");
    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    if (status < 0) printf("ERROR: clCreateContext: %d\n", status);
    else printf("Success: clCreateContext\n");


    // 5) CREATE COMMAND QUEUE, and associate it with the first device 
    printf("5) Create Command Queue\n");
    cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    if (status < 0) printf("ERROR: clCreateCommandQueue: %d\n", status);
    else printf("Success: clCreateContext:\n");


    // 6) CREATE BUFFERS
    printf("6) Create Buffers\n");
    // Call 1: create buffer for Cout
    cl_mem bufCout;
    size_t cout_size = NUM * IMROW * IMROW * sizeof(float);
    bufCout = clCreateBuffer(context, CL_MEM_READ_ONLY, cout_size, NULL, &status);
    if (status < 0) printf("ERROR: clCreateBuffer (Cout): %d\n", status);
    else printf("Success: clCreateBuffer (Cout)\n");

    // Call 2: create buffer for Cin
    cl_mem bufCin;
    size_t cin_size = NUM * INIMROW * INIMROW * sizeof(float);
    bufCin = clCreateBuffer(context, CL_MEM_READ_ONLY, cin_size, NULL, &status);
    if (status < 0) printf("ERROR: clCreateBuffer (Cin): %d\n", status);
    else printf("Success: clCreateBuffer (Cin)\n");

	// Call 3: create buffer for weight
    cl_mem bufWeight;
    size_t weight_size = NUM * NUM * KERNEL * KERNEL * sizeof(float);
    bufWeight = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_size, NULL, &status);
    if (status < 0) printf("ERROR: clCreateBuffer (weight): %d\n", status);
    else printf("Success: clCreateBuffer (weight)\n");

	// Call 4: create buffer for bias
    cl_mem bufBias;
    size_t bias_size = NUM * sizeof(float);
    bufBias = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size, NULL, &status);
    if (status < 0) printf("ERROR: clCreateBuffer (bias): %d\n", status);
    else printf("Success: clCreateBuffer (bias)\n");


    // 7) WRITE BUFFERS
    printf("7) Write Buffers\n");
    // Call 1: write Cin to bufCin
    status = clEnqueueWriteBuffer(cmdQueue, bufCin, CL_FALSE, 
        0, cin_size, Cin, 0, NULL, NULL);
    if (status < 0) printf("ERROR: clEnqueueWriteBuffer (Cin): %d\n", status);
    else printf("Success: clEnqueueWriteBuffer (Cin)\n");

    // Call 2: write weight to bufWeight
    status = clEnqueueWriteBuffer(cmdQueue, bufWeight, CL_FALSE, 
        0, weight_size, weight, 0, NULL, NULL);
    if (status < 0) printf("ERROR: clEnqueueWriteBuffer (weight): %d\n", status);
    else printf("Success: clEnqueueWriteBuffer (weight)\n");

    // Call 3: write bias to bufBias
    status = clEnqueueWriteBuffer(cmdQueue, bufBias, CL_FALSE, 
        0, bias_size, bias, 0, NULL, NULL);
    if (status < 0) printf("ERROR: clEnqueueWriteBuffer (bias): %d\n", status);
    else printf("Success: clEnqueueWriteBuffer (bias)\n");


    // 8) CREATE KERNEL
    printf("8) Create Kernel\n");
    // Create a program from kernel_cl.h source code
    cl_program program = clCreateProgramWithSource(context, 1, 
        (const char**)&kernel_cl, NULL, &status);
    if (status < 0) printf("ERROR: clCreateProgramWithSource: %d\n", status);
    else printf("Success: clCreateProgramWithSource\n");

    // Build (compile) the program for the device
    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    if (status < 0) printf("ERROR: clBuildProgram: %d\n", status);
    else printf("Success: clBuildProgram\n");

    // Create a kernel from the CONV function in the program (kernel_cl.h)
    cl_kernel kernel;
    kernel = clCreateKernel(program, "CONV", &status);
    if (status < 0) printf("ERROR: clCreateKernel: %d\n", status);
    else printf("Success clCreateKernel\n");

    // Associate the input and output buffers with the kernel 
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufCout);
    if (status < 0) printf("ERROR: clSetKernelArg (Cout): %d\n", status);
    else printf("Success clSetKernelArg (Cout)\n");

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufCin);
    if (status < 0) printf("ERROR: clSetKernelArg (Cin): %d\n", status);
    else printf("Success clSetKernelArg (Cin)\n");

    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufWeight);
    if (status < 0) printf("ERROR: clSetKernelArg (weight): %d\n", status);
    else printf("Success clSetKernelArg (weight)\n");

    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufBias);
    if (status < 0) printf("ERROR: clSetKernelArg (bias): %d\n", status);
    else printf("Success clSetKernelArg (bias)\n");

    // size_t clocal_size = (N_THREADS / THREADS_PER_WARP) * IMROW * IMROW * sizeof(float);
    // status = clSetKernelArg(kernel, 4, clocal_size, NULL);
    // if (status < 0) printf("ERROR: clSetKernelArg (Clocal): %d\n", status);
    // else printf("Success clSetKernelArg (Clocal)\n");



    // 9) EXECUTE KERNEL
    printf("9) Execute Kernel\n");
    // Define an index space (global work size) of work items for execution. 
    size_t globalWorkSize[1];   
    // total number of work items
    globalWorkSize[0] = N_THREADS;

    size_t localWorkSize[1];
    localWorkSize[0] = THREADS_PER_WARP;
    //Make sure all queued events are finished
    clFinish(cmdQueue);
    
    // Execute the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
    if (status < 0) printf("ERROR: clEnqueueNDRangeKernel: %d\n", status);
    else printf("Success: clEnqueueNDRangeKernel\n");
    //Make sure kernel execution has finished
    clWaitForEvents(1, &event);

    //Kernel Profiling
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("clEnqueueNDRangeKernel = %0.3f (sec)\n", (total_time / 1000000000.0));


    // 10) READ BUFFER
    //Make sure all queued events are finished
    clFinish(cmdQueue);
    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufCout, CL_TRUE, 0, cout_size, Cout, 0, NULL, &event);
    if (status < 0) printf("ERROR: clEnqueueReadBuffer (Cout): %d\n", status);
    else printf("Success: clEnqueueReadBuffer\n");
    //Make sure kernel execution has finished
    clWaitForEvents(1, &event);

    //Read Profiling
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = time_end - time_start;
    printf("clEnqueueReadBuffer = %0.3f (sec)\n", (total_time / 1000000000.0));


    // 11) VERIFY RESULT
    int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

    // 12) FREE OpenCL RESOURCES
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufCout);
    clReleaseMemObject(bufCin);
    clReleaseMemObject(bufWeight);
    clReleaseMemObject(bufBias);
    clReleaseContext(context);

	return 0;
}

