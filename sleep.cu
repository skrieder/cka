#include <stdio.h>

//This file will run nkernel many kernels concurrently and each
//  of them will sleep for Kernel_time ms. This will work correctly
//  until clock() ~ 2.15 billion clicks. On my system, where the GPU
//  run at 1.56 GHz, this happens in 1.37 seconds.
//Therefore this cannot be used to run tests that will call clock()
//  after more than ~1.37 seconds.


// This is a kernel that does no real work but runs at least for a specified number of clocks
__global__ void clock_block(int kernel_time, int clockRate)
{ 
    int temp;
    int finish_clock;
    for(temp=0; temp<kernel_time; temp++){
        int start_time = clock();
        finish_clock = start_time + clockRate;
        bool wrapped = finish_clock < start_time;
        while( clock() < finish_clock || wrapped) wrapped = clock()>0 && wrapped;
    }
}

int main(int argc, char **argv)
{
    int nkernels = 4;              // number of concurrent kernels
    int nstreams = nkernels + 1;   // use one more stream than concurrent kernel
    int kernel_time = 2500;        // time the kernel should run in ms
    int cuda_device = 0;

    //nkernels = atoi(argv[1]);       //could be used to pass in parameters
    //kernel_time = atoi(argv[2]);



    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);	

    cudaGetDeviceProperties(&deviceProp, cuda_device);

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    for(int i = 1; i < nstreams; i++)
        cudaStreamCreate(&(streams[i]));

    //////////////////////////////////////////////////////////////////////

    int clockRate = deviceProp.clockRate; 

    //I am starting this at i=1 because the default stream is 0.
    for( int i=1; i<nkernels+1; ++i)
    {
        printf("starting kernel:  %d\n", i);
        clock_block<<<1,1,1,streams[i]>>>(kernel_time, clockRate);
    }

    //Find an errors that the gpu kernels had
    cudaError cuda_error = cudaDeviceSynchronize();

    if(cuda_error==cudaSuccess){
        printf( "  Running the concurrentKernels was a success\n");
    }else{
        if(cuda_error==cudaErrorLaunchTimeout ){
            printf( "  A thread was stopped for reaching time limit\n" );
        }else{
            printf( "  An error happened while running the wait\n" );
        }
    }

    // release resources
    for(int i = 1; i < nstreams; i++)
        cudaStreamDestroy(streams[i]); 
 
    free(streams);
  return 0;    
}
