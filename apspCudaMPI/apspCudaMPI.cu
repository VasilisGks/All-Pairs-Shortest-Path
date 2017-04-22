
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#define BLOCKSIZE 16
#define CELLS_PER_THREAD 16
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "header1.h" 
//#define INFINITY 1500000000000
/* ********************************************************************************/
/* ShortestPath1 function implements Cuda Kernel for finding the shortest path WITHOUT using Shared memory */
/*					Assigns one element of 2d array to each thread						    														 */
/* ***************************************************************************************************************/


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void print(float *A, char Name, int N, int N0);

__global__ void ShortestPath2(float *Arr1,float *Arr2,float *recv,int N,int rows, int k,int rank,int owner){
											
	int col=blockIdx.x * blockDim.x + threadIdx.x;
	int row=blockIdx.y * blockDim.y + threadIdx.y;

	int index=row*N+col;
	int index_ik = row*N+k;


	if(Arr1[index]>(Arr1[index_ik]+recv[col])){
		Arr2[index]=Arr1[index_ik]+recv[col];	
	}
	__syncthreads();

}



__global__ void ShortestPath1(float *Arr1,float *Arr2,int N,int rows, int rank){     //rowNum is number of rows for each process (full assigned to process) 
                                                                            //Arr1 input array,Holds  of (u,v)	//Arr2 output array	
	int k;
	int col=blockIdx.x * blockDim.x + threadIdx.x;
	int row=blockIdx.y * blockDim.y + threadIdx.y;
	
	int offset=rows*rank;
	
	int index=row*N+col;
	int index_ik,index_kj;
	
	Arr2[index]=Arr1[index];              
	for(k=rank*rows; k<((rank+1)*rows); k++){
			
		index_ik = row*N+k;
		index_kj = (k-offset)*N+col;
		
		if(Arr1[index]>(Arr1[index_ik]+Arr1[index_kj])){
			Arr2[index]=Arr1[index_ik]+Arr1[index_kj];	
		}
      __syncthreads();
	}
}

extern "C" float *shortest(float *Ar1,int n,int numtasks, int rank){
      
    cudaSetDevice(rank);
    
    if (rank == 0){
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
	     prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
	     prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
	     2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
  } 
     
    
    float *d_Ar1, *d_Ar2, *d_Ar6 ,*d_recv;
    int N=(n*n/numtasks),i;

    int size0=N*sizeof(float),k;
	int recv_size=n*sizeof(float);
    struct timeval start0,end0; //,start1,end1,start2,end2;
    
    float *recv=(float*)malloc(n*sizeof(float));
    float *Ar2=(float*)malloc(N*sizeof(float));
    float *Ar6=(float*)malloc(N*sizeof(float));
    float *result1=(float*)malloc(N*sizeof(float));
    
    for(i=0;i<N;i++){
		Ar2[i]=0;Ar6[i]=0;
    }
    
    cudaMalloc((void**)&d_Ar1,size0);				//Dynamic allocation in cuda Kernel within the Host code
    cudaMalloc((void**)&d_Ar2,size0);
    cudaMalloc((void**)&d_Ar6,size0);
	cudaMalloc((void**)&d_recv,recv_size);
    
    gpuErrchk ( cudaMemcpy(d_Ar2,Ar2,size0,cudaMemcpyHostToDevice) ); 
    
    int rows=n/numtasks;  //Each task's rows = 
    
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);									//Block and thread size for 1st and 2nd question of project
	dim3 dimGrid((int)ceil(n/BLOCKSIZE),(int)ceil(rows/BLOCKSIZE));
	if (rank==0)
	{
		printf("\nBlocksize=%dx%d and Grid=%dx%d\n\n",BLOCKSIZE,BLOCKSIZE,(rows/BLOCKSIZE),(n/BLOCKSIZE));
	}
    
    gettimeofday(&start0, NULL);         						//Start timer
    gpuErrchk( cudaMemcpy(d_Ar1,Ar1,size0,cudaMemcpyHostToDevice) );   
   
    ShortestPath1<<<dimGrid,dimBlock>>>(d_Ar1,d_Ar2,n,rows,rank);           //Calling 1st Cuda kernel ( No use of shared memory)
 
    gpuErrchk( cudaMemcpy(Ar6,d_Ar2,size0,cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(d_Ar6,Ar6,size0,cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_Ar1,Ar1,size0,cudaMemcpyHostToDevice) );
	int owner;
	k=0;
	for (int i=0; i<numtasks; i++){
		for (int j=0; j<rows; j++){
			
			owner=i;
			recv=SendtoProc(Ar1,n,k,owner);
				
			/*Copy received row*/
			gpuErrchk( cudaMemcpy(d_recv ,recv ,recv_size ,cudaMemcpyHostToDevice) );
			/*Run Kernel*/
			ShortestPath2<<<dimGrid,dimBlock>>>(d_Ar1,d_Ar6,d_recv,n,rows,k,rank,owner);		
			k++;			
		}
		 
	} 

      gettimeofday(&end0, NULL);
    //Printing results for parallel version after computations complete
    if(rank==0){
    printf("FOR PARALLEL VERSION TIME IS : %ld microSeconds \n", ((end0.tv_sec * 1000000 + end0.tv_usec)
    - (start0.tv_sec * 1000000 + start0.tv_usec)));
    }

	/*Copy Output*/
	gpuErrchk( cudaMemcpy(Ar6,d_Ar6,size0,cudaMemcpyDeviceToHost) );	
	return Ar6;
}
//Debug
void print(float *A, char Name, int N, int N0)
{
	int i,j;
	//printf("\n %c =\n", Name);
	for (i=0; i<N0;i++)
	{
		for (j=0; j<N; ++j)
		{
			if (A[i*N+j] < INFINITY)
			{
				printf("%5.2f ",A[i*N+j]);
			}
			else
			{
				printf(" Inf ");
			} 
			
		}
		printf("\n");
	}
}

