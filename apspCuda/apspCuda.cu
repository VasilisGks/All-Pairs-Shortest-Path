#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#define BLOCKSIZE 4
#define CELLS_PER_THREAD 4     // Stride length
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);	
   }
}

float *fmDist(float *Ar,int n){
	
	float *dist=(float *)malloc(n*n*sizeof(float));
	int i,j,k;
	
	for(i=0;i<n;i++){        //Initialize
		for(j=0;j<n;j++){
			dist[i*n+j]=Ar[i*n+j];
		}
	}
	
	for(k=0;k<n;k++){
		for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				if(Ar[i*n+j]>Ar[i*n+k]+Ar[k*n+j])
					dist[i*n+j]=Ar[i*n+k]+Ar[k*n+j];
			}
		}
	}
	return dist;
}

/* This Function constructs array with random values.Initializes Shortest paths   */
/* ********************************************************************************/
void *makeAdjacency(int n,float p,int w,float *A){	
	int i,j;float r;
//	float *A=malloc((n*n)*sizeof(float));
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
				r=(float)rand()/RAND_MAX;
				if(r>p){
		//			printf("THIS IS %F r \n",r);
					A[i*n+j]=INFINITY;
				}
				else{
					A[i*n+j]=r*w;
				}
		}
		A[i*n+j]=0;
	}
	return A;
}
/* ********************************************************************************/
/* ShortestPath3 function implements Cuda Kernel for finding the shortest path using Shared memory ,and  assigns N/8 threads in each Row of array*/
/*											   Using grid stride loop      */
/* ********************************************************************************/
__global__ void ShortestPath3(float *Arr1,float *Arr2,int N){//Arr1 input array,Holds weights
																//Arr2 output array
	unsigned int k;	
	int row=blockIdx.x;
	int col=threadIdx.x;
	
	if(row >= N || col >= N) return;
	
	int index=row*N+col;					//Index for Elements of global memory array
	extern __shared__ float sArr[];			//Dynamic share memory allocation in Array
	int stride=N/CELLS_PER_THREAD;			// Stride for each index of arrays (both global and shared)
	
	for(stride=0;stride<N;stride+=N/CELLS_PER_THREAD){
		if(threadIdx.x+stride>N) break;						//Copy elements of same ROW in shared memory (SHARED memory indexes = ThreadIdx.x + stride)
		sArr[threadIdx.x+stride]=Arr1[index+stride];		//"jump" to the next index according to stride .Write to SM
	}
	__syncthreads();										//Wait all threads in current block

	
	for(stride=0;stride<N;stride+=N/CELLS_PER_THREAD){
		
		for(k=0;k<N;k++){
			if(k==threadIdx.x+stride)continue;	
			if(sArr[threadIdx.x+stride]>(sArr[k]+Arr1[k*N+col+stride])){		//Return
				Arr2[index+stride]=(sArr[k]+Arr1[k*N+col+stride]);		//If Condition true,write to shared memory
		//	
			}
		}
	//						//Copy results to global memory and return to HOST
			
	}
}
/* ********************************************************************************/
/* ShortestPath2 function implements Cuda Kernel for finding the shortest path using Shared memory  and assigns one thread to each array's cell*/
/*																									   */
/* *************************************************************************************************************************/
__global__ void ShortestPath2(float *Arr1,float *Arr2,int N){   //Arr1 input array,Holds weights
																//Arr2 output array		
	unsigned int k;

	int row=blockIdx.x;
	int col=threadIdx.x;
	if(row >= N || col >= N) return;
	
	int index=row*N+col;						//Index for Elements of global memory array
	extern __shared__ float sArr[];				//Dynamic share memory allocation in Array
	Arr2[index]=Arr1[index];
	sArr[threadIdx.x]=Arr1[index];				//Copy elements of same ROW in shared memory (SHARED memory indexes = ThreadIdx.x )
	__syncthreads();							//Wait threads in block to finish
	
	for(k=0;k<N;k++){
		if(k==threadIdx.x) continue;						//If out of bounds , just current loop iteration
		if(sArr[threadIdx.x]>(sArr[k]+Arr1[k*N+col])){		//If condition true write in shared memory new value
			Arr2[index]=(sArr[k]+Arr1[k*N+col]);			
		}
                 	
	}
}
/* ********************************************************************************/
/* ShortestPath1 function implements Cuda Kernel for finding the shortest path WITHOUT using Shared memory */
/*					Assigns one element of 2d array to each thread						    														 */
/* ***************************************************************************************************************/
__global__ void ShortestPath1(float *Arr1,float *Arr2,int N){
															   //Arr1 input array,Holds  of (u,v)
																//Arr2 output array	
	int k;
	int col=blockIdx.x * blockDim.x + threadIdx.x;
	int row=blockIdx.y * blockDim.y + threadIdx.y;
	int index=row*N+col;
	
	if((row<N)&&(col<N)){	
		Arr2[index]=Arr1[index];              
			
		for(k=0;k<N;k++){
			
			if(Arr1[index]>(Arr1[row*N+k]+Arr1[N*k+col])){
				Arr2[index]=Arr1[row*N+k]+Arr1[N*k+col];	
				//	printf("ENTERED %f \n",Arr2[index]);
			}
		}
	}
	
}

/* ***********************STARTING MAIN PROGRAM HERE**************************************************/

int main(int argc,char *argv[]){
	srand (time(NULL));
	
	struct timeval start0,end0,start1,end1,start2,end2;
	
	int n,i;	
	float *d_Ar1, *d_Ar2, *d_Ar3, *d_Ar4, *d_Ar5;
	float *Ar1, *Ar2,*Ar3,*Ar4,*Ar5,*Ar6;
	if(argc!=2){
		printf("ERROR ARGUEMENTS MUST BE POWER OF TWO\n"), 
		exit(1);
	}
	n = 1<<atoi(argv[1]);	
	int N=n*n;
	int size0=N*sizeof(float);

	Ar1=(float*)malloc(N*sizeof(float));
	Ar2=(float*)malloc(N*sizeof(float));      //Dynamic allocation memory for HOST's arrays
	Ar3=(float*)malloc(N*sizeof(float));
	Ar4=(float*)malloc(N*sizeof(float));
	Ar5=(float*)malloc(N*sizeof(float));
	Ar6=(float*)malloc(N*sizeof(float));
	
	for(i=0;i<N;i++){
		Ar1[i]=0;Ar2[i]=0;Ar3[i]=0;Ar4[i]=0;Ar5[i]=0;Ar6[i]=0;
	}
	
	cudaMalloc((void**)&d_Ar1,size0);
	cudaMalloc((void**)&d_Ar2,size0);				//Dynamic allocation in cuda Kernel within the Host code
	cudaMalloc((void**)&d_Ar3,size0);
	cudaMalloc((void**)&d_Ar4,size0);
	cudaMalloc((void**)&d_Ar5,size0);
		
	float p=0.33;int w=30;	
	makeAdjacency(n,p,w,Ar1);                     //Create makeadjency array , Holding weights With given w , and p

	/* ************************************************ SERIAL CODE :  ********************************************************************/
	/* ***************************************************************************************************/
	
	gettimeofday(&start1, NULL);			//Start timer
	Ar3=fmDist(Ar1,n);
	gettimeofday(&end1, NULL);
	printf("FOR SERIAL VERSION OF ALGORITHM TIME IS : %ld microSeconds \n", ((end1.tv_sec * 1000000 + end1.tv_usec)
		  - (start1.tv_sec * 1000000 + start1.tv_usec)));
	
	/* ********************FIRST PART BELOW  : PARALLEL WITHOUT USING SHARED MEMORY*******************************************************************************/
	/* ***************************************************************************************************/
	
	cudaMemcpy(d_Ar2,Ar2,size0,cudaMemcpyHostToDevice);
	
	 dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);									//Block and thread size for 1st and 2nd question of project
	 dim3 dimGrid((int)ceil(n/BLOCKSIZE),(int)ceil(n/BLOCKSIZE));
	 
	 //gpuErrchk( cudaPeekAtLastError() );
	 
	 gettimeofday(&start0, NULL);         						//Start timer
	 cudaMemcpy(d_Ar1,Ar1,size0,cudaMemcpyHostToDevice);
	 ShortestPath1<<<dimGrid,dimBlock>>>(d_Ar1,d_Ar2,n);           //Calling 1st Cuda kernel ( No use of shared memory)
	 cudaMemcpy(Ar6,d_Ar2,size0,cudaMemcpyDeviceToHost);
	 gettimeofday(&end0, NULL);
	 
	 printf("FOR FIRST KERNEL, VERSION WITHOUT SHARED MEMORY TIME IS : %ld microSeconds \n", ((end0.tv_sec * 1000000 + end0.tv_usec)
		  - (start0.tv_sec * 1000000 + start0.tv_usec)));
		

		
	int counter02=0;
	for(i=0;i<N;i++){
		float result=abs(Ar6[i]-Ar3[i]);
//printf("THIS IS DIFFERENCE  (WITHOUT SHARED MEM) = %f , SERIAL =%f\n",Ar6[i],Ar3[i]);
		if(abs(result)<0.01)
			counter02+=1;
			
	}

	float percent1=(counter02/(float)N)*100;
	printf("**In %f percent of points**  absolute difference between parallel(WITHOUT Shared Memory use) and serial output is less than 0.01\n",percent1);
		

	/* *******************************Second PART BELOW : PARALLEL WITH USING OF SHARED MEMORY********************************************************************/
	/* ***********************************************************************************************************************************/
									
	gettimeofday(&start2, NULL);												   //Start timer
	cudaMemcpy(d_Ar1,Ar1,size0,cudaMemcpyHostToDevice);
	ShortestPath2<<<n, n, n * sizeof(float)>>>(d_Ar1,d_Ar3,n);                    //Calling 2nd Cuda kernel ( With use of shared memory)
	cudaMemcpy(Ar4,d_Ar3,size0,cudaMemcpyDeviceToHost);
	gettimeofday(&end2, NULL);
	
	printf("FOR SECOND KERNEL ,SHARED MEMORY VERSION TIME IS : %ld microSeconds  \n", ((end2.tv_sec * 1000000 + end2.tv_usec)
		  - (start2.tv_sec * 1000000 + start2.tv_usec)));
		
		int counter2=0;
	for(i=0;i<N;i++){
	float result00=Ar4[i]-Ar3[i];
//		printf("THIS IS DIFFERENCE  PARALLEL(WITH SHARED MEMORY) = %f , SERIAL =%f\n",Ar4[i],Ar3[i]);
		
		if(abs(result00)<0.1)
			counter2+=1;
			
	}
	//	printf("TOTAL FROM %d points ,differ between serial and parallel implementation IN %d\n",N,counter);
		float percent=(counter2/(float)N)*100;
		printf("**In %f percent of points**  absolute difference between parallel(with shared memory use) and serial output is less than 0.01\n",percent);
	
//printf("THIS IS DIF  PARALLEL(WITH SHARED MEMORY) = %f ,SERIAL =%f\n",Ar4[i],Ar3[i]);

	
	
	/* *************THIRD PART START BELOW : PARALLEL IMPLEMENTATION USING SHARED MEMORY .N/8 THREADS IN ROW(OF SIZE N) *************************************/
	/* ***********************************************************************************************************************************/
	struct timeval start3,end3;
	gettimeofday(&start3, NULL);
	cudaMemcpy(d_Ar1,Ar1,size0,cudaMemcpyHostToDevice);
	ShortestPath3<<<n, n/CELLS_PER_THREAD, n * sizeof(float)>>>(d_Ar1,d_Ar4,n);
	cudaMemcpy(Ar5,d_Ar4,size0,cudaMemcpyDeviceToHost);
	 gettimeofday(&end3, NULL);
	
	printf("FOR 3rd KERNEL, VERSION WITH SHARED MEMORY (MULTIPLE POINTS PER THREAD) TIME IS : %ld microSeconds \n", ((end3.tv_sec * 1000000 + end3.tv_usec)
		  - (start3.tv_sec * 1000000 + start3.tv_usec)));
	
	int counter22=0;
	for(i=0;i<N;i++){
		float result01=abs(Ar5[i]-Ar3[i]);
 //printf("THIS IS DIFFERENCE(WITHOUT SHARED MEM and with n/8 threads per block) = %f , SERIAL =%f\n",Ar6[i],Ar3[i]);
		
		if(abs(result01)<0.01)
			counter22+=1;
			
	}
	
//	printf("TOTAL FROM %d points ,PARALLEL BETTER OR SAME WITH SERIAL IN %d\n",N,counter11);
		percent1=(counter22/(float)N)*100;
		printf("**In %f percent of points**  absolute difference between parallel(with shared memory use and n/8 threads per block) and serial output is less than 0.01\n",percent1);
	/* ***************************** THIRD PART END HERE*************************************/
	free(Ar1);free(Ar2);free(Ar3);
	cudaFree(d_Ar1);cudaFree(d_Ar2);
	return EXIT_SUCCESS;
}

