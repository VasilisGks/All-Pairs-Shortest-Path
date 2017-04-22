//************** ---------------/*AUTHORS GKITSAS VASILEIOS : SOTIRIS DIMOS : GIWRGOS LOOPAS*/******************-----------------
#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "header.h"
#include <string.h>
#define INFINITY 1500000000000

void print(float *A, char Name, int N, int N0);

float *SendtoProc(float *arr,int n,int k,int owner){

  int numTasks,rank,i;
  
   MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
  int n0=n/numTasks;         //Total number of elements for each task 
  int number=n*n/numTasks;
  int offset=rank*n0;
 
  float *data=(float *)malloc(n*sizeof(float));
  
  /*Fill the send buffer*/
  k=k-offset;
  if (rank==owner){
	  for(i=0;i<n;i++)
	  {
		data[i]=arr[k*n+i];  
	  }
  }
   MPI_Bcast(data,n,MPI_FLOAT,owner,MPI_COMM_WORLD);

    return data;
}

/*Serial Code*/
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
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
				r=(float)rand()/RAND_MAX;
				if(r>p){
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


/* ***********************STARTING MAIN PROGRAM HERE**************************************************/

int main(int argc,char *argv[]){
  
  int numTasks,rank;
  MPI_Status status[4];
  MPI_Request request;
  int rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
  printf ("Error starting MPI program. Terminating.\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  }
  
  time_t timee	; 
	int n,i;	  
  if(argc!=2){
		printf("ERROR ARGUMENTS MUST BE POWER OF TWO\n"), 
		exit(1);
	}
	n = 1<<atoi(argv[1]);	
    
  MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	srand ((unsigned) time(&timee)+200*rank);
	
	struct timeval start0,end0,start1,end1,start2,end2;
	
	float *d_Ar1, *d_Ar2, *d_Ar3, *d_Ar4, *d_Ar5;
	float *Ar1, *Ar2,*Ar3,*Ar4,*Ar5,*Ar6, *Ar0;
	
	int N=n*n;
	int N0=N/numTasks;
	int size0=N*sizeof(float);
	int size1=N0*sizeof(float);

    Ar0=(float*)malloc(N0*sizeof(float));
	Ar1=(float*)malloc(N*sizeof(float));
	Ar2=(float*)malloc(N*sizeof(float));      //Dynamic allocation memory for HOST's arrays
	Ar3=(float*)malloc(N*sizeof(float));
	Ar4=(float*)malloc(N*sizeof(float));
	Ar5=(float*)malloc(N*sizeof(float));
	Ar6=(float*)malloc(N*sizeof(float));
	
	for(i=0;i<N;i++){
		Ar1[i]=0;Ar2[i]=0;Ar3[i]=0;Ar4[i]=0;Ar5[i]=0;Ar6[i]=0;
	}
  
  //Initialize in root process array with ALL the elements
  if(rank==0){
  	float p=0.7;int w=10;	
  	makeAdjacency(n,p,w,Ar1);                     //Create makeadjency array , Holding weights With given w , and p 
  	/* ************************************************ SERIAL CODE :  ********************************************************************/
  	/* ***************************************************************************************************/
  	gettimeofday(&start1, NULL);			//Start timer
  	Ar3=fmDist(Ar1,n);

  	gettimeofday(&end1, NULL);
  	printf("FOR SERIAL VERSION OF ALGORITHM TIME IS : %ld microSeconds \n", ((end1.tv_sec * 1000000 + end1.tv_usec)
  		  - (start1.tv_sec * 1000000 + start1.tv_usec)));
           
	}
		/* ************************************************ END OF SERIAL CODE :  ********************************************************************/
   MPI_Barrier(MPI_COMM_WORLD);
 
   
    int numb=N/numTasks;     //Total number that belong to current process
      
      
    MPI_Datatype rowtype;
    MPI_Type_contiguous(numb,MPI_FLOAT,&rowtype);      //Contiguous buffer (MPI Derived Data type)
    MPI_Type_commit(&rowtype);
   
      //Rank 0 , Splits the array to other tasks
      MPI_Scatter(Ar1,numb,MPI_FLOAT,Ar0,numb,
                  MPI_FLOAT,0,MPI_COMM_WORLD);
      /*Ar0 is local local array of weight for each process */
      
      if(rank==0)
        Ar0=memcpy(Ar0,Ar1,numb*sizeof(float));              //Copy elements from global array to TASK 0
      
/* ***************************************************START MPI-Cuda CODE *************************************************** *************************************************** */
/* ***************************************************START MPI-Cuda CODE *************************************************** *************************************************** */
  float *result;
 
  result = shortest (Ar0,n,numTasks,rank);   //Call cuda (via extern C function)
 
  int j;
  int n0=n/numTasks;
  
  float *D=(float*)malloc(N*sizeof(float)); //Final Matrix
  MPI_Gather(result, n*n0, MPI_FLOAT, D, n*n0, MPI_FLOAT, 0, MPI_COMM_WORLD );   //Gather all results to TASK 0 
  
  
  //Compare parallel and serial results
  if (rank==0){
	  for (i=0; i<n; i++)
		{
			for (j=0; j<n; j++)
			{
	//			printf("This is SERIAL :%f THIS IS PARALLEL %f \n",Ar3[i*n+j],D[i*n+j]);
				if ( Ar3[i*n+j] != D[i*n+j] )
				{
					printf("\n Test FAILED\n");       
					return;                            
				}
			}
		}
		printf("\n Test PASSED\n");	
	}
  MPI_Barrier(MPI_COMM_WORLD);
  
  if(rank==0)
  printf("FINISHED\n");


  free(Ar1);free(Ar2);free(Ar3);free(Ar4);free(Ar5);free(Ar6);
  MPI_Finalize();
  return 0;
}

//Debug
void print(float *A, char Name, int N, int N0)
{
	int i,j;
	printf("\n %c =\n", Name);
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
