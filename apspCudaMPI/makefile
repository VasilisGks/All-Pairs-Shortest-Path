
SHELL := /bin/bash

####################
## FILENAMES
####################

CUDA_MAIN  := cuda
MPI_MAIN   := mpi
OUTPUT     := main

####################
## COMPILER
####################

MPICC      := /usr/lib64/openmpi/1.4-gcc/bin/mpicc
NVCC 	   := /usr/local/cuda-5.5/bin/nvcc
CUDA_FLAGS := -arch=sm_13
LINK_FLAGS := -L/usr/local/cuda/lib64 -lcuda -lcudart
MPI_FLAGS  := 


OBJECTS := $(MPI_MAIN).o $(CUDA_MAIN).o

all: $(OUTPUT)

$(OUTPUT): $(OBJECTS)
	$(MPICC) -o $(OUTPUT) $(OBJECTS) $(LINK_FLAGS)

$(CUDA_MAIN).o:
	$(NVCC) -c $(CUDA_MAIN).cu $(CUDA_FLAGS)

$(MPI_MAIN).o:
	$(MPICC) -c $(MPI_MAIN).c $(MPI_FLAGS)


clean:
	rm -f *.o
	rm -f *~
