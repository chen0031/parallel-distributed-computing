#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv){
	//MPI_init takes a pointer to argc and argv
	MPI_Init(&argc, &argv);

	//MPI_Comm_Size takes a communication and returns the number of processes in it
	int world_size;
	//Using the world communication
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//MPI_Comm_rank returns the rank of a process in a communication
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	//Print out each processors id
	printf("Hello world from processor %d\n", world_rank);

	//Set up a barrier for all processes before moving to next line
	MPI_Barrier(MPI_COMM_WORLD);

	//Print total number of processes
	if (world_rank == 0){
		printf("%d processes said Hello!\n", world_size);
	}

	//Finalize MPI
	MPI_Finalize();
}