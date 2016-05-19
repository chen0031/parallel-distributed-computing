#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define A(i,k) A[i*n+k]
#define A_local(i,k) A_local[i*n+k]
#define B(k,j) B[k*n+j]
#define B_local(k,j) B_local[k*n+j]
#define C(i,j) C[i*n+j]
#define C_local(i,j) C_local[i*n+j]
#define BLOCK_SIZE 256

// Please modify this function
void mmul(float *A, float *B, float *C, int n)
{
	int pnum, pid;

	MPI_Comm_size(MPI_COMM_WORLD, &pnum);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	//Scatter matrix A to all processes (each matrix gets n / pnum rows)
	int A_rows, A_local_size;
	float *A_local;

	A_rows = n / pnum;
	A_local_size = A_rows * n;
	A_local = (float*)malloc(sizeof(float) * A_local_size);
	MPI_Scatter(A, A_local_size, MPI_FLOAT, A_local, A_local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//Broadcast matrix B to all processes
	int B_local_size;
		B_local_size = n*n;
	float *B_local;
	if (pid == 0) B_local = B;
	else B_local = (float*)malloc(sizeof(float) * B_local_size);
	MPI_Bcast(B_local, B_local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//Do blocked matrix multiplication
	int C_rows, C_local_size;
	float *C_local;
	C_rows = A_rows;
	C_local_size = A_local_size;
	C_local = (float*)malloc(sizeof(float) * C_local_size);

	int ii,jj,kk;
	int i,j,k;
	int cmpi,cmpj,cmpk;

	for (ii = 0; ii < C_rows; ii += BLOCK_SIZE){
		cmpi = (ii + BLOCK_SIZE) > C_rows ? C_rows : (ii + BLOCK_SIZE);

		for (jj = 0; jj < n; jj += BLOCK_SIZE){
			cmpj = (jj + BLOCK_SIZE) > n ? n : (jj + BLOCK_SIZE);

			for (i=ii; i < cmpi; i++) {
				for (j=jj; j < cmpj; j++) {
					C_local(i,j) = 0;
				}
			}

			for (kk=0; kk < n; kk += BLOCK_SIZE) {
				 cmpk = (kk + BLOCK_SIZE) > n ? n : (kk + BLOCK_SIZE);

				for (i=ii; i < cmpi; i++){
					for (k = kk; k < cmpk; k++){
						for (j=jj; j < cmpj; j++){
							C_local(i,j) += A_local(i,k)*B_local(k,j);
						}
					}
				}
			}
		}
	}

	//Gather C from all processes
	MPI_Gather(C_local, C_local_size, MPI_FLOAT, C, C_local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
}
