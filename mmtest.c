#include <stdio.h>

void mmul(float A[ni][nk], float B[nk][nj], float C[ni][nj]){
	int i,j,k;
	for (i=0; i<ni; i++) {
		for (j=0; j<nj; j++) {
			C[i][j] = 0;
			for (k=0; k<nk; k++) {
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
}

void block_mmul(float A[ni][nk], float B[nk][nj], float C[ni][nj]){
	int i,j,k;
	int ii,jj,kk;
	for (i=0; i<ni; i+=2) {
		for (j=0; j<nj; j+=2) {
			for (k=0;k<nk;k+=2){
				for (ii=i;ii<(i+2);i++){
					for (jj=j;jj<(j+2);j++){
						for (kk=k;kk<(k+2);k++){
							C[i][j] += A[ii][kk] * B[kk][jj];
						}
					}
				}
			}
		}
	}

}

float compute_diff(float C[ni][nj], float Cans[ni][nj])
{
	int cnt = 0;
	int i, j;
	float diff = 0.0;
	#pragma omp parallel for private(j)
	for (i=0; i<ni; i++) {
		for (j=0; j<nj; j++) {
			diff += (C[i][j]-Cans[i][j])*(C[i][j]-Cans[i][j]);
		}
	}
	return diff;
}

int main(int argc, char *argv[]){
	float A[4][4] = {{1,1,2,2},{1,1,2,2},{3,3,4,4},{3,3,4,4}};
	float B[4][4] = {{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4}};
	float C[4][4];
	float D[4][4];

	mmul(A,B,C);
	block_mmul(A,B,D);
	float diff = compute_diff(C,D);
	printf("Difference %f\n",diff);

	// int nthreads, nprocs, tid;
	// nprocs = omp_get_num_procs();
	// printf("Number of processors %d\n",nprocs);
	// omp_set_num_threads(16n);
	// #pragma omp parallel private(nthreads,tid)
	// {
	// 	tid = omp_get_thread_num();
	// 	if (tid == 0){
	// 		nthreads = omp_get_num_threads();
	// 		printf("Number of threads %d\n",nthreads);
	// 	}
	// 	else{
	// 		printf("Hello from thread %d!\n",tid);
	// 	}
		
	// }
}