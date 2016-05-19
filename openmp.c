#include <stdio.h>

int main(int argc, char *argv[]){
	int nthreads, nprocs, tid;
	nprocs = omp_get_num_procs();
	printf("Number of processors %d\n",nprocs);
	omp_set_num_threads(16);
	#pragma omp parallel private(nthreads,tid)
	{
		tid = omp_get_thread_num();
		if (tid == 0){
			nthreads = omp_get_num_threads();
			printf("Number of threads %d\n",nthreads);
		}
		else{
			printf("Hello from thread %d!\n",tid);
		}
		
	}
}