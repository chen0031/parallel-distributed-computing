#include "const.h"

#define BLOCK_SIZE 32


void mmul2(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
	int i, j, k;
	int ii,jj,kk;
    int cmpi,cmpj,cmpk;

	#pragma omp parallel for private(i,j,k,ii,jj,kk,cmpi,cmpj,cmpk) schedule(static)
	//Advance A Block Rows
    for (i=0; i<ni; i += BLOCK_SIZE){
        cmpi = (i + BLOCK_SIZE) > ni ? ni : (i + BLOCK_SIZE);
    	//Advance B Block Columns
    	for (j=0; j<nj; j += BLOCK_SIZE){
            cmpj = (j + BLOCK_SIZE) > nj ? nj : (j + BLOCK_SIZE);

    	   //Clear Block
    		for (ii=i; ii < cmpi; ii++){
    			for (jj=j; jj < cmpj; jj++){
    				C[ii][jj] = 0;
    			}
    		}
    		//Advance on block entries in A row and B col
    		for (k=0; k<nk; k += BLOCK_SIZE){
                cmpk = (k + BLOCK_SIZE) > nk ? nk : (k + BLOCK_SIZE);
    			//Multiply Blocks
    			for (ii=i; ii < cmpi; ii++){
    				for (kk = k; kk < cmpk; kk++){
                        for (jj = j; jj < cmpj; jj++){
	                        C[ii][jj] += A[ii][kk]*B[kk][jj];
	                    }
	                }
	            }
    		}
    	}
    }
}



