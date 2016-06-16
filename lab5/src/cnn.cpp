#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cnn.h"

void kernel(float *Cout, float *Cin, float *weight, float *bias);

int main()
{
	static float Cout[NUM * IMROW * IMROW];
	static float Cin[NUM * INIMROW * INIMROW];
	static float weight[NUM * NUM * KERNEL * KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

    printf("start cnn computation\n");
    long beginTime = clock();

	#pragma ACCEL task
    kernel(Cout, Cin, weight, bias);
   
    long endTime=clock();
    printf("time: %f\n", (float)(endTime - beginTime) / (float) CLOCKS_PER_SEC);

	
    //VERIFY RESULT
    int error = Verify(Cout);
	if(error != 0){
		fprintf(stderr, "error ocurrs %d\n", error);
	}
	else{
		fprintf(stderr, "all right!\n");
	}
	return 0;
}

