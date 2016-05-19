#ifndef _CNN_H_
#define _CNN_H_

#include <math.h>

#define GEN 0
#define GoldInput 1
#define GoldOutput 2

#define NUM 512
#define IMROW 224
#define INIMROW 226
#define KERNEL 3

float rcmp(float a, float b)
{
	return fabs((a-b)/(a+b));
}

void LoadData(float Cin[NUM][INIMROW][INIMROW], float weight[NUM][NUM][KERNEL][KERNEL],
              float bias[NUM])
{
	fprintf(stderr, "start load input& weight\n");
	FILE *fw, *fb, *fi;
	fw = fopen("weight.bin", "rb");
	fb = fopen("bias.bin", "rb");
	double* t_bias = (double *)malloc(sizeof(double) * NUM);
	double* t_wght = (double *)malloc(sizeof(double) * NUM * NUM * KERNEL * KERNEL);
	fread(t_wght, NUM * NUM * KERNEL * KERNEL, sizeof(double), fw);
	fread(t_bias, NUM, sizeof(double), fb);

	for(int i=0; i<NUM; i++) {
		bias[i] = t_bias[i];
		for(int j=0; j<NUM; j++) {
			for(int k=0; k<KERNEL; k++) {
				for(int s=0; s<KERNEL; s++)
					weight[i][j][k][s] = (float)t_wght[i*NUM*KERNEL*KERNEL + j*KERNEL*KERNEL + k*KERNEL + s];
			}
		}
	}
	fprintf(stderr, "finish load weight\n");
	free(t_bias);
	free(t_wght);
	fclose(fw);
	fclose(fb);

	double* t_in = (double *)malloc(sizeof(double) * NUM * INIMROW * INIMROW);
	fi = fopen("input.bin", "rb");
	fread(t_in, NUM * INIMROW * INIMROW, sizeof(double), fi);
	for(int i=0; i<NUM; i++) {
		for(int j=0; j<INIMROW; j++) {
			for(int k=0; k<INIMROW; k++)
				Cin[i][j][k] = (float)t_in[i*INIMROW*INIMROW + j*INIMROW + k];
		}
	}
	fprintf(stderr, "finish load Cin\n");
	free(t_in);
	fclose(fi);
}

int Verify(float Cout[NUM][IMROW][IMROW])
{
	int error=0;
	FILE *fo;
	fo = fopen("output.bin", "rb");
	double* t_out = (double *)malloc(sizeof(double) * NUM * IMROW * IMROW);
	fread(t_out, NUM * IMROW * IMROW, sizeof(double), fo);
	for(int i=0; i<NUM; i++) {
		for(int j=0; j<IMROW; j++) {
			for(int k=0; k<IMROW; k++) {
				if(rcmp(Cout[i][j][k], (float)t_out[i*IMROW*IMROW + j*IMROW + k]) > 1e-3)
					error++;
			}
		}
	}
	free(t_out);
	fclose(fo);
	return error;
}


#endif
