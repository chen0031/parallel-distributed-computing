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

void LoadData(float *Cin, float *weight, float *bias)
{
	int i, j, k, s;
	fprintf(stderr, "start loading input & weight\n");
	FILE *fw, *fb, *fi;
	fw = fopen("/usr/local/cs133/archive/CnnCase/weight.bin", "rb");
	fb = fopen("/usr/local/cs133/archive/CnnCase/bias.bin", "rb");
	double* t_bias = (double *)malloc(sizeof(double) * NUM);
	double* t_wght = (double *)malloc(sizeof(double) * NUM * NUM * KERNEL * KERNEL);
	fread(t_wght, NUM * NUM * KERNEL * KERNEL, sizeof(double), fw);
	fread(t_bias, NUM, sizeof(double), fb);

	for(i=0; i<NUM; i++) {
		bias[i] = t_bias[i];
		for(j=0; j<NUM; j++) {
			for(k=0; k<KERNEL; k++) {
				for(s=0; s<KERNEL; s++)
					weight[i * (NUM * KERNEL * KERNEL) + j * (KERNEL * KERNEL) + k * KERNEL + s] = 
						(float)t_wght[i*NUM*KERNEL*KERNEL + j*KERNEL*KERNEL + k*KERNEL + s];
			}
		}
	}
	fprintf(stderr, "finish loading weight\n");
	free(t_bias);
	free(t_wght);
	fclose(fw);
	fclose(fb);

	double* t_in = (double *)malloc(sizeof(double) * NUM * INIMROW * INIMROW);
	fi = fopen("/usr/local/cs133/archive/CnnCase/input.bin", "rb");
	fread(t_in, NUM * INIMROW * INIMROW, sizeof(double), fi);
	for(i=0; i<NUM; i++) {
		for(j=0; j<INIMROW; j++) {
			for(k=0; k<INIMROW; k++)
				Cin[i * (INIMROW * INIMROW) + j * INIMROW + k] = (float)t_in[i*INIMROW*INIMROW + j*INIMROW + k];
		}
	}
	fprintf(stderr, "finish loading Cin\n");
	free(t_in);
	fclose(fi);
}

int Verify(float *Cout)
{
	int error=0;
	FILE *fo;
	double* t_out = (double *)malloc(sizeof(double) * NUM * IMROW * IMROW);
	fo = fopen("/usr/local/cs133/archive/CnnCase/output.bin", "rb");
	fread(t_out, NUM * IMROW * IMROW, sizeof(double), fo);
	for(int i=0; i<NUM; i++) {
		for(int j=0; j<IMROW; j++) {
			for(int k=0; k<IMROW; k++) {
				float err = rcmp(Cout[i*IMROW*IMROW + j*IMROW + k], (float)t_out[i*IMROW*IMROW + j*IMROW + k]);
				if(err > 1e-3) {
					error++;
					#ifdef SHOW_ERROR_VALUE
	        		fprintf(stderr, "%f,\t%f,\t%f\n", Cout[i*IMROW*IMROW + j*IMROW + k], t_out[i*IMROW*IMROW + j*IMROW + k],
  						rcmp(Cout[i*IMROW*IMROW + j*IMROW + k], (float)t_out[i*IMROW*IMROW + j*IMROW + k]));
	        		#endif
				}
			}
		}
	}
	free(t_out);
	fclose(fo);
	return error;
}

// int
// load_file_to_memory(const char *filename, char **result)
// { 
//     size_t size = 0;
//     FILE *f = fopen(filename, "rb");
//     if (f == NULL) 
//         { 
//             *result = NULL;
//             return -1; // -1 means file opening fail 
//         } 
//     fseek(f, 0, SEEK_END);
//     size = ftell(f);
//     fseek(f, 0, SEEK_SET);
//     *result = (char *)malloc(size+1);
//     if (size != fread(*result, sizeof(char), size, f)) 
//         { 
//             free(*result);
//             return -2; // -2 means file reading fail 
//         } 
//     fclose(f);
//     (*result)[size] = 0;
//     return size;
// }

#endif
