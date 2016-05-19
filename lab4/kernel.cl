#define NUM 512
#define IMROW 224
#define INIMROW 226
#define KERNEL 3

__kernel 
void CONV(
  __global float * Cout,
	__global float * Cin,
	__global float * weight,
	__global float * bias) {

  int gid = get_global_id(0);
  int g_size = get_global_size(0);

  float private_bias = bias[gid];


  float private_Cout[IMROW * IMROW];
  for (int h = 0; h < IMROW; h++) {
    for (int w = 0; w < IMROW; w++) {
      private_Cout[(h * IMROW) + w] = private_bias;
    }
  }

  for (int j = 0; j < NUM; j++) {
    for (int h = 0; h < IMROW; h++) {
      for (int w = 0; w < IMROW; w++) {
        for (int p = 0; p < KERNEL; p++) {
          for (int q = 0; q < KERNEL; q++) {
            private_Cout[(h * IMROW) + w] +=
              (weight[(gid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q] *
              Cin[(j * INIMROW * INIMROW) + ((h + p) * INIMROW) + (w + q)]);
          }
        }
      }
    }
  }

  for (int h = 0; h < IMROW; h++) {
    for (int w = 0; w < IMROW; w++) {
      Cout[(gid * IMROW * IMROW) + (h * IMROW) + w] = private_Cout[(h * IMROW) + w];
    }
  }
}
