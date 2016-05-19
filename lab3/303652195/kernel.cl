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
  int tnum = get_global_size(0);
  int start = gid * (NUM / tnum);
  int end = start + (NUM / tnum);

  for (int i = start; i < end; i++) {
    for (int h = 0; h < IMROW; h++) {
      for (int w = 0; w < IMROW; w++) {
        Cout[i * (IMROW * IMROW) + h * IMROW + w] = bias[i];
      }
    }
  }

  for (int i = start; i < end; i++) {
    for (int j = 0; j < NUM; j++) {
      for (int h = 0; h < IMROW; h++) {
        for (int w = 0; w < IMROW; w++) {
          for (int p = 0; p < KERNEL; p++) {
            for (int q = 0; q < KERNEL; q++) {
              Cout[i * (IMROW * IMROW) + h * IMROW + w] += weight[i * (NUM * KERNEL * KERNEL) + j * (KERNEL * KERNEL) + p * KERNEL + q] * Cin[j * (INIMROW * INIMROW) + (h + p) * INIMROW + (w + q)];
            }
          }
        }
      }
    }
  }
}
