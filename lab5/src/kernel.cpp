#define NUM 512
#define IMROW 224
#define INIMROW 226
#define KERNEL 3

void kernel(float *Cout, float *Cin, float *weight, float *bias) {
  int i, j, h, w, p, q;
  for (i = 0; i < NUM; i++) {
    #pragma ACCEL pipeline flatten
    for (h = 0; h < IMROW; h++) {
      for (w = 0; w < IMROW; w++) {
        Cout[i * (IMROW * IMROW) + h * IMROW + w] = bias[i];
      }
    }
  }

  for (i = 0; i < NUM; i++) {
    for (j = 0; j < NUM; j++) {
      for (p = 0; p < KERNEL; p++) {
        for (q = 0; q < KERNEL; q++) {
          #pragma ACCEL pipeline flatten
          for (h = 0; h < IMROW; h++) {
            for (w = 0; w < IMROW; w++) {
              Cout[i * (IMROW * IMROW) + h * IMROW + w] += weight[i * (NUM * KERNEL * KERNEL) + j * (KERNEL * KERNEL) + p * KERNEL + q] * Cin[j * (INIMROW * INIMROW) + (h + p) * INIMROW + (w + q)];
            }
          }
        }
      }
    }
  }
}