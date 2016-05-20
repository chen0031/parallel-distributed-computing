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

  int lid = get_local_id(0);
  int l_size = get_local_size(0);

  float private_bias = bias[gid];


  float private_Cout[IMROW * IMROW];
  for (int h = 0; h < IMROW; h++) {
    for (int w = 0; w < IMROW; w++) {
      private_Cout[(h * IMROW) + w] = private_bias;
    }
  }

  //BASIC ALGORITHM
  // for (int j = 0; j < NUM; j++) {
  //   for (int h = 0; h < IMROW; h++) {
  //     for (int w = 0; w < IMROW; w++) {
  //       for (int p = 0; p < KERNEL; p++) {
  //         for (int q = 0; q < KERNEL; q++) {
  //           private_Cout[(h * IMROW) + w] +=
  //             (weight[(gid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q] *
  //             Cin[(j * INIMROW * INIMROW) + ((h + p) * INIMROW) + (w + q)]);
  //         }
  //       }
  //     }
  //   }
  // }

  //PRIVATE WEIGHT
  // float private_weight[NUM * KERNEL * KERNEL];
  // for (int j = 0; j < NUM; j++){
  //   for (int p = 0; p < KERNEL; p++){
  //     for (int q = 0; q < KERNEL; q++){
  //       private_weight[(j * KERNEL * KERNEL) + (p * KERNEL) + q] =
  //         weight[(gid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q];
  //     }
  //   }
  // }

  float private_weight[KERNEL * KERNEL];
  for (int j = 0; j < NUM; j++) {
    for (int p = 0; p < KERNEL; p++){
      for (int q = 0; q < KERNEL; q++){
        private_weight[(p * KERNEL) + q] = weight[(gid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q];
      }
    }

    for (int h = 0; h < IMROW; h++) {
      for (int w = 0; w < IMROW; w++) {
        for (int p = 0; p < KERNEL; p++) {
          for (int q = 0; q < KERNEL; q++) {
            private_Cout[(h * IMROW) + w] +=
              (private_weight[(p * KERNEL) + q] *
              Cin[(j * INIMROW * INIMROW) + ((h + p) * INIMROW) + (w + q)]);
          }
        }
      }
    }
  }


  // LOCAL WEIGHT
  // for (int j = 0; j < NUM; j++){
  //   for (int p = 0; p < KERNEL; p++){
  //     for (int q = 0; q < KERNEL; q++){
  //       weight_local[(lid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q] =
  //         weight[(gid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q];
  //     }
  //   }
  // }

  

  for (int h = 0; h < IMROW; h++) {
    for (int w = 0; w < IMROW; w++) {
      Cout[(gid * IMROW * IMROW) + (h * IMROW) + w] = private_Cout[(h * IMROW) + w];
    }
  }


  // LOCAL COut
  // for (int h = 0; h < IMROW; h++) {
  //   for (int w = 0; w < IMROW; w++) {
  //     Clocal[(lid * IMROW * IMROW) + (h * IMROW) + w] = private_Cout[(h * IMROW) + w];
  //   }
  // }
  // barrier(CLK_LOCAL_MEM_FENCE);

  // if (lid == 0){
  //   for (int l = 0; l < l_size; l++){
  //     for (int h = 0; h < IMROW; h++) {
  //       for (int w = 0; w < IMROW; w++) {
  //         Cout[((gid + l) * IMROW * IMROW) + (h * IMROW) + w] = Clocal[(l * IMROW * IMROW) + (h * IMROW) + w];
  //       }
  //     }
  //   }
  // }

}
