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


  // **** LARGE PRIVATE COUT ****
  // float private_Cout[IMROW * IMROW];
  // for (int h = 0; h < IMROW; h++) {
  //   for (int w = 0; w < IMROW; w++) {
  //     private_Cout[(h * IMROW) + w] = private_bias;
  //   }
  // }

  
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

  // for (int h = 0; h < IMROW; h++) {
  //   for (int w = 0; w < IMROW; w++) {
  //     Cout[(gid * IMROW * IMROW) + (h * IMROW) + w] = private_Cout[(h * IMROW) + w];
  //   }
  // }


  // **** LARGE PRIVATE COUT, SMALL PRIVATE WEIGHT *****
  float private_Cout[IMROW * IMROW];
  for (int h = 0; h < IMROW; h++) {
    for (int w = 0; w < IMROW; w++) {
      private_Cout[(h * IMROW) + w] = private_bias;
    }
  }

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

  for (int h = 0; h < IMROW; h++) {
    for (int w = 0; w < IMROW; w++) {
      Cout[(gid * IMROW * IMROW) + (h * IMROW) + w] = private_Cout[(h * IMROW) + w];
    }
  }


  // **** LARGE PRIVATE COUT, SMALL PRIVATE WEIGHT, SMALL PRIVATE CIN *****
  // float private_Cout[IMROW * IMROW];
  // for (int h = 0; h < IMROW; h++) {
  //   for (int w = 0; w < IMROW; w++) {
  //     private_Cout[(h * IMROW) + w] = private_bias;
  //   }
  // }

  // float private_weight[KERNEL * KERNEL];
  // float private_Cin[INIMROW];
  // for (int j = 0; j < NUM; j++) {

  //   for (int h = 0; h < INIMROW; h++){
  //     for (int w = 0; w < INIMROW; w++){
  //       private_Cin[(h * INIMROW) + w] = Cin[(j * INIMROW * INIMROW) + (h * INIMROW) + w];
  //     }
  //   }


  //   for (int p = 0; p < KERNEL; p++){
  //     for (int q = 0; q < KERNEL; q++){
  //       private_weight[(p * KERNEL) + q] = weight[(gid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q];
  //     }
  //   }

  //   for (int h = 0; h < IMROW; h++) {
  //     for (int w = 0; w < IMROW; w++) {
  //       for (int p = 0; p < KERNEL; p++) {
  //         for (int q = 0; q < KERNEL; q++) {
  //           private_Cout[(h * IMROW) + w] +=
  //             (private_weight[(p * KERNEL) + q] *
  //             private_Cin[((h + p) * INIMROW) + (w + q)]);
  //         }
  //       }
  //     }
  //   }
  // }

  // for (int h = 0; h < IMROW; h++) {
  //   for (int w = 0; w < IMROW; w++) {
  //     Cout[(gid * IMROW * IMROW) + (h * IMROW) + w] = private_Cout[(h * IMROW) + w];
  //   }
  // }


  //*** SMALL PRIVATE COUT, LARGE PRIVATE WEIGHT ***
  // float private_weight[NUM * KERNEL * KERNEL];
  // for (int j = 0; j < NUM; j++){
  //   for (int p = 0; p < KERNEL; p++){
  //     for (int q = 0; q < KERNEL; q++){
  //       private_weight[(j * KERNEL * KERNEL) + (p * KERNEL) + q] =
  //         weight[(gid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q];
  //     }
  //   }
  // }

  // float private_Cout[IMROW];
  // for (int j = 0; j < NUM; j++) {
  //   for (int h = 0; h < IMROW; h++) {

  //     for (int w = 0; w < IMROW; w++){
  //       private_Cout[w] = private_bias;
  //     }

  //     for (int w = 0; w < IMROW; w++) {
  //       for (int p = 0; p < KERNEL; p++) {
  //         for (int q = 0; q < KERNEL; q++) {
  //           private_Cout[w] +=
  //             (private_weight[(j * KERNEL * KERNEL) + (p * KERNEL) + q] *
  //             Cin[(j * INIMROW * INIMROW) + ((h + p) * INIMROW) + (w + q)]);
  //         }
  //       }
  //     }

  //     for (int w = 0; w < IMROW; w++){
  //       Cout[(gid * IMROW * IMROW) + (h * IMROW) + w] = private_Cout[w];
  //     }

  //   }
  // }


  


  // LOCAL WEIGHT
  // for (int j = 0; j < NUM; j++){
  //   for (int p = 0; p < KERNEL; p++){
  //     for (int q = 0; q < KERNEL; q++){
  //       weight_local[(lid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q] =
  //         weight[(gid * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q];
  //     }
  //   }
  // }


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
