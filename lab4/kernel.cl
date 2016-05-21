#define NUM 512
#define IMROW 224
#define INIMROW 226
#define KERNEL 3

__kernel 
void CONV(
  __global float * Cout,
	__global float * Cin,
	__global float * weight,
	__global float * bias,
  __local float * Cout_loc,
  __local float * weight_loc,
  __local float * Cin_loc) {

  int gid = get_global_id(0);
  int g_size = get_global_size(0);

  int lid = get_local_id(0);
  int l_size = get_local_size(0);

  int group_id = get_group_id(0); //i

  float private_bias = bias[group_id];

  // **** LARGE PRIVATE COUT ****
  for (int h = 0; h < IMROW; h++) {
    for (int w = lid; w < IMROW; w += l_size){
      Cout[(group_id * IMROW * IMROW) + (h * IMROW) + w] = private_bias;
    }
  }
  
  for (int j = 0; j < NUM; j++) {
    //Load local weight
    for (int p = 0; p < KERNEL; p++){
      for (int q = lid; q < KERNEL; q += l_size){
        weight_loc[(p * KERNEL) + q] = weight[(group_id * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // int cout_row;
    for (int h = 0; h < IMROW; h++) {
      // cout_row = h - (h / 28);
      // //Load local Cout
      // if (cout_row == 0){
      //   for (int c = 0; c < 28; c++){
      //     for (int w = lid; w < IMROW; w += l_size){
      //       Cout_loc[(c * IMROW) + w] = Cout[(group_id * IMROW * IMROW) + ((h + c) * IMROW) + w];
      //     }
      //   }
      //   barrier(CLK_LOCAL_MEM_FENCE);
      // }

      //Load local Cin
      for (int k = 0; k < KERNEL; k++){
        for (int w = lid; w < INIMROW; w += l_size){
          Cin_loc[(k * INIMROW) + w] = Cin[(j * INIMROW * INIMROW) + ((h + k) * INIMROW) + w];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      //Computation
      for (int w = lid; w < IMROW; w += l_size) {
        for (int p = 0; p < KERNEL; p++) {
          for (int q = 0; q < KERNEL; q++) {
            Cout[(group_id * IMROW * IMROW) + (h * IMROW) + w] +=
            // Cout_loc[(cout_row * IMROW) + w] +=
              (weight_loc[(p * KERNEL) + q] * Cin_loc[(p * INIMROW) + (w + q)]);
          }
        }
      }
      // barrier(CLK_LOCAL_MEM_FENCE);

      //Copy back local Cout
      // if (cout_row == 27){
      //   for (int c = 0; c < 28; c ++){
      //     for (int w = lid; w < IMROW; w += l_size){
      //       Cout[(group_id * IMROW * IMROW) + ((h + c) * IMROW) + w] += Cout_loc[(c * IMROW) + w];
      //     }
      //   }
      //   barrier(CLK_LOCAL_MEM_FENCE);
      // }
    }
  }




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
  // float private_Cout[IMROW * IMROW];
  // for (int h = 0; h < IMROW; h++) {
  //   for (int w = 0; w < IMROW; w++) {
  //     private_Cout[(h * IMROW) + w] = private_bias;
  //   }
  // }

  // float private_weight[KERNEL * KERNEL];
  // for (int j = 0; j < NUM; j++) {
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
