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
  __local float * weight_loc,
  __local float * Cin_loc) {

  // int gid = get_global_id(0);
  // int g_size = get_global_size(0);

  int group_id = get_group_id(0); //i

  int lid = get_local_id(0);
  int l_size = get_local_size(0);

  float private_bias = bias[group_id];

  for (int h = 0; h < IMROW; h++) {
    for (int w = lid; w < IMROW; w += l_size){
      Cout[(group_id * IMROW * IMROW) + (h * IMROW) + w] = private_bias;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (int j = 0; j < NUM; j++) {
    //Load local weight
    for (int p = 0; p < KERNEL; p++){
      for (int q = lid; q < KERNEL; q += l_size){
        weight_loc[(p * KERNEL) + q] = weight[(group_id * NUM * KERNEL * KERNEL) + (j * KERNEL * KERNEL) + (p * KERNEL) + q];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int h = 0; h < IMROW; h++) {
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
              (weight_loc[(p * KERNEL) + q] * Cin_loc[(p * INIMROW) + (w + q)]);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
