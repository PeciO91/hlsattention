#include "gemm.hpp"

void gemm(
    const data_t A[M][K],
    const data_t B[K][N],
    acc_t C[M][N]
) {
    data_t A_local[M][K];
    data_t B_local[K][N];

#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=1

    // Load A into local memory
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A_local[i][k] = A[i][k];
        }
    }

    // Load B into local memory
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            B_local[k][j] = B[k][j];
        }
    }

    // Matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            acc_t acc = 0;

            for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                acc += A_local[i][k] * B_local[k][j];
            }

            C[i][j] = acc;
        }
    }
}