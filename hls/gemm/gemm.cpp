#include "gemm.hpp"

void gemm(
    const data_t A[M][K],
    const data_t B[K][N],
    acc_t C[M][N]
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            acc_t acc = 0;

            for (int k = 0; k < K; k++) {
                acc += A[i][k] * B[k][j];
            }

            C[i][j] = acc;
        }
    }
}