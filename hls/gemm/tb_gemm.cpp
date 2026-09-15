#include "gemm.hpp"

#include <iostream>
#include <cstdint>

int main() {

    data_t A[M][K];
    data_t B[K][N];
    acc_t C[M][N];

    int32_t golden[M][N];

    // Fill matrix A with deterministic values
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A[i][k] = ((i * 3 + k * 5) % 17) - 8;
        }
    }

    // Fill matrix B with deterministic values
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            B[k][j] = ((k * 7 + j * 2) % 19) - 9;
        }
    }

    // Golden software reference
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            int32_t acc = 0;

            for (int k = 0; k < K; k++) {
                acc += static_cast<int>(A[i][k])
                     * static_cast<int>(B[k][j]);
            }

            golden[i][j] = acc;
        }
    }

    // Run implementation under test
    gemm(A, B, C);

    // Compare results
    int errors = 0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            if (static_cast<int32_t>(C[i][j]) != golden[i][j]) {

                std::cout
                    << "Mismatch at C[" << i << "][" << j << "]"
                    << ": got " << C[i][j]
                    << ", expected " << golden[i][j]
                    << std::endl;

                errors++;
            }
        }
    }

    if (errors == 0) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    }

    std::cout << "TEST FAILED: "
              << errors
              << " errors"
              << std::endl;

    return 1;
}