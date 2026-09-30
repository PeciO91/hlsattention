#include "gemm_tiled.hpp"

#include <cstdint>
#include <iostream>

int main() {

  data_t A[M * K];
  data_t B[K * N];
  acc_t C[M * N];

  int32_t golden[M * N];

  // Fill matrix A with deterministic values
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      A[i * K + k] = ((i * 3 + k * 5) % 17) - 8;
    }
  }

  // Fill matrix B with deterministic values
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < N; j++) {
      B[k * N + j] = ((k * 7 + j * 2) % 19) - 9;
    }
  }

  // Golden software reference
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {

      int32_t acc = 0;

      for (int k = 0; k < K; k++) {
        acc += static_cast<int>(A[i * K + k]) * static_cast<int>(B[k * N + j]);
      }

      golden[i * N + j] = acc;
    }
  }

  // Run implementation under test
  gemm_tiled(A, B, C);

  // Compare results
  int errors = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {

            if (static_cast<int32_t>(C[i * N + j]) != golden[i * N + j]) {
        std::cout << "Mismatch at C[" << i << "][" << j << "]"
                  << ": got " << C[i * N + j] << ", expected "
                  << golden[i * N + j] << std::endl;

        errors++;
            }
    }
  }

  if (errors == 0) {
    std::cout << "TEST PASSED" << std::endl;
    return 0;
  }

  std::cout << "TEST FAILED: " << errors << " errors" << std::endl;

  return 1;
}