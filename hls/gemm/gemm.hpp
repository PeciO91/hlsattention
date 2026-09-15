#ifndef GEMM_HPP
#define GEMM_HPP

#include <ap_int.h>

constexpr int M = 16;
constexpr int K = 16;
constexpr int N = 16;

using data_t = ap_int<8>;
using acc_t  = ap_int<32>;

void gemm(
    const data_t A[M][K],
    const data_t B[K][N],
    acc_t C[M][N]
);

#endif