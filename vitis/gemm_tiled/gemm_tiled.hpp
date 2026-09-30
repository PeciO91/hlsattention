#pragma once
#include <ap_int.h>

constexpr int M = 64;
constexpr int K = 64;
constexpr int N = 64;
constexpr int TM = 16;
constexpr int TK = 16;
constexpr int TN = 16;

using data_t = ap_int<8>;
using acc_t = ap_int<32>;

void gemm_tiled(
    const data_t *A,
    const data_t *B,
    acc_t *C
);