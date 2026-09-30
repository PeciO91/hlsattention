#include "gemm_tiled.hpp"

// =========================================================================
// 1. Reset local accumulator buffer C_local
// =========================================================================
static void reset_tile_c(acc_t C_local[TM][TN]) {
    #pragma HLS INLINE
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            #pragma HLS PIPELINE II=1
            C_local[i][j] = 0;
        }
    }
}

// =========================================================================
// 2. Load tile A (TM x TK) from global DDR memory into on-chip BRAM
//    - Memory offset: row (ti + i), column (tk + k)
//    - Global row stride in DDR is K
// =========================================================================
static void load_tile_a(const data_t *A, data_t A_local[TM][TK], int ti, int tk) {
    #pragma HLS INLINE
    for (int i = 0; i < TM; i++) {
        for (int k = 0; k < TK; k++) {
            #pragma HLS PIPELINE II=1
            int ddr_idx = (ti + i) * K + (tk + k);
            A_local[i][k] = A[ddr_idx];
        }
    }
}

// =========================================================================
// 3. Load tile B (TK x TN) from global DDR memory into on-chip BRAM
//    - Memory offset: row (tk + k), column (tj + j)
//    - Global row stride in DDR is N
// =========================================================================
static void load_tile_b(const data_t *B, data_t B_local[TK][TN], int tk, int tj) {
    #pragma HLS INLINE
    for (int k = 0; k < TK; k++) {
        for (int j = 0; j < TN; j++) {
            #pragma HLS PIPELINE II=1
            int ddr_idx = (tk + k) * N + (tj + j);
            B_local[k][j] = B[ddr_idx];
        }
    }
}

// =========================================================================
// 4. Compute Engine (Multiply-Accumulate Tile MAC)
//    - Multiplies submatrices TM x TK and TK x TN
//    - Accumulates the partial product into C_local
// =========================================================================
static void compute_tile(
    const data_t A_local[TM][TK],
    const data_t B_local[TK][TN],
    acc_t C_local[TM][TN]
) {
    #pragma HLS INLINE

    // Memory partitioning to support parallel reads across unrolled loop
    #pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=8 dim=1

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = C_local[i][j];

            for (int k = 0; k < TK; k++) {
                #pragma HLS UNROLL factor=8
                sum += A_local[i][k] * B_local[k][j];
            }

            C_local[i][j] = sum;
        }
    }
}

// =========================================================================
// 5. Store completed tile C (TM x TN) from on-chip BRAM back to DDR
//    - Memory offset: row (ti + i), column (tj + j)
//    - Global row stride in DDR is N
// =========================================================================
static void store_tile_c(acc_t *C, const acc_t C_local[TM][TN], int ti, int tj) {
    #pragma HLS INLINE
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            #pragma HLS PIPELINE II=1
            int ddr_idx = (ti + i) * N + (tj + j);
            C[ddr_idx] = C_local[i][j];
        }
    }
}

// =========================================================================
// Top-Level GEMM Accelerator Function
// =========================================================================
void gemm_tiled(const data_t *A, const data_t *B, acc_t *C) {
    // AXI Master interfaces for external DDR memory access
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=4096
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=4096
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=4096

    // AXI-Lite interface for control registers and base pointer addresses
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local on-chip BRAM buffers for current tiles
    data_t A_local[TM][TK];
    data_t B_local[TK][TN];
    acc_t  C_local[TM][TN];

    // Tiling coordinate loops
    LOOP_TI: for (int ti = 0; ti < M; ti += TM) {
        LOOP_TJ: for (int tj = 0; tj < N; tj += TN) {

            // Clear local accumulation buffer before processing tile column
            reset_tile_c(C_local);

            // Accumulate partial products along reduction dimension K
            LOOP_TK: for (int tk = 0; tk < K; tk += TK) {
                load_tile_a(A, A_local, ti, tk);
                load_tile_b(B, B_local, tk, tj);
                compute_tile(A_local, B_local, C_local);
            }

            // Write finished tile C directly back to global memory
            store_tile_c(C, C_local, ti, tj);
        }
    }
}