# FPGA HLS Attention Accelerator

FPGA implementation and optimization of a quantized attention accelerator using AMD Vitis HLS.

The project will progressively implement:

- INT8 GEMM accelerator
- tiled and pipelined GEMM
- fixed-point softmax
- quantized attention
- YOLO attention workload
- deployment on AMD Kria KV260

## Current status

Initial HLS GEMM implementation.