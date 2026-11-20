# CUDA C++ full-warp FP32 sum

## Purpose

Reduce FP32 values with `__shfl_xor_sync` at deltas 1, 2, 4, 8 and 16. The final sum is present in every lane.

## Minimal runnable example

Run the [CUDA probe](probes/async_copy_and_warp_sum.cu):

```sh
nvcc -arch=sm_120a -O3 async_copy_and_warp_sum.cu -o cpp_sm120
./cpp_sm120
```

## Gotchas

For a partial row, assign zero to out-of-range positions. All lanes named by the mask must reach the shuffle.

## Sources

- [CUDA Programming Guide, warp shuffle](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions).
- CUDA 13.0.88 nvcc on RTX PRO 5000 72GB Blackwell: `PROBE PASS max_diff=0.0`; `cuobjdump` showed `sm_120a` cubins and `SHFL.BFLY`.
