# SM120 (Blackwell) CUDA C++ interface index

CUDA task target: `sm_120a`.

| Interface | When to use it | Status |
| --- | --- | --- |
| [`cp_async_partial_zero_fill`](cp_async_partial_zero_fill.md) | Stage a 16-byte vector and zero-fill a partial fragment | ✅ Verified |
| [`warp_fp32_sum`](warp_fp32_sum.md) | Reduce FP32 values across a full warp | ✅ Verified |

## Resource limits

RTX PRO 5000 72GB Blackwell: 100 KiB shared memory per SM and 99 KiB per CTA with opt-in.

## Sources

- [CUDA Programming Guide compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
