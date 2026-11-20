# Block-scaled FP4 warp MMA interface

## Purpose

Choose the SM120 FP4 warp MMA atom and derive the operand and scale layouts from it. The API contract below is documented by NVIDIA. Both formats also passed the upstream dense GEMM example on the local SM120 card.

## When needed

Use it to plan a block-scaled GEMM for dense or grouped workloads. Start with a single dense tile before adding routing, per-expert offsets, a persistent schedule, or a fused epilogue.

| Format | Atom | FP4 input | Scale type in CuTe DSL | K values per scale | Atom M×N×K |
| --- | --- | --- | --- | --- | --- |
| MXFP4 | `warp.MmaMXF4Op` | E2M1 | `Float8E8M0FNU` | 32 | 16×8×64 |
| NVFP4 | `warp.MmaMXF4NVF4Op` | E2M1 | `Float8E4M3FN` | 16 | 16×8×64 |

Both documented atoms use FP32 accumulation. With the warp-level interface the values and their scale factors are placed in registers and passed to `cute.gemm` as paired operands `[a, sfa]` and `[b, sfb]`.

## Minimal runnable example

Use the [CUTLASS v4.8.0 persistent ping-pong example](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py)
with [its dispatch helper](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/blockscaled_gemm_dispatch.py)
in the same directory. Run this command for NVFP4:

```sh
bash gpu_entry.sh cmd python3 /path/to/dense_blockscaled_gemm_persistent_pingpong.py \
  --mnkl 256,256,256,1 --tile_shape_mnk 128,128,128 --epi_tile 64,32 \
  --a_dtype Float4E2M1FN --b_dtype Float4E2M1FN --c_dtype Float16 --acc_dtype Float32 \
  --sf_dtype Float8E4M3FN --sf_vec_size 16 --iterations 2
```

For MXFP4 use `--sf_dtype Float8E8M0FNU --sf_vec_size 32`. On RTX PRO 5000
72GB Blackwell with CuTe DSL 4.8.1, both runs exited 0 and printed `PASS`
with the example's result check enabled. That establishes this dense ping-pong
configuration with FP16 output and FP32 accumulation.

## Gotchas

- Test each format separately. Use distinct scales by operand, row or column, and K group, so a scale permutation cannot hide behind values of one.
- Compare with an FP32 product formed by dequantizing the stored values and their actual scales. Then inspect the generated instruction and verify its block-scaling kind.
- For unequal group shapes, use the [grouped NVFP4 scale-addressing probe](grouped_nvfp4_baseline.md).

## Sources

- [NVIDIA CuTe DSL warp API, version 4.8.0](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html) and [public CUTLASS SM120 dispatch helpers](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/blockscaled_gemm_dispatch.py).
- Porting basis for test design: SM100 `tcgen05_blockscaled_mma_sf_tmem`. The dense verification above used the upstream example.
