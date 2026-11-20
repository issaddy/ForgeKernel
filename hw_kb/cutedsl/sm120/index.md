# SM120 (Blackwell, compute capability 12.0) interface index

CuTe DSL target: `sm_120a`. Verified entries cite SM120 GPU probes below.

## Find by symptom

| Observation | Check first | Relevant entries |
| --- | --- | --- |
| Correct output but the expected tensor/copy instruction is absent | Dump the PTX and cubin for the exact variant, then verify the opcode family, dtype and layout | [Instruction checks](profiling_and_instruction_checks.md), [warp MMA](warp_mma_f16_bf16.md), [stmatrix writeback](stmatrix_gemm_epilogue.md) |
| Shared-memory load wavefronts are high | Compare bank conflicts at equal work, load/store width and shared-memory allocation | [Padding and XOR swizzle](shared_bank_stride.md), [writeback comparison](stmatrix_gemm_epilogue.md) |
| A smaller register budget raises occupancy but slows down the kernel | Confirm the actual registers per thread and compare local load/store sectors before deciding | [Register pressure and spills](stmatrix_gemm_epilogue.md#register-pressure-and-spills) |
| Partial tiles contain stale or invalid data | Verify whether the load was skipped, explicitly zero-filled or covered by a TMA descriptor boundary | [cp.async source-size](cp_async_partial_zero_fill.md), [TMA descriptor tail](tma_tile_load.md), [GEMM tails](grouped_warp_gemm.md) |
| Irregular row loads dominate | Separate index load lead distance from data-pipeline stages and check invalid-index behavior | [Indexed gather](indexed_gather_lookahead.md) |
| A pipeline stalls at a later iteration or corrupts a reused tile | Check phase per slot, expected transaction bytes, and the consumer release before overwrite | [TMA stage reuse](tma_stage_reuse.md), [TMA store slot reuse](tma_store_slot_reuse.md) |
| Decode device timing follows CPU enqueue time | Pair the CPU enqueue clock with nonempty graph replay or full-launch-sequence NCU readings per workload | [Profiling and timing](profiling_and_instruction_checks.md) |
| Softmax fails only for all-masked rows or changing tile maxima | Trace the max, denominator and weighted accumulator through the boundary | [Online normalization](online_weighted_softmax.md) |
| Quantized results fail only for certain groups or tails | Check the scale-group address separately from the data address | [Grouped NVFP4 scale offsets](grouped_nvfp4_baseline.md), [FP4 scale layout](block_scaled_warp_mma.md) |

## Interfaces

| Interface | When to use it | Status |
| --- | --- | --- |
| [`bf16_aligned_vector_copy`](bf16_aligned_vector_copy.md) | Copy contiguous BF16 elements between global memory and registers in 16-byte groups | ✅ Verified |
| [`warp_fp32_sum`](warp_fp32_sum.md) | Reduce 32 lane-local FP32 values and broadcast the result within one warp | ✅ Verified |
| [`cta_fp32_sum`](cta_fp32_sum.md) | Combine partial sums across four warps through shared memory | ✅ Verified |
| [`warp_mma_f16_bf16`](warp_mma_f16_bf16.md) | Feed a warp MMA with `ldmatrix` and write its FP32 accumulator | ✅ Verified |
| [`warp_mma_fp8`](warp_mma_fp8.md) | Use an unscaled E4M3 or E5M2 warp MMA | ✅ Verified |
| [`cp_async_aligned_tile`](cp_async_aligned_tile.md) | Stage aligned global-to-shared tiles and wait for their data | ✅ Verified |
| [`tma_tile_load`](tma_tile_load.md) | TMA global-to-shared load, mbarrier completion and descriptor-boundary zero fill | ✅ Verified |
| [`tma_descriptor_promotion`](tma_descriptor_promotion.md) | Use equivalent driver-encoded tensor maps with independent L2 promotion settings | ✅ Verified |
| [`masked_warp_softmax`](masked_warp_softmax.md) | Stable maximum and sum across two segments, including fully masked rows | ✅ Verified |
| [`dsl_tracing`](dsl_tracing.md) | Trace scope and compile-time specialization | SM120 specialization example |
| [`profiling_and_instruction_checks`](profiling_and_instruction_checks.md) | Use paired variants to separate numerical, codegen, and instrumentation findings | SM120 profiling example |
| [`block_scaled_warp_mma`](block_scaled_warp_mma.md) | Select an MXFP4 or NVFP4 atom and derive scale layout | Upstream dense GEMM passed on SM120 |
| [`tma_stage_reuse`](tma_stage_reuse.md) | Reuse TMA slots across phases with one warp or dedicated producer/consumer warps | ✅ Verified |
| [`shared_bank_stride`](shared_bank_stride.md) | Isolate bank conflicts and shared-memory wavefronts with a controlled pitch change | ✅ Verified |
| [`grouped_warp_gemm`](grouped_warp_gemm.md) | Map unequal group shapes to BF16 MMA tiles and validate M/N/K tails | ✅ Verified |
| [`online_weighted_softmax`](online_weighted_softmax.md) | Combine masked tiles with FP32 denominator and numerator rescaling | ✅ Verified |
| [`stmatrix_gemm_epilogue`](stmatrix_gemm_epilogue.md) | Compare direct and staged GEMM epilogue writeback at the same shape | ✅ Verified |
| [`cp_async_partial_zero_fill`](cp_async_partial_zero_fill.md) | Fill partial aligned fragments using a runtime source byte count | ✅ Verified |
| [`tma_store_slot_reuse`](tma_store_slot_reuse.md) | Publish shared writes to TMA and retire the store before reuse | ✅ Verified |
| [`grouped_nvfp4_baseline`](grouped_nvfp4_baseline.md) | Check uneven expert ranges and measure the hardware block-scaled grouped GEMM path | ✅ Verified |
| [`indexed_gather_lookahead`](indexed_gather_lookahead.md) | Stage irregular gathers and compare loading the next index in advance | ✅ Verified |

## RTX PRO 5000 72GB Blackwell limits

`cuDeviceGetAttribute`, driver 590.44.01:

| Resource | Limit |
| --- | ---: |
| Shared memory per SM | 100 KiB (102,400 B) |
| Shared memory per CTA, opt-in | 99 KiB (101,376 B) |
| Resident threads per SM | 1,536 (48 warps) |
| 32-bit registers per SM | 65,536 |

Use SM120 warp MMA and register accumulators. The available TMA probes use single-CTA transfers.

## Sources

- [NVIDIA CUTLASS hardware compatibility](https://docs.nvidia.com/cutlass/latest/overview.html#hardware)
- [NVIDIA CuTe DSL warp-level MMA programming guide](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/guides/mma/wmma_programming.html)
- [CUDA Programming Guide, compute capability and memory capacity tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#shared-memory-capacity-per-compute-capability)
- [NVIDIA CUTLASS SM120 GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
