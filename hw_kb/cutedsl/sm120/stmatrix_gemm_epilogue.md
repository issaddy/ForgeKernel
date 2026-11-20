# BF16 GEMM epilogue with stmatrix staging

## When needed

The warp MMA accumulator is laid out by fragment ownership, but the epilogue needs a shared tile before writing a contiguous output. Compare the complete staged writeback with the simplest correct direct writeback.

## Minimal runnable example

[The GEMM probe](probes/stmatrix_gemm_epilogue.py) holds the BF16 MMA main loop and output transform fixed while comparing the two writebacks. The transform applies an FP32 affine operation and converts once to BF16. The staged branch uses `StMatrix8x8x16bOp(False, 2)` tied to `make_tiled_copy_C`. It retiles the BF16 fragment into a 16-byte-aligned, N-contiguous shared tile, synchronizes, then stores adjacent output elements from adjacent lanes.

```sh
bash gpu_entry.sh cmd python3 /path/to/stmatrix_gemm_epilogue.py
bash gpu_entry.sh cmd 'ncu --csv --kernel-name "regex:.*writeback.*" --launch-count 10 --metrics gpu__time_active.sum,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum python3 /path/to/stmatrix_gemm_epilogue.py'
```

## Measured comparison

RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1, NCU 2025.3.1. BF16 inputs/output, FP32 accumulation, M=N=512 and K=128, one warp per 16×8 tile. The GPU active-time medians and ranges are from five profiled launches per path in a single NCU invocation.

| Metric | Direct fragment writeback | stmatrix plus shared-memory readback |
| --- | ---: | ---: |
| GPU active time, median (range) | 10.816 µs (10.368–10.848) | 10.720 µs (10.560–10.944) |
| Registers per thread | 80 | 80 |
| Dynamic shared memory per CTA | 1,024 B | 1,024 B |
| Local load/store sectors | 0 / 0 | 0 / 0 |
| Shared load wavefronts | 196,608 | 204,800 |
| Shared store wavefronts | 196,608 | 200,704 |

The timing ranges overlap. Use the readings to account for the extra shared-memory traffic. A performance decision requires additional repetitions at representative shapes, especially when the epilogue has more work than this affine transform.

## Four-warp CTA comparison

Pass `--multi-warp` to the same probe and NCU command. This uses M=1,024, N=1,536, K=256. Four warps own different 16×8 tiles in the same CTA, with independent shared faces. Both writeback variants make the same copies and reserve the same shared memory.

| Metric | Direct | stmatrix |
| --- | ---: | ---: |
| GPU active time, median (range) | 75.456 µs (75.136–75.776) | 77.216 µs (77.152–77.280) |
| Registers per thread | 72 | 72 |
| Dynamic shared memory per CTA | 4,096 B | 4,096 B |
| Local load/store sectors | 0 / 0 | 0 / 0 |
| Shared load wavefronts, median | 2,366,276 | 2,414,593 |
| Shared store wavefronts, median | 2,484,101 | 2,511,256 |

Five profiled launches per variant. This configuration favored the direct path. The paired result preserves the shape, warp count, and shared-memory allocation, which makes it a stronger writeback comparison than comparing last-stage instruction counts alone.

### Register pressure and spills

On this four-warp shape, forcing 48 registers per thread increased active warps but caused local-memory traffic and made both paths slower. These median readings came from five launches of each path per NCU invocation, on the same RTX PRO 5000 with BF16 inputs/output and FP32 accumulation.

| Writeback | Register allocation | GPU active time | Local load sectors | Local store sectors |
| --- | ---: | ---: | ---: | ---: |
| Direct, default | 72 | 75.456 µs | 0 | 0 |
| Direct, capped | 48 | 131.296 µs | 11,108,352 | 3,833,856 |
| stmatrix, default | 72 | 77.216 µs | 0 | 0 |
| stmatrix, capped | 48 | 137.696 µs | 11,993,088 | 3,735,552 |

Pass `--multi-warp --max-registers 48` to the probe to reproduce the capped run. CuTe DSL 4.8.1 emitted a PTX launch directive for this kernel. The cap alone still produced 72 registers, so the probe passes `--override-directive-values` with the cap and verifies the actual register allocation in NCU. Both capped variants passed the numerical check.

## Gotchas

- The `stmatrix` instruction writes shared memory. Include the subsequent rendezvous, reload, and final global store when measuring.
- Use compatible source and destination partitions from the same MMA-linked tiled copy. Alignment and orientation are part of the layout contract.
- An N-major result and an M-major result can require different store atoms. This probe verifies the N-contiguous, non-transposing branch.

## Sources

- [CuTe DSL warp API](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html). Porting basis: SM90 `ldmatrix_stmatrix`, `stsm_transpose_exchange`, and `uniform_reg_budget`; SM100 `role_loop_state_spill_free`.
- GPU certification: both paths returned `PROBE PASS max_diff=0.0` at both shapes against CPU FP32 GEMM followed by BF16 conversion. The single-warp staged PTX emitted `stmatrix.sync.aligned.m8n8.x2.shared.b16`.
