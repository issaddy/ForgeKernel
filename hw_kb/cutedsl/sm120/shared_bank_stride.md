# Shared-memory bank conflicts, padding and XOR swizzle

## Purpose

Use equal-volume accesses at two row pitches and an XOR swizzle to distinguish bank conflict from extra shared-memory work.

## Minimal runnable example

[The stride probe](probes/shared_bank_stride.py) stores an FP32 tile by rows, synchronizes, then reads it by columns. Each warp reads different rows, so the load cannot use broadcast. All variants run 128 CTAs of one warp on the same 32×32 logical tile, with 4,224 bytes of dynamic shared memory per CTA. It compares row pitches of 32 and 33 FP32 elements with an unpadded `make_swizzle(5, 0, 5)` variant. For this tile the XOR swizzle maps logical column `c` of row `r` to physical column `c ^ r`.

```sh
bash gpu_entry.sh cmd 'ncu --csv --kernel-name "regex:.*stride_probe.*" --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,launch__shared_mem_per_block_dynamic,gpu__time_active.sum python3 /path/to/shared_bank_stride.py'
```

Observed in one NCU 2025.3.1 profile on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1:

| Layout | Shared-load bank conflicts | Shared-load wavefronts | Dynamic SMEM per CTA | GPU active time |
| --- | ---: | ---: | ---: | ---: |
| 32-word pitch | 28,672 | 32,768 | 4,224 B | 4,064 ns |
| 33-word pitch | 0 | 4,096 | 4,224 B | 3,200 ns |
| 32-word pitch with XOR swizzle | 0 | 4,096 | 4,224 B | 3,104 ns |

Each layout also recorded 4,096 shared-store wavefronts and zero shared-store bank conflicts.

## Gotchas

- A warp reading the same word can use a broadcast. For a conflict control, make each lane read a distinct word.
- Compare the load and store sides separately. More wavefronts can also come from wider or additional instructions. Check the instruction mix and work volume when changing a production kernel.
- Padding and the tested XOR swizzle make successive rows land in distinct banks for this FP32 column access. Recompute the mapping and layout compatibility for BF16, vectorized operations, TMA descriptors, or matrix-copy instructions.
- The timings are one profile of a toy access pattern. The paired bank and wavefront counts are the primary evidence.

## Sources

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) and [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/).
- Porting basis: SM90/SM100 `smem_traffic_triage`.
- GPU certification: the linked probe reported `PROBE PASS max_diff=0.0` for all three layouts and produced the load and store metrics above.
