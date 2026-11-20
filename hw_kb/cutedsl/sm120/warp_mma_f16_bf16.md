# FP16 and BF16 warp MMA with register fragments

## Purpose

Use one warp to compute an `m16n8k16` tile, with FP16 or BF16 operands and FP32 accumulation.

## When needed

Use this as a baseline for dense GEMM or for the matrix-multiply portions of Attention and grouped GEMM. Start with one warp and one K tile before introducing larger tiling or a pipeline.

## Minimal runnable example

[The matrix-tile probe](probes/warp_mma_and_async_copy.py) uses `warp.MmaF16BF16Op`, `cute.make_tiled_mma`, and two non-transposing `ldmatrix` atoms. A is stored as `(M,K)`, B as `(N,K)`, both with contiguous K. A's 16×16 shared tile uses the x4 member, B's 8×16 tile uses x2. The copies are linked to the MMA using `make_tiled_copy_A/B` and write through `retile` into the MMA-owned fragments. The FP32 accumulator starts at zero and writes through the thread's `partition_C`.

```sh
bash gpu_entry.sh cmd python3 /path/to/warp_mma_and_async_copy.py --dtype bf16
bash gpu_entry.sh cmd python3 /path/to/warp_mma_and_async_copy.py --dtype fp16
```

## Gotchas

- All 32 lanes participate in `ldmatrix` and `mma.sync`. Predicate the staging or writeback of partial tiles, or pad them. Keep the warp collectives converged.
- Shared-memory rows consumed by these `ldmatrix` atoms are 16-byte aligned. Preserve that alignment when changing row pitch or taking a displaced tile view.
- The probe checks one output tile, using CPU FP32 `A @ B.T` on exactly representable inputs. It establishes the fragment and coordinate mapping. Partial tiles, multiple K tiles, and arbitrary numeric tolerances need their own checks.
- BF16 and FP16 dumps from CuTe DSL 4.8.1 can share a name. Dump each dtype in a different directory.

## Sources

- [NVIDIA warp MMA programming guide](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/guides/mma/wmma_programming.html). Porting basis: SM90 `warp_mma_f16` and SM100 `warp_mma_ldsm_operand_displaced_view`.
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: BF16/FP16 with sync/async staging all returned `PROBE PASS max_diff=0.0`. PTX for each dtype contained `ldmatrix.sync` x4 and x2 and the corresponding `mma.sync.aligned.m16n8k16` dtype variant. The FP16 cubin showed `LDSM` and `HMMA.16816.F32`.
