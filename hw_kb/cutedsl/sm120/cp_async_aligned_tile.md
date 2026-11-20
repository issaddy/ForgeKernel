# Aligned cp.async staging and CTA handoff

## Purpose

Stage complete 16-byte groups from global into shared memory with per-thread `cp.async`. Make the complete tile available to warp consumers.

## Minimal runnable example

Run [the matrix-tile probe](probes/warp_mma_and_async_copy.py). It keeps layout, input, MMA, and output fixed while specializing the staging atom to `CopyUniversalOp` or `cpasync.CopyG2SOp`.

```sh
bash gpu_entry.sh cmd python3 /path/to/warp_mma_and_async_copy.py --dtype bf16
```

Each issuing lane commits its own copies. The async path calls `cute.arch.cp_async_commit_group()` and `cp_async_wait_group(0)` before the CTA rendezvous. This probe uses the stock non-bulk wrappers in CuTe DSL 4.8.1.

## Gotchas

- Group completion is per issuing thread. Use the CTA barrier before consuming data staged by other threads.
- This entry certifies complete aligned groups. A predicate that skips the issue leaves the destination unchanged. Test explicit zero-filling before using the path for an incomplete tile.
- The probe drains a single stage before compute. An overlapped ring also needs a slot reuse rule so producers cannot overwrite in-use shared memory.

## Sources

- [CUDA PTX async copy instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) and [CuTe DSL architecture API](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_arch.html). Porting basis: SM90 `cp_async`.
- On RTX PRO 5000 72GB Blackwell with CuTe DSL 4.8.1, both controls and async variants returned `PROBE PASS max_diff=0.0` for FP16 and BF16. The async PTX contained two 16-byte `cp.async.ca` sites, commit, and wait. The FP16 async cubin showed `LDGSTS.E.128`; the synchronous control had no `LDGSTS`.
