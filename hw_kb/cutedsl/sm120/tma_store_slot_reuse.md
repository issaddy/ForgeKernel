# TMA store and shared slot reuse

## Purpose

Write a shared tile to global memory with a TMA store. Publish the shared-memory writes to the async proxy and wait before overwriting the slot.

## Minimal runnable example

[The async boundary probe](probes/async_copy_boundaries.py) uses `CopyBulkTensorTileS2GOp` with one warp, one 2×32 BF16 shared tile and two distinct destination tiles. For each tile it fills shared memory, executes `fence_view_async_shared()` and a CTA rendezvous, then calls the warp-uniform CuTe copy. A bulk commit and wait followed by a CTA rendezvous protect the slot before reuse.

```sh
bash gpu_entry.sh cmd python3 /path/to/async_copy_boundaries.py
```

## Gotchas

- The proxy fence publishes ordinary shared stores to the async proxy. The CTA barrier coordinates the threads that produced those stores.
- Let the CuTe TMA copy elect its issuing lane from the converged warp. Use the **bulk** commit and wait wrappers for this store.
- The test drains every store before reuse, which is a simple correctness baseline. An overlapped epilogue needs separate slot ownership and retirement rules.
- The probe covers two complete, aligned boxes. Strides, swizzled faces, and partial store boxes need matching descriptors and their own checks.

## Sources

- [CuTe DSL TMA store API](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html) and [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). Porting basis: SM90 `tma_store_epilogue`.
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: `PROBE PASS max_diff=0.0` for the reused slot. PTX showed two `cp.async.bulk.tensor.1d.global.shared::cta` store sites.
