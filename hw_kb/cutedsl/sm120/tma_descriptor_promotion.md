# Driver-encoded TMA descriptors and L2 promotion

## Purpose

Vary L2 promotion independently of the shape and layout of a TMA transfer. Encode descriptors with the Driver API and keep the CuTe tile coordinates compatible with them.

## Minimal runnable example

[The descriptor probe](probes/tma_descriptor_promotion.py) uses a three-row BF16 view with logical width 32 and physical row pitch 48. The CuTe atom and external descriptors all fetch 2×32 elements and use the same transaction barrier. The probe compares the inline atom with driver-encoded promotion settings NONE, 128 B and 256 B.

```sh
bash gpu_entry.sh cmd python3 /path/to/tma_descriptor_promotion.py
```

The padded row pitch preserves a rank-2 descriptor. The encoder receives dimensions in contiguous-first order, physical row stride in bytes, and box sizes in elements. The map is uploaded before launch. `cute.copy` receives the external descriptor through `tma_desc_ptr` in the generic address space.

## Gotchas

- Encode addresses with the current source allocation. Reusing a map after its base pointer changes can read the old allocation.
- The atom's coordinate tensor must describe the same box and dimensionality as the external descriptor. Check in-bounds data and the descriptor boundary before collecting timings.
- Treat the map as opaque. Change L2 promotion through the driver enum.
- A fair cache comparison isolates promotion after confirming equivalent external maps. The probe verifies addressing, tail fill and legal promotion settings. For performance, compare paired external descriptors at workload size and record the cache state.

## Sources

- [CUDA Driver API tensor map encoder](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html). Porting basis: SM100 `tma_descriptor_l2_promotion`.
- GPU verification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: all four descriptor variants returned `PROBE PASS max_diff=0.0`, including the partial final tile.
