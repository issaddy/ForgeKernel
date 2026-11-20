# Grouped GEMM tile mapping and boundary handling

## Purpose

Map a flattened tile ID to one of several unequal GEMMs, then run BF16 warp MMA with FP32 accumulation on the selected group's tile.

## Minimal runnable example

[The grouped GEMM probe](probes/grouped_warp_gemm.py) packs each group into flat buffers, with separate base offsets for A, B and C. Group metadata carries its M, N, K and the first global tile ID. The kernel derives a local tile by subtracting the group's first ID, then uses that group's N extent to find the M/N tile coordinates.

```sh
bash gpu_entry.sh cmd python3 /path/to/grouped_warp_gemm.py
```

The probe processes four `(M,N,K)` shapes: `(1,3,1)`, `(17,19,23)`, `(33,9,16)` and `(4,17,29)`. It compares one CTA per tile against a three-CTA grid-stride launch over the same 16 tiles. Every K iteration zero-fills out-of-bounds staging elements. Output stores are masked by the group's bounds.

## Gotchas

- Advance each buffer's base by its actual packed element count. A group's next output address is independent of its tile count.
- Compute the tile count with ceil division over M and N. A tail in K affects the inner loop and staging masks, not the outer tile count.
- Keep all lanes converged for warp MMA, including boundary tiles. Predicate individual loads and stores.
- Reinitialize the accumulator for every tile. A grid-stride CTA can process different groups in successive iterations.
- This is a correctness baseline with a small group lookup. Measure scheduler overhead and per-CTA work imbalance before adopting it for a large number of experts. Routing, gate weights and fused MoE accumulation need separate checks.

## Sources

- [NVIDIA CUTLASS grouped scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html) and [CuTe DSL warp MMA guide](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/guides/mma/wmma_programming.html).
- Porting basis: SM90 `warp_mma_f16` and SM100 `launch_identity_stamp` for comparing launch mappings.
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: both schedules returned `PROBE PASS max_diff=0.0` against CPU FP32 products of the BF16 inputs. Both PTX files contained `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
