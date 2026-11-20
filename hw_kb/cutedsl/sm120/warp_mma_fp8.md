# Unscaled FP8 warp MMA

## Purpose

Use `warp.MmaFP8Op` to multiply an `m16n8k32` tile with FP32 accumulation. Both operands use the same FP8 format, E4M3 or E5M2.

## Minimal runnable example

[The FP8 probe](probes/warp_mma_fp8.py) partitions each K-contiguous operand by the tiled MMA and copies it to the register fragment. This small control reads the fragments directly from global memory so it isolates the MMA contract.

```sh
bash gpu_entry.sh cmd python3 /path/to/warp_mma_fp8.py
```

## Gotchas

- Compare against the product of the already-quantized inputs converted to FP32. Comparing against the pre-quantization inputs mixes quantization error into the instruction check.
- The FP16/BF16 tile and fragment sizes differ. Use the selected FP8 atom's partition to obtain the register mapping.
- This uses no scale operands. For scale factors grouped along K, see [block-scaled warp MMA](block_scaled_warp_mma.md).
- The exactly representable probe inputs isolate correctness. Test application-level input ranges, quantization and tolerance separately.

## Sources

- [NVIDIA CuTe DSL warp API](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html).
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: E4M3 and E5M2 both returned `PROBE PASS max_diff=0.0`. The E5M2 PTX dump contained `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32`.
