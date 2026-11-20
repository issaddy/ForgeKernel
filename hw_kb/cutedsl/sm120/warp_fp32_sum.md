# Full-warp FP32 sum and BF16 store

## Purpose

Accumulate a full warp's lane-local values in FP32 and leave the sum in every participating lane. Narrow the final value when the output contract calls for BF16.

## When needed

One warp owns a reduction segment. Each lane can reduce its own input elements before the warp exchange. A larger tile spanning multiple warps needs a second reduction over warp partials.

## Minimal runnable example

Run [the probe](probes/bf16_copy_and_reductions.py) on an SM120 GPU through a generated card's job entry:

```sh
bash gpu_entry.sh cmd python3 /path/to/bf16_copy_and_reductions.py
```

The kernel widens BF16 input to FP32, adds peers at XOR deltas 1, 2, 4, 8, and 16 with `cute.arch.shuffle_sync_bfly`, then stores BF16. The probe checks every lane of 16 independent warps against the expected sum after conversion.

## Gotchas

- Keep all 32 lanes converged for the full-warp butterfly. For a partial row, feed zeros for out-of-range elements while keeping those lanes active at the shuffle.
- The shuffle exchanges registers within one warp. CTA-wide aggregation requires a separate, synchronized handoff.
- This probe uses exactly representable integer inputs to check lane mapping. Validate numerical tolerances separately for floating-point workloads with a different accumulation order.

## Sources

- [CuTe DSL architecture API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html), `cute.arch.shuffle_sync_bfly` and `cutlass.range_constexpr`. The full-warp shuffle pattern also appears in the SM90 `softmax_segment_ilp` recipe.
- GPU certification: [probe](probes/bf16_copy_and_reductions.py) on NVIDIA RTX PRO 5000 72GB Blackwell, compute capability 12.0, CuTe DSL 4.8.1. `PROBE PASS max_diff=0.0  # warp_fp32_sum_and_bf16_store`.
