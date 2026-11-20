# Four-warp FP32 sum

## Purpose

Combine four independent warp sums through a small shared-memory exchange.

## When needed

A CTA owns a reduction tile and each of its four warps contributes one partial sum.

## Minimal runnable example

Run [the probe](probes/bf16_copy_and_reductions.py) on an SM120 GPU through a generated card's job entry:

```sh
bash gpu_entry.sh cmd python3 /path/to/bf16_copy_and_reductions.py
```

Lane 0 of each warp writes an FP32 partial to its own shared-memory slot. All 128 threads
reach `cute.arch.sync_threads()`. Warp 0 then reduces the four partials, with lanes 4–31
contributing zero. Its lane 0 stores the CTA result. The probe checks four independent
CTAs against sums over distinct input segments.

## Gotchas

- Every CTA thread must reach the barrier. Put the warp-0 consumer branch after it.
- The butterfly runs with all 32 lanes in warp 0. Guard shared-memory reads for lanes 4–31.
- For a loop reusing the shared slots, synchronize readers and writers before the next overwrite.
- The tested geometry is four full warps. Resize the shared layout and consumer mask together for another geometry.

## Sources

- [CuTe DSL kernel launch and allocator documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) and [architecture API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html), `cutlass.memory.SmemAllocator` and `cute.arch.sync_threads`.
- GPU certification: [probe](probes/bf16_copy_and_reductions.py) on NVIDIA RTX PRO 5000 72GB Blackwell, compute capability 12.0, CuTe DSL 4.8.1. `PROBE PASS max_diff=0.0  # cta_fp32_sum`.
