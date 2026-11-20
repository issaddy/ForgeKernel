# Aligned BF16 vector copy

## Purpose

Use `cute.autovec_copy` to move eight contiguous BF16 values per lane between global memory and a compact register fragment.

## When needed

Adjacent lanes handle adjacent groups of eight BF16 values, and every group starts at a 16-byte-aligned address. This is the aligned interior path for a contiguous row. Handle an incomplete group with a separate guarded path.

## Minimal runnable example

Run [the probe](probes/bf16_copy_and_reductions.py) on an SM120 GPU through a generated card's job entry:

```sh
bash gpu_entry.sh cmd python3 /path/to/bf16_copy_and_reductions.py
```

The copy kernel gives each lane a disjoint group of eight values, uses `cute.assume(offset, divby=8)` on the element offset, and calls `cute.autovec_copy` for both directions. The probe compares the entire output with the input bit for bit.

## Gotchas

- `from_dlpack(..., assumed_align=16)` and `cute.assume` describe guarantees already established by the caller. A sliced tensor or a row stride that is not a multiple of eight BF16 elements can break them.
- `autovec_copy` establishes a copy pattern. Inspect generated code when instruction width is an optimization requirement.
- Certification covers a contiguous BF16 layout with complete groups. Benchmark tail handling and other layouts independently.

## Sources

- [CuTe DSL tensor operations](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html), `cute.autovec_copy`, `cute.assume`, `cute.make_rmem_tensor`, and `cutlass.cute.runtime.from_dlpack`.
- GPU certification: [probe](probes/bf16_copy_and_reductions.py) on NVIDIA RTX PRO 5000 72GB Blackwell, compute capability 12.0, CuTe DSL 4.8.1. `PROBE PASS max_diff=0.0  # aligned_bf16_copy`.
