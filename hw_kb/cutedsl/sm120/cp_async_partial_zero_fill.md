# Partial cp.async with an explicit source byte count

## Purpose

Stage an aligned 16-byte fragment that may hold fewer than eight BF16 values. Use the PTX `src-size` operand so the remainder of the shared-memory fragment is initialized to zero.

## Minimal runnable example

[The async boundary probe](probes/async_copy_boundaries.py) gives each lane its own aligned fragment. It passes a runtime byte count of 0, 2, …, 14 to `cp.async.cg.shared.global`, commits and waits, and then reads shared memory. Shared memory is prefilled with a nonzero sentinel so a skipped store is observable.

```sh
bash gpu_entry.sh cmd python3 /path/to/async_copy_boundaries.py
```

## Gotchas

- Count **bytes**, not elements, in the instruction. This probe uses two bytes per BF16 element.
- Keep the source and destination base addresses aligned to the copy width. A partial count limits bytes read from an aligned fragment; alignment is still required.
- Use the regular full-width copy for a complete fragment. A predicate that skips an incomplete copy leaves the shared destination unchanged.
- Wait for group completion before reading the result. The probe tests valid in-range source pointers with variable valid byte counts. Use a valid aligned source pointer for the zero-byte case.

## Sources

- [PTX ISA non-bulk cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async). Porting basis: SM90 `cp_async_predicated_zfill`.
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: `PROBE PASS max_diff=0.0` across all eight partial counts. PTX retained a register in the source-size operand.
