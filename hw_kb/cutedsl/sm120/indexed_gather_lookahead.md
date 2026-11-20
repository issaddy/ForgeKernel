# Indexed row staging with lookahead

## Purpose

Stage an irregular row selected by one warp-uniform index. Compare loading the next row's index before issuing the current transfer versus loading each index just before its transfer.

## Minimal runnable example

[The BF16 gather probe](probes/indexed_gather_lookahead.py) uses 1,031 source rows, 4,096 output rows, width 256, and four rows per CTA. Lane 0 loads each index and broadcasts it with a shuffle. Valid rows use aligned 16-byte `cp.async` groups; invalid indices select a zero-byte copy from a valid aligned source. The test has repeated indices and both negative and out-of-range positions.

```sh
bash gpu_entry.sh cmd python3 /path/to/indexed_gather_lookahead.py
bash gpu_entry.sh cmd 'ncu --csv --kernel-name "regex:.*cutlass_gather.*" --launch-count 10 --metrics gpu__time_active.sum,launch__registers_per_thread python3 /path/to/indexed_gather_lookahead.py'
```

## Observed on SM120

RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1, NCU 2025.3.1. Each path passed with `PROBE PASS max_diff=0.0`. Five profiled launches per path gave:

| Index schedule | GPU active time, median (range) | Registers/thread |
| --- | ---: | ---: |
| Load index at use | 5.312 µs (5.184–5.984) | 22 |
| One-row lookahead | 5.280 µs (5.024–5.792) | 24 |

The ranges overlap. Before choosing lookahead, test an index distribution and row width representative of the workload and inspect whether the generated schedule moves the index load ahead.

## Gotchas

- Look ahead only when a subsequent row is in the CTA's range.
- A negative or too-large index must never be used to form a global row pointer. Explicit zero fill also prevents stale shared values when reusing the slot.
- This compares index lead distance. It uses one data staging slot, so it does not compare overlapped groups of row transfers.

## Sources

- Porting basis: SM90 `gather_index_prefetch_predicated` and SM100 `gather_index_lane_owned`.
- [PTX cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) and [partial zero fill probe](cp_async_partial_zero_fill.md).
