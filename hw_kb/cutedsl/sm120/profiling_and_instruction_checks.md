# Profiling and instruction checks on SM120

## When needed

Use this sequence when a candidate is slow, when a copy/compute variant has the same numerical output as its control, or when the device timing window tracks CPU enqueue time.

## Minimal runnable example

Run short, separate processes for environment checks, numerical verification, and profiling. An earlier launch failure can poison the CUDA context and make later results ambiguous.

```sh
bash gpu_entry.sh cmd python3 /path/to/warp_mma_and_async_copy.py --dtype bf16
bash gpu_entry.sh cmd 'ncu --csv --kernel-name "regex:.*matrix_tile.*" --metrics gpu__time_active.sum,launch__registers_per_thread python3 /path/to/warp_mma_and_async_copy.py --dtype bf16'
```

For code generation, run with `CUTE_DSL_KEEP_PTX=1`, `CUTE_DSL_KEEP_CUBIN=1`, and a dedicated `CUTE_DSL_DUMP_DIR`. Read PTX, then use `cuobjdump --dump-sass` on the emitted cubin. Match instruction families and check address space, vector width, or modifiers only when they carry the mechanism being tested. Use a same-shape control to catch false positives. The sync and async variants of the warp MMA probe provide one example.

## Interpreting the evidence

| Question | Evidence to collect |
| --- | --- |
| Did the intended mechanism lower? | Compiled PTX plus SASS family, with the work and launch matched across variants. |
| Did the path spill or run out of resident warps? | Registers per thread, grid/block shape, static/dynamic shared memory, local load/store sectors, achieved occupancy. Inspect local accesses before attributing all local traffic to spills. |
| Did staging improve the memory path? | DRAM/L2 traffic and shared bank conflicts, alongside elapsed GPU time at equal work. |
| Is the device window dominated by host submission? | Compare host enqueue, end-to-end, and device readings per workload. If graph capture is nonempty and covers the whole call, compare graph replay time. Alternatively use NCU over the full launch sequence and normalize by call count. |

A tiny probe verifies a mechanism and its instrumentation. Measure profitability with a representative workload and steady-state launch sizes. Keep timing and profiler runs separate so profiling replay does not become part of the benchmark.

For the BF16 matrix-tile probe, one NCU run on the RTX PRO 5000 recorded 26 registers per thread for the synchronous variant and 18 for the async variant. Both launches had one CTA of 32 threads and the same 16×8×16 tile. The run confirms that the profiling comparison works on this card. It does not establish a steady-state latency or throughput advantage.

## Sources

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) and [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html).
- Porting basis: SM90 `ptx_sass_census_criteria` and SM90/SM100 `gpu_probe_ladder` and `smem_traffic_triage`.
- On RTX PRO 5000 72GB Blackwell, NCU 2025.3.1 has returned kernel time and register counts for the BF16 copy and reduction probes. The broader diagnosis table is a measurement procedure, with conclusions conditional on each new workload.
