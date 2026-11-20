# TMA tile load with an mbarrier

## Purpose

Move a BF16 tile into one CTA's shared memory via a tensor map. A transaction barrier protects the shared-memory consumer and the tensor map supplies zero fill for the tested boundary.

## Minimal runnable example

Run [the TMA probe](probes/tma_tile_load.py) through a generated card on an SM120 GPU.

```sh
TASK_TIMEOUT=30 bash gpu_entry.sh cmd python3 /path/to/tma_tile_load.py
```

The probe builds a non-multicast G2S atom, partitions the coordinate tensor and shared tile, and copies a 2×32 BF16 box. An elected lane initializes the mbarrier and registers 128 expected bytes. The warp collectively reaches the CuTe copy site. An elected lane then arrives on the barrier. The consumers wait for phase 0 before reading shared memory.

## Gotchas

- Use the coordinate tensor returned by `make_tiled_tma_atom`. Its descriptor basis can merge contiguous axes. Here a full-width tile lowers to a 1D transfer and the final incomplete tile ends at the descriptor boundary.
- Register the full box size in bytes, including the out-of-bounds region that the hardware fills. The test copies three physical rows into four logical rows and checks the last row is zero.
- Let CuTe handle issuing at the warp-uniform TMA copy site. In an earlier probe variant, combining a lane-0 guard around the copy with a different barrier-arrival sequence timed out. The verified sequence uses separate `expect_tx` and `arrive`. The experiment did not isolate which change was sufficient.
- The probe covers one transfer per CTA, no multicast, and one mbarrier phase. For a reused pipeline slot, test phase rollover and buffer ownership independently.

## Sources

- [NVIDIA CuTe DSL TMA tutorial](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v0.py) and [PTX TMA instruction documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor). Porting basis: SM90 `tma_ndim_oob_zerofill` and SM100 `mbarrier_ring_rolled_loop`.
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: `PROBE PASS max_diff=0.0` for the full tile and partial tile. The dump contained `cp.async.bulk.tensor.1d.shared::cta` and `mbarrier.try_wait.parity`.
