# TMA stage reuse and barrier phase rollover

## Purpose

Queue multiple non-multicast TMA loads, then reuse a shared-memory slot only after its current consumers finish. The barrier phase disambiguates successive transfers to one slot.

## Minimal runnable example

[The stage-reuse probe](probes/tma_stage_reuse.py) compares one slot, two slots, and a two-slot producer/consumer-warp variant using the same seven BF16 tiles.

```sh
TASK_TIMEOUT=45 bash gpu_entry.sh cmd python3 /path/to/tma_stage_reuse.py
```

At startup, allocate one transaction barrier per slot and fill all slots. In tile order:

1. Derive the slot as `tile % stages` and its current phase as `(tile // stages) % 2`.
2. Wait on that slot's transaction barrier, then read the tile.
3. Synchronize the consuming CTA before issuing the next tile for that slot. If all remaining tiles are already issued, drain the in-flight slots.

The one-slot variant issues the next load only after consuming the current tile. The two-slot variant can have a load pending while processing the preceding tile. Each slot uses the publication sequence from [TMA tile load](tma_tile_load.md).

In the split-warp variant each slot has a full transaction barrier and an empty barrier. Warp 0 issues TMA. Before reusing a slot it waits for the empty phase of the previous generation. Warp 1 waits for the full phase of the current generation, reads and stores the tile, then elects one lane to signal empty after a warp rendezvous.

## Gotchas

- A phase alternates per **reuse of the same slot**. Incrementing it once per tile is wrong for multiple slots.
- Account for all bytes in the descriptor box. The last tile in the probe includes an out-of-bounds row, and still completes the full-byte transaction.
- The split-warp probe has exactly one consumer warp. With multiple consumers, set the empty barrier's arrival count to match the release protocol, then verify every consumer releases before reuse.
- The three variants verify ordering and the opportunity for overlap. Measure the actual overlap and its value on the target workload.

## Sources

- [CUTLASS pipeline synchronization](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) and [CuTe DSL pipeline API](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/pipeline.html).
- Porting basis: SM100 `mbarrier_ring_rolled_loop` for stage/phase indexing and SM90 `cp_async_mbarrier_ring` for consumption and slot reuse.
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: `PROBE PASS max_diff=0.0` for all three variants across seven tiles, including the partial final tile.
