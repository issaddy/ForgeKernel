"""Exercise one- and two-stage TMA slot reuse across changing barrier phases."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

TILES = 7
ROWS = 2
COLS = 32


@cute.jit
def issue(atom, coords, slot, barrier, tile):
    gmem = cute.local_tile(coords, (ROWS, COLS), (tile, 0))
    shared_view, global_view = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(slot, 0, 2),
        cute.group_modes(gmem, 0, 2),
    )
    with cute.arch.elect_one():
        cute.arch.mbarrier_expect_tx(barrier, ROWS * COLS * 2)
    cute.arch.sync_warp()
    cute.copy(atom, global_view, shared_view, tma_bar_ptr=barrier)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(barrier)


@cute.kernel
def copy_ring(atom: cute.CopyAtom, coords: cute.Tensor, output: cute.Tensor,
              stages: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    smem = SmemAllocator()
    barriers = smem.allocate_tensor(cutlass.Int64, cute.make_layout(stages)).iterator
    storage = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((ROWS, COLS, stages), stride=(COLS, 1, ROWS * COLS)),
        byte_alignment=128,
    )
    with cute.arch.elect_one():
        for stage in cutlass.range_constexpr(stages):
            cute.arch.mbarrier_init(barriers + stage, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    for stage in cutlass.range_constexpr(stages):
        issue(atom, coords, storage[None, None, stage], barriers + stage, stage)

    for tile in cutlass.range(TILES):
        stage = tile % stages
        phase = (tile // stages) % 2
        slot = storage[None, None, stage]
        cute.arch.mbarrier_wait(barriers + stage, phase)
        for row in cutlass.range_constexpr(ROWS):
            output[tile * ROWS + row, lane] = slot[row, lane]
        cute.arch.sync_threads()
        if tile + stages < TILES:
            issue(atom, coords, slot, barriers + stage, tile + stages)


@cute.kernel
def copy_ring_split_warps(atom: cute.CopyAtom, coords: cute.Tensor, output: cute.Tensor):
    stages = 2
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = SmemAllocator()
    full = smem.allocate_tensor(cutlass.Int64, cute.make_layout(stages)).iterator
    empty = smem.allocate_tensor(cutlass.Int64, cute.make_layout(stages)).iterator
    storage = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((ROWS, COLS, stages), stride=(COLS, 1, ROWS * COLS)),
        byte_alignment=128,
    )
    if warp == 0:
        with cute.arch.elect_one():
            for stage in cutlass.range_constexpr(stages):
                cute.arch.mbarrier_init(full + stage, 1)
                cute.arch.mbarrier_init(empty + stage, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    if warp == 0:
        for tile in cutlass.range(TILES):
            stage = tile % stages
            if tile >= stages:
                # The consumer must release the previous generation of this slot.
                cute.arch.mbarrier_wait(empty + stage, (tile // stages - 1) % 2)
            issue(atom, coords, storage[None, None, stage], full + stage, tile)
    else:
        for tile in cutlass.range(TILES):
            stage = tile % stages
            cute.arch.mbarrier_wait(full + stage, (tile // stages) % 2)
            slot = storage[None, None, stage]
            for row in cutlass.range_constexpr(ROWS):
                output[tile * ROWS + row, lane] = slot[row, lane]
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(empty + stage)


@cute.jit
def launch(source: cute.Tensor, output: cute.Tensor, stages: cutlass.Constexpr,
           split_warps: cutlass.Constexpr):
    face = cute.make_layout((ROWS, COLS), stride=(COLS, 1))
    atom, coords = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), source, face, (ROWS, COLS)
    )
    if cutlass.const_expr(split_warps):
        copy_ring_split_warps(atom, coords, output).launch(grid=(1, 1, 1), block=(64, 1, 1))
    else:
        copy_ring(atom, coords, output, stages).launch(grid=(1, 1, 1), block=(32, 1, 1))


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    # The last tile is incomplete. Distinct row patterns expose stale slot reads.
    source = torch.arange((TILES * ROWS - 1) * COLS, device="cuda")
    source = (source % 251 + 1).reshape(-1, COLS).to(torch.bfloat16)
    expected = torch.cat((source, torch.zeros_like(source[:1])))
    output = torch.empty_like(expected)
    views = tuple(from_dlpack(t, assumed_align=16) for t in (source, output))
    for name, stages, split_warps in (
        ("one_slot", 1, False), ("two_slots", 2, False), ("split_warps", 2, True)
    ):
        output.fill_(float("nan"))
        cute.compile(launch, *views, stages, split_warps)(*views)
        error = (output.float() - expected.float()).abs().max().item()
        if not torch.equal(output, expected):
            raise AssertionError(f"PROBE FAIL max_diff={error}  # {name}")
        print(f"PROBE PASS max_diff={error}  # {name}")


if __name__ == "__main__":
    main()
