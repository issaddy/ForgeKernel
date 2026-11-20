"""Check a non-multicast TMA load and zero fill at a descriptor boundary."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator


@cute.kernel
def tile_load(atom: cute.CopyAtom, coordinates: cute.Tensor, output: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    tile, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    barrier = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1)).iterator
    shared = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((2, 32), stride=(32, 1)), byte_alignment=128
    )
    source_tile = cute.local_tile(coordinates, (2, 32), (tile, 0))
    shared_view, source_view = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(shared, 0, 2),
        cute.group_modes(source_tile, 0, 2),
    )

    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_expect_tx(barrier, 2 * 32 * 2)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    cute.copy(atom, source_view, shared_view, tma_bar_ptr=barrier)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(barrier)
    cute.arch.mbarrier_wait(barrier, 0)
    for row in cutlass.range_constexpr(2):
        output[tile * 2 + row, lane] = shared[row, lane]


@cute.jit
def launch(source: cute.Tensor, output: cute.Tensor):
    layout = cute.make_layout((2, 32), stride=(32, 1))
    atom, coordinates = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), source, layout, (2, 32)
    )
    tile_load(atom, coordinates, output).launch(grid=(2, 1, 1), block=(32, 1, 1))


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")

    source = (torch.arange(96, device="cuda").reshape(3, 32) + 1).to(torch.bfloat16)
    output = torch.full((4, 32), float("nan"), device="cuda", dtype=torch.bfloat16)
    views = tuple(from_dlpack(t, assumed_align=16) for t in (source, output))
    cute.compile(launch, *views)(*views)
    expected = torch.cat((source, torch.zeros_like(source[:1])))
    for name, actual, wanted in [
        ("in_bounds", output[:2], expected[:2]),
        ("partial_tile_zero_fill", output[2:], expected[2:]),
    ]:
        error = (actual.float() - wanted.float()).abs().max().item()
        if not torch.equal(actual, wanted):
            raise AssertionError(f"PROBE FAIL max_diff={error}  # {name}")
        print(f"PROBE PASS max_diff={error}  # {name}")


if __name__ == "__main__":
    main()
