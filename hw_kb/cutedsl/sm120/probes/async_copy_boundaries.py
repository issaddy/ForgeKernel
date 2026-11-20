"""Exercise non-bulk async tail zero-fill and TMA store slot reuse."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.memory import SmemAllocator


@dsl_user_op
def copy_partial(dst, src, size, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(dst.toint()).ir_value(loc=loc, ip=ip),
            cutlass.Int64(src.toint()).ir_value(loc=loc, ip=ip),
            cutlass.Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r,~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.kernel
def zero_fill(source: cute.Tensor, lengths: cute.Tensor, out: cute.Tensor):
    lane = cute.arch.lane_idx()
    shared = SmemAllocator().allocate_tensor(
        cutlass.BFloat16, cute.make_layout((32, 8), stride=(8, 1)), byte_alignment=16
    )
    for col in cutlass.range_constexpr(8):
        shared[lane, col] = cutlass.BFloat16(-99)
    cute.arch.fence_view_async_shared()
    offset = cute.assume(lane * 8, divby=8)
    copy_partial(shared.iterator + offset, source.iterator + offset, lengths[lane] * 2)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    for col in cutlass.range_constexpr(8):
        out[lane, col] = shared[lane, col]


@cute.kernel
def store_tiles(atom: cute.CopyAtom, coords: cute.Tensor, source: cute.Tensor):
    lane = cute.arch.lane_idx()
    shared = SmemAllocator().allocate_tensor(
        cutlass.BFloat16, cute.make_layout((2, 32), stride=(32, 1)), byte_alignment=128
    )
    for tile in cutlass.range(2):
        for row in cutlass.range_constexpr(2):
            shared[row, lane] = source[tile * 2 + row, lane]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        dst = cute.local_tile(coords, (2, 32), (tile, 0))
        shared_view, global_view = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), cute.group_modes(shared, 0, 2),
            cute.group_modes(dst, 0, 2),
        )
        cute.copy(atom, shared_view, global_view)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)
        cute.arch.sync_threads()


@cute.jit
def launch_zero_fill(source: cute.Tensor, lengths: cute.Tensor, output: cute.Tensor):
    zero_fill(source, lengths, output).launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.jit
def launch_store(source: cute.Tensor, output: cute.Tensor):
    atom, coords = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), output,
        cute.make_layout((2, 32), stride=(32, 1)), (2, 32),
    )
    store_tiles(atom, coords, source).launch(grid=(1, 1, 1), block=(32, 1, 1))


def check(fn, inputs, expected, tag):
    output = torch.full_like(expected, float("nan"))
    views = tuple(from_dlpack(t, assumed_align=16) for t in (*inputs, output))
    cute.compile(fn, *views)(*views)
    error = (output.float() - expected.float()).abs().max().item()
    if not torch.equal(output, expected):
        raise AssertionError(f"PROBE FAIL max_diff={error}  # {tag}")
    print(f"PROBE PASS max_diff={error}  # {tag}")


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    source = (torch.arange(256, device="cuda").reshape(32, 8) + 1).to(torch.bfloat16)
    lengths = (torch.arange(32, device="cuda", dtype=torch.int32) % 8)
    columns = torch.arange(8, device="cuda")
    masked = source.masked_fill(columns[None, :] >= lengths[:, None], 0)
    check(launch_zero_fill, (source, lengths), masked, "cp_async_partial_0_to_7")
    tile_data = source.flatten()[:128].reshape(4, 32)
    check(launch_store, (tile_data,), tile_data, "tma_store_reused_slot")


if __name__ == "__main__":
    main()
