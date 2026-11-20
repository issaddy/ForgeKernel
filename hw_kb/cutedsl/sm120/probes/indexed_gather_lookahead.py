"""Check indexed BF16 staging with a one-row-ahead index load."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

from async_copy_boundaries import copy_partial

ROWS = 1031
WIDTH = 256
GROUP = 4
OUTPUT_ROWS = 4096


@cute.jit
def row_index(indices: cute.Tensor, row, lane):
    index = cutlass.Int32(0)
    if lane == 0:
        index = indices[row]
    return cute.arch.shuffle_sync(index, 0)


@cute.kernel
def gather(source: cute.Tensor, indices: cute.Tensor, output: cute.Tensor,
           lookahead: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    block, _, _ = cute.arch.block_idx()
    shared = SmemAllocator().allocate_tensor(
        cutlass.BFloat16, cute.make_layout(WIDTH), byte_alignment=16
    )
    first_row = block * GROUP
    offset = cute.assume(lane * 8, divby=8)
    stage = shared.iterator + offset
    forward = cutlass.Int32(0)
    if cutlass.const_expr(lookahead):
        forward = row_index(indices, first_row, lane)

    for step in cutlass.range_constexpr(GROUP):
        row = first_row + step
        if cutlass.const_expr(lookahead):
            index = forward
            if cutlass.const_expr(step + 1 < GROUP):
                forward = row_index(indices, row + 1, lane)
        else:
            index = row_index(indices, row, lane)
        src_row = cutlass.Int32(0)
        if 0 <= index and index < ROWS:
            src_row = index
            # Full-width non-bulk copy from an aligned fragment.
            src = source.iterator + cute.assume(src_row * WIDTH + offset, divby=8)
            vector = cute.make_layout(8)
            atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), cutlass.BFloat16,
                                       num_bits_per_copy=128)
            cute.copy(atom, cute.make_tensor(src, vector), cute.make_tensor(stage, vector))
        else:
            copy_partial(stage, source.iterator + offset, 0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        # Each lane reads only the fragment it staged.
        for col in cutlass.range_constexpr(8):
            output[row, lane * 8 + col] = shared[lane * 8 + col]


@cute.jit
def launch(source: cute.Tensor, indices: cute.Tensor, output: cute.Tensor,
           lookahead: cutlass.Constexpr):
    gather(source, indices, output, lookahead).launch(
        grid=(OUTPUT_ROWS // GROUP, 1, 1), block=(32, 1, 1)
    )


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    ids = torch.arange(OUTPUT_ROWS, device="cuda", dtype=torch.int32)
    indices = ((ids * 997) % (ROWS + 3) - 1).contiguous()
    source = torch.arange(ROWS * WIDTH, device="cuda", dtype=torch.int32)
    source = (source % 239 + 1).reshape(ROWS, WIDTH).to(torch.bfloat16)
    valid = (indices >= 0) & (indices < ROWS)
    expected = source[indices.clamp(0, ROWS - 1).long()].masked_fill(~valid[:, None], 0)
    output = torch.empty_like(expected)
    views = tuple(from_dlpack(t, assumed_align=16) for t in (source, indices, output))
    for lookahead in (False, True):
        output.fill_(float("nan"))
        fn = cute.compile(launch, *views, lookahead)
        for _ in range(5):
            fn(*views)
        error = (output.float() - expected.float()).abs().max().item()
        if not torch.equal(output, expected):
            raise AssertionError(f"PROBE FAIL max_diff={error} # lookahead={lookahead}")
        print(f"PROBE PASS max_diff={error} # lookahead={lookahead}")


if __name__ == "__main__":
    main()
