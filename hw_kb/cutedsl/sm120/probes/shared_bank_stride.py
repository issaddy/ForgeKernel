"""Compare plain, padded, and swizzled FP32 shared-memory reads at equal work."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

BLOCKS = 128
WIDTH = 32


@cute.kernel
def stride_probe(source: cute.Tensor, output: cute.Tensor, pitch: cutlass.Constexpr,
                 swizzled: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    block, _, _ = cute.arch.block_idx()
    # All variants reserve the same storage, so resource usage is controlled.
    storage = SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout(WIDTH * (WIDTH + 1)), byte_alignment=16
    )
    layout = cute.make_layout((WIDTH, WIDTH), stride=(pitch, 1))
    if cutlass.const_expr(swizzled):
        layout = cute.make_composed_layout(cute.make_swizzle(5, 0, 5), 0, layout)
    tile = cute.make_tensor(storage.iterator, layout)
    for row in cutlass.range(WIDTH):
        tile[row, lane] = source[block, row, lane]
    cute.arch.sync_threads()
    # Each lane reads a distinct row of one column. This defeats a broadcast and
    # forces the cross-lane handoff through shared memory.
    total = cutlass.Float32(0)
    for column in cutlass.range(WIDTH):
        total += tile[lane, column]
    output[block, lane] = total


@cute.jit
def launch(source: cute.Tensor, output: cute.Tensor, pitch: cutlass.Constexpr,
           swizzled: cutlass.Constexpr):
    stride_probe(source, output, pitch, swizzled).launch(grid=(BLOCKS, 1, 1), block=(WIDTH, 1, 1))


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    source = torch.arange(BLOCKS * WIDTH * WIDTH, device="cuda", dtype=torch.int32)
    source = (source % 191 - 95).reshape(BLOCKS, WIDTH, WIDTH).float()
    output = torch.empty((BLOCKS, WIDTH), device="cuda", dtype=torch.float32)
    views = tuple(from_dlpack(t, assumed_align=16) for t in (source, output))
    expected = source.sum(dim=-1)
    for name, pitch, swizzled in (("pitch_32", 32, False), ("pitch_33", 33, False),
                                  ("xor_swizzle", 32, True)):
        output.fill_(float("nan"))
        cute.compile(launch, *views, pitch, swizzled)(*views)
        error = (output - expected).abs().max().item()
        if not torch.equal(output, expected):
            raise AssertionError(f"PROBE FAIL max_diff={error}  # {name}")
        print(f"PROBE PASS max_diff={error}  # {name}")


if __name__ == "__main__":
    main()
