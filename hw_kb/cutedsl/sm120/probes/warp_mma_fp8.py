"""Check an unscaled FP8 m16n8k32 warp MMA on SM120."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def matrix_tile(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor):
    lane = cute.arch.lane_idx()
    op = warp.MmaFP8Op(a.element_type, cutlass.Float32, (16, 8, 32))
    mma = cute.make_tiled_mma(op)
    thread_mma = mma.get_slice(lane)
    part_a = thread_mma.partition_A(a)
    part_b = thread_mma.partition_B(b)
    frag_a = thread_mma.make_fragment_A(part_a)
    frag_b = thread_mma.make_fragment_B(part_b)
    cute.autovec_copy(part_a, frag_a)
    cute.autovec_copy(part_b, frag_b)

    out = thread_mma.partition_C(output)
    acc = thread_mma.make_fragment_C(out)
    acc.fill(0.0)
    cute.gemm(mma, acc, frag_a, frag_b, acc)
    cute.autovec_copy(acc, out)


@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor):
    matrix_tile(a, b, output).launch(grid=(1, 1, 1), block=(32, 1, 1))


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        a = (torch.arange(512, device="cuda").reshape(16, 32) % 11 - 5).to(dtype)
        b = (torch.arange(256, device="cuda").reshape(8, 32) % 7 - 3).to(dtype)
        expected = a.cpu().float() @ b.cpu().float().T
        output = torch.full((16, 8), float("nan"), device="cuda", dtype=torch.float32)
        views = tuple(from_dlpack(t, assumed_align=16) for t in (a, b, output))
        cute.compile(launch, *views)(*views)
        actual = output.cpu()
        error = (actual - expected).abs().max().item()
        if not torch.equal(actual, expected):
            raise AssertionError(f"PROBE FAIL max_diff={error}  # {dtype}")
        print(f"PROBE PASS max_diff={error}  # {dtype}")


if __name__ == "__main__":
    main()
