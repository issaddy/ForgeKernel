"""Exercise aligned BF16 copies and FP32 reductions on SM120."""
import sys
from importlib.metadata import version

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator


@cute.kernel
def vector_copy(source: cute.Tensor, target: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    offset = cute.assume(block * 256 + lane * 8, divby=8)
    src = cute.make_tensor(source.iterator + offset, cute.make_layout(8))
    dst = cute.make_tensor(target.iterator + offset, cute.make_layout(8))
    registers = cute.make_rmem_tensor(8, cutlass.BFloat16)
    cute.autovec_copy(src, registers)
    cute.autovec_copy(registers, dst)


@cute.jit
def sum_lanes(value: cutlass.Float32):
    for delta in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, 1 << delta)
    return value


@cute.kernel
def warp_sum(source: cute.Tensor, target: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 32 + lane
    target[index] = sum_lanes(source[index].to(cutlass.Float32)).to(cutlass.BFloat16)


@cute.kernel
def cta_sum(source: cute.Tensor, target: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = thread % 32
    warp = thread // 32
    shared = SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout(4))
    partial = sum_lanes(source[block * 128 + thread].to(cutlass.Float32))
    if lane == 0:
        shared[warp] = partial
    cute.arch.sync_threads()
    if warp == 0:
        value = cutlass.Float32(0)
        if lane < 4:
            value = shared[lane]
        total = sum_lanes(value)
        if lane == 0:
            target[block] = total


@cute.jit
def launch_copy(source: cute.Tensor, target: cute.Tensor):
    vector_copy(source, target).launch(grid=(2, 1, 1), block=(32, 1, 1))


@cute.jit
def launch_sum(source: cute.Tensor, target: cute.Tensor):
    warp_sum(source, target).launch(grid=(16, 1, 1), block=(32, 1, 1))


@cute.jit
def launch_cta_sum(source: cute.Tensor, target: cute.Tensor):
    cta_sum(source, target).launch(grid=(4, 1, 1), block=(128, 1, 1))


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")

    # Vary both lanes and warps so indexing mistakes cannot pass with constant input.
    indices = torch.arange(512, device="cuda", dtype=torch.int32)
    source = ((indices % 23 - 11) * (1 + indices // 32)).to(torch.bfloat16)
    copied = torch.empty_like(source)
    summed = torch.empty_like(source)
    cta_summed = torch.empty(4, device="cuda", dtype=torch.float32)
    source_view = from_dlpack(source, assumed_align=16)
    copy_view = from_dlpack(copied, assumed_align=16)
    sum_view = from_dlpack(summed)
    cta_view = from_dlpack(cta_summed)

    cute.compile(launch_copy, source_view, copy_view)(source_view, copy_view)
    cute.compile(launch_sum, source_view, sum_view)(source_view, sum_view)
    cute.compile(launch_cta_sum, source_view, cta_view)(source_view, cta_view)
    torch.cuda.synchronize()

    expected_sum = source.float().reshape(-1, 32).sum(dim=1, keepdim=True)
    expected_sum = expected_sum.expand(-1, 32).reshape(-1).to(torch.bfloat16)
    print(torch.cuda.get_device_name(), "CuTe DSL", version("nvidia-cutlass-dsl"))
    for name, actual, expected in [
        ("aligned_bf16_copy", copied, source),
        ("warp_fp32_sum_and_bf16_store", summed, expected_sum),
        ("cta_fp32_sum", cta_summed, source.float().reshape(-1, 128).sum(dim=1)),
    ]:
        error = (actual.float() - expected.float()).abs().max().item()
        if not torch.equal(actual, expected):
            print(f"PROBE FAIL max_diff={error}  # {name}")
            return 1
        print(f"PROBE PASS max_diff={error}  # {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
