"""Check register-fragment MMA and synchronous/asynchronous staging on SM120."""
import argparse

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator


@cute.jit
def stage_vector(atom: cute.CopyAtom, src: cute.Tensor, dst: cute.Tensor, lane):
    offset = cute.assume(lane * 8, divby=8)
    group = cute.make_layout(8)
    cute.copy(
        atom,
        cute.make_tensor(src.iterator + offset, group),
        cute.make_tensor(dst.iterator + offset, group),
    )


@cute.kernel
def matrix_tile(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor, use_async: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    allocator = SmemAllocator()
    shared_a = allocator.allocate_tensor(
        a.element_type, cute.make_layout((16, 16), stride=(16, 1)), byte_alignment=16
    )
    shared_b = allocator.allocate_tensor(
        b.element_type, cute.make_layout((8, 16), stride=(16, 1)), byte_alignment=16
    )
    if cutlass.const_expr(use_async):
        stage = cute.make_copy_atom(cpasync.CopyG2SOp(), a.element_type, num_bits_per_copy=128)
    else:
        stage = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), a.element_type, num_bits_per_copy=128)
    stage_vector(stage, a, shared_a, lane)
    if lane < 16:
        stage_vector(stage, b, shared_b, lane)
    if cutlass.const_expr(use_async):
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(a.element_type, cutlass.Float32, (16, 8, 16)))
    thread_mma = mma.get_slice(lane)
    frag_a = thread_mma.make_fragment_A(thread_mma.partition_A(shared_a))
    frag_b = thread_mma.make_fragment_B(thread_mma.partition_B(shared_b))
    load_a = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), a.element_type), mma
    )
    load_b = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), b.element_type), mma
    )
    thread_load_a = load_a.get_slice(lane)
    thread_load_b = load_b.get_slice(lane)
    cute.copy(load_a, thread_load_a.partition_S(shared_a), thread_load_a.retile(frag_a))
    cute.copy(load_b, thread_load_b.partition_S(shared_b), thread_load_b.retile(frag_b))

    out = thread_mma.partition_C(output)
    acc = thread_mma.make_fragment_C(out)
    acc.fill(0.0)
    cute.gemm(mma, acc, frag_a, frag_b, acc)
    cute.autovec_copy(acc, out)


@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor, use_async: cutlass.Constexpr):
    matrix_tile(a, b, output, use_async).launch(grid=(1, 1, 1), block=(32, 1, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "both"), default="both")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    # Small integers make the FP32 result exact. Distinct nonconstant faces test
    # both the operand and output coordinate mappings.
    dtypes = {"bf16": (torch.bfloat16,), "fp16": (torch.float16,),
              "both": (torch.bfloat16, torch.float16)}
    for dtype in dtypes[args.dtype]:
        a = (torch.arange(256, device="cuda").reshape(16, 16) % 11 - 5).to(dtype)
        b = (torch.arange(128, device="cuda").reshape(8, 16) % 7 - 3).to(dtype)
        expected = a.cpu().float() @ b.cpu().float().T
        out = torch.empty((16, 8), device="cuda", dtype=torch.float32)
        views = tuple(from_dlpack(t, assumed_align=16) for t in (a, b, out))
        for use_async in (False, True):
            out.fill_(float("nan"))
            compiled = cute.compile(launch, *views, use_async)
            compiled(*views)
            actual = out.cpu()
            name = f"{dtype}_{'async' if use_async else 'sync'}"
            error = (actual - expected).abs().max().item()
            if not torch.equal(actual, expected):
                raise AssertionError(f"PROBE FAIL max_diff={error}  # {name}")
            print(f"PROBE PASS max_diff={error}  # {name}")


if __name__ == "__main__":
    main()
