"""Compare direct BF16 GEMM writeback against an stmatrix shared staging path."""
import argparse

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

M = 512
N = 512
K = 128
TM = 16
TN = 8
TK = 16


@cute.kernel
def writeback(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, use_stmatrix: cutlass.Constexpr,
              warps: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    block_n, block_m, _ = cute.arch.block_idx()
    warp_index = cute.arch.warp_idx()
    row_tile = block_m * warps + warp_index
    g_a = cute.local_tile(a, (TM, TK), (row_tile, None))
    g_b = cute.local_tile(b, (TN, TK), (block_n, None))
    g_out = cute.local_tile(out, (TM, TN), (row_tile, block_n))
    smem = SmemAllocator()
    shared_a = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((TM, TK, warps), stride=(TK, 1, TM * TK)), byte_alignment=16,
    )[None, None, warp_index]
    shared_b = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((TN, TK, warps), stride=(TK, 1, TN * TK)), byte_alignment=16,
    )[None, None, warp_index]
    shared_out = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((TM, TN, warps), stride=(TN, 1, TM * TN)), byte_alignment=16,
    )[None, None, warp_index]
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (TM, TN, TK)))
    thr = mma.get_slice(lane)
    frag_a = thr.make_fragment_A(thr.partition_A(shared_a))
    frag_b = thr.make_fragment_B(thr.partition_B(shared_b))
    copy_a = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16), mma
    )
    copy_b = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16), mma
    )
    thr_a = copy_a.get_slice(lane)
    thr_b = copy_b.get_slice(lane)
    out_fragment = thr.partition_C(g_out)
    acc = thr.make_fragment_C(out_fragment)
    acc.fill(0.0)

    for step in cutlass.range(a.shape[1] // TK):
        for vector in cutlass.range_constexpr(TM * TK // 32):
            index = vector * 32 + lane
            row, col = index // TK, index % TK
            shared_a[row, col] = g_a[row, col, step]
        for vector in cutlass.range_constexpr(TN * TK // 32):
            index = vector * 32 + lane
            row, col = index // TK, index % TK
            shared_b[row, col] = g_b[row, col, step]
        cute.arch.sync_threads()
        cute.copy(copy_a, thr_a.partition_S(shared_a), thr_a.retile(frag_a))
        cute.copy(copy_b, thr_b.partition_S(shared_b), thr_b.retile(frag_b))
        cute.gemm(mma, acc, frag_a, frag_b, acc)
        cute.arch.sync_threads()

    result = cute.make_rmem_tensor_like(acc, cutlass.BFloat16)
    result.store((acc.load() * 1.25 + 0.5).to(cutlass.BFloat16))
    if cutlass.const_expr(use_stmatrix):
        store = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), cutlass.BFloat16), mma
        )
        thread_store = store.get_slice(lane)
        cute.copy(store, thread_store.retile(result), thread_store.partition_D(shared_out))
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(TM * TN // 32):
            index = i * 32 + lane
            row, col = index // TN, index % TN
            g_out[row, col] = shared_out[row, col]
    else:
        cute.autovec_copy(result, out_fragment)


@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, use_stmatrix: cutlass.Constexpr,
           warps: cutlass.Constexpr):
    writeback(a, b, out, use_stmatrix, warps).launch(
        grid=(b.shape[0] // TN, a.shape[0] // (TM * warps), 1), block=(32 * warps, 1, 1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-warp", action="store_true")
    parser.add_argument("--max-registers", type=int)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    m, n, k, warps = (1024, 1536, 256, 4) if args.multi_warp else (M, N, K, 1)
    rng = torch.Generator().manual_seed(402)
    a = torch.randint(-7, 8, (m, k), generator=rng).to(torch.bfloat16)
    b = torch.randint(-7, 8, (n, k), generator=rng).to(torch.bfloat16)
    expected = (a.float() @ b.float().T * 1.25 + 0.5).to(torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    device_inputs = (a.cuda(), b.cuda(), out)
    views = tuple(from_dlpack(t, assumed_align=16) for t in device_inputs)
    options = ""
    if args.max_registers is not None:
        options = f"--ptxas-options='--override-directive-values --maxrregcount={args.max_registers}'"
    for name, variant in (("direct", False), ("stmatrix", True)):
        out.fill_(float("nan"))
        kernel = cute.compile(launch, *views, variant, warps, options=options)
        for _ in range(5):
            kernel(*views)
        actual = out.cpu()
        error = (actual.float() - expected.float()).abs().max().item()
        if not torch.equal(actual, expected):
            raise AssertionError(f"PROBE FAIL max_diff={error}  # {name}")
        print(f"PROBE PASS max_diff={error}  # {name}")


if __name__ == "__main__":
    main()
