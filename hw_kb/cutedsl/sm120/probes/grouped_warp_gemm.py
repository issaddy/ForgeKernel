"""Verify grouped BF16 GEMM tile mapping, offsets, and M/N/K tails on SM120."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

SHAPES = ((1, 3, 1), (17, 19, 23), (33, 9, 16), (4, 17, 29))
TILE_M = 16
TILE_N = 8
TILE_K = 16


@cute.kernel
def grouped_gemm(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor, groups: cute.Tensor,
                 num_tiles: cutlass.Constexpr, grid_size: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    block, _, _ = cute.arch.block_idx()
    allocator = SmemAllocator()
    shared_a = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1))
    )
    shared_b = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((TILE_N, TILE_K), stride=(TILE_K, 1))
    )
    op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (TILE_M, TILE_N, TILE_K))
    mma = cute.make_tiled_mma(op)
    thread_mma = mma.get_slice(lane)
    part_a = thread_mma.partition_A(shared_a)
    part_b = thread_mma.partition_B(shared_b)
    frag_a = thread_mma.make_fragment_A(part_a)
    frag_b = thread_mma.make_fragment_B(part_b)
    tile_coords = thread_mma.partition_C(cute.make_identity_tensor((TILE_M, TILE_N)))
    acc = cute.make_rmem_tensor(tile_coords.shape, cutlass.Float32)

    for tile in cutlass.range(block, num_tiles, grid_size):
        group = cutlass.Int32(0)
        for candidate in cutlass.range_constexpr(1, len(SHAPES)):
            if tile >= groups[candidate, 6]:
                group = candidate
        m = groups[group, 0]
        n = groups[group, 1]
        k = groups[group, 2]
        a_base = groups[group, 3]
        b_base = groups[group, 4]
        c_base = groups[group, 5]
        local_tile = tile - groups[group, 6]
        tile_cols = (n + TILE_N - 1) // TILE_N
        first_row = local_tile // tile_cols * TILE_M
        first_col = local_tile % tile_cols * TILE_N

        acc.fill(0.0)
        for k_tile in cutlass.range((k + TILE_K - 1) // TILE_K):
            for idx in cutlass.range_constexpr(TILE_M * TILE_K // 32):
                pos = lane + idx * 32
                row = pos // TILE_K
                col = pos % TILE_K
                actual_k = k_tile * TILE_K + col
                value = cutlass.BFloat16(0)
                if first_row + row < m and actual_k < k:
                    value = a[a_base + (first_row + row) * k + actual_k]
                shared_a[row, col] = value
            for idx in cutlass.range_constexpr(TILE_N * TILE_K // 32):
                pos = lane + idx * 32
                row = pos // TILE_K
                col = pos % TILE_K
                actual_k = k_tile * TILE_K + col
                value = cutlass.BFloat16(0)
                if first_col + row < n and actual_k < k:
                    value = b[b_base + (first_col + row) * k + actual_k]
                shared_b[row, col] = value
            cute.arch.sync_threads()
            cute.autovec_copy(part_a, frag_a)
            cute.autovec_copy(part_b, frag_b)
            cute.gemm(mma, acc, frag_a, frag_b, acc)
            cute.arch.sync_threads()

        for slot in cutlass.range_constexpr(cute.size(acc)):
            row, col = tile_coords[slot]
            row += first_row
            col += first_col
            if row < m and col < n:
                output[c_base + row * n + col] = acc[slot]


@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor, groups: cute.Tensor,
           num_tiles: cutlass.Constexpr, grid_size: cutlass.Constexpr):
    grouped_gemm(a, b, output, groups, num_tiles, grid_size).launch(
        grid=(grid_size, 1, 1), block=(32, 1, 1)
    )


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")

    groups = []
    a_parts, b_parts, expected_parts = [], [], []
    a_offset = b_offset = c_offset = num_tiles = 0
    for group, (m, n, k) in enumerate(SHAPES):
        a = (torch.arange(m * k).reshape(m, k) % 13 - 6 + group).to(torch.bfloat16)
        b = (torch.arange(n * k).reshape(n, k) % 11 - 5 - group).to(torch.bfloat16)
        a_parts.append(a.flatten())
        b_parts.append(b.flatten())
        expected_parts.append((a.float() @ b.float().T).flatten())
        groups.append((m, n, k, a_offset, b_offset, c_offset, num_tiles))
        a_offset += m * k
        b_offset += n * k
        c_offset += m * n
        num_tiles += ((m + TILE_M - 1) // TILE_M) * ((n + TILE_N - 1) // TILE_N)

    inputs = [torch.cat(a_parts).cuda(), torch.cat(b_parts).cuda()]
    expected = torch.cat(expected_parts)
    output = torch.empty_like(expected, device="cuda")
    metadata = torch.tensor(groups, dtype=torch.int32, device="cuda")
    views = tuple(from_dlpack(t, assumed_align=16) for t in (*inputs, output, metadata))
    for name, grid_size in (("one_cta_per_tile", num_tiles), ("grid_stride", 3)):
        output.fill_(float("nan"))
        cute.compile(launch, *views, num_tiles, grid_size)(*views)
        actual = output.cpu()
        error = (actual - expected).abs().max().item()
        if not torch.equal(actual, expected):
            raise AssertionError(f"PROBE FAIL max_diff={error}  # {name}")
        print(f"PROBE PASS max_diff={error}  # {name}, {num_tiles} tiles, {grid_size} CTAs")


if __name__ == "__main__":
    main()
