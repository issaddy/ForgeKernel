"""Verify QK, online normalization, and multi-component PV in one SM120 kernel."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

QUERIES = 5
KEYS = 96
HEAD = 16
VALUES = 8
TILE = 32


@cute.jit
def sum_lanes(value: cutlass.Float32):
    for step in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, 1 << step)
    return value


@cute.jit
def issue_tile(atom, coords, shared, barrier, batch, tile, width: cutlass.Constexpr):
    gmem = cute.local_tile(coords, (1, TILE, width), (batch, tile, 0))
    shared_view, global_view = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(shared, 0, 2),
        cute.group_modes(gmem, 0, 3),
    )
    cute.copy(atom, global_view, shared_view, tma_bar_ptr=barrier)


@cute.jit
def stage_kv(atom_k, tma_k, atom_v, tma_v, slot_k, slot_v, barrier, batch, tile):
    with cute.arch.elect_one():
        cute.arch.mbarrier_expect_tx(barrier, TILE * (HEAD + VALUES) * 2)
    cute.arch.sync_warp()
    issue_tile(atom_k, tma_k, slot_k, barrier, batch, tile, HEAD)
    issue_tile(atom_v, tma_v, slot_v, barrier, batch, tile, VALUES)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(barrier)


@cute.kernel
def attention(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, limits: cute.Tensor,
              offsets: cute.Tensor, output: cute.Tensor, lse: cute.Tensor,
              causal: cutlass.Constexpr, stages: cutlass.Constexpr,
              atom_k, tma_k, atom_v, tma_v):
    lane = cute.arch.lane_idx()
    query, batch, _ = cute.arch.block_idx()
    limit = limits[batch]
    if cutlass.const_expr(causal):
        limit = cute.min(limit, offsets[batch] + query + 1)
    if cutlass.const_expr(stages > 0):
        smem = SmemAllocator()
        barriers = smem.allocate_tensor(cutlass.Int64, cute.make_layout(stages)).iterator
        shared_k = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((TILE, HEAD, stages), stride=(HEAD, 1, TILE * HEAD)),
            byte_alignment=128,
        )
        shared_v = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((TILE, VALUES, stages), stride=(VALUES, 1, TILE * VALUES)),
            byte_alignment=128,
        )
        with cute.arch.elect_one():
            for slot in cutlass.range_constexpr(stages):
                cute.arch.mbarrier_init(barriers + slot, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        for slot in cutlass.range_constexpr(stages):
            if slot * TILE < limit:
                stage_kv(atom_k, tma_k, atom_v, tma_v, shared_k[None, None, slot],
                         shared_v[None, None, slot], barriers + slot, batch, slot)

    m = cutlass.Float32(float("-inf"))
    norm = cutlass.Float32(0)
    weighted = cute.make_rmem_tensor(VALUES, cutlass.Float32)
    weighted.fill(0.0)

    for tile in cutlass.range(KEYS // TILE):
        if tile * TILE < limit:
            if cutlass.const_expr(stages > 0):
                slot = tile % stages
                cute.arch.mbarrier_wait(barriers + slot, (tile // stages) % 2)
            key = tile * TILE + lane
            score = cutlass.Float32(float("-inf"))
            if key < limit:
                dot = cutlass.Float32(0)
                for d in cutlass.range_constexpr(HEAD):
                    if cutlass.const_expr(stages > 0):
                        key_value = shared_k[lane, d, slot]
                    else:
                        key_value = k[batch, key, d]
                    dot += q[batch, query, d].to(cutlass.Float32) * key_value.to(cutlass.Float32)
                score = dot * 0.25   # 1 / sqrt(HEAD)

            tile_max = score
            for step in cutlass.range_constexpr(5):
                tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, 1 << step))
            new_max = cute.arch.fmax(m, tile_max)
            carry = cute.math.exp(m - new_max)
            weight = cute.math.exp(score - new_max)
            norm = norm * carry + sum_lanes(weight)
            for d in cutlass.range_constexpr(VALUES):
                value = cutlass.Float32(0)
                if key < limit:
                    if cutlass.const_expr(stages > 0):
                        value = shared_v[lane, d, slot].to(cutlass.Float32)
                    else:
                        value = v[batch, key, d].to(cutlass.Float32)
                weighted[d] = weighted[d] * carry + sum_lanes(weight * value)
            m = new_max
            if cutlass.const_expr(stages > 0):
                cute.arch.sync_threads()
                next_tile = tile + stages
                if next_tile * TILE < limit:
                    stage_kv(atom_k, tma_k, atom_v, tma_v, shared_k[None, None, slot],
                             shared_v[None, None, slot], barriers + slot, batch, next_tile)

    if lane == 0:
        normalizer = cutlass.Float32(float("-inf"))
        for d in cutlass.range_constexpr(VALUES):
            value = cutlass.Float32(0)
            if limit > 0:
                value = weighted[d] / norm
            output[batch, query, d] = value
        if limit > 0:
            normalizer = m + cute.math.log(norm)
        lse[batch, query] = normalizer


@cute.jit
def launch(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, limits: cute.Tensor,
           offsets: cute.Tensor, output: cute.Tensor, lse: cute.Tensor, causal: cutlass.Constexpr,
           stages: cutlass.Constexpr):
    atom_k, tma_k, atom_v, tma_v = None, None, None, None
    if cutlass.const_expr(stages > 0):
        atom_k, tma_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), k,
            cute.make_layout((1, TILE, HEAD), stride=(TILE * HEAD, HEAD, 1)), (1, TILE, HEAD),
        )
        atom_v, tma_v = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), v,
            cute.make_layout((1, TILE, VALUES), stride=(TILE * VALUES, VALUES, 1)), (1, TILE, VALUES),
        )
    attention(q, k, v, limits, offsets, output, lse, causal, stages, atom_k, tma_k, atom_v, tma_v).launch(
        grid=(QUERIES, q.shape[0], 1), block=(32, 1, 1)
    )


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    rng = torch.Generator().manual_seed(521)
    limits = torch.tensor([0, 33, 65, 96], dtype=torch.int32)
    offsets = torch.tensor([0, 30, 61, 93], dtype=torch.int32)
    batches = len(limits)
    q = torch.randn(batches, QUERIES, HEAD, generator=rng).to(torch.bfloat16)
    k = torch.randn(batches, KEYS, HEAD, generator=rng).to(torch.bfloat16)
    v = torch.randn(batches, KEYS, VALUES, generator=rng).to(torch.bfloat16)
    columns = torch.arange(KEYS)[None, None, :]
    pad = columns[:, 0, :] >= limits[:, None]
    data = (q.cuda(), k.masked_fill(pad[:, :, None], float("nan")).cuda(),
            v.masked_fill(pad[:, :, None], float("nan")).cuda(), limits.cuda(), offsets.cuda())
    scores = q.float() @ k.float().transpose(1, 2) * 0.25
    out = torch.empty((batches, QUERIES, VALUES), device="cuda", dtype=torch.float32)
    lse = torch.empty((batches, QUERIES), device="cuda", dtype=torch.float32)
    views = tuple(from_dlpack(t, assumed_align=16) for t in (*data, out, lse))
    for causal in (False, True):
        valid = columns < limits[:, None, None]
        if causal:
            valid = valid & (columns <= offsets[:, None, None] + torch.arange(QUERIES)[None, :, None])
        masked = scores.masked_fill(~valid, float("-inf"))
        expected = masked.softmax(dim=-1).masked_fill(~valid, 0.0) @ v.float()
        expected_lse = masked.logsumexp(dim=-1)
        for stages in (0, 1, 2):
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            kernel = cute.compile(launch, *views, causal, stages)
            for _ in range(5):
                kernel(*views)
            actual, actual_lse = out.cpu(), lse.cpu()
            finite = torch.isfinite(expected_lse)
            error = max((actual - expected).abs().max().item(),
                        (actual_lse[finite] - expected_lse[finite]).abs().max().item())
            label = f"{'causal' if causal else 'noncausal'}_stages_{stages}"
            if not (torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)
                    and torch.allclose(actual_lse, expected_lse, rtol=2e-5, atol=2e-6)):
                raise AssertionError(f"PROBE FAIL max_diff={error}  # {label}")
            print(f"PROBE PASS max_diff={error}  # {label}")


if __name__ == "__main__":
    main()
