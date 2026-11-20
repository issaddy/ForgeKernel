"""Check online normalization and weighted accumulation across changing tile maxima."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

WIDTH = 96


@cute.jit
def warp_sum(value: cutlass.Float32):
    for step in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, 1 << step)
    return value


@cute.kernel
def weighted_softmax(scores: cute.Tensor, values: cute.Tensor, lengths: cute.Tensor,
                     output: cute.Tensor, normalizer: cute.Tensor):
    lane = cute.arch.lane_idx()
    row, _, _ = cute.arch.block_idx()
    length = lengths[row]
    running_max = cutlass.Float32(float("-inf"))
    running_sum = cutlass.Float32(0)
    running_weighted = cutlass.Float32(0)

    for tile in cutlass.range(WIDTH // 32):
        # A tile with no valid columns must not form exp(-inf - -inf).
        if tile * 32 < length:
            col = tile * 32 + lane
            score = cutlass.Float32(float("-inf"))
            value = cutlass.Float32(0)
            if col < length:
                score = scores[row, col]
                value = values[row, col]

            tile_max = score
            for step in cutlass.range_constexpr(5):
                tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, 1 << step))
            merged_max = cute.arch.fmax(running_max, tile_max)
            scale_old = cute.math.exp(running_max - merged_max)
            weight = cute.math.exp(score - merged_max)
            running_sum = running_sum * scale_old + warp_sum(weight)
            running_weighted = running_weighted * scale_old + warp_sum(weight * value)
            running_max = merged_max

    if lane == 0:
        normalized = cutlass.Float32(0)
        log_sum_exp = cutlass.Float32(float("-inf"))
        if length > 0:
            normalized = running_weighted / running_sum
            log_sum_exp = running_max + cute.math.log(running_sum)
        output[row] = normalized
        normalizer[row] = log_sum_exp


@cute.jit
def launch(scores: cute.Tensor, values: cute.Tensor, lengths: cute.Tensor,
           output: cute.Tensor, normalizer: cute.Tensor):
    weighted_softmax(scores, values, lengths, output, normalizer).launch(
        grid=(scores.shape[0], 1, 1), block=(32, 1, 1)
    )


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    lengths = torch.tensor([0, 1, 32, 33, 64, 65, 96], dtype=torch.int32)
    columns = torch.arange(WIDTH)
    valid = columns[None, :] < lengths[:, None]
    # Alternate increasing and decreasing tile maxima across rows.
    direction = torch.where(torch.arange(len(lengths)) % 2 == 0, 1, -1)[:, None]
    phase = torch.arange(len(lengths) * WIDTH, dtype=torch.float32).reshape(-1, WIDTH)
    scores = (phase * 0.19).sin() * 3 + direction * (columns // 32) * 7
    values = (phase * 0.27).cos() * 5
    masked = scores.masked_fill(~valid, float("-inf"))
    expected = torch.zeros(len(lengths))
    expected[1:] = (masked[1:].softmax(dim=1) * values[1:]).sum(dim=1)
    expected_lse = masked.logsumexp(dim=1)
    source = scores.masked_fill(~valid, float("nan")).cuda()
    value = values.masked_fill(~valid, float("nan")).cuda()
    sizes = lengths.cuda()
    output = torch.full(expected.shape, float("nan"), device="cuda")
    lse = torch.full(expected.shape, float("nan"), device="cuda")
    views = tuple(from_dlpack(t, assumed_align=16) for t in (source, value, sizes, output, lse))
    cute.compile(launch, *views)(*views)
    actual = output.cpu()
    actual_lse = lse.cpu()
    error = max((actual - expected).abs().max().item(),
                (actual_lse[1:] - expected_lse[1:]).abs().max().item())
    if not (torch.allclose(actual, expected, rtol=2e-6, atol=1e-6)
            and torch.allclose(actual_lse, expected_lse, rtol=2e-6, atol=1e-6)):
        raise AssertionError(f"PROBE FAIL max_diff={error}  # online_weighted_softmax")
    print(f"PROBE PASS max_diff={error}  # online_weighted_softmax")


if __name__ == "__main__":
    main()
