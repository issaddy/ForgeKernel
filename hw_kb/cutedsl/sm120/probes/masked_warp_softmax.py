"""Check warp reductions and bounds masking across two 32-element segments."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def warp_softmax(source: cute.Tensor, lengths: cute.Tensor, output: cute.Tensor):
    lane = cute.arch.lane_idx()
    row, _, _ = cute.arch.block_idx()
    length = lengths[row]
    first = cutlass.Float32(float("-inf"))
    second = cutlass.Float32(float("-inf"))
    if lane < length:
        first = source[row, lane]
    if lane + 32 < length:
        second = source[row, lane + 32]

    # All lanes execute both reductions, even for empty rows.
    largest = cute.arch.fmax(first, second)
    for step in cutlass.range_constexpr(5):
        largest = cute.arch.fmax(largest, cute.arch.shuffle_sync_bfly(largest, 1 << step))

    weight_first = cutlass.Float32(0)
    weight_second = cutlass.Float32(0)
    if length > 0:
        weight_first = cute.math.exp(first - largest)
        weight_second = cute.math.exp(second - largest)
    total = weight_first + weight_second
    for step in cutlass.range_constexpr(5):
        total += cute.arch.shuffle_sync_bfly(total, 1 << step)

    # Avoid 0/0 for empty rows; their contract is an all-zero output.
    if length > 0:
        weight_first /= total
        weight_second /= total
    output[row, lane] = weight_first
    output[row, lane + 32] = weight_second


@cute.jit
def launch(source: cute.Tensor, lengths: cute.Tensor, output: cute.Tensor):
    warp_softmax(source, lengths, output).launch(grid=(source.shape[0], 1, 1), block=(32, 1, 1))


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    lengths = torch.tensor([0, 1, 31, 32, 33, 63, 64], dtype=torch.int32)
    column = torch.arange(64)[None, :]
    valid = column < lengths[:, None]
    values = torch.sin(torch.arange(448, dtype=torch.float32)).reshape(-1, 64) * 11
    expected = torch.zeros_like(values)
    expected[1:] = values[1:].masked_fill(~valid[1:], float("-inf")).softmax(dim=1)

    # NaN in the padded input detects accidentally including the masked tail.
    source = values.masked_fill(~valid, float("nan")).cuda()
    device_lengths = lengths.cuda()
    output = torch.empty_like(source)
    views = tuple(from_dlpack(t, assumed_align=16) for t in (source, device_lengths, output))
    cute.compile(launch, *views)(*views)
    actual = output.cpu()
    error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, rtol=2e-6, atol=2e-7) or actual[~valid].count_nonzero():
        raise AssertionError(f"PROBE FAIL max_diff={error}  # masked_warp_softmax")
    print(f"PROBE PASS max_diff={error}  # masked_warp_softmax")


if __name__ == "__main__":
    main()
