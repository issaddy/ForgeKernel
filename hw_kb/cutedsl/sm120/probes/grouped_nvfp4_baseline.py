"""Verify uneven hardware NVFP4 grouped GEMM and scale-row offsets on SM120."""
import argparse

import torch
from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

ROWS = (4, 127, 129, 511)
OUT = 512
K = 512
SCALE_ALIGNMENT = 128


def quantize(matrix):
    global_scale = (448 * 6) / matrix.float().abs().amax()
    data, scales = nvfp4_quantize(
        matrix, global_scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
    )
    return data, scales, global_scale


def inputs():
    generator = torch.Generator(device="cuda").manual_seed(608)
    rand = lambda shape: torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    original = [(rand((rows, K)), rand((OUT, K))) for rows in ROWS]
    a_parts = []
    b_parts = []
    a_scale_offsets = []
    alpha = []
    row_offsets = [0]
    for group, (a, b) in enumerate(original):
        packed_a, a_scales, a_global = quantize(a)
        packed_b, b_scales, b_global = quantize(b)
        a_parts.append(packed_a)
        b_parts.append((packed_b, b_scales))
        scale_start = (row_offsets[-1] + group * (SCALE_ALIGNMENT - 1)) // SCALE_ALIGNMENT
        a_scale_offsets.append((scale_start * SCALE_ALIGNMENT, a_scales))
        alpha.append((a_global * b_global).reciprocal())
        row_offsets.append(row_offsets[-1] + a.shape[0])

    last_offset, last_scales = a_scale_offsets[-1]
    a_descale = last_scales.new_zeros((last_offset + last_scales.shape[0], last_scales.shape[1]))
    for offset, scales in a_scale_offsets:
        a_descale[offset:offset + scales.shape[0]] = scales
    b_data, b_descale = zip(*b_parts)
    packed = (
        torch.cat(a_parts), torch.stack(b_data), a_descale, torch.stack(b_descale),
        torch.tensor(row_offsets, dtype=torch.int32, device="cuda"), torch.stack(alpha).float(),
    )
    return packed, original, row_offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    packed, original, offsets = inputs()
    for swap in (False, True):
        def run():
            return group_gemm_nvfp4_nt_groupwise(
                *packed, tile_m=128, tile_n=64, tile_k=128,
                swap_ab=swap, out_dtype=torch.bfloat16,
            )

        output = run()
        torch.cuda.synchronize()
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
            output = run()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            continue
        similarities = []
        for group, (a, b) in enumerate(original):
            target = a.float() @ b.float().T
            actual = output[offsets[group]:offsets[group + 1]].float()
            similarities.append(torch.nn.functional.cosine_similarity(
                target.flatten(), actual.flatten(), dim=0
            ).item())
        worst = min(similarities)
        if worst <= 0.97:
            raise AssertionError(f"PROBE FAIL max_diff={1 - worst} # swap_ab={swap}: {similarities}")
        print(f"PROBE PASS max_diff={1 - worst} # swap_ab={swap}; cosines={similarities}")


if __name__ == "__main__":
    main()
