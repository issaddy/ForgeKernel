# Masked warp softmax over two segments

## Purpose

Use the same warp for two 32-element segments and combine them in a stable maximum and sum. This is a reduction building block for Attention.

## Minimal runnable example

Run [the probe](probes/masked_warp_softmax.py) on SM120.

```sh
bash gpu_entry.sh cmd python3 /path/to/masked_warp_softmax.py
```

One warp owns a row of 64 values. Invalid positions enter the maximum as negative infinity. All lanes reduce the maximum in FP32. Each lane computes two exponentials, then reduces the sum of both segments. An empty row bypasses exponentiation and division and writes zeros.

## Gotchas

- Keep all 32 lanes at the butterfly even for short or empty rows. Guard the invalid loads and use a warp-uniform empty-row branch.
- The probe poisons the input padding with NaNs and checks that invalid outputs are exactly zero. It assumes finite valid inputs.
- This bounded two-segment example computes a single row maximum. An online multi-tile iteration additionally needs to rescale the previous sum and output accumulator when its maximum changes.

## Sources

- [CuTe DSL architecture API](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_arch.html). Porting basis: SM90/SM100 `softmax_segment_ilp` and `branch_free_select_mask`.
- GPU certification on RTX PRO 5000 72GB Blackwell with CuTe DSL 4.8.1: `PROBE PASS max_diff=2.9802322387695312e-08` against CPU PyTorch for valid lengths 0, 1, 31, 32, 33, 63, and 64.
