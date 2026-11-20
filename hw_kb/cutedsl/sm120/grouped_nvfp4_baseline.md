# Hardware NVFP4 grouped GEMM with uneven rows

## Purpose

Verify per-group scale addressing in a hardware block-scaled grouped GEMM. Treat the group-argument setup kernel and Tensor Core GEMM as separate sources of cost.

## Minimal runnable example

[The FlashInfer probe](probes/grouped_nvfp4_baseline.py) compares the SM120 backend's two `swap_ab` modes. It packs the activation row ranges consecutively, but pads and places activation scale rows separately per group. For `128×4` scale swizzle, the row location of group `g` is `128 * floor((m_indptr[g] + 127*g) / 128)` for this backend. The probe checks four unequal M extents 4/127/129/511, N=512, K=512 and 16 values per NVFP4 scale.

```sh
bash gpu_entry.sh cmd python3 /path/to/grouped_nvfp4_baseline.py
bash gpu_entry.sh cmd 'ncu --csv --profile-from-start off --metrics gpu__time_active.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed python3 /path/to/grouped_nvfp4_baseline.py --profile'
```

## Observed on SM120

RTX PRO 5000 72GB Blackwell, FlashInfer 0.7.3, NCU 2025.3.1, tile 128×64×128 and BF16 output. Each group is compared separately with its original BF16-input FP32 product. Both modes passed the upstream cosine criterion of 0.97. The maximum per-group cosine distance was 0.00914431, so the minimum group cosine was about 0.99086. This includes quantization error.

| swap_ab | GPU argument setup | GPU grouped GEMM | GEMM tensor pipe active |
| --- | ---: | ---: | ---: |
| False | 3.776 µs | 11.168 µs | 6.21% |
| True | 2.720 µs | 11.072 µs | 5.47% |

One profiled invocation per mode, with prepacked FP4 inputs.

## Gotchas

- Distinguish the start of each activation row range, the padded scale-row offset, and the per-expert B scale tensor. They are different address spaces even when a uniform test accidentally makes them line up.
- Check each expert independently. Aggregating cosine first can hide corruption in small expert batches.
- A library timing should specify whether quantization, routing, argument setup and output combine are inside the measurement.

## Sources

- [FlashInfer v0.7.3 grouped NVFP4 test](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.3/tests/gemm/test_group_gemm_fp4.py) and API symbol `flashinfer.gemm.group_gemm_nvfp4_nt_groupwise`.
- GPU certification: both modes logged `PROBE PASS max_diff=0.009144306182861328`. NCU reported the SM120 grouped mainloop and nonzero Tensor Core activity.
