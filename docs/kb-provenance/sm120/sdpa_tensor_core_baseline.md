# Tensor Core SDPA performance baseline on SM120

## Purpose

Measure the cost of a known fused implementation at realistic head dimensions before optimizing a custom Attention kernel. Force the backend and check actual Tensor Core activity.

## Reproduce

[The probe](probes/sdpa_tensor_core_baseline.py) compares forced PyTorch math SDPA with forced cuDNN SDPA. Both consume the same BF16 Q/K/V tensors and mask configuration. It uses five batches of 30 CUDA Graph replays for timing. NCU runs separately and captures the selected cuDNN launches.

```sh
bash gpu_entry.sh cmd python3 /path/to/sdpa_tensor_core_baseline.py
bash gpu_entry.sh cmd 'ncu --csv --profile-from-start off --metrics gpu__time_active.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed python3 /path/to/sdpa_tensor_core_baseline.py --profile'
```

## Observed on RTX PRO 5000 72GB Blackwell

PyTorch 2.13.0+cu130, cuDNN 9.20.0, NCU 2025.3.1. BF16 inputs/output, B=1, H=8, head dimension 128. These graph medians include the entire SDPA call. The NCU tensor counter comes from a separate profiling run.

| Q/KV length | Mask | Math graph median | cuDNN graph median | Max absolute error | Tensor pipe active |
| --- | --- | ---: | ---: | ---: | ---: |
| 1,024/1,024 | Noncausal | 0.307435 ms | 0.035826 ms | 0.001953125 | 41.09% |
| 2,048/2,048 | Causal | 1.897315 ms | 0.078505 ms | 0.0078125 | 41.11% |

Both cases passed `atol=0.004, rtol=0.01`. NCU selected different fused kernels, with names containing `wmma` for the first case and `sm120_flash_fprop` for the second. A backend name or its kernel-name prefix alone cannot establish the Tensor Core path.

## Gotchas

- Force one backend when comparing implementations. An automatic selector can silently change the implementation across shapes.
- Capture after initialization and verify the replay returns a correct output. Keep profiler replay separate from timing.
- Use this as a library baseline. The speedup combines multiple changes such as fusion, scheduling, and Tensor Core execution. For a specific architecture decision, isolate the change in a paired experiment.

## Sources

- [NVIDIA cuDNN SDPA support matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention/) and [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- GPU certification: `PROBE PASS max_diff=0.001953125` and `PROBE PASS max_diff=0.0078125` for the two rows, respectively.
