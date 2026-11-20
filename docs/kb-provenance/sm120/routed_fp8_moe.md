# Routed FP8 expert scale addressing and fused accumulation

## Purpose

Verify per-expert scale addressing and shared intermediate ownership in a single routed launch. This is a correctness baseline for the expert compute and combine part of a fused MoE.

## Minimal runnable example

[The routed expert probe](probes/routed_fp8_moe.py) takes two preselected expert IDs and weights per token. One CTA processes each token. For each selected expert it decodes E4M3 weights with FP32 scales, computes the gate and up products and SwiGLU, publishes FP32 hidden values through shared memory, then computes the down product and adds the route-weighted contribution to the output accumulator.

```sh
bash gpu_entry.sh cmd python3 /path/to/routed_fp8_moe.py
```

The inputs use five BF16 tokens, four experts, two routes, widths 32→13→17, and distinct scales for each expert, row and group of eight K elements. The 13-element down-projection input covers a partial scale group. The cases include repeated use of experts across tokens, an unused expert, and a zero route weight. CPU PyTorch independently dequantizes the stored weights and executes the three projections.

## Gotchas

- A scale index combines the selected expert, the output row, and `K_index // group_size`. For the down projection, K is the hidden width, which uses separate scales.
- Keep the barrier that publishes hidden values and the barrier that protects the next expert's overwrite. Both barriers run in all CTA lanes.
- The probe takes supplied routing decisions and uses SIMT FP32 arithmetic after dequantization. Router/top-k generation, hardware block-scaled MMA, expert-parallel transport and workload throughput need their own checks.
- Use [grouped GEMM](../../../hw_kb/cutedsl/sm120/grouped_warp_gemm.md) for tiled output mapping and [block-scaled warp MMA](../../../hw_kb/cutedsl/sm120/block_scaled_warp_mma.md) for the FP4 scale-layout contract.

## Sources

- [NVIDIA SM120 routed NVFP4 implementation](https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/main/kernelSrcs/nvfp4_fused_moe_cutedsl) for further hardware-path comparison.
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: `PROBE PASS max_diff=1.1920928955078125e-07` against CPU dequantized SwiGLU and weighted accumulation.
