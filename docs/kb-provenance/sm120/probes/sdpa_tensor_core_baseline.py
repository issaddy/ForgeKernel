"""Verify and time a forced cuDNN SDPA path on SM120 at practical head sizes."""
import argparse
import statistics

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def attn(backend, q, k, v, causal):
    with sdpa_kernel(backend):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def graph_samples(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 30)
    return output, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    print(f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()}")

    generator = torch.Generator(device="cuda").manual_seed(630)
    cases = ((1024, 1024, False), (2048, 2048, True))
    for queries, keys, causal in cases:
        q = torch.randn((1, 8, queries, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
        k, v = (torch.randn((1, 8, keys, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
                for _ in range(2))
        if args.profile:
            attn(SDPBackend.CUDNN_ATTENTION, q, k, v, causal)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            output = attn(SDPBackend.CUDNN_ATTENTION, q, k, v, causal)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            assert output.shape == q.shape
            continue
        with torch.inference_mode():
            expected, math_ms = graph_samples(lambda: attn(SDPBackend.MATH, q, k, v, causal))
            output, cudnn_ms = graph_samples(lambda: attn(SDPBackend.CUDNN_ATTENTION, q, k, v, causal))
        error = (output.float() - expected.float()).abs().max().item()
        if not torch.allclose(output.float(), expected.float(), atol=0.004, rtol=0.01):
            raise AssertionError(f"PROBE FAIL max_diff={error} # {queries}x{keys}, causal={causal}")
        print(f"PROBE PASS max_diff={error} # {queries}x{keys}, causal={causal}")
        for label, samples in (("math", math_ms), ("cudnn", cudnn_ms)):
            print(f"{label} median={statistics.median(samples):.6f} ms "
                  f"range={min(samples):.6f}..{max(samples):.6f} ms")


if __name__ == "__main__":
    main()
