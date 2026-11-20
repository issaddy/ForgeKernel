"""Validate FP8 weight scales, routed SwiGLU experts, and weighted accumulation."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

EXPERTS = 4
IN = 32
HIDDEN = 13
OUT = 17
GROUP = 8
TOP_K = 2


@cute.kernel
def routed_experts(x: cute.Tensor, routes: cute.Tensor, route_weights: cute.Tensor,
                   gate: cute.Tensor, up: cute.Tensor, down: cute.Tensor,
                   gate_scales: cute.Tensor, up_scales: cute.Tensor, down_scales: cute.Tensor,
                   output: cute.Tensor):
    lane = cute.arch.lane_idx()
    token, _, _ = cute.arch.block_idx()
    intermediate = SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout(HIDDEN))
    result = cutlass.Float32(0)

    for route in cutlass.range_constexpr(TOP_K):
        expert = routes[token, route]
        if lane < HIDDEN:
            g = cutlass.Float32(0)
            u = cutlass.Float32(0)
            for group in cutlass.range_constexpr(IN // GROUP):
                g_scale = gate_scales[expert, lane, group]
                u_scale = up_scales[expert, lane, group]
                for i in cutlass.range_constexpr(GROUP):
                    position = group * GROUP + i
                    value = x[token, position].to(cutlass.Float32)
                    g += value * (gate[expert, lane, position].to(cutlass.Float32) * g_scale)
                    u += value * (up[expert, lane, position].to(cutlass.Float32) * u_scale)
            intermediate[lane] = g / (1.0 + cute.math.exp(-g)) * u

        cute.arch.sync_threads()
        if lane < OUT:
            value = cutlass.Float32(0)
            for position in cutlass.range_constexpr(HIDDEN):
                scale = down_scales[expert, lane, position // GROUP]
                w = down[expert, lane, position].to(cutlass.Float32) * scale
                value += intermediate[position] * w
            result += value * route_weights[token, route]
        cute.arch.sync_threads()

    if lane < OUT:
        output[token, lane] = result


@cute.jit
def launch(x: cute.Tensor, routes: cute.Tensor, route_weights: cute.Tensor,
           gate: cute.Tensor, up: cute.Tensor, down: cute.Tensor, gate_scales: cute.Tensor,
           up_scales: cute.Tensor, down_scales: cute.Tensor, output: cute.Tensor):
    routed_experts(x, routes, route_weights, gate, up, down, gate_scales, up_scales,
                   down_scales, output).launch(grid=(x.shape[0], 1, 1), block=(32, 1, 1))


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")
    rng = torch.Generator().manual_seed(503)
    x = torch.randn((5, IN), generator=rng).to(torch.bfloat16)
    routes = torch.tensor([[0, 2], [2, 1], [1, 0], [2, 0], [0, 1]], dtype=torch.int32)
    route_weights = torch.tensor(
        [[0.25, 0.75], [0.9, 0.1], [0.0, 1.0], [0.37, 0.63], [0.62, 0.38]]
    )

    def weights(rows, columns):
        shape = (EXPERTS, rows, columns)
        scale_shape = (EXPERTS, rows, (columns + GROUP - 1) // GROUP)
        values = torch.randn(shape, generator=rng).to(torch.float8_e4m3fn)
        scales = 0.05 + torch.rand(scale_shape, generator=rng) * 0.15
        return values, scales

    def unpack(weight, scales):
        return weight.float() * scales.repeat_interleave(GROUP, dim=-1)[..., :weight.shape[-1]]

    gate, gate_scales = weights(HIDDEN, IN)
    up, up_scales = weights(HIDDEN, IN)
    down, down_scales = weights(OUT, HIDDEN)
    g, u, d = (unpack(gate, gate_scales), unpack(up, up_scales), unpack(down, down_scales))
    expected = torch.zeros((len(routes), OUT))
    for token, selected in enumerate(routes):
        for rank, expert in enumerate(selected.tolist()):
            hidden = torch.nn.functional.silu(g[expert] @ x[token].float())
            hidden *= u[expert] @ x[token].float()
            expected[token] += route_weights[token, rank] * (d[expert] @ hidden)

    inputs = tuple(t.cuda() for t in (
        x, routes, route_weights, gate, up, down, gate_scales, up_scales, down_scales
    ))
    output = torch.full(expected.shape, float("nan"), device="cuda")
    views = tuple(from_dlpack(t, assumed_align=16) for t in (*inputs, output))
    cute.compile(launch, *views)(*views)
    actual = output.cpu()
    error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, rtol=2e-5, atol=2e-6):
        raise AssertionError(f"PROBE FAIL max_diff={error}  # routed_fp8_swiglu")
    print(f"PROBE PASS max_diff={error}  # routed_fp8_swiglu")


if __name__ == "__main__":
    main()
