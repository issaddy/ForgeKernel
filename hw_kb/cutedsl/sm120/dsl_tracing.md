# CuTe DSL tracing and compile-time specialization

## When needed

A branch needs to specialize on a dtype, tile shape, stage count, or implementation switch. Or lowering fails because a Python name was introduced inside a dynamic region and used outside it.

## Minimal runnable example

The [warp MMA probe](probes/warp_mma_and_async_copy.py) compiles the same kernel with `use_async: cutlass.Constexpr` set to `False` or `True`. In the kernel, `if cutlass.const_expr(use_async)` picks the copy atom at trace time. The `lane < 16` condition depends on the running GPU lane and remains ordinary runtime control flow.

```sh
CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR="$PWD/ptx-bf16" \
  bash gpu_entry.sh cmd python3 /path/to/warp_mma_and_async_copy.py --dtype bf16
```

## Gotchas

- Declare specialization inputs as `cutlass.Constexpr` and mark compile-time branches explicitly. For a known trip count, use `cutlass.range_constexpr` when unrolling is wanted.
- Initialize dynamic result values before a runtime branch and update them inside it. Keep Python objects such as copy atoms in trace-time decisions.
- Put compiler dumps for different dtypes in separate directories. In CuTe DSL 4.8.1, this probe's BF16 and FP16 artifacts can share a filename.
- A correct result does not distinguish two implementations. Inspect the PTX or cubin for each variant separately.

## Sources

- [NVIDIA CuTe DSL control flow](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html).
- Porting basis: SM90/SM100 `cute_dsl_trace_scope` and `cute_dsl_const_expr_if`. Certification here covers the specialization exercised by the linked SM120 probe.
