# cp.async commit→HMMA→partial-wait

状态：✅ Verified — PROBE PASS max_diff=0.0, job 1795350256

## Purpose

在未来 global→shared 拷贝提交后，用已经就绪的寄存器 fragment 继续当前 warp MMA 依赖链，再部分等待下一次 shared 读取所需的旧组。

## When needed (trigger)

`cpasync.CopyG2SOp` 与 warp MMA 共存，但 commit 后立即 wait，独立的当前计算没有放入供给窗口。适用前提是当前 operand 已进入寄存器、未来拷贝不覆盖仍需读取的 shared slot，并且保留 consumer rendezvous。

## Minimal runnable example

在含认证探针及 `notes/sass_tools.py` 的工作目录运行以下完整入口；需要 numpy、torch、CuTe DSL 和 `cuobjdump`：

```python
import runpy

runpy.run_path("notes/cp_async_hmma_partial_wait.py", run_name="__main__")
```

探针用一个 warp、三个 shared slot、九组不同的 BF16 小整数面验证组节拍、slot 环绕和 drain。每组包含 16×16 与 8×16 两个面，输出为 `repeats + 1` 次相同 warp MMA 累加的 FP32 结果；同一编译实例分别传入运行时 `cutlass.Int32(1/4/16)`，numpy 独立计算对应倍数的 tile 乘积。这些尺寸和重复次数只用于硬件接口测试，不是吞吐优化建议。

精确接口构造：

- `cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)`；`cute.copy(atom, src_view, dst_view)` 发射每 lane 的八元素向量。
- `SmemAllocator().allocate_tensor(cutlass.BFloat16, cute.make_layout(3 * 384), byte_alignment=16)` 分配环形 shared 存储。
- `cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)))` 与 `mma.get_slice(lane)` 创建 warp 及线程映射。
- `thread_mma.make_fragment_A/B(thread_mma.partition_A/B(shared_face))` 创建寄存器 operand。
- `cute.make_tiled_copy_A/B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4/2), cutlass.BFloat16), mma)` 创建 A/B 的 LDSM 路径；线程 slice 的 `partition_S(shared_face)` 与 `retile(fragment)` 必须保持同一 MMA 映射。

以下为探针的稳态窗口；`stage(..., False)` 只发射未来一组拷贝，`frag_a/frag_b` 已在前一轮 wait 与 barrier 后加载：

```python
for tile in cutlass.range_constexpr(7):
    acc.fill(0.0)
    cute.gemm(mma, acc, frag_a, frag_b, acc)
    stage(source, arena, tile + 2, lane, False)
    cute.arch.cp_async_commit_group()
    for step in cutlass.range(repeats, unroll=1):
        cute.gemm(mma, acc, frag_a, frag_b, acc)
    current_c = cute.make_tensor(output.iterator + tile * 128, layout_c)
    cute.autovec_copy(acc, thread_mma.partition_C(current_c))
    cute.arch.cp_async_wait_group(1)
    cute.arch.sync_threads()
    next_offset = (tile + 1) % 3 * 384
    next_a = cute.make_tensor(arena.iterator + next_offset, layout_a)
    next_b = cute.make_tensor(arena.iterator + next_offset + 256, layout_b)
    cute.copy(load_a, thread_load_a.partition_S(next_a), thread_load_a.retile(frag_a))
    cute.copy(load_b, thread_load_b.partition_S(next_b), thread_load_b.retile(frag_b))
```

`repeats` 为 kernel 与 launch 的 `cutlass.Int32` 运行时参数；`cute.compile(launch, *views, cutlass.Int32(16))` 编译后可用同一实例传入不同次数。`cutlass.range(repeats, unroll=1)` 使用动态循环，第一段 MMA 与循环中的 MMA 共用 accumulator 依赖链；不通过添加 shuffle 或 commit 后的额外 CTA barrier 强制发射顺序。动态循环用于测试调度边界，其分支开销不代表优化收益，也不能通过在实际路径中添加冗余计算来宣称优化。运行时循环与源码顺序仍不是最终调度证书，实际 cubin 必须通过下述检查。

组账本：预热提交组 0、1，`cp_async_wait_group(1)` 后 `sync_threads()`，再加载组 0。第 i 轮提交 i+2，完成寄存器中的组 i，部分等待组 i+1 后加载它；最新组 i+2 可以仍 pending。循环完成后计算组 7，`cp_async_wait_group(0)` 与 barrier drain 全部拷贝，再加载并计算组 8。所有组执行 `repeats + 1` 次相同 MMA，便于用精确整数结果验证 accumulator 链。

认证检查保存本次 cubin，以 `cuobjdump -sass` 与 `sass_tools.parse_sass` 验证七个静态窗口：每次稳态 `LDGDEPBAR` 之后、`DEPBAR.LE ..., 0x1` 之前必须有 HMMA；下一组 LDSM 必须在 wait 后的 CTA barrier 之后。数值与 SASS 同时通过才打印唯一一条 `PROBE PASS max_diff=<实际最大绝对误差>`；非数值失败打印 `PROBE FAIL max_diff=n/a`。GPU/SASS 认证通过：commits=9、windows=7；确认 `LDGDEPBAR → HMMA → DEPBAR.LE SB0, 0x1 → BAR.SYNC.DEFER_BLOCKING → LDSM`，drain 使用 `DEPBAR.LE SB0, 0x0` 后接 barrier 与 LDSM。此证据验证发射次序与数值正确性，不证明完全重叠或通用吞吐收益。

## Gotchas

- `cp_async_wait_group(N)` 允许最新 N 个已提交组仍 pending，不是等待 N 组，也不是 slot 编号。组计数属于线程；不同 lane 即使拷贝数量不同，也须执行一致的逻辑 commit 节拍。
- 深度 1 仅由本例的组账本推导。更改预热、每轮 commit 数量或 drain 时，必须重新确认 consumer 要读取哪一组，不能只改立即数。
- wait 完成每个生产线程的目标组；随后 `sync_threads()` 建立跨线程读取条件。barrier 不能代替异步拷贝完成，不能删除 wait 后的 consumer rendezvous。
- commit 后只能消费已经加载到寄存器的当前 fragment，不得提前 LDSM 读取 pending 面。slot 覆盖必须晚于此前 shared 读取完成；跨 warp 情形另需证明 release/覆盖关系。
- 每 lane 的八个 BF16 源/目的元素须真实满足 16-byte 对齐。`cute.assume(offset, divby=8)` 只是声明元素偏移条件，不修正地址；同一组不能存在重叠的异步目的写入。
- LDSM 与 MMA 的 partition/retile 必须来自同一 tiled MMA；只保证面尺寸相同不足以保证线程所有权一致。
- 已有 accumulator 链与当前 fragment 在窗口中保持存活；不能为延后 MMA 而重新读 pending shared，也不能改变 BF16 输入或 FP32 累加精度。
- 本路径是 warp `mma.sync` 和普通 `cp.async`，不能套用 WGMMA 或 bulk/TMA 的 commit/wait 计数。
- 两段 `cute.gemm` 之间的 accumulator 依赖不自动证明它们相对于 commit 的最终次序。不同编译版本或更大 fragment 组织需重新检查 SASS；某窗口缺少 HMMA 就是不满足隐藏关系，不能放宽为只检查全 kernel 出现过 HMMA。
- SASS 发射顺序不证明运行时完全重叠，也不承诺通用加速比例。本探针仅覆盖 SM120、单 warp CTA、对齐整 tile 与上述组节拍。

## Sources

- [NVIDIA PTX ISA：cp.async.commit_group](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group)
- [NVIDIA PTX ISA：cp.async.wait_group / cp.async.wait_all](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)
- [NVIDIA PTX ISA：bar.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar)
- [NVIDIA CuTe DSL warp MMA 编程指南](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/wmma_programming.html)
- 公共 API：`cutlass.cute.arch.cp_async_commit_group`、`cutlass.cute.arch.cp_async_wait_group`、`cutlass.cute.arch.sync_threads`、`cutlass.cute.nvgpu.cpasync.CopyG2SOp`、`cutlass.cute.nvgpu.warp.MmaF16BF16Op`、`cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp`、`cutlass.cute.gemm`、`cutlass.memory.SmemAllocator`。
- 认证探针：`notes/cp_async_hmma_partial_wait.py`；PROBE PASS max_diff=0.0, job 1795350256。
- Detailed provenance archive: docs/kb-provenance/cp_async_hmma_partial_wait.md
