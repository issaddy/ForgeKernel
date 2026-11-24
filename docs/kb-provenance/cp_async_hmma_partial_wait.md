# cp_async_hmma_partial_wait 溯源归档

## 输入证据与边界

本次诊断为 G3，deficient_locus 为 D0.4。要求保持既定契约，补充 cp.async commit→HMMA→partial-wait 的实际接口；当前 stage4 best 的 priced_by=hmma_after_commit 双 workload 合并收益为 +3.70%，分别 +4.05%/+3.35%。旧树 +8.53% 只属于历史输入，不能迁移到当前树。当前定价同样只作为诊断输入，未在本轮独立复测，不写入公共配方，也不用于宣称通用收益。

提供的 ORACLE 位于 `/home/relay/songjiayun/ForgeKernel/.artifacts/dsv41_wo_a_bf16_prefill_261122-164703-6b7c/task/reference/run.py`。其函数 `run` 仅调用 torch.einsum 并 contiguous；没有可提取的 cp.async、warp MMA 或同步接口。因而未从该实现推断任何硬件细节，也没有复制其算法到配方或探针。

## 公共接口依据

读取本机 NVIDIA CuTe DSL 官方安装包 `cutlass/cute/arch/nvvm_wrappers.py`：`cp_async_commit_group` 文档说明提交此前发射但未提交的拷贝，直接降低到 `nvvm.cp_async_commit_group`；`cp_async_wait_group(n)` 文档说明等待直到仅指定数量的组 pending，降低到 `nvvm.cp_async_wait_group(n)`。两者的官方 PTX 文档链接收入公共配方。

官方文档网页在本机 fetch 被 403 拒绝；未复制任何在线代码。接口构造还核对了仓库已认证公共示例 `hw_kb/cutedsl/sm120/probes/warp_mma_and_async_copy.py` 的 warp MMA、x4/x2 LDSM、partition/retile 和 DLPack 调用。新探针自行编写三槽、九组小矩阵流，不包含所提供算子的分组、shape、dispatch 或算法。

本机官方包 `cutlass/base_dsl/env_manager.py` 明确支持 `CUTE_DSL_KEEP_CUBIN`、`CUTE_DSL_DUMP_DIR`、`CUTE_DSL_NO_CACHE`；以新进程在导入 cutlass 前设置，避免已有模块缓存使开关不生效。

## 认证计划与实际状态

探针路径：`notes/cp_async_hmma_partial_wait.py`。它只运行自身的 DSL kernel，CPU numpy 对九个独立小整数 tile 计算最大绝对误差；不导入 kernel.py，不读取 answer branch，不运行评估器或其他计算库作 GPU 对照。

静态组账本：预提交 0、1；等待保留 1 后读 0；每轮提交 i+2、用寄存器 operand 计算 i、等待保留 1 后读 i+1；七轮后计算 7，再 drain 全部并计算 8。三槽复用至少跨越两次环绕。实际 SASS 使用工作区预置 `sass_tools.parse_sass` 解析，检查七个 commit→HMMA→partial depth 1 wait 窗口与 wait 后 barrier-before-LDSM 条件。

首次 GPU 返回 `PROBE FAIL max_diff=n/a`，不是编译失败。已读取该次 `/tmp/cp_async_hmma_partial_wait_ze2s1t92/schedule.sass`：首轮 HMMA 在地址 0240，第三次 LDGDEPBAR 在 0280，partial wait 在 02c0；首轮被提前到 commit 前，其余六个稳态窗口顺序正确。原检查没有返回指令证据，现改为在 stderr 输出所有相关指令与明确失败原因。

本次修复保持三槽组账本与七窗口验收不变，不放宽 SASS 条件。稳态拷贝不再由 stage 自行 commit，而用 `dsl_user_op` 封装有副作用的 `llvm.inline_asm`，将 `cp.async.commit_group` 与 XOR 偏移为零的 `shfl.sync.bfly.b32` 放在同一个 PTX 块；返回值写回当前 A fragment 的首个 32-bit packed word，由 HMMA 消费。仅做位模式原样传递，无量化、无精度改变；此依赖写法仍需实际 GPU/SASS 再认证，不以 inline PTX 的文本顺序宣称已成立。

接口核对了本机官方包 `cutlass/cute/arch/nvvm_wrappers.py` 的 shuffle 参数、`cutlass/cute/tensor.py` 的 recast_tensor 位解释与共享存储语义，以及工作区已有 `notes/timeline_probe.py` 的 inline_asm 调用签名。官方 PTX 网页再次返回 403，未读取或复制在线代码。增加 shuffle 的发射成本不计入诊断的 +8.53% 读数，不声称该读数认证了新增写法。

第二次 GPU 返回 `PROBE FAIL max_diff=n/a`。读取 `/tmp/cp_async_hmma_partial_wait_rgla9wlh/schedule.sass` 确认首轮 SHFL 在 0250、HMMA 在 0280，而第三个 LDGDEPBAR 在 02b0；其余六个窗口仍满足顺序。失败不是窗口计数误判：ptxas 可以拆开 inline PTX 块并把 commit 排到 dependent shuffle 与 HMMA 之后，原样 operand 依赖只约束 shuffle→HMMA，未约束 commit→shuffle。

本轮在 `commit_and_forward` 返回值写回后、HMMA 前插入 `cute.arch.sync_warp()`，保留同一三槽组账本、七窗口验收、partial depth 1 与 CTA barrier；不以等待全部拷贝或更改深度绕开目标。新边界是否保留并约束最终调度仍需 GPU/SASS 认证。本机官方安装包 `cutlass/cute/arch/nvvm_wrappers.py` 的 `sync_warp(mask=FULL_MASK)` 直接调用 `nvvm.bar_warp_sync`，文档为带可选掩码的 warp 同步；官方 PTX 网页访问再次被 403 拒绝，未复制在线代码。配方注明该同步边界不是 cp.async 完成信号，且不能宣称额外同步与 shuffle 没有成本。

第三次 GPU 返回 `PROBE FAIL max_diff=n/a`，日志明确指出首窗口缺少 commit 后的 HMMA；后续可见窗口有 LDGDEPBAR、SHFL、HMMA、depth 1 wait 和 barrier。日志尾部不足以确定首窗口全部地址，未将此失败归因于数值，也不放宽七窗口验收。warp 同步未能获得所需证书，本轮将 commit 后的 `sync_warp()` 替换为 `sync_threads()`，保留 operand 位模式传递、三槽组账本和 partial depth 1，试验一个显式 CTA 同步边界。该边界不消费 pending shared 数据，不是完成信号；是否阻止首 HMMA 提前仍以 GPU/SASS 为准，不作预先承诺。

核对本机官方包 `/opt/sglang/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/cute/arch/nvvm_wrappers.py`：`sync_threads()` 调用 `nvvm.barrier`，`sync_warp()` 调用 `nvvm.bar_warp_sync`。官方网页仍不可访问，未复制在线代码。新增 CTA barrier 只作为单 warp 接口认证尝试，其成本未被诊断的 +8.53% 性能读数覆盖；配方不得将它推广为其他流水线的必需 barrier 或通用收益。

## 当前重新认证方案

重新读取注入的 run.py，确认其仅为定义层 torch 运算，无法提供目标硬件接口；本轮依据本机官方 CuTe DSL 包的 `cp_async_commit_group`、`cp_async_wait_group`、`sync_threads` 实现与文档，以及既有公共 warp MMA 探针核对调用签名，未复制在线代码。

工作区 kernel.py 的两段 HMMA 在未来 copy 批次之间与 commit 之前，当前诊断要求延后现有 accumulator 链的最后一段，不增加同步。因此弃用历史探针的 inline PTX shuffle 与 commit 后额外 barrier，改为第一段 MMA→未来 copies→官方 commit API→第二段依赖同一 accumulator 的 MMA→partial wait→原有 barrier。两个 MMA 使用相同已就绪的小整数 operand，numpy 期望值改为两倍乘积；这是接口验证，不是算子算法，未修改 kernel.py 或 milestone。

保留三槽九组测试与七个稳态窗口硬验收，并加强 SASS 账本：partial wait 的 commit 序列必须为 2 到 9，每次 wait（包含预热及最终 drain）之后必须先有 barrier 再有 LDSM，且必须出现最终 depth 0 drain。源码 accumulator 依赖不能预先证明最终 commit→HMMA 次序，仍等待本次实际 GPU verdict；若次序失败，按实际指令证据修正，不允许只检验 kernel 内存在 HMMA。

当前修正版等待 GPU verdict，尚未认证，不修改 sm120/index.md 为 Verified。GPU PASS 返回后，须把实际 verdict、作业标识和必要的调度证据补入本归档，再更新配方状态与索引。

## 短 accumulator 链的调度失败与修订

本次驱动返回 `PROBE FAIL max_diff=n/a`，明确诊断第一个稳态窗口没有 commit 后 HMMA。日志末尾仍可见其他窗口的 LDGDEPBAR→HMMA→DEPBAR.LE depth 1→BAR→LDSM，以及最终 depth 0 drain；不能凭日志尾部为首窗口补造地址，也不能把失败解释为数值误差。仅两个 MMA 的短链不保证所需调度，本次保留七窗口、九次 commit 与 partial/drain 账本验收不变。

修订只将 commit 后的相同 accumulator 链延长为十六个 warp MMA，commit 前保留一个；组 7、8 同样累加十七次，numpy 期望值同步改为十七倍乘积。所有 operand 仍预先加载到寄存器，无新增 shuffle、同步、shared 读取或等待深度变化。重复计算仅是扩大独立计算窗口的接口实验，不建议学生向实际路径添加冗余 MMA，不将诊断的 +3.70% 迁移为这段测试的收益。是否存在每个 commit 后 HMMA 仍交由实际 GPU/SASS verdict，不宣称源码依赖保证次序。

再次读取提供的 run.py，确认没有此硬件接口；本机官方 `cutlass/cute/nvgpu/warp/mma.py` 的 `MmaF16BF16Op` 文档列出 BF16→FP32、(16,8,16)、warp-uniform `cute.gemm` 和固定 K-major operand 布局，修订沿用该公开接口而不复制在线代码。索引保持待认证。

## 静态长链失败后的动态循环试验

驱动返回 `PROBE FAIL max_diff=n/a`，诊断仍为首个窗口缺少 commit 后 HMMA；日志尾部最后一组连续十七条 HMMA 只能证明该段计算存在，不能替首窗口建立顺序证据。延长静态链未获得七窗口证书，保留原验收，不以此声称数值失败。

本次将 commit 后静态展开的十六次 MMA 改为 `cutlass.range(repeats, unroll=1)`，`repeats` 是 `cutlass.Int32` kernel/launch 参数。依据本机官方 `cutlass/base_dsl/ast_helpers.py` 的 `range` 文档，该接口生成动态循环，unroll=0/1 表示不展开。此变更测试控制流边界能否避免静态 HMMA 链跨 commit 移动；这是待认证假设，不是已证明的调度保证。不新增 inline PTX、shuffle、barrier 或 wait，不修改算子 kernel 与 milestone。

同一 compiled kernel 分别运行 repeats=1、4、16；两组尾部同样用 repeats+1 次计算，CPU numpy 期望值为对应倍数乘积，max_diff 取全部运行的真实最大绝对误差。SASS 仍须满足九个 commit、七个稳态 commit→HMMA→partial wait 窗口、partial commit 序列 2…9、最终 drain 与 barrier-before-LDSM；增加分支指令到 stderr 诊断以便 GPU 返回后核对动态循环。运行时参数均正，零次路径不属于本认证范围。动态循环有额外控制流成本，不把当前 +3.70% 定价归给它，也不建议向真实计算添加重复 MMA。GPU verdict 尚待返回，索引保持未认证。
