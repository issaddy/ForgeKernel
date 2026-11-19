# MyFlicker in a GPU container

Run ForgeKernel, MyFlicker, and the GPU jobs in the same Linux container. Use a
checkout in a persistent directory if you plan to resume runs after recreating
the container.

1. Install the Linux MyFlicker CLI from your distribution channel and sign in
   inside the container. Check `myflicker --version` in the shell that runs
   ForgeKernel. Use the same shell to launch `forge` so it can call `myflicker` from PATH.
2. Install Bubblewrap (`bwrap`) and ensure the container permits the user, mount,
   PID, and network namespaces needed by the shell sandbox.
3. Use Python 3.11 or newer with CUDA PyTorch and, for CuTe DSL tasks,
   `nvidia-cutlass-dsl`. Follow the [quick start](../../README.md#quick-start-in-a-gpu-container)
   to install ForgeKernel in that environment.
4. Copy `local.env.example` on first setup. Keep `FK_DELIVERY=local` and set
   `FK_GPU` to the CUDA device index visible in the container. For H800, use
   `arch = "sm90"` in `task.toml`.

Both agents default to `wanqing/gpt-6.1-sol` with `xhigh`. The native
provider uses the account configured in MyFlicker. Override the model or effort
in the task's `[agent]`, `[agent.teacher]`, or `[agent.student]` settings.

## Start a run

```bash
forge new tasks/rmsnorm --dry-run
forge new tasks/rmsnorm
```

Run the `forge run` command printed by `forge new`. Use `forge status` with the
same run directory to see progress. Repeat the original `forge run` command to
resume. Run state and reports live in the checkout's `.artifacts/` directory.

RMSNorm includes a PyTorch reference. For tasks without a reference, use a MyFlicker
version that accepts `--input-format stream-json`. ForgeKernel sends each round
to the same MyFlicker session, checks the report and GPU results, and sends the
check feedback if another round is needed. Start these tasks with the same
`forge run` command. They use the configured reference-generation budget.

## Use the shared startup skill

From the checkout inside the container:

```bash
mkdir -p .agents/skills
cp -R integrations/claude-code/forge-start .agents/skills/
```

Open MyFlicker in the checkout and ask it to use `forge-start` to preflight
`tasks/rmsnorm` without starting a run. To start the run, give the task, the
container's CUDA device index, and the budget. The launcher resolves both role
configurations and uses `--role both` for the container setup.
