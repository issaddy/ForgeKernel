# Codex integration

ForgeKernel defaults to [MyFlicker in a GPU container](../myflicker/README.md).
Codex can invoke the shared startup skill and can also be selected explicitly as
an agent backend.

## Use the startup skill from Codex

From the ForgeKernel checkout:

```bash
mkdir -p .agents/skills
cp -R integrations/claude-code/forge-start .agents/skills/
```

Use `~/.agents/skills/` for a user-wide install. In Codex, ask
`$forge-start Preflight tasks/rmsnorm in the GPU container. Do not start a run.`
The launch script uses the task's backend settings and repository defaults.

## Select Codex as an agent backend

Install and authenticate the Codex CLI where `forge` runs. In the task's `task.toml`:

```toml
[agent]
backend = "codex"
provider = "native"
model = "gpt-6.2-sol"
effort = "medium"

[agent.teacher]
effort = "xhigh"
```

Set `FK_CODEX_BIN` in `local.env` if the CLI is outside PATH. Configure a custom
Responses API gateway in `global_config/backends.toml` and select it with
`provider` if needed. The launcher can reuse a CLI supplied with
`--codex /path/to/codex` or install it with npm.

Regular turns use `codex exec`. Reference generation uses the native `/goal` in tmux
and resumes by its recorded session ID. Skills are staged in `.agents/skills/`.
