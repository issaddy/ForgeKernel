# Claude Code integration

ForgeKernel defaults to [Antigravity CLI in a GPU container](../../docs/design/v1-launch.md#antigravity-cli).
Claude Code can invoke the shared [forge-start skill](forge-start/SKILL.md) and
can also be selected explicitly as an agent backend.

## Use the startup skill from Claude Code

```bash
mkdir -p ~/.claude/skills
cp -R integrations/claude-code/forge-start ~/.claude/skills/
```

In Claude Code, type `/forge-start Preflight tasks/rmsnorm in the GPU container.
Do not start a run.` The launch script uses the task's backend settings and
repository defaults. The repository's `skills/` directory serves the internal
architecture and coding agents.

## Select Claude Code as an agent backend

Install Claude Code where `forge` runs. Sign in there or set `ANTHROPIC_API_KEY`
or `CLAUDE_CODE_OAUTH_TOKEN` in `local.env`. Set `backend = "claude"` and a Claude
model available to your account in the task's `[agent]` table. Role-specific
settings belong in `[agent.teacher]` or `[agent.student]`. Set `FK_CLAUDE_BIN`
in `local.env` if the CLI is outside PATH.

Run the startup scripts' CPU-only tests with Python 3.11 or newer:

```bash
python3 -m unittest discover -s integrations/claude-code/forge-start/tests
```
