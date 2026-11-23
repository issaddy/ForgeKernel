"""Build an agent command from the task settings for each role."""

from __future__ import annotations

import shlex
import sys

from forgekernel import provider as provider_mod
from forgekernel.paths import repo_root

AGENT_SCRIPT = "scripts/agent-claude.sh"
CODEX_SCRIPT = "scripts/agent-codex.py"
MYFLICKER_SCRIPT = "scripts/agent-myflicker.py"
MYFLICKER_POLICY_FILE = ".agents/forgekernel-myflicker.json"
AGY_SCRIPT = "scripts/agent-agy.py"
AGY_POLICY_FILE = ".agents/forgekernel-agy.json"

CODEX_POLICY_FILE = ".agents/forgekernel-turn.json"


def python_agent_script(cmd: str | None) -> str | None:
    try:
        argv = shlex.split(cmd or "")
        return next((script for script in (CODEX_SCRIPT, MYFLICKER_SCRIPT, AGY_SCRIPT)
                     if str(repo_root() / script) in argv), None)
    except ValueError:
        return None


def is_codex_command(cmd: str | None) -> bool:
    return python_agent_script(cmd) == CODEX_SCRIPT


def agent_command(spec, role: str, env: dict | None = None) -> str:
    """The shell command the loop runs for one ``role`` turn (prompt on stdin). A provider
    other than ``native`` rides as environment assignments in front of it: the endpoint and
    the name of the variable that holds the key, never the key."""
    if spec.backend == "claude":
        launcher = ["bash", str(repo_root() / AGENT_SCRIPT)]
    elif spec.backend == "codex":
        launcher = [sys.executable, str(repo_root() / CODEX_SCRIPT)]
    elif spec.backend == "myflicker":
        launcher = [sys.executable, str(repo_root() / MYFLICKER_SCRIPT)]
    elif spec.backend == "agy":
        launcher = [sys.executable, str(repo_root() / AGY_SCRIPT)]
    else:
        raise ValueError(f"backend {spec.backend!r} is not supported")
    grant = provider_mod.grant(spec.backend, spec.provider, env)
    argv = [*launcher, spec.model, spec.effort or "", str(spec.timeout_s)]
    prefix = ["env", *(f"{k}={v}" for k, v in grant.items())] if grant else []
    return " ".join(shlex.quote(a) for a in [*prefix, *argv])
