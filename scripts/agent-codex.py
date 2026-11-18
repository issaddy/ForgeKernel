#!/usr/bin/env python3
"""Launch a Codex turn or a native Goal session for ForgeKernel."""

import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def event_error(events) -> str:
    reconnect_error = ""
    for line in events:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "turn.failed":
            return line.strip()
        if kind == "error":
            reconnect_error = line.strip()
        elif kind == "turn.completed":
            reconnect_error = ""
    return reconnect_error


def run_turn(cmd: list[str], timeout: int) -> int:
    run = runpy.run_path(str(Path(__file__).with_name("timeout-run.py")))["run"]
    with tempfile.TemporaryDirectory(prefix="forge-codex-turn-") as tmp:
        reply = Path(tmp) / "reply.txt"
        with (Path(tmp) / "events.jsonl").open("w+", encoding="utf-8") as events:
            rc = run([*cmd, "--json", "--output-last-message", str(reply), "-"], timeout,
                     stdout=events)
            events.seek(0)
            error = event_error(events)
        if error:
            print(f"agent-codex: {error}", file=sys.stderr)
        if rc or error:
            return rc or 1
        final = reply.read_text(encoding="utf-8") if reply.is_file() else ""
        if not final.strip():
            print("agent-codex: the turn returned no final message", file=sys.stderr)
            return 1
        sys.stdout.write(final)
        return 0


def main() -> int:
    model, effort, timeout_s = sys.argv[1:4]
    binary = os.environ.get("FK_CODEX_BIN") or "codex"

    policy = json.loads(Path(".agents/forgekernel-turn.json").read_text(encoding="utf-8"))
    filesystem = ", ".join(
        f"{quote(path)} = {quote(access)}" for path, access in policy["filesystem"].items()
    )
    config = {
        "approval_policy": '"never"',
        "default_permissions": '"forgekernel"',
        "permissions.forgekernel.extends": '":workspace"',
        "permissions.forgekernel.filesystem": "{" + filesystem + "}",
        "permissions.forgekernel.network.enabled": json.dumps(policy["network_enabled"]),
        "web_search": quote(policy["web_search"]),
        "features.apps": "false",
    }
    if effort:
        config["model_reasoning_effort"] = quote(effort)
    excludes = ["*API_KEY*", "*TOKEN*", "*PASSWORD*", "*SECRET*"]
    if base_url := os.environ.get("FK_CODEX_BASE_URL"):
        key_env = os.environ["FK_PROVIDER_KEY_ENV"]
        if not os.environ.get(key_env):
            sys.exit(f"agent-codex: {key_env} is required by the selected provider")
        excludes.append(key_env)
        config.update({
            "model_provider": '"forgekernel"',
            "model_providers.forgekernel.name": '"ForgeKernel gateway"',
            "model_providers.forgekernel.base_url": quote(base_url),
            "model_providers.forgekernel.env_key": quote(key_env),
            "model_providers.forgekernel.wire_api": '"responses"',
        })
    config["shell_environment_policy.exclude"] = json.dumps(excludes, ensure_ascii=False)

    goal = os.environ.get("FK_AGENT_DRIVER") == "goal"
    flags = ["--strict-config", "--model", model]
    if goal:
        cmd = [binary, "--no-daemon", "--no-alt-screen", *flags]
        config["features.goals"] = "true"
        config["tui.status_line"] = '["session-id"]'
        config["disable_paste_burst"] = "true"
        config["developer_instructions"] = quote(sys.stdin.read())
        config["projects"] = "{" + quote(str(Path.cwd())) + '={trust_level="untrusted"}}'
    else:
        cmd = [binary, "exec", "--color", "never", *flags,
               "--ignore-user-config", "--skip-git-repo-check", "--ephemeral"]
    for key, value in config.items():
        cmd.extend(("-c", f"{key}={value}"))
    if goal:
        from codex_goal import run_goal

        return run_goal(cmd, Path(sys.argv[4]))
    return run_turn(cmd, int(timeout_s))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError) as exc:
        sys.exit(f"agent-codex: {exc}")
