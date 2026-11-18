#!/usr/bin/env python3
"""Run MyFlicker's native coding backend on a Linux ForgeKernel controller."""

import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from myflicker_goal import run_goal


def config_dir():
    if path := os.environ.get("TAKUMI_CONFIG_DIR"):
        return Path(path).resolve()
    return Path(os.environ.get("TAKUMI_CONFIG_HOME") or Path.home()) / ".myflicker"


def prepare_shell(directory, policy):
    """Isolate tool Bash without taking the model client offline."""
    if not (bwrap := shutil.which("bwrap")):
        raise ValueError("install bubblewrap on the Linux controller")
    filesystem = policy["filesystem"]
    argv = [bwrap, "--die-with-parent", "--unshare-pid", "--ro-bind", "/", "/",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    if not policy["network"]:
        argv.append("--unshare-net")
    shell_tmp = directory / "shell-tmp"
    shell_tmp.mkdir()
    os.environ["TMPDIR"] = str(shell_tmp)
    for p in [shell_tmp, Path.cwd(), *map(Path, filesystem["allowWrite"])]:
        if p.exists():
            argv.extend(("--bind", str(p), str(p)))
    for p in map(Path, filesystem["denyWrite"]):
        if p.exists():
            argv.extend(("--ro-bind", str(p), str(p)))
    for p in [config_dir(), *map(Path, filesystem["denyRead"])]:
        if p.is_dir():
            argv.extend(("--tmpfs", str(p)))
        elif p.is_file():
            argv.extend(("--ro-bind", "/dev/null", str(p)))
    for name in os.environ:
        if any(part in name.upper() for part in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
            argv.extend(("--unsetenv", name))
    argv.extend(("--", "/bin/bash"))
    shell = directory / "forge-bash"
    shell.write_text("#!/bin/sh\nexec " + shlex.join(argv) + ' "$@"\n', encoding="utf-8")
    shell.chmod(0o700)
    subprocess.run([str(shell), "-c", "true"], check=True, stdin=subprocess.DEVNULL)
    return shell


def run_turn(cmd, timeout, prompt, directory):
    run = runpy.run_path(str(Path(__file__).with_name("timeout-run.py")))["run"]
    message = {"type": "user", "message": {"role": "user", "content": prompt}}
    request = directory / "turn.jsonl"
    request.write_text(json.dumps(message, ensure_ascii=False) + "\n", encoding="utf-8")
    with request.open() as input_file, tempfile.TemporaryFile(mode="w+") as output_file:
        rc = run([*cmd, "-q", "--input-format", "stream-json", "--output-format", "stream-json"],
                 timeout, stdout=output_file, stdin=input_file)
        output_file.seek(0)
        result = None
        for line in output_file:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                result = event
    if result and result.get("subtype") == "success" and rc == 0:
        if reply := result.get("result") or result.get("content"):
            sys.stdout.write(reply)
            return 0
    message = (result or {}).get("result") or (result or {}).get("content")
    print(f"agent-myflicker: {message or 'no successful terminal result'}", file=sys.stderr)
    return rc or 1


def main():
    model, effort, timeout = sys.argv[1:4]
    policy = json.loads(Path(".agents/forgekernel-myflicker.json").read_text(encoding="utf-8"))
    binary = os.environ.get("FK_MYFLICKER_BIN") or "myflicker"
    cmd = [binary, "--backend", "myflicker", "--cwd", str(Path.cwd()), "--model", model,
           "--approval-mode", "yolo", "--mcp-config", '{"mcpServers":{}}']
    if effort:
        cmd.extend(("--thinking-level", effort))
    prompt = sys.stdin.read()
    # Keep the shell launcher outside the writable workspace and visible across tmpfs.
    with tempfile.TemporaryDirectory(prefix=".fk-myflicker-", dir=Path.home()) as tmp:
        directory = Path(tmp)
        os.environ["TAKUMI_SHELL"] = str(prepare_shell(directory, policy))
        settings = policy["config"]
        auth = config_dir()
        settings["permissions"]["deny"].extend([f"Read(/{auth})", f"Read(/{auth}/**)"])
        goal = os.environ.get("FK_AGENT_DRIVER") == "goal"
        if goal:
            settings.update(goal=True, appendSystemPrompt=prompt)
        settings_path = directory / "config.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")
        cmd.extend(("--config", str(settings_path)))
        if goal:
            return run_goal(cmd, Path(sys.argv[4]), directory, auth)
        return run_turn(cmd, int(timeout), prompt, directory)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"agent-myflicker: {exc}")
