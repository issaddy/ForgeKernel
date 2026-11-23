#!/usr/bin/env python3
"""Run an Antigravity CLI turn or a streamed reference-generation session."""

import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

from stream_goal import agy_ok, run_goal


def run_turn(cmd, timeout):
    run = runpy.run_path(str(Path(__file__).with_name("timeout-run.py")))["run"]
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        rc = run([*cmd, "--input-format", "text", "--output-format", "json"],
                 timeout, stdout=output)
        if rc == 124:
            return rc
        output.seek(0)
        raw = output.read()
        if rc and not raw:
            return rc
        result = json.loads(raw)
    if rc == 0 and agy_ok(result) and result.get("response"):
        sys.stdout.write(result["response"])
        return 0
    message = result.get("error") or result.get("finished_reason") or "no successful terminal result"
    print(f"agent-agy: {message}", file=sys.stderr)
    return rc or 1


def main():
    model, effort, timeout = sys.argv[1:4]
    settings = Path(".agents/forgekernel-agy.json").resolve()
    cmd = [os.environ.get("FK_AGY_BIN") or "agy", "--model", model, "--mode", "accept-edits",
           "--sandbox", "--settings", str(settings)]
    if effort:
        cmd.extend(("--effort", effort))
    if os.environ.get("FK_AGENT_DRIVER") == "goal":
        return run_goal(cmd, Path(sys.argv[4]), backend="agy", prompt=sys.stdin.read())
    return run_turn(cmd, int(timeout))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(f"agent-agy: {exc}")
