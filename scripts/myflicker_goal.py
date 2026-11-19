"""Drive reference generation through ordinary turns in one MyFlicker session."""

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path


def user_turn(prompt):
    event = {"type": "user", "message": {"role": "user", "content": prompt}}
    return (json.dumps(event) + "\n").encode()


def check_deliverable(spec, deadline):
    check = subprocess.run(spec["check"], shell=True, capture_output=True, text=True,
                           timeout=min(5, max(0, deadline - time.monotonic())))
    output = check.stdout.strip()
    if check.returncode == 0 and "TURN-DELIVERABLE-OK" in output:
        return True, output
    if check.returncode == 1 and output:
        return False, output
    raise ValueError(f"reference check failed: {check.stderr.strip() or output or check.returncode}")


def drive(proc, spec, deadline, session_file, session_id):
    reply = Path(spec["reply"])
    prompt = spec["condition"]
    if session_id:
        prompt = "Continue the reference task from the current workspace and evaluation records.\n\n" + prompt
    pending = user_turn(prompt)
    stdin, stdout = proc.stdin.fileno(), proc.stdout.fileno()
    os.set_blocking(stdin, False)
    partial = b""
    next_check = time.monotonic()
    while (now := time.monotonic()) < deadline:
        # Accept a finished report even if the CLI has not finished its current turn.
        if now >= next_check:
            next_check = now + 2
            if reply.is_file() and check_deliverable(spec, deadline)[0]:
                return 0
        readable, writable, _ = select.select(
            [stdout], [stdin] if pending else [], [], min(2, max(0, deadline - time.monotonic()))
        )
        if writable:
            pending = pending[os.write(stdin, pending):]
        if not readable:
            continue
        chunk = os.read(stdout, 65536)
        if not chunk:
            if reply.is_file() and check_deliverable(spec, deadline)[0]:
                return 0
            print("agent-myflicker: stream ended before the reference passed its check", file=sys.stderr)
            return proc.poll() or 1
        *lines, partial = (partial + chunk).split(b"\n")
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if not session_id and (sid := event.get("session_id") or event.get("sessionId")):
                session_id = sid
                temporary = session_file.with_suffix(".tmp")
                temporary.write_text(session_id + "\n", encoding="utf-8")
                temporary.replace(session_file)
            if event.get("type") != "result":
                continue
            accepted, feedback = check_deliverable(spec, deadline)
            if accepted:
                return 0
            if event.get("subtype") != "success":
                message = event.get("result") or event.get("content") or "no successful terminal result"
                print(f"agent-myflicker: {message}", file=sys.stderr)
                return 1
            pending = user_turn(
                "Continue working on the reference in this session. ForgeKernel's check found:\n"
                + feedback
            )
    return 124


def run_goal(cmd, turn_dir):
    seconds = float(os.environ["FK_GOAL_DEADLINE"]) - time.time()
    if seconds <= 0:
        return 124
    deadline = time.monotonic() + seconds
    spec = json.loads((turn_dir / "spec.json").read_text(encoding="utf-8"))
    session_file = Path(spec["session_id_file"])
    session_id = session_file.read_text(encoding="utf-8").strip() if session_file.is_file() else ""
    argv = [*cmd, "-q", "--input-format", "stream-json", "--output-format", "stream-json"]
    if session_id:
        argv.extend(("--resume", session_id))
    # Keep the CLI in the controller's process group so its existing timeout and
    # interruption handling also reaches the CLI and its tool processes.
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        try:
            rc = drive(proc, spec, deadline, session_file, session_id)
            if rc == 0:
                sys.stdout.write(Path(spec["reply"]).read_text(encoding="utf-8"))
            return rc
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                return 124
            raise
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
