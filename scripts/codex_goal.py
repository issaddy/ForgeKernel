"""Drive Codex's interactive /goal and verify the ForgeKernel deliverable."""

import json
import os
import re
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

READY = re.compile(r"^\s*[›❯]\s", re.M)
SESSION_ID = re.compile(r"^\s*([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})(?=\s|$)", re.I)


class Interrupted(Exception):
    pass


def send_goal(tmux, session, objective):
    # Type the slash prefix separately. A large paste can collapse to a composer
    # placeholder, hiding its prefix from Codex's slash-command parser.
    tmux.run("send-keys", "-t", f"={session}:", "-l", "/goal ", check=True)
    tmux.paste(session, objective)


def watch(tmux, session, spec, deadline, state_file):
    objective = spec["condition"]
    started = last_action = time.monotonic()
    poll = float(os.environ.get("FK_GOAL_POLL_S") or 2)
    startup = float(os.environ.get("FK_GOAL_STARTUP_S") or 180)
    settle = float(os.environ.get("FK_GOAL_SETTLE_S") or 15)
    idle = float(os.environ.get("FK_GOAL_IDLE_S") or 2400)
    last_change, last_screen = started, ""
    sent = False
    nudges = 0
    saved_session = state_file.is_file()
    sid = None
    while True:
        now = time.monotonic()
        dead, exit_status = tmux.pane_dead(session)
        screen = tmux.capture(session, 0)
        pane = screen.casefold()
        active = "pursuing goal" in pane
        paused = "goal paused" in pane or "goal stalled" in pane
        limited = any(text in pane for text in ("goal hit usage limits", "goal unmet (", "goal abandoned"))
        achieved = "goal achieved" in pane
        if not saved_session:
            # Codex may hide the footer while working. Capture the ID when
            # visible, then save it as soon as the native Goal is visible.
            if sid is None:
                for line in reversed(screen.rstrip().splitlines()[-4:]):
                    if match := SESSION_ID.match(line):
                        sid = match[1]
                        break
            if sid and (active or paused or limited or achieved):
                state_file.write_text(sid + "\n", encoding="utf-8")
                saved_session = True
        if screen != last_screen:
            last_change, last_screen = now, screen
        # This is the same external check used by the Claude Goal driver.
        if Path(spec["reply"]).is_file():
            result = subprocess.run(spec["check"], shell=True, capture_output=True,
                                    text=True, timeout=5)
            if result.returncode == 0 and "TURN-DELIVERABLE-OK" in result.stdout:
                return 0
        if now >= deadline:
            return 124
        if dead:
            print(f"agent-codex: tmux pane exited ({exit_status}): {screen.strip()[-600:]}",
                  file=sys.stderr)
            return 1

        ready = bool(READY.search(screen))
        if not sent:
            if active:
                sent = True
            elif ready:
                send_goal(tmux, session, "resume" if paused or limited else objective)
                sent, last_action = True, now
            elif now - started >= startup:
                print(f"agent-codex: CLI did not become ready: {screen.strip()[-600:]}",
                      file=sys.stderr)
                return 1
        elif now - last_action >= settle:
            if limited:
                print("agent-codex: Goal reached a usage or token limit", file=sys.stderr)
                return 1
            if ready and (achieved or paused):
                if nudges >= 2:
                    print("agent-codex: Goal stopped before the deliverable check passed",
                          file=sys.stderr)
                    return 1
                send_goal(tmux, session, "resume" if paused else objective)
                nudges += 1
                last_action = now
            elif now - last_change >= idle:
                print(f"agent-codex: Goal stopped making progress: {screen.strip()[-600:]}",
                      file=sys.stderr)
                return 1
        time.sleep(min(poll, max(0.0, deadline - now)))


def run_goal(cmd, turn_dir):
    seconds = float(os.environ["FK_GOAL_DEADLINE"]) - time.time()
    if seconds <= 0:
        return 124
    spec = json.loads((turn_dir / "spec.json").read_text(encoding="utf-8"))
    state_file = Path(spec["session_id_file"])
    if state_file.is_file():
        sid = state_file.read_text(encoding="utf-8").strip()
        cmd = [cmd[0], "resume", sid, *cmd[1:]]
    helpers = runpy.run_path(str(Path(__file__).with_name("claude-goal.py")))
    tmux = helpers["Tmux"](os.environ.get("FK_GOAL_TMUX_SOCKET", ""))
    session = f"fkgoalref-codex-{os.getpid()}"

    def stop(_signum, _frame):
        raise Interrupted()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    with tempfile.TemporaryDirectory(prefix="forge-codex-goal-") as temp:
        try:
            tmux.start_agent(session, cmd, Path.cwd(), os.environ, Path(temp))
            rc = watch(tmux, session, spec, time.monotonic() + seconds, state_file)
            if rc == 0:
                sys.stdout.write(Path(spec["reply"]).read_text(encoding="utf-8"))
            return rc
        except Interrupted:
            return 143
        finally:
            tmux.kill(session)
