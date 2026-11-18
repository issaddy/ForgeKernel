"""Drive MyFlicker's native /goal and resume a previously recorded session."""

import hashlib
import json
import os
import re
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

from codex_goal import Interrupted, send_goal

READY = re.compile(r"^[ \t]*[›❯>](?:[ \t]|$)", re.M)


def project_journals(config_root):
    cwd = str(Path.cwd())
    name = re.sub("[^a-zA-Z0-9]+", "-", cwd).strip("-").lower()
    if len(name) > 80:
        name = name[:80] + "-" + hashlib.sha1(cwd.encode()).hexdigest()[:8]
    return config_root / "projects" / name


def goal_outcome(journal, objective):
    if not journal.is_file():
        return None
    outcome = None
    with journal.open(encoding="utf-8") as records:
        for line in records:
            if '"goal_status"' not in line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            content = entry.get("content")
            if (entry.get("role") == "attachment" and isinstance(content, dict)
                    and content.get("type") == "goal_status"
                    and content.get("condition") == objective):
                outcome = content["outcome"]
    return outcome


def watch(tmux, session, spec, deadline, state_file, journals, previous, sid):
    objective = spec["condition"]
    poll = float(os.environ.get("FK_GOAL_POLL_S") or 2)
    startup = float(os.environ.get("FK_GOAL_STARTUP_S") or 180)
    last_send = None
    resumed_outcome = None
    started = time.monotonic()
    while (now := time.monotonic()) < deadline:
        if Path(spec["reply"]).is_file():
            result = subprocess.run(spec["check"], shell=True, capture_output=True,
                                    text=True, timeout=5)
            if result.returncode == 0 and "TURN-DELIVERABLE-OK" in result.stdout:
                return 0

        dead, status = tmux.pane_dead(session)
        screen = tmux.capture(session, 0)
        if dead:
            print(f"agent-myflicker: tmux pane exited ({status}): {screen.strip()[-600:]}",
                  file=sys.stderr)
            return 1
        if sid is None:
            for journal in set(journals.glob("*.jsonl")) - previous:
                if goal_outcome(journal, objective) is not None:
                    sid = journal.stem
                    state_file.write_text(sid + "\n", encoding="utf-8")
                    break

        outcome = goal_outcome(journals / f"{sid}.jsonl", objective) if sid else None
        if last_send is None:
            if READY.search(screen):
                if outcome == "set":
                    # Resume restores the active goal but needs a new user turn.
                    tmux.paste(session, "Continue working toward the current goal.")
                else:
                    action = "resume" if outcome in {"paused", "blocked", "usageLimited"} else objective
                    send_goal(tmux, session, action)
                    resumed_outcome = outcome if action == "resume" else None
                last_send = now
            elif now - started >= startup:
                print(f"agent-myflicker: CLI did not become ready: {screen.strip()[-600:]}",
                      file=sys.stderr)
                return 1
        elif now - last_send >= startup and outcome is None:
            print("agent-myflicker: native Goal was not recorded", file=sys.stderr)
            return 1
        elif (now - last_send >= 15 and outcome != resumed_outcome
              and outcome in {"blocked", "paused", "usageLimited", "met"}):
            if READY.search(screen):
                print(f"agent-myflicker: Goal stopped ({outcome}) before ForgeKernel accepted it",
                      file=sys.stderr)
                return 1
        time.sleep(min(poll, max(0.0, deadline - now)))
    return 124


def run_goal(cmd, turn_dir, directory, config_root):
    seconds = float(os.environ["FK_GOAL_DEADLINE"]) - time.time()
    if seconds <= 0:
        return 124
    spec = json.loads((turn_dir / "spec.json").read_text(encoding="utf-8"))
    state_file = Path(spec["session_id_file"])
    sid = state_file.read_text(encoding="utf-8").strip() if state_file.is_file() else None
    if sid:
        cmd.extend(("--resume", sid))
    journals = project_journals(config_root)
    previous = set(journals.glob("*.jsonl"))
    helpers = runpy.run_path(str(Path(__file__).with_name("claude-goal.py")))
    tmux = helpers["Tmux"](os.environ.get("FK_GOAL_TMUX_SOCKET", ""))
    session = f"fkgoalref-myflicker-{os.getpid()}"

    def stop(_signum, _frame):
        raise Interrupted()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        tmux.start_agent(session, cmd, Path.cwd(), os.environ, directory)
        rc = watch(tmux, session, spec, time.monotonic() + seconds, state_file, journals,
                   previous, sid)
        if rc == 0:
            sys.stdout.write(Path(spec["reply"]).read_text(encoding="utf-8"))
        return rc
    except Interrupted:
        return 143
    finally:
        tmux.kill(session)
