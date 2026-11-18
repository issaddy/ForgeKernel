"""Exercise the MyFlicker adapter without a model, GPU, or real sandbox."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from forgekernel.agents import MYFLICKER_POLICY_FILE
from forgekernel.loop.agent_policy import from_config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("subtype, returncode", [("success", 0), ("error", 1)])
def test_myflicker_turn_uses_text_stdin_and_returns_terminal_reply(
    tmp_path, subtype, returncode
):
    agent = tmp_path / "myflicker"
    agent.write_text(f"""#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
assert json.loads(pathlib.Path(args[args.index('--config') + 1]).read_text())['agentBackend'] == 'myflicker'
assert args[args.index('--thinking-level') + 1] == 'xhigh'
assert '--input-format' not in args and '--backend' not in args
assert args[args.index('--output-format') + 1] == 'stream-json' and '-q' in args
assert '--unshare-net' in pathlib.Path(os.environ['TAKUMI_SHELL']).read_text()
assert sys.stdin.read() == {'one' + chr(10) + '二'!r}
print(json.dumps({{'type': 'assistant', 'message': 'intermediate'}}))
print(json.dumps({{'type': 'result', 'subtype': {subtype!r}, 'content': 'terminal reply'}}))
""", encoding="utf-8")
    agent.chmod(0o755)
    sandbox = tmp_path / "bwrap"
    sandbox.write_text("#!/bin/sh\nexit 0\n")
    sandbox.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cmd = f"{sys.executable} {SCRIPTS / 'agent-myflicker.py'} model xhigh 5"
    policy = from_config({}, oracle_mode="source", repo_root=tmp_path, teacher_trees=[])
    policy.prepare_turn(cmd, "student", workspace)
    stored = json.loads((workspace / MYFLICKER_POLICY_FILE).read_text())
    assert not stored["network"] and str(tmp_path / "local.env") in stored["filesystem"]["denyRead"]
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "agent-myflicker.py"), "model", "xhigh", "5"],
        input="one\n二", text=True, capture_output=True, cwd=workspace,
        env={**os.environ, "HOME": str(tmp_path),
             "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        timeout=15,
    )
    assert result.returncode == returncode, result.stderr
    if returncode:
        assert "terminal reply" in result.stderr and not result.stdout
    else:
        assert result.stdout == "terminal reply"


def test_goal_persists_exact_session_and_checks_deliverable(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import myflicker_goal

    journals = tmp_path / "journals"
    journals.mkdir()
    reply = tmp_path / "reply"
    state_file = tmp_path / "sid"
    objective = "evaluated objective"
    spec = {"condition": objective, "reply": str(reply),
            "check": "printf TURN-DELIVERABLE-OK"}
    previous = journals / "old.jsonl"
    previous.touch()

    class Tmux:
        polls = 0

        def pane_dead(self, _session):
            self.polls += 1
            if self.polls == 3:
                reply.write_text("done")
            return False, ""

        def capture(self, _session, _lines):
            return "❯ "

        def run(self, *_args, **_kw):
            pass

        def paste(self, _session, text):
            assert text == objective
            entry = {"type": "message", "role": "attachment",
                     "content": {"type": "goal_status", "condition": objective,
                                 "outcome": "set"}}
            (journals / "new.jsonl").write_text(json.dumps(entry) + "\n")

    monkeypatch.setenv("FK_GOAL_POLL_S", "0.01")
    assert myflicker_goal.watch(Tmux(), "test", spec, myflicker_goal.time.monotonic() + 2,
                                state_file, journals, {previous}, None) == 0
    assert state_file.read_text().strip() == "new"


def test_goal_launch_resumes_saved_session_id(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import myflicker_goal

    sid = "8bb2caca-0bb9-4cef-93f7-bf155cdeec70"
    saved = tmp_path / "saved"
    saved.write_text(sid + "\n")
    (tmp_path / "spec.json").write_text(json.dumps({"session_id_file": str(saved)}))
    events = []

    class Tmux:
        def start_agent(self, _session, argv, _cwd, _env, _directory):
            events.append(argv)

        def kill(self, _session):
            events.append("stop")

    monkeypatch.setattr(myflicker_goal.runpy, "run_path", lambda _src: {"Tmux": lambda _: Tmux()})
    monkeypatch.setattr(myflicker_goal.signal, "signal", lambda _sig, _callback: None)
    monkeypatch.setattr(myflicker_goal, "watch", lambda *_args: 124)
    monkeypatch.setenv("FK_GOAL_DEADLINE", str(myflicker_goal.time.time() + 10))
    assert myflicker_goal.run_goal(["myflicker"], tmp_path, tmp_path, tmp_path) == 124
    assert events == [["myflicker", "--resume", sid], "stop"]
