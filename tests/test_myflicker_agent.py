"""Exercise the MyFlicker adapter without a model, GPU, or real sandbox."""

import json
import os
import shlex
import subprocess
import sys
import time
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



@pytest.mark.parametrize("resume", [False, True])
def test_reference_turns_share_one_session_until_check_passes(tmp_path, resume):
    agent = tmp_path / "myflicker"
    agent.write_text(f"""#!{sys.executable}
import json, pathlib, sys
args = sys.argv[1:]
config = json.loads(pathlib.Path(args[args.index('--config') + 1]).read_text())
assert config['goal'] is False
assert config['appendSystemPrompt'] == 'reference task'
assert '-q' in args
assert args[args.index('--input-format') + 1] == 'stream-json'
assert args[args.index('--output-format') + 1] == 'stream-json'
sid = 'abcd1234'
if {resume!r}:
    assert args[args.index('--resume') + 1] == sid
else:
    assert '--resume' not in args
first = json.loads(sys.stdin.readline())
assert first['type'] == 'user' and first['message']['role'] == 'user'
assert first['message']['content'].endswith('generate the reference')
print(json.dumps({{'type': 'system', 'subtype': 'init', 'session_id': sid}}), flush=True)
pathlib.Path('report').write_text('first report')
print(json.dumps({{'type': 'result', 'subtype': 'success', 'session_id': sid}}), flush=True)
second = json.loads(sys.stdin.readline())
assert 'TURN-DELIVERABLE-MISSING: run an evaluation' in second['message']['content']
assert pathlib.Path('turn/sid').read_text().strip() == sid
pathlib.Path('report').write_text('checked report')
pathlib.Path('evaluated').touch()
print(json.dumps({{'type': 'result', 'subtype': 'success', 'session_id': sid}}), flush=True)
pathlib.Path('after-acceptance').write_text(sys.stdin.read())
""", encoding="utf-8")
    agent.chmod(0o755)
    sandbox = tmp_path / "bwrap"
    sandbox.write_text("#!/bin/sh\nexit 0\n")
    sandbox.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    turn = workspace / "turn"
    turn.mkdir()
    sid_file = turn / "sid"
    if resume:
        sid_file.write_text("abcd1234\n")
    check = (
        "if test -f evaluated; then echo TURN-DELIVERABLE-OK; "
        "else echo 'TURN-DELIVERABLE-MISSING: run an evaluation'; exit 1; fi"
    )
    (turn / "spec.json").write_text(json.dumps({
        "condition": "generate the reference", "reply": str(workspace / "report"),
        "check": check, "session_id_file": str(sid_file),
    }))

    launcher = [sys.executable, str(SCRIPTS / "agent-myflicker.py"), "model", "xhigh", "5"]
    policy = from_config({}, oracle_mode="source", repo_root=tmp_path, teacher_trees=[])
    policy.prepare_turn(shlex.join(launcher), "student", workspace)
    result = subprocess.run(
        [*launcher, str(turn)], input="reference task", cwd=workspace, text=True,
        capture_output=True, timeout=15,
        env={**os.environ, "HOME": str(tmp_path),
             "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
             "FK_MYFLICKER_BIN": str(agent), "FK_AGENT_DRIVER": "goal",
             "FK_GOAL_DEADLINE": str(time.time() + 10)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "checked report"
    assert sid_file.read_text().strip() == "abcd1234"
    assert (workspace / "after-acceptance").read_text() == ""


def test_reference_turn_obeys_deadline_when_cli_stops_reading(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import myflicker_goal

    (tmp_path / "spec.json").write_text(json.dumps({
        "condition": "x" * 1_000_000, "reply": str(tmp_path / "missing"),
        "check": "exit 1", "session_id_file": str(tmp_path / "sid"),
    }))
    monkeypatch.setenv("FK_GOAL_DEADLINE", str(time.time() + 0.3))
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    assert myflicker_goal.run_goal(cmd, tmp_path) == 124
