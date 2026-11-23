"""Verify the agy command and NDJSON exchange without a model or GPU."""

import json
import os
import shlex
import subprocess
import sys
import time

import pytest

from forgekernel.agents import AGY_POLICY_FILE, agent_command
from forgekernel.loop.agent_policy import AgentPolicy
from forgekernel.loop.cli import goal_leg_findings
from forgekernel.task import AgentSpec


@pytest.mark.parametrize("mode", ["turn", "timeout", "wall-timeout", "reference", "resume"])
def test_agy_backend(tmp_path, mode):
    timeout = 1 if mode == "wall-timeout" else 5
    argv = shlex.split(agent_command(AgentSpec("agy", "model", "high", timeout_s=timeout), "student"))
    assert goal_leg_findings([shlex.join(argv)], {}) == []
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = workspace / "hidden.py"
    ledger = workspace / "ledger.jsonl"
    policy = AgentPolicy("source", tmp_path, ref_source_paths=(reference,), deny_write=(ledger,))
    policy.prepare_turn(shlex.join(argv), "student", workspace)
    settings = json.loads((workspace / AGY_POLICY_FILE).read_text())
    assert {"unsandboxed(*)", "read_url(*)", f"read_file({reference})", f"write_file({ledger})"} <= set(
        settings["permissions"]["deny"]
    )
    assert not (workspace / ".claude/settings.json").exists()
    hooks = json.loads((workspace / ".agents/hooks.json").read_text())
    hook = hooks["forgekernel-agent-policy"]["PreToolUse"][0]["hooks"][0]["command"]
    check = subprocess.run(shlex.split(hook), text=True, capture_output=True, check=True,
                           input=json.dumps({"toolCall": {"name": "search_web", "args": {"query": "test"}}}))
    assert json.loads(check.stdout)["decision"] == "deny"

    agent = tmp_path / "agy"
    agent.write_text(f"""#!{sys.executable}
import json, sys, time
from pathlib import Path

args = sys.argv[1:]
mode = {mode!r}
def flag(name):
    return args[args.index(name) + 1]
assert flag('--model') == 'model' and flag('--effort') == 'high'
assert '--sandbox' in args and Path(flag('--settings')).is_file()
if mode in ('turn', 'timeout', 'wall-timeout'):
    assert flag('--input-format') == 'text' and flag('--output-format') == 'json'
    assert sys.stdin.read() == 'work please'
    if mode == 'wall-timeout':
        print('{{"status":', flush=True)
        time.sleep(30)
    result = {{'status': 'SUCCESS', 'response': 'partial' if mode == 'timeout' else 'text reply'}}
    if mode == 'timeout':
        result['finished_reason'] = 'print_timeout'
    print(json.dumps(result), flush=True)
    sys.exit(0)

assert flag('--input-format') == flag('--output-format') == 'stream-json'
if mode == 'resume':
    assert flag('--conversation') == 'conversation-1'
else:
    assert '--conversation' not in args
first = json.loads(sys.stdin.readline())
assert first['event'] == 'user' and first['message']['content'].endswith('work please\\n\\nwrite the report')
print(json.dumps({{'event': 'init', 'conversation_id': 'conversation-1'}}), flush=True)
result = {{'event': 'result', 'result': {{'status': 'SUCCESS', 'response': 'text only'}}}}
print(json.dumps(result), flush=True)
second = json.loads(sys.stdin.readline())
assert 'TURN-DELIVERABLE-MISSING: report needed' in second['message']['content']
Path('reply.md').write_text('approved report')
print(json.dumps(result), flush=True)
Path('after-approval').write_text(sys.stdin.read())
""")
    agent.chmod(0o755)
    env = {**os.environ, "FK_AGY_BIN": str(agent)}
    env.pop("FK_AGENT_DRIVER", None)
    if mode in ("reference", "resume"):
        sid = tmp_path / "sid"
        if mode == "resume":
            sid.write_text("conversation-1\n")
        (tmp_path / "spec.json").write_text(json.dumps({
            "condition": "write the report", "reply": str(workspace / "reply.md"),
            "session_id_file": str(sid),
            "check": "if test -f reply.md; then echo TURN-DELIVERABLE-OK; "
                     "else echo 'TURN-DELIVERABLE-MISSING: report needed'; exit 1; fi",
        }))
        argv.append(str(tmp_path))
        env.update(FK_AGENT_DRIVER="goal", FK_GOAL_DEADLINE=str(time.time() + 10))

    result = subprocess.run(argv, input="work please", text=True, capture_output=True,
                            cwd=workspace, env=env, timeout=15)
    if mode == "wall-timeout":
        assert result.returncode == 124
        assert not result.stdout
    elif mode == "timeout":
        assert result.returncode == 1
        assert not result.stdout and "print_timeout" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        if mode == "turn":
            assert result.stdout == "text reply"
        else:
            assert result.stdout == "approved report"
            assert sid.read_text().strip() == "conversation-1"
            assert (workspace / "after-approval").read_text() == ""


def test_agy_grants_teacher_the_detached_run_trees(tmp_path):
    workspace = tmp_path / "run" / "ws"
    operators = tmp_path / "run" / "operators"
    card = operators / "op"
    recon = operators / "_opref"
    source = tmp_path / "run" / "task" / "reference"
    policy = AgentPolicy("source", tmp_path / "checkout", teacher_trees=(recon, source),
                         operator_tree=card)

    settings = policy.agy_settings("teacher", workspace)
    assert settings["enableTerminalSandbox"] and settings["toolPermission"] == "proceed-in-sandbox"
    teacher = settings["permissions"]
    assert {f"read_file({source})", f"write_file({card})", f"write_file({recon})"} <= set(
        teacher["allow"])
    assert f"write_file({source})" not in teacher["allow"]

    student = policy.agy_settings("student", workspace)["permissions"]
    assert f"read_file({card})" in student["allow"]
    assert f"write_file({card})" not in student["allow"]
    assert {f"read_file({source})", f"read_file({recon})"} <= set(student["deny"])


def test_agy_online_reference_policy_allows_unrelated_docs(tmp_path):
    cmd = agent_command(AgentSpec("agy", "model", "high"), "student")
    policy = AgentPolicy("source", tmp_path, student_online=True,
                         ref_source_urls=("github.com/example/protected",))
    policy.prepare_turn(cmd, "student", tmp_path)
    settings = json.loads((tmp_path / AGY_POLICY_FILE).read_text())
    assert all("read_url(*)" not in rules for rules in settings["permissions"].values())
    hooks = json.loads((tmp_path / ".agents/hooks.json").read_text())
    hook = shlex.split(hooks["forgekernel-agent-policy"]["PreToolUse"][0]["hooks"][0]["command"])
    for url, expected in (("https://docs.python.org", "auto_approve"),
                          ("https://github.com/example/protected", "deny")):
        event = {"toolCall": {"name": "read_url_content", "args": {"Url": url}}}
        result = subprocess.run(hook, input=json.dumps(event), capture_output=True, check=True, text=True)
        assert json.loads(result.stdout)["decision"] == expected
