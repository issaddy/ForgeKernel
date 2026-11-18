import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class StartTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('start', SCRIPTS / 'start.py')
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_start_runs_after_creation(self):
        create = Mock(return_value=Path('/run/one'))
        run = Mock(return_value=0)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.m.run_once(Path(d) / 'session.json', '/task', create, run), 0)
        create.assert_called_once()
        run.assert_called_once_with(Path('/run/one'))

    def test_repeat_resumes_same_run(self):
        create = Mock(return_value=Path('/run/one'))
        run = Mock(return_value=2)
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'session.json'
            self.m.run_once(state, '/task', create, run)
            self.m.run_once(state, '/task', create, run)
        create.assert_called_once()
        self.assertEqual(run.call_count, 2)

    def test_different_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'session.json'
            self.m.run_once(state, '/task', lambda: Path('/run/one'), lambda _: 0)
            with self.assertRaises(ValueError):
                self.m.run_once(state, '/other', Mock(), Mock())

    def test_corrupt_state_is_not_new_run(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'session.json'; state.write_text('broken')
            create = Mock()
            with self.assertRaises(ValueError):
                self.m.run_once(state, '/task', create, Mock())
            create.assert_not_called()

    def test_gpu_probe_reports_remote_error(self):
        import subprocess
        failure = subprocess.CalledProcessError(1, ['ssh'], stderr='Missing remote environment')
        with patch.object(self.m.subprocess, 'run', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'Missing remote environment'):
                self.m.probe_gpu(['ssh'])

    def test_arch_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.m.check_arch('sm90', [8, 0])
        self.m.check_arch('sm90', [9, 0])

    def test_codex_launch_exports_the_selected_cli_without_looking_up_claude(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            codex = root / 'codex'
            codex.touch()
            codex.chmod(0o755)
            role = SimpleNamespace(backend='codex')
            task = SimpleNamespace(teacher=role, student=role)
            machine = SimpleNamespace(env={'FK_CODEX_BIN': str(codex)})
            argv = ['start.py', '--repo', str(root), '--env', str(root), '--role', 'controller',
                    '--task', str(root / 'op'), '--session', str(root / 'session.json')]
            with (patch.object(self.m, 'launch_context', return_value=(task, machine)),
                  patch.object(self.m, 'execute', return_value=0) as execute,
                  patch.object(self.m.deploy.shutil, 'which', side_effect=AssertionError),
                  patch.object(sys, 'argv', argv), patch.object(sys, 'prefix', str(root)),
                  patch.dict(os.environ, {}, clear=True)):
                self.assertEqual(self.m.main(), 0)
                self.assertEqual(os.environ['FK_CODEX_BIN'], str(codex))
                self.assertNotIn('FK_CLAUDE_BIN', os.environ)
                self.assertIs(execute.call_args.args[1], task)

    def test_resuming_resolves_agent_roles_from_the_saved_task(self):
        repo = SCRIPTS.parents[3]
        with (tempfile.TemporaryDirectory() as d,
              patch.object(sys, 'path', [str(repo / 'src'), *sys.path])):
            from forgekernel.runs import new_run
            from forgekernel.task import load_task

            root = Path(d).resolve()
            task_dir = root / 'rmsnorm'
            shutil.copytree(repo / 'tasks/rmsnorm', task_dir)
            with (task_dir / 'task.toml').open('a') as task_file:
                task_file.write('\n[agent.teacher]\nbackend = "claude"\nmodel = "claude-opus-5-5"\n')
            task = load_task(task_dir)
            run = new_run(task, root=root / 'runs')
            session = root / 'session.json'
            self.m.run_once(session, str(task_dir), lambda: run.dir, lambda _: 0)
            task_dir.rename(root / 'original-task')
            selected, _ = self.m.launch_context(SimpleNamespace(repo=repo, task=task_dir,
                                                                session=session))
            self.assertEqual((selected.teacher.backend, selected.student.backend), ('claude', 'myflicker'))
            self.assertEqual(selected.name, 'rmsnorm')


if __name__ == '__main__':
    unittest.main()
