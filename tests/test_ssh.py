"""Tests for the SSH/SCP wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vast_helper.errors import SSHError
from vast_helper.ssh import SSHSession


def _session() -> SSHSession:
    return SSHSession("1.2.3.4", 2222, "~/.ssh/id_ed25519")


class TestSSHArgs:
    def test_include_key_port_user_and_timeout(self):
        args = _session()._ssh_args()

        assert "-i" in args
        assert args[args.index("-i") + 1].endswith(".ssh/id_ed25519")
        assert args[args.index("-p") + 1] == "2222"
        assert args[-1] == "root@1.2.3.4"
        assert "ConnectTimeout=5" in args
        assert "BatchMode=yes" in args
        assert "-t" not in args

    def test_tty_flag_precedes_destination(self):
        args = _session()._ssh_args(tty=True)

        assert args.index("-t") < args.index("root@1.2.3.4")

    def test_ssh_command_is_interactive_equivalent(self):
        command = _session().ssh_command()

        assert command.startswith("ssh ")
        assert "-i" in command
        assert "-p 2222" in command
        assert command.endswith("root@1.2.3.4")


class TestTestConnection:
    def test_returns_true_on_ready(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return SimpleNamespace(returncode=0, stdout="READY\n")

        monkeypatch.setattr("vast_helper.ssh.subprocess.run", fake_run)

        assert _session().test_connection() is True

    def test_returns_false_on_nonzero_exit(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return SimpleNamespace(returncode=255, stdout="")

        monkeypatch.setattr("vast_helper.ssh.subprocess.run", fake_run)

        assert _session().test_connection() is False

    def test_returns_false_on_timeout(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=15)

        monkeypatch.setattr("vast_helper.ssh.subprocess.run", fake_run)

        assert _session().test_connection() is False


class TestExecQuiet:
    def test_merges_stderr_and_strips(self, monkeypatch):
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(stdout="  hello\n", returncode=0)

        monkeypatch.setattr("vast_helper.ssh.subprocess.run", fake_run)

        assert _session().exec_quiet("echo hello") == "hello"
        assert captured["stderr"] == subprocess.STDOUT


class TestExecStreaming:
    def test_returns_exit_code_and_echoes_lines(self, monkeypatch, capsys):
        class FakeProc:
            def __init__(self):
                self.stdout = iter(["first\n", "second\r\n"])
                self.returncode = 0

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                pass

        monkeypatch.setattr(
            "vast_helper.ssh.subprocess.Popen",
            lambda *args, **kwargs: FakeProc(),
        )

        code, output = _session().exec_streaming("run me")

        assert code == 0
        assert output == "first\nsecond"
        printed = capsys.readouterr().out
        assert "first" in printed
        assert "second" in printed

    def test_timeout_kills_process_and_raises(self, monkeypatch):
        class FakeProc:
            def __init__(self):
                self.stdout = iter([])
                self.returncode = -9
                self.killed = False
                self.wait_calls = 0

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired(cmd="ssh", timeout=1)
                return self.returncode

            def kill(self):
                self.killed = True

        proc = FakeProc()
        monkeypatch.setattr(
            "vast_helper.ssh.subprocess.Popen",
            lambda *args, **kwargs: proc,
        )

        with pytest.raises(SSHError, match="timed out"):
            _session().exec_streaming("sleep 100", timeout=1)

        assert proc.killed is True


class TestDownloadDir:
    def test_builds_scp_command_and_creates_local_dir(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

        monkeypatch.setattr("vast_helper.ssh.subprocess.run", fake_run)
        local = tmp_path / "out"

        result = _session().download_dir("/workspace/runs", local)

        assert result == local
        assert local.is_dir()
        cmd = captured["cmd"]
        assert cmd[0] == "scp"
        assert "-r" in cmd
        assert "-P" in cmd
        assert cmd[cmd.index("-P") + 1] == "2222"
        assert cmd[-2] == "root@1.2.3.4:/workspace/runs/*"
        assert cmd[-1] == str(local)
        assert captured["kwargs"]["check"] is True

    def test_scp_failure_raises_ssh_error(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr("vast_helper.ssh.subprocess.run", fake_run)

        with pytest.raises(SSHError, match="scp failed"):
            _session().download_dir("/workspace/runs", tmp_path / "out")


def test_context_manager_returns_session():
    session = _session()

    with session as entered:
        assert entered is session


def test_key_path_is_expanded(tmp_path):
    key = tmp_path / "key"
    key.write_text("fake")

    session = SSHSession("host", 22, str(key))

    assert Path(session.key_filename) == key
