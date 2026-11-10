"""Tests for the CLI surface."""

from __future__ import annotations

import json

import pytest

from vast_helper.cli import (
    _command,
    _gb,
    _instance_payload,
    _money,
    _parse_env,
    _rel,
    build_parser,
    main,
)
from vast_helper.constants import DEFAULT_MIN_VRAM_GB, DEFAULT_SSH_KEY
from vast_helper.errors import VastHelperError
from vast_helper.provision import InstanceInfo

from helpers import FakeClient


class TestParser:
    def test_search_defaults(self):
        args = build_parser().parse_args(["search"])

        assert args.min_vram_gb == DEFAULT_MIN_VRAM_GB
        assert args.allow_headed is False
        assert args.limit is None

    def test_search_flags(self):
        args = build_parser().parse_args(
            [
                "search",
                "--max-price", "0.2",
                "--min-vram-gb", "16",
                "--allow-headed",
                "--limit", "5",
            ]
        )

        assert args.max_price == 0.2
        assert args.min_vram_gb == 16
        assert args.allow_headed is True
        assert args.limit == 5

    def test_provision_requires_template(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["provision"])

    def test_provision_flags(self):
        args = build_parser().parse_args(
            [
                "provision",
                "--template-hash", "abc",
                "--env", "A=1",
                "--env", "B=2",
                "--no-wait-for-ssh",
                "--ssh-key", "key",
                "--keep-alive",
            ]
        )

        assert args.template_hash == "abc"
        assert args.env == ["A=1", "B=2"]
        assert args.wait_for_ssh is False
        assert args.ssh_key == "key"
        assert args.keep_alive is True

    def test_run_command_remainder(self):
        args = build_parser().parse_args(
            ["run", "--template-hash", "abc", "python", "train.py"]
        )

        assert args.command == ["python", "train.py"]

    def test_run_command_after_separator(self):
        args = build_parser().parse_args(
            ["run", "--template-hash", "abc", "--", "echo", "hi"]
        )

        assert _command(args.command) == "echo hi"

    def test_ssh_defaults(self):
        args = build_parser().parse_args(["ssh"])

        assert args.instance_id is None
        assert args.ssh_key == DEFAULT_SSH_KEY

    def test_destroy_accepts_multiple_ids(self):
        args = build_parser().parse_args(["destroy", "1", "2", "3"])

        assert args.instance_ids == [1, 2, 3]


class TestHelpers:
    def test_parse_env(self):
        assert _parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}

    def test_parse_env_rejects_missing_equals(self):
        with pytest.raises(VastHelperError, match="KEY=VALUE"):
            _parse_env(["BROKEN"])

    def test_parse_env_rejects_empty_key(self):
        with pytest.raises(VastHelperError, match="key is empty"):
            _parse_env(["=value"])

    def test_command_strips_separator(self):
        assert _command(["--", "echo", "hi"]) == "echo hi"

    def test_command_quotes_spaces(self):
        assert _command(["python", "-c", "print('hi there')"]) == (
            "python -c 'print('\"'\"'hi there'\"'\"')'"
        )

    def test_command_empty(self):
        assert _command([]) == ""

    def test_formatting_helpers(self):
        assert _gb(24576) == "24G"
        assert _money(0.1234) == "$0.123"
        assert _rel(0.987) == "99%"
        assert _gb(None) == "?"
        assert _money("bad") == "?"


class TestCommands:
    def test_search_json_output(self, monkeypatch, capsys):
        client = FakeClient(offers=[{"id": 1, "dph_total": 0.1}])
        monkeypatch.setattr("vast_helper.cli._client", lambda args: client)

        code = main(["search", "--json"])

        assert code == 0
        assert json.loads(capsys.readouterr().out) == [{"id": 1, "dph_total": 0.1}]

    def test_search_table_output(self, monkeypatch, capsys):
        client = FakeClient(
            offers=[
                {
                    "id": 1,
                    "gpu_name": "RTX 3090",
                    "gpu_ram": 24576,
                    "dph_total": 0.12,
                    "reliability": 0.99,
                    "cuda_max_good": 12.8,
                    "disk_space": 100,
                    "pcie_bw": 16,
                    "public_ipaddr": "1.2.3.4",
                }
            ]
        )
        monkeypatch.setattr("vast_helper.cli._client", lambda args: client)

        code = main(["search"])

        assert code == 0
        out = capsys.readouterr().out
        assert "RTX 3090" in out
        assert "24G" in out
        assert "$0.120" in out

    def test_list_running_only(self, monkeypatch, capsys):
        client = FakeClient(
            instances=[
                {"id": 1, "actual_status": "running", "gpu_name": "A"},
                {"id": 2, "actual_status": "stopped", "gpu_name": "B"},
            ]
        )
        monkeypatch.setattr("vast_helper.cli._client", lambda args: client)

        code = main(["list", "--running"])

        assert code == 0
        out = capsys.readouterr().out
        assert "running" in out
        assert "stopped" not in out

    def test_verify_failure_returns_one(self, monkeypatch, capsys):
        from vast_helper.provision import ConnectionCheck

        monkeypatch.setattr(
            "vast_helper.cli.verify_connection",
            lambda api_key=None: ConnectionCheck(api_ok=False),
        )

        assert main(["verify"]) == 1
        assert "did not authenticate" in capsys.readouterr().err

    def test_errors_return_one(self, monkeypatch, capsys):
        def boom(api_key=None):
            raise VastHelperError("nope")

        monkeypatch.setattr("vast_helper.cli.verify_connection", boom)

        assert main(["verify"]) == 1
        assert "error: nope" in capsys.readouterr().err


def test_instance_payload():
    info = InstanceInfo(
        id=7,
        host="1.2.3.4",
        port=22002,
        gpu_name="RTX 3090",
        price_per_hour=0.12,
        reliability=0.99,
        ssh_key="key",
    )

    payload = _instance_payload(info)

    assert payload["id"] == 7
    assert payload["ssh"] == "ssh -i key -p 22002 root@1.2.3.4"
    assert payload["destroy"] == "vast-helper destroy 7"
