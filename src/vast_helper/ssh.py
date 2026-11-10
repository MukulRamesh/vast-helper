"""SSH and SCP helpers backed by the system OpenSSH client.

Using the system ``ssh``/``scp`` binaries keeps behavior identical to what
a user does in a terminal and avoids a paramiko dependency.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from vast_helper.constants import DEFAULT_SSH_KEY
from vast_helper.errors import SSHError

logger = logging.getLogger(__name__)


class SSHSession:
    """Run commands on a remote host over SSH.

    Parameters
    ----------
    host:
        Hostname or IP address.
    port:
        SSH port.  On vast.ai, read this from
        ``ports["22/tcp"][0]["HostPort"]`` rather than the top-level
        ``ssh_port`` field.
    key_filename:
        Path to the private key.  ``~`` is expanded.
    user:
        Remote user.  vast.ai containers run as ``root``.
    connect_timeout:
        Per-attempt SSH connection timeout in seconds.
    """

    def __init__(
        self,
        host: str,
        port: int,
        key_filename: str = DEFAULT_SSH_KEY,
        *,
        user: str = "root",
        connect_timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.user = user
        self._key = os.path.expanduser(str(key_filename))
        self._connect_timeout = connect_timeout

    @property
    def key_filename(self) -> str:
        return self._key

    def _ssh_args(self, *, tty: bool = False) -> list[str]:
        args = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self._connect_timeout:g}",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "TCPKeepAlive=yes",
            "-i", self._key,
            "-p", str(self.port),
        ]
        if tty:
            args.append("-t")
        args.append(f"{self.user}@{self.host}")
        return args

    def ssh_command(self) -> str:
        """Return an equivalent interactive ``ssh`` command."""
        return (
            f"ssh -i {self._key} -p {self.port} "
            f"{self.user}@{self.host}"
        )

    def test_connection(self, timeout: float = 15.0) -> bool:
        """Return ``True`` if the host accepts SSH and runs a command."""
        try:
            result = subprocess.run(
                [*self._ssh_args(), "echo", "READY"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return result.returncode == 0 and "READY" in result.stdout

    def exec_streaming(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> tuple[int, str]:
        """Run *command*, echo each output line, and return the exit code.

        Returns
        -------
        (exit_code, output)
            The remote exit code and the captured output with stderr merged.
        """
        proc = subprocess.Popen(
            [*self._ssh_args(tty=True), command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    stripped = line.rstrip("\r\n")
                    print(stripped, flush=True)
                    lines.append(stripped)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise SSHError(
                f"Command timed out after {timeout}s: {command}"
            ) from None
        return int(proc.returncode), "\n".join(lines)

    def exec_quiet(self, command: str, *, timeout: float = 30.0) -> str:
        """Run *command* and return trimmed stdout with stderr merged."""
        result = subprocess.run(
            [*self._ssh_args(), command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()

    def download_dir(
        self,
        remote_path: str,
        local_path: str | Path,
        *,
        timeout: float = 300.0,
    ) -> Path:
        """Recursively copy ``remote_path/*`` into *local_path*."""
        local = Path(local_path)
        local.mkdir(parents=True, exist_ok=True)
        cmd = [
            "scp",
            "-r",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-i", self._key,
            "-P", str(self.port),
            f"{self.user}@{self.host}:{remote_path}/*",
            str(local),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            raise SSHError(
                f"scp failed (exit {exc.returncode}) downloading "
                f"{remote_path} to {local}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SSHError(
                f"scp timed out after {timeout}s downloading {remote_path}"
            ) from exc
        return local

    def close(self) -> None:
        """No-op: subprocess connections close on their own."""

    def __enter__(self) -> "SSHSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
