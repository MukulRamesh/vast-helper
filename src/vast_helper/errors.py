"""Exceptions raised by vast-helper."""

from __future__ import annotations


class VastHelperError(Exception):
    """Base class for all vast-helper errors."""


class AuthError(VastHelperError):
    """The vast.ai API key is missing, invalid, or lacks permission."""


class NoOffersError(VastHelperError):
    """No GPU offers matched the search criteria."""


class ProvisionTimeoutError(VastHelperError):
    """An instance did not reach ``running`` before the timeout."""


class InstanceNotFoundError(VastHelperError):
    """The requested instance does not exist or is not visible."""


class SSHTimeoutError(VastHelperError):
    """SSH did not become available before the timeout."""


class SSHError(VastHelperError):
    """An SSH or SCP operation failed."""


class CommandError(SSHError):
    """A remote command exited with a non-zero status."""

    def __init__(self, message: str, *, exit_code: int, output: str = "") -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.output = output
