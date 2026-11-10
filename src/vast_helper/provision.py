"""Provision, attach to, and tear down vast.ai GPU instances."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests

from vast_helper.client import OfferCriteria, VastClient
from vast_helper.constants import (
    API_POLL_TIMEOUT_S,
    DEFAULT_SSH_KEY,
    DESTROY_RETRIES,
    DESTROY_RETRY_DELAY_S,
    POLL_INTERVAL_S,
    SSH_CONNECT_TIMEOUT_S,
)
from vast_helper.errors import (
    AuthError,
    NoOffersError,
    ProvisionTimeoutError,
    SSHTimeoutError,
    VastHelperError,
)
from vast_helper.ssh import SSHSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectionCheck:
    """Result of :func:`verify_connection`.

    Attributes
    ----------
    api_ok:
        Whether the API key authenticated.
    user_email:
        Email of the authenticated user, if available.
    balance:
        Account credit balance in dollars, if available.
    """

    api_ok: bool
    user_email: str | None = None
    balance: float | None = None


@dataclass
class InstanceInfo:
    """A provisioned instance and the information needed to reach it.

    Attributes
    ----------
    id:
        vast.ai instance ID.
    host:
        Public IP or SSH hostname.
    port:
        External SSH port.
    gpu_name:
        GPU model reported by the offer or instance.
    price_per_hour:
        Price in dollars per hour, if known.
    reliability:
        Host reliability from the offer, if known.
    ssh_key:
        Private key used for SSH.
    user:
        Remote SSH user.
    raw:
        The instance record returned by the API.
    """

    id: int
    host: str
    port: int
    gpu_name: str | None = None
    price_per_hour: float | None = None
    reliability: float | None = None
    ssh_key: str = DEFAULT_SSH_KEY
    user: str = "root"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(
        cls,
        data: dict[str, Any],
        *,
        ssh_key: str = DEFAULT_SSH_KEY,
        user: str = "root",
        offer: dict[str, Any] | None = None,
    ) -> "InstanceInfo":
        """Build an :class:`InstanceInfo` from an instance API record."""
        instance_id = data.get("id") or data.get("instance_id")
        if not instance_id:
            raise VastHelperError(f"Instance record has no id: {data}")

        host = data.get("public_ipaddr") or data.get("ssh_host")
        if not host:
            raise VastHelperError(
                f"Instance {instance_id} has no SSH host. "
                f"Status: {instance_status(data)}"
            )

        offer = offer or {}
        return cls(
            id=int(instance_id),
            host=str(host),
            port=extract_ssh_port(data),
            gpu_name=data.get("gpu_name") or offer.get("gpu_name"),
            price_per_hour=_first_number(
                data.get("dph_total"), offer.get("dph_total")
            ),
            reliability=_first_number(
                offer.get("reliability"), data.get("reliability")
            ),
            ssh_key=ssh_key,
            user=user,
            raw=dict(data),
        )

    def ssh(self) -> SSHSession:
        """Return an SSH session for this instance."""
        return SSHSession(
            self.host,
            self.port,
            self.ssh_key,
            user=self.user,
        )

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    def destroy_command(self) -> str:
        """Return the CLI command that destroys this instance."""
        return f"vast-helper destroy {self.id}"


@dataclass
class ProvisionConfig:
    """Configuration for :func:`provision_instance`.

    Attributes
    ----------
    template_hash:
        vast.ai template hash ID.  The template supplies the Docker image,
        disk size, and onstart script.
    criteria:
        GPU offer search filters.
    env:
        Environment variables passed to the instance.  These merge over
        template variables; account-level variables cannot be overridden.
    ssh_key:
        Private key registered with vast.ai.
    user:
        Remote SSH user.
    api_poll_timeout_s:
        How long to wait for the instance to reach ``running``.
    ssh_connect_timeout_s:
        How long to wait for SSH to accept connections.
    poll_interval_s:
        Base polling interval for status and SSH checks.
    wait_for_ssh:
        If ``True`` (default), do not return until SSH is ready.
    keep_alive:
        If ``True``, leave the instance running if provisioning fails.
        Useful for debugging a host that never became reachable.
    """

    template_hash: str
    criteria: OfferCriteria = field(default_factory=OfferCriteria)
    env: dict[str, str] = field(default_factory=dict)
    ssh_key: str = DEFAULT_SSH_KEY
    user: str = "root"
    api_poll_timeout_s: float = API_POLL_TIMEOUT_S
    ssh_connect_timeout_s: float = SSH_CONNECT_TIMEOUT_S
    poll_interval_s: float = POLL_INTERVAL_S
    wait_for_ssh: bool = True
    keep_alive: bool = False


# ---------------------------------------------------------------------------
# Account / discovery
# ---------------------------------------------------------------------------


def verify_connection(
    api_key: str | None = None,
    *,
    client: VastClient | None = None,
) -> ConnectionCheck:
    """Check that the API key authenticates.

    Network failures and rejected keys both return ``api_ok=False`` rather
    than raising, so this is safe to use as a pre-flight check.
    """
    try:
        vast = client or VastClient(api_key)
        user = vast.check_auth()
    except (AuthError, requests.RequestException):
        return ConnectionCheck(api_ok=False)
    return ConnectionCheck(
        api_ok=True,
        user_email=user.get("email"),
        balance=user.get("balance"),
    )


def find_running_instances(
    client: VastClient,
) -> list[dict[str, Any]]:
    """Return instances whose ``actual_status`` is ``running``."""
    return [
        inst for inst in client.list_instances()
        if inst.get("actual_status") == "running"
    ]


def select_running_instance(
    client: VastClient,
    instance_id: int | None = None,
    *,
    ssh_key: str = DEFAULT_SSH_KEY,
    user: str = "root",
    connect: bool = False,
    timeout_s: float = SSH_CONNECT_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> InstanceInfo:
    """Attach to *instance_id*, or auto-select the only running instance.

    If *instance_id* is ``None`` and several instances are running, this
    raises :class:`VastHelperError` listing the IDs instead of guessing.
    """
    if instance_id is not None:
        return attach_instance(
            client,
            instance_id,
            ssh_key=ssh_key,
            user=user,
            connect=connect,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )

    running = find_running_instances(client)
    if not running:
        raise VastHelperError(
            "No running instances found. Provision one first."
        )

    for inst in running:
        logger.info(
            "Running: id=%s gpu=%s host=%s status=%s",
            inst.get("id", "?"),
            inst.get("gpu_name", "unknown"),
            inst.get("public_ipaddr") or inst.get("ssh_host") or "?",
            inst.get("actual_status", "?"),
        )

    if len(running) > 1:
        ids = ", ".join(str(inst.get("id", "?")) for inst in running)
        raise VastHelperError(
            f"Multiple running instances found (ids: {ids}). "
            f"Pass an instance ID to choose one."
        )

    chosen = int(running[0]["id"])
    logger.info("Only one running instance; selecting id=%s.", chosen)
    return attach_instance(
        client,
        chosen,
        ssh_key=ssh_key,
        user=user,
        connect=connect,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )


def attach_instance(
    client: VastClient,
    instance_id: int,
    *,
    ssh_key: str = DEFAULT_SSH_KEY,
    user: str = "root",
    connect: bool = False,
    timeout_s: float = SSH_CONNECT_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> InstanceInfo:
    """Look up an existing instance and optionally wait for SSH."""
    data = client.show_instance(instance_id)
    info = InstanceInfo.from_api(data, ssh_key=ssh_key, user=user)
    if connect:
        wait_for_ssh(
            info.ssh(),
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
    return info


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def provision_instance(
    client: VastClient,
    config: ProvisionConfig,
) -> InstanceInfo:
    """Find an offer, create an instance, and wait until it is reachable.

    On any failure after creation, the instance is destroyed unless
    ``config.keep_alive`` is set.  This keeps a timed-out provision from
    billing indefinitely.

    Raises
    ------
    NoOffersError
        No offer matched ``config.criteria``.
    ProvisionTimeoutError
        The instance did not reach ``running`` in time.
    SSHTimeoutError
        The instance never accepted SSH.
    VastHelperError
        Any other provisioning error.
    """
    logger.info("Searching for GPU offers %s", config.criteria.describe())
    offers = client.search_offers(config.criteria)
    if not offers:
        raise NoOffersError(
            "No GPU offers matched the search criteria "
            f"{config.criteria.describe()}."
        )

    best = offers[0]
    logger.info(
        "Selected %s (%s MB VRAM, %s%% reliability, $%s/hr)",
        best.get("gpu_name", "unknown"),
        best.get("gpu_ram", "?"),
        _percent(best.get("reliability")),
        best.get("dph_total", "?"),
    )

    created = client.create_instance(
        int(best["id"]),
        template_hash=config.template_hash,
        env=dict(config.env),
    )
    instance_id = _extract_instance_id(created)
    logger.info("Instance %s provisioning...", instance_id)

    try:
        data = wait_for_running(
            client,
            instance_id,
            timeout_s=config.api_poll_timeout_s,
            poll_interval_s=config.poll_interval_s,
        )
        info = InstanceInfo.from_api(
            data,
            ssh_key=config.ssh_key,
            user=config.user,
            offer=best,
        )
        if config.wait_for_ssh:
            wait_for_ssh(
                info.ssh(),
                timeout_s=config.ssh_connect_timeout_s,
                poll_interval_s=config.poll_interval_s,
            )
        return info
    except BaseException:
        # Also covers KeyboardInterrupt: an interrupted provision must not
        # leave a GPU billing in the background.
        destroy_unless_kept(
            client,
            instance_id,
            config.keep_alive,
            success=False,
        )
        raise


@contextmanager
def managed_instance(
    client: VastClient,
    config: ProvisionConfig,
) -> Iterator[InstanceInfo]:
    """Provision an instance, yield it, and destroy it on exit.

    The instance is destroyed whether the body succeeds or raises, unless
    ``config.keep_alive`` is set.
    """
    info = provision_instance(client, config)
    success = False
    try:
        yield info
        success = True
    finally:
        destroy_unless_kept(
            client,
            info.id,
            config.keep_alive,
            success=success,
        )


# ---------------------------------------------------------------------------
# Waiting
# ---------------------------------------------------------------------------


def wait_for_running(
    client: VastClient,
    instance_id: int,
    *,
    timeout_s: float = API_POLL_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> dict[str, Any]:
    """Poll until the instance is ``running`` and has port mappings."""
    deadline = time.monotonic() + timeout_s
    last_status: str | None = None
    waiting_for_ports = False

    while time.monotonic() < deadline:
        data = client.show_instance(instance_id)
        status = instance_status(data)

        if status == "running" and data.get("ports"):
            return data

        if status != last_status:
            logger.info("Instance %s status: %s", instance_id, status)
            last_status = status

        if status == "running" and not waiting_for_ports:
            logger.info(
                "Instance %s is running; waiting for port mappings...",
                instance_id,
            )
            waiting_for_ports = True

        time.sleep(poll_interval_s)

    raise ProvisionTimeoutError(
        f"Instance {instance_id} did not reach 'running' within "
        f"{timeout_s:g}s. Last status: {last_status}"
    )


def wait_for_ssh(
    ssh: SSHSession,
    *,
    timeout_s: float = SSH_CONNECT_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> None:
    """Retry SSH until the host accepts a command or the timeout expires."""
    deadline = time.monotonic() + timeout_s
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        if ssh.test_connection():
            logger.info(
                "SSH ready on %s:%s (attempt %d).",
                ssh.host,
                ssh.port,
                attempt,
            )
            return
        if attempt == 1:
            logger.info("Waiting for SSH on %s:%s...", ssh.host, ssh.port)
        time.sleep(max(poll_interval_s * 2, 0.1))

    raise SSHTimeoutError(
        f"SSH did not become available on {ssh.host}:{ssh.port} "
        f"within {timeout_s:g}s ({attempt} attempts)."
    )


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def destroy_instance(
    client: VastClient,
    instance_id: int,
    *,
    retries: int = DESTROY_RETRIES,
    delay_s: float = DESTROY_RETRY_DELAY_S,
) -> bool:
    """Destroy an instance, retrying transient API failures.

    Returns ``True`` if the instance is gone, ``False`` if every attempt
    failed.
    """
    for attempt in range(1, retries + 1):
        try:
            client.destroy_instance(instance_id)
        except requests.RequestException as exc:
            status = _http_status(exc)
            if attempt < retries:
                logger.warning(
                    "Destroy attempt %d/%d failed (HTTP %s); "
                    "retrying in %gs...",
                    attempt,
                    retries,
                    status,
                    delay_s,
                )
                time.sleep(delay_s)
                continue
            logger.error(
                "Could not destroy instance %s after %d attempts "
                "(HTTP %s). Destroy it manually: vast-helper destroy %s",
                instance_id,
                retries,
                status,
                instance_id,
            )
            return False
        logger.info(
            "Instance %s destroyed (attempt %d).",
            instance_id,
            attempt,
        )
        return True
    return False


def destroy_unless_kept(
    client: VastClient,
    instance_id: int | None,
    keep_alive: bool,
    *,
    success: bool = True,
) -> bool:
    """Destroy *instance_id* unless *keep_alive* is set.

    Returns ``True`` if an instance was destroyed.
    """
    if instance_id is None:
        return False

    logger.info(
        "Instance %s (destroy manually: vast-helper destroy %s)",
        instance_id,
        instance_id,
    )
    if keep_alive:
        logger.info("keep_alive set; not destroying instance %s.", instance_id)
        return False

    logger.info(
        "Work %s; destroying instance %s...",
        "completed" if success else "failed",
        instance_id,
    )
    return destroy_instance(client, instance_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def instance_status(data: dict[str, Any]) -> str:
    """Return the most reliable status field from an instance record."""
    return str(
        data.get("actual_status")
        or data.get("cur_state")
        or data.get("status")
        or "unknown"
    )


def extract_ssh_port(instance_data: dict[str, Any]) -> int:
    """Return the external port mapped to the container's SSH server.

    Templates with several exposed ports get a contiguous block of
    external ports.  The top-level ``ssh_port`` can point at the first
    port in that block rather than the one mapped to ``22/tcp``, so the
    ``ports`` mapping is preferred.
    """
    try:
        port = int(instance_data["ports"]["22/tcp"][0]["HostPort"])
        logger.debug("SSH port from ports mapping: %d", port)
        return port
    except (KeyError, IndexError, TypeError, ValueError):
        fallback = int(instance_data.get("ssh_port") or 22)
        logger.debug("SSH port from ssh_port field: %d", fallback)
        return fallback


def _extract_instance_id(response: dict[str, Any]) -> int:
    for key in ("new_contract", "instance_id", "id"):
        value = response.get(key)
        if value:
            return int(value)
    raise VastHelperError(
        f"Could not determine instance ID from response: {response}"
    )


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _percent(value: Any) -> str:
    try:
        return str(int(float(value) * 100))
    except (TypeError, ValueError):
        return "?"


def _http_status(exc: requests.RequestException) -> Any:
    return getattr(getattr(exc, "response", None), "status_code", "?")
