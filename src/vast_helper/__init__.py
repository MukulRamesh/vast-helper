"""Provision, connect to, and manage vast.ai GPU instances."""

from vast_helper.client import OfferCriteria, VastClient, api_key_from_env
from vast_helper.constants import (
    API_POLL_TIMEOUT_S,
    API_KEY_ENV_VARS,
    DEFAULT_MAX_PRICE,
    DEFAULT_MIN_CUDA,
    DEFAULT_MIN_DISK_GB,
    DEFAULT_MIN_PCIE_BW,
    DEFAULT_MIN_PRICE,
    DEFAULT_MIN_RELIABILITY,
    DEFAULT_MIN_VRAM_GB,
    DEFAULT_SSH_KEY,
    DESTROY_RETRIES,
    DESTROY_RETRY_DELAY_S,
    POLL_INTERVAL_S,
    SSH_CONNECT_TIMEOUT_S,
    VAST_API_BASE,
    VAST_API_V1,
)
from vast_helper.errors import (
    AuthError,
    CommandError,
    InstanceNotFoundError,
    NoOffersError,
    ProvisionTimeoutError,
    SSHError,
    SSHTimeoutError,
    VastHelperError,
)
from vast_helper.provision import (
    ConnectionCheck,
    InstanceInfo,
    ProvisionConfig,
    attach_instance,
    destroy_instance,
    destroy_unless_kept,
    extract_ssh_port,
    find_running_instances,
    instance_status,
    managed_instance,
    provision_instance,
    select_running_instance,
    verify_connection,
    wait_for_running,
    wait_for_ssh,
)
from vast_helper.ssh import SSHSession

__version__ = "0.1.0"

__all__ = [
    # client
    "OfferCriteria",
    "VastClient",
    "api_key_from_env",
    # provisioning and lifecycle
    "ConnectionCheck",
    "InstanceInfo",
    "ProvisionConfig",
    "attach_instance",
    "destroy_instance",
    "destroy_unless_kept",
    "extract_ssh_port",
    "find_running_instances",
    "instance_status",
    "managed_instance",
    "provision_instance",
    "select_running_instance",
    "verify_connection",
    "wait_for_running",
    "wait_for_ssh",
    # ssh
    "SSHSession",
    # errors
    "AuthError",
    "CommandError",
    "InstanceNotFoundError",
    "NoOffersError",
    "ProvisionTimeoutError",
    "SSHError",
    "SSHTimeoutError",
    "VastHelperError",
    # constants
    "API_KEY_ENV_VARS",
    "API_POLL_TIMEOUT_S",
    "DEFAULT_MAX_PRICE",
    "DEFAULT_MIN_CUDA",
    "DEFAULT_MIN_DISK_GB",
    "DEFAULT_MIN_PCIE_BW",
    "DEFAULT_MIN_PRICE",
    "DEFAULT_MIN_RELIABILITY",
    "DEFAULT_MIN_VRAM_GB",
    "DEFAULT_SSH_KEY",
    "DESTROY_RETRIES",
    "DESTROY_RETRY_DELAY_S",
    "POLL_INTERVAL_S",
    "SSH_CONNECT_TIMEOUT_S",
    "VAST_API_BASE",
    "VAST_API_V1",
    "__version__",
]
