"""Defaults shared across vast-helper.

The search defaults target LLM inference: enough VRAM for an 8B model at
fp16 plus KV cache, enough disk for model weights and datasets, and a PCIe
link fast enough that weight loading is not the bottleneck.
"""

from __future__ import annotations

VAST_API_BASE = "https://console.vast.ai/api/v0"
VAST_API_V1 = "https://console.vast.ai/api/v1"

#: Environment variables consulted for the API key, in order.
API_KEY_ENV_VARS = ("VAST_API_KEY", "VASTAI_API_KEY")

DEFAULT_SSH_KEY = "~/.ssh/id_ed25519"

DEFAULT_MAX_PRICE = 0.50
DEFAULT_MIN_PRICE = 0.15  # filter out suspiciously cheap hosts
DEFAULT_MIN_VRAM_GB = 24
DEFAULT_MIN_RELIABILITY = 0.95
DEFAULT_MIN_CUDA = 12.4
DEFAULT_MIN_DISK_GB = 50
DEFAULT_MIN_PCIE_BW = 10

POLL_INTERVAL_S = 5.0
API_POLL_TIMEOUT_S = 600.0  # provisioning can be slow
SSH_CONNECT_TIMEOUT_S = 600.0  # Docker boot and onstart can take a while
DESTROY_RETRIES = 3
DESTROY_RETRY_DELAY_S = 5.0
