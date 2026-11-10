"""HTTP client for the vast.ai REST API.

Only the endpoints needed to find a GPU, create an instance, inspect it,
and destroy it are covered.  The client deliberately wraps raw HTTP rather
than a generated SDK so every call maps 1:1 to the published API reference.

Docs: https://vast.ai/docs/api-reference
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

from vast_helper.constants import (
    API_KEY_ENV_VARS,
    DEFAULT_MAX_PRICE,
    DEFAULT_MIN_CUDA,
    DEFAULT_MIN_DISK_GB,
    DEFAULT_MIN_PCIE_BW,
    DEFAULT_MIN_PRICE,
    DEFAULT_MIN_RELIABILITY,
    DEFAULT_MIN_VRAM_GB,
    VAST_API_BASE,
    VAST_API_V1,
)
from vast_helper.errors import (
    AuthError,
    InstanceNotFoundError,
    VastHelperError,
)

logger = logging.getLogger(__name__)


def api_key_from_env() -> str | None:
    """Return the first non-empty API key found in the environment."""
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class OfferCriteria:
    """Filters for :meth:`VastClient.search_offers`.

    The vast.ai search endpoint handles ``gpu_ram``, ``reliability``,
    ``dph_total``, and the verified/rentable/rented flags.  CUDA version,
    disk space, PCIe bandwidth, and headless status are filtered locally
    after the API call because the API either ignores those filters or
    does not expose them.

    Attributes
    ----------
    min_vram_gb:
        Minimum GPU memory in GB.
    max_price:
        Maximum price in dollars per hour.
    min_price:
        Minimum price in dollars per hour.  Filters out suspiciously
        cheap hosts, which tend to be unreliable.
    min_reliability:
        Minimum host reliability as a fraction (0.0-1.0).
    min_cuda:
        Minimum CUDA version supported by the host driver.  Set to 0 to
        disable the filter.
    min_disk_gb:
        Minimum disk space in GB.  Set to 0 to disable the filter.
    min_pcie_bw:
        Minimum PCIe bandwidth in GB/s.  Set to 0 to disable the filter.
    require_headless:
        Exclude hosts with a display attached, which steals VRAM from the
        framebuffer.
    limit:
        Truncate the result list after filtering.
    """

    min_vram_gb: float = DEFAULT_MIN_VRAM_GB
    max_price: float = DEFAULT_MAX_PRICE
    min_price: float = DEFAULT_MIN_PRICE
    min_reliability: float = DEFAULT_MIN_RELIABILITY
    min_cuda: float = DEFAULT_MIN_CUDA
    min_disk_gb: float = DEFAULT_MIN_DISK_GB
    min_pcie_bw: float = DEFAULT_MIN_PCIE_BW
    require_headless: bool = True
    limit: int | None = None

    def describe(self) -> str:
        """Return a human-readable summary for logs."""
        parts = [
            f"${self.min_price:.2f}-${self.max_price:.2f}/hr",
            f">={self.min_vram_gb:g} GB VRAM",
            f">={self.min_reliability:.0%} reliability",
        ]
        if self.min_cuda > 0:
            parts.append(f"CUDA >= {self.min_cuda:g}")
        if self.min_disk_gb > 0:
            parts.append(f"disk >= {self.min_disk_gb:g} GB")
        if self.min_pcie_bw > 0:
            parts.append(f"PCIe >= {self.min_pcie_bw:g} GB/s")
        if self.require_headless:
            parts.append("headless")
        return "(" + ", ".join(parts) + ")"

    def api_body(self) -> dict[str, Any]:
        """Build the ``POST /bundles/`` request body."""
        return {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "type": "ondemand",
            "gpu_ram": {"gte": int(round(self.min_vram_gb * 1024))},
            "reliability": {"gte": self.min_reliability},
            "dph_total": {"gte": self.min_price, "lte": self.max_price},
            "order": [["dph_total", "asc"], ["reliability", "desc"]],
        }


class VastClient:
    """Minimal vast.ai REST client.

    Parameters
    ----------
    api_key:
        vast.ai API key.  Defaults to ``VAST_API_KEY``, then the legacy
        ``VASTAI_API_KEY``, from the environment.
    session:
        Optional pre-configured :class:`requests.Session`.  Useful for
        tests and for callers that already manage connection pooling.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or api_key_from_env()
        if not key:
            raise AuthError(
                "No vast.ai API key. Pass api_key= or set VAST_API_KEY "
                "(VASTAI_API_KEY is also accepted)."
            )
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
        )
        self._timeout = timeout

    # -- account --------------------------------------------------------

    def check_auth(self) -> dict[str, Any]:
        """Verify the API key and return the current user record.

        Raises
        ------
        AuthError
            If vast.ai rejects the key.
        requests.HTTPError
            For any other non-2xx response.
        """
        resp = self._session.get(
            f"{VAST_API_BASE}/users/current/",
            timeout=self._timeout,
        )
        if resp.status_code in (401, 403):
            raise AuthError(
                f"vast.ai rejected the API key (HTTP {resp.status_code})."
            )
        resp.raise_for_status()
        return resp.json()

    # -- offers ---------------------------------------------------------

    def search_offers(
        self,
        criteria: OfferCriteria | None = None,
    ) -> list[dict[str, Any]]:
        """Search for GPU offers matching *criteria*.

        Returns offers sorted cheapest-first, with reliability as the
        tiebreaker.  Only verified, rentable, unrented on-demand offers
        are returned.
        """
        criteria = criteria or OfferCriteria()
        resp = self._session.post(
            f"{VAST_API_BASE}/bundles/",
            json=criteria.api_body(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        offers = list(resp.json().get("offers", []))
        return _apply_local_filters(offers, criteria)

    # -- instances ------------------------------------------------------

    def create_instance(
        self,
        offer_id: int,
        *,
        template_hash: str,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Accept *offer_id* and create an instance from a template.

        The template supplies the Docker image, disk size, and onstart
        script.  *env* merges with the template's environment variables;
        vast.ai account-level environment variables are injected beneath
        both and cannot be overridden.
        """
        body: dict[str, Any] = {
            "template_hash_id": template_hash,
            "env": env or {},
        }
        resp = self._session.put(
            f"{VAST_API_BASE}/asks/{offer_id}/",
            json=body,
            timeout=self._timeout,
        )
        if not resp.ok:
            raise VastHelperError(
                f"Failed to create instance from offer {offer_id} "
                f"(HTTP {resp.status_code}): {_response_detail(resp)}"
            )
        return resp.json()

    def show_instance(self, instance_id: int) -> dict[str, Any]:
        """Return the current state of one instance.

        Uses the v1 list endpoint, which includes the ``ports`` mapping
        needed to resolve the external SSH port.
        """
        resp = self._session.get(
            f"{VAST_API_V1}/instances",
            timeout=self._timeout,
        )
        resp.raise_for_status()
        for inst in resp.json().get("instances", []):
            if inst.get("id") == instance_id:
                return inst
        raise InstanceNotFoundError(f"Instance {instance_id} not found.")

    def list_instances(self) -> list[dict[str, Any]]:
        """Return every instance visible to the account."""
        resp = self._session.get(
            f"{VAST_API_V1}/instances",
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return list(resp.json().get("instances", []))

    def destroy_instance(self, instance_id: int) -> None:
        """Destroy an instance.  Idempotent: a 404 is treated as success.

        Raises
        ------
        requests.HTTPError
            For any non-2xx response other than 404.
        """
        resp = self._session.delete(
            f"{VAST_API_BASE}/instances/{instance_id}/",
            timeout=self._timeout,
        )
        if resp.status_code == 404:
            logger.info("Instance %s is already destroyed.", instance_id)
            return
        resp.raise_for_status()

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "VastClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_local_filters(
    offers: list[dict[str, Any]],
    criteria: OfferCriteria,
) -> list[dict[str, Any]]:
    """Apply the filters that the vast.ai API does not reliably enforce."""
    if criteria.min_cuda > 0:
        before = len(offers)
        offers = [
            offer for offer in offers
            if offer.get("cuda_max_good", 0) >= criteria.min_cuda
        ]
        _log_filter("CUDA", before, offers, criteria.min_cuda, "cuda_max_good")

    if criteria.min_disk_gb > 0:
        before = len(offers)
        offers = [
            offer for offer in offers
            if offer.get("disk_space", 0) >= criteria.min_disk_gb
        ]
        _log_filter("Disk", before, offers, criteria.min_disk_gb, "disk_space")

    if criteria.min_pcie_bw > 0:
        before = len(offers)
        offers = [
            offer for offer in offers
            if offer.get("pcie_bw", 0) >= criteria.min_pcie_bw
        ]
        _log_filter("PCIe", before, offers, criteria.min_pcie_bw, "pcie_bw")

    if criteria.require_headless:
        before = len(offers)
        offers = [
            offer for offer in offers
            if not offer.get("gpu_display_active", False)
        ]
        removed = before - len(offers)
        if removed:
            logger.info(
                "Headless filter: removed %d/%d (gpu_display_active == True)",
                removed,
                before,
            )

    if criteria.limit is not None:
        offers = offers[: criteria.limit]

    return offers


def _log_filter(
    label: str,
    before: int,
    offers: list[dict[str, Any]],
    threshold: float,
    field: str,
) -> None:
    removed = before - len(offers)
    if removed:
        logger.info(
            "%s filter: removed %d/%d (%s < %g)",
            label,
            removed,
            before,
            field,
            threshold,
        )


def _response_detail(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text
