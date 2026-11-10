"""Live tests against the vast.ai API.

Skipped unless an API key is set.  Run explicitly with::

    pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from vast_helper.client import OfferCriteria, VastClient
from vast_helper.provision import verify_connection

pytestmark = pytest.mark.integration


def _api_key() -> str | None:
    return os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")


def test_api_key_authenticates():
    if not _api_key():
        pytest.skip("VAST_API_KEY not set")

    check = verify_connection()

    assert check.api_ok, "vast.ai rejected the configured API key"
    assert check.user_email is not None


def test_search_returns_offers_or_empty_list():
    if not _api_key():
        pytest.skip("VAST_API_KEY not set")

    client = VastClient()
    offers = client.search_offers(
        OfferCriteria(
            min_vram_gb=24,
            max_price=0.20,
            min_cuda=0,
            min_disk_gb=0,
            min_pcie_bw=0,
            limit=5,
        )
    )

    assert isinstance(offers, list)
    for offer in offers:
        assert "id" in offer
        assert "dph_total" in offer
        assert "gpu_name" in offer
