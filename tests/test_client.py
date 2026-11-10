"""Tests for the vast.ai REST client."""

from __future__ import annotations

import os

import pytest
import requests

from vast_helper.client import OfferCriteria, VastClient
from vast_helper.constants import VAST_API_BASE, VAST_API_V1
from vast_helper.errors import AuthError, InstanceNotFoundError, VastHelperError

from helpers import FakeResponse, FakeSession


def _client(*responses: FakeResponse, **kwargs: object) -> tuple[VastClient, FakeSession]:
    session = FakeSession(*responses)
    return VastClient("test-key", session=session, **kwargs), session


class TestOfferCriteria:
    def test_api_body_uses_megabytes_and_expected_flags(self):
        criteria = OfferCriteria(
            min_vram_gb=24,
            max_price=0.2,
            min_price=0.1,
            min_reliability=0.95,
        )

        body = criteria.api_body()

        assert body["verified"] == {"eq": True}
        assert body["rentable"] == {"eq": True}
        assert body["rented"] == {"eq": False}
        assert body["type"] == "ondemand"
        assert body["gpu_ram"] == {"gte": 24576}
        assert body["reliability"] == {"gte": 0.95}
        assert body["dph_total"] == {"gte": 0.1, "lte": 0.2}
        assert body["order"] == [
            ["dph_total", "asc"],
            ["reliability", "desc"],
        ]

    def test_describe_includes_active_filters(self):
        text = OfferCriteria(
            min_cuda=12.4,
            min_disk_gb=50,
            min_pcie_bw=10,
            require_headless=True,
        ).describe()

        assert "CUDA >= 12.4" in text
        assert "disk >= 50 GB" in text
        assert "PCIe >= 10 GB/s" in text
        assert "headless" in text


class TestVastClientInit:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        monkeypatch.delenv("VASTAI_API_KEY", raising=False)

        with pytest.raises(AuthError, match="VAST_API_KEY"):
            VastClient()

    def test_legacy_env_var_is_accepted(self, monkeypatch):
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        monkeypatch.setenv("VASTAI_API_KEY", "legacy-key")
        session = FakeSession()

        VastClient(session=session)

        assert session.headers["Authorization"] == "Bearer legacy-key"

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("VAST_API_KEY", "env-key")
        session = FakeSession()

        VastClient("explicit-key", session=session)

        assert session.headers["Authorization"] == "Bearer explicit-key"


class TestSearchOffers:
    def test_posts_criteria_and_returns_offers(self):
        offer = {"id": 1, "dph_total": 0.15}
        client, session = _client(FakeResponse(payload={"offers": [offer]}))

        offers = client.search_offers(
            OfferCriteria(
                min_vram_gb=16,
                min_reliability=0.9,
                max_price=0.2,
                min_cuda=0,
                min_disk_gb=0,
                min_pcie_bw=0,
                require_headless=False,
            )
        )

        assert offers == [offer]
        method, url, kwargs = session.calls[0]
        assert method == "POST"
        assert url == f"{VAST_API_BASE}/bundles/"
        assert kwargs["json"]["gpu_ram"] == {"gte": 16384}

    def test_applies_local_filters(self):
        offers = [
            {"id": 1, "cuda_max_good": 12.0, "disk_space": 100, "pcie_bw": 16},
            {"id": 2, "cuda_max_good": 12.8, "disk_space": 20, "pcie_bw": 16},
            {"id": 3, "cuda_max_good": 12.8, "disk_space": 100, "pcie_bw": 4},
            {"id": 4, "cuda_max_good": 12.8, "disk_space": 100, "pcie_bw": 16, "gpu_display_active": True},
            {"id": 5, "cuda_max_good": 12.8, "disk_space": 100, "pcie_bw": 16, "gpu_display_active": False},
        ]
        client, _ = _client(FakeResponse(payload={"offers": offers}))

        result = client.search_offers(
            OfferCriteria(
                min_cuda=12.4,
                min_disk_gb=50,
                min_pcie_bw=10,
                require_headless=True,
            )
        )

        assert [offer["id"] for offer in result] == [5]

    def test_limit_truncates_after_filtering(self):
        offers = [{"id": i, "cuda_max_good": 13, "disk_space": 100, "pcie_bw": 16} for i in range(5)]
        client, _ = _client(FakeResponse(payload={"offers": offers}))

        result = client.search_offers(
            OfferCriteria(min_cuda=0, min_disk_gb=0, min_pcie_bw=0, limit=2)
        )

        assert [offer["id"] for offer in result] == [0, 1]

    def test_empty_payload_returns_empty_list(self):
        client, _ = _client(FakeResponse(payload={}))

        assert client.search_offers(OfferCriteria()) == []


class TestCreateInstance:
    def test_puts_template_and_env(self):
        client, session = _client(FakeResponse(payload={"instance_id": 7}))

        created = client.create_instance(
            42,
            template_hash="abc123",
            env={"A": "1"},
        )

        assert created == {"instance_id": 7}
        method, url, kwargs = session.calls[0]
        assert method == "PUT"
        assert url == f"{VAST_API_BASE}/asks/42/"
        assert kwargs["json"] == {
            "template_hash_id": "abc123",
            "env": {"A": "1"},
        }

    def test_error_includes_response_detail(self):
        client, _ = _client(
            FakeResponse(status_code=400, payload={"error": "offer gone"})
        )

        with pytest.raises(VastHelperError, match="offer gone"):
            client.create_instance(1, template_hash="abc")


class TestShowInstance:
    def test_returns_matching_instance(self):
        instance = {"id": 99, "actual_status": "running"}
        client, session = _client(
            FakeResponse(payload={"instances": [{"id": 1}, instance]})
        )

        assert client.show_instance(99) == instance
        assert session.calls[0][1] == f"{VAST_API_V1}/instances"

    def test_raises_when_missing(self):
        client, _ = _client(FakeResponse(payload={"instances": []}))

        with pytest.raises(InstanceNotFoundError):
            client.show_instance(99)


class TestListInstances:
    def test_returns_all_instances(self):
        instances = [{"id": 1}, {"id": 2}]
        client, _ = _client(FakeResponse(payload={"instances": instances}))

        assert client.list_instances() == instances


class TestDestroyInstance:
    def test_sends_delete(self):
        client, session = _client(FakeResponse(status_code=204))

        client.destroy_instance(99)

        method, url, _ = session.calls[0]
        assert method == "DELETE"
        assert url == f"{VAST_API_BASE}/instances/99/"

    def test_404_is_silent(self):
        client, _ = _client(FakeResponse(status_code=404))

        client.destroy_instance(99)

    def test_other_error_raises(self):
        client, _ = _client(FakeResponse(status_code=500))

        with pytest.raises(requests.HTTPError):
            client.destroy_instance(99)


class TestCheckAuth:
    def test_returns_user(self):
        client, _ = _client(FakeResponse(payload={"email": "a@b.c"}))

        assert client.check_auth() == {"email": "a@b.c"}

    def test_401_raises_auth_error(self):
        client, _ = _client(FakeResponse(status_code=401))

        with pytest.raises(AuthError):
            client.check_auth()

    def test_500_raises_http_error(self):
        client, _ = _client(FakeResponse(status_code=500))

        with pytest.raises(requests.HTTPError):
            client.check_auth()


def test_context_manager_closes_session():
    session = FakeSession()
    closed = []
    session.close = lambda: closed.append(True)  # type: ignore[method-assign]

    with VastClient("key", session=session):
        pass

    assert closed == [True]


def test_env_key_is_stripped(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "  spaced-key  ")
    session = FakeSession()

    VastClient(session=session)

    assert session.headers["Authorization"] == "Bearer spaced-key"
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    assert os.environ.get("VAST_API_KEY") is None
