"""Tests for provisioning, waiting, and teardown."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from vast_helper.errors import (
    AuthError,
    NoOffersError,
    ProvisionTimeoutError,
    SSHTimeoutError,
    VastHelperError,
)
from vast_helper.provision import (
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

from helpers import FakeClient

OFFER = {
    "id": 42,
    "gpu_name": "RTX 3090",
    "gpu_ram": 24576,
    "reliability": 0.99,
    "dph_total": 0.12,
}

RUNNING = {
    "id": 7,
    "actual_status": "running",
    "public_ipaddr": "1.2.3.4",
    "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "22002"}]},
}


class FlakyDestroyClient(FakeClient):
    def __init__(
        self,
        failures: int,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.failures = failures
        self.error = error or requests.HTTPError(
            response=SimpleNamespace(status_code=502)  # type: ignore[arg-type]
        )

    def destroy_instance(self, instance_id: int) -> None:
        if self.failures:
            self.failures -= 1
            raise self.error
        self.destroyed.append(instance_id)


class TestExtractSSHPort:
    def test_prefers_ports_mapping(self):
        data = {
            "ssh_port": 22001,
            "ports": {"22/tcp": [{"HostPort": "22002"}]},
        }

        assert extract_ssh_port(data) == 22002

    def test_falls_back_to_ssh_port(self):
        assert extract_ssh_port({"ssh_port": 22001}) == 22001

    def test_defaults_to_22(self):
        assert extract_ssh_port({}) == 22


class TestInstanceInfo:
    def test_from_api_uses_offer_fallbacks(self):
        info = InstanceInfo.from_api(
            RUNNING,
            offer=OFFER,
            ssh_key="key",
        )

        assert info.id == 7
        assert info.host == "1.2.3.4"
        assert info.port == 22002
        assert info.gpu_name == "RTX 3090"
        assert info.price_per_hour == 0.12
        assert info.reliability == 0.99
        assert info.ssh_target == "root@1.2.3.4"
        assert info.destroy_command() == "vast-helper destroy 7"

    def test_from_api_falls_back_to_ssh_host(self):
        data = {"id": 1, "ssh_host": "ssh5.vast.ai", "ssh_port": 1234}

        info = InstanceInfo.from_api(data)

        assert info.host == "ssh5.vast.ai"
        assert info.port == 1234

    def test_missing_host_raises(self):
        with pytest.raises(VastHelperError, match="no SSH host"):
            InstanceInfo.from_api({"id": 1, "actual_status": "loading"})

    def test_missing_id_raises(self):
        with pytest.raises(VastHelperError, match="no id"):
            InstanceInfo.from_api({"public_ipaddr": "1.2.3.4"})


class TestWaitForRunning:
    def test_returns_when_running_with_ports(self):
        client = FakeClient(states=[RUNNING])

        data = wait_for_running(client, 7, timeout_s=1, poll_interval_s=0)

        assert data == RUNNING

    def test_waits_for_ports_after_running(self):
        client = FakeClient(
            states=[
                {"id": 7, "actual_status": "running"},
                RUNNING,
            ]
        )

        data = wait_for_running(client, 7, timeout_s=1, poll_interval_s=0)

        assert data == RUNNING

    def test_times_out(self):
        client = FakeClient(states=[{"id": 7, "actual_status": "provisioning"}])

        with pytest.raises(ProvisionTimeoutError, match="did not reach 'running'"):
            wait_for_running(client, 7, timeout_s=0.01, poll_interval_s=0)


class TestWaitForSSH:
    def test_returns_when_ready(self):
        class ReadySSH:
            host = "h"
            port = 22

            def test_connection(self):
                return True

        wait_for_ssh(ReadySSH(), timeout_s=1, poll_interval_s=0)  # type: ignore[arg-type]

    def test_times_out(self):
        class DeadSSH:
            host = "h"
            port = 22

            def test_connection(self):
                return False

        with pytest.raises(SSHTimeoutError, match="did not become available"):
            wait_for_ssh(  # type: ignore[arg-type]
                DeadSSH(), timeout_s=0.01, poll_interval_s=0
            )


class TestProvisionInstance:
    def test_happy_path(self):
        client = FakeClient(
            offers=[OFFER],
            created={"new_contract": 7},
            states=[RUNNING],
        )
        config = ProvisionConfig(
            template_hash="template",
            env={"A": "1"},
            wait_for_ssh=False,
        )

        info = provision_instance(client, config)

        assert info.id == 7
        assert info.host == "1.2.3.4"
        assert info.port == 22002
        assert info.gpu_name == "RTX 3090"
        assert client.created_with == {
            "offer_id": 42,
            "template_hash": "template",
            "env": {"A": "1"},
        }

    def test_waits_for_ssh_when_configured(self, monkeypatch):
        client = FakeClient(
            offers=[OFFER],
            created={"instance_id": 7},
            states=[RUNNING],
        )
        calls = []

        def fake_wait(ssh, **kwargs):
            calls.append(ssh)

        monkeypatch.setattr("vast_helper.provision.wait_for_ssh", fake_wait)
        config = ProvisionConfig(template_hash="template", wait_for_ssh=True)

        provision_instance(client, config)

        assert len(calls) == 1
        assert calls[0].host == "1.2.3.4"
        assert calls[0].port == 22002

    def test_no_offers_raises(self):
        client = FakeClient(offers=[])

        with pytest.raises(NoOffersError, match="No GPU offers"):
            provision_instance(
                client,
                ProvisionConfig(template_hash="template"),
            )

    def test_accepts_all_instance_id_response_keys(self):
        for key in ("new_contract", "instance_id", "id"):
            client = FakeClient(
                offers=[OFFER],
                created={key: 7},
                states=[RUNNING],
            )

            info = provision_instance(
                client,
                ProvisionConfig(template_hash="t", wait_for_ssh=False),
            )

            assert info.id == 7

    def test_timeout_destroys_instance(self):
        client = FakeClient(
            offers=[OFFER],
            created={"instance_id": 7},
            states=[{"id": 7, "actual_status": "provisioning"}],
        )
        config = ProvisionConfig(
            template_hash="t",
            wait_for_ssh=False,
            api_poll_timeout_s=0.01,
            poll_interval_s=0,
        )

        with pytest.raises(ProvisionTimeoutError):
            provision_instance(client, config)

        assert client.destroyed == [7]

    def test_timeout_keep_alive_leaves_instance(self):
        client = FakeClient(
            offers=[OFFER],
            created={"instance_id": 7},
            states=[{"id": 7, "actual_status": "provisioning"}],
        )
        config = ProvisionConfig(
            template_hash="t",
            wait_for_ssh=False,
            api_poll_timeout_s=0.01,
            poll_interval_s=0,
            keep_alive=True,
        )

        with pytest.raises(ProvisionTimeoutError):
            provision_instance(client, config)

        assert client.destroyed == []

    def test_unreachable_instance_is_destroyed(self):
        client = FakeClient(
            offers=[OFFER],
            created={"instance_id": 7},
            states=[{"id": 7, "actual_status": "running", "ports": {}}],
        )
        config = ProvisionConfig(
            template_hash="t",
            wait_for_ssh=False,
            api_poll_timeout_s=0.01,
            poll_interval_s=0,
        )

        with pytest.raises(ProvisionTimeoutError):
            provision_instance(client, config)

        assert client.destroyed == [7]


class TestDestroy:
    def test_retries_transient_failures(self):
        client = FlakyDestroyClient(failures=2)

        assert destroy_instance(client, 7, retries=3, delay_s=0) is True
        assert client.destroyed == [7]

    def test_gives_up_after_retries(self):
        client = FlakyDestroyClient(failures=3)

        assert destroy_instance(client, 7, retries=3, delay_s=0) is False
        assert client.destroyed == []

    def test_retries_network_errors(self):
        client = FlakyDestroyClient(
            failures=2,
            error=requests.ConnectionError("transient"),
        )

        assert destroy_instance(client, 7, retries=3, delay_s=0) is True
        assert client.destroyed == [7]

    def test_destroy_unless_kept_honours_flag(self):
        client = FakeClient()

        assert destroy_unless_kept(client, 7, keep_alive=True) is False
        assert client.destroyed == []

    def test_destroy_unless_kept_destroys(self):
        client = FakeClient()

        assert destroy_unless_kept(client, 7, keep_alive=False) is True
        assert client.destroyed == [7]

    def test_destroy_unless_kept_none_is_noop(self):
        client = FakeClient()

        assert destroy_unless_kept(client, None, keep_alive=False) is False
        assert client.destroyed == []


class TestManagedInstance:
    def _config(self, **overrides):
        values = {
            "template_hash": "t",
            "wait_for_ssh": False,
        }
        values.update(overrides)
        return ProvisionConfig(**values)

    def _client(self):
        return FakeClient(
            offers=[OFFER],
            created={"instance_id": 7},
            states=[RUNNING],
        )

    def test_destroys_on_success(self):
        client = self._client()

        with managed_instance(client, self._config()) as info:
            assert info.id == 7

        assert client.destroyed == [7]

    def test_destroys_on_body_error(self):
        client = self._client()

        with pytest.raises(RuntimeError):
            with managed_instance(client, self._config()):
                raise RuntimeError("boom")

        assert client.destroyed == [7]

    def test_keep_alive_leaves_instance(self):
        client = self._client()

        with managed_instance(client, self._config(keep_alive=True)):
            pass

        assert client.destroyed == []


class TestDiscovery:
    def test_find_running_instances(self):
        client = FakeClient(
            instances=[
                {"id": 1, "actual_status": "running"},
                {"id": 2, "actual_status": "stopped"},
                {"id": 3, "actual_status": "running"},
            ]
        )

        assert [i["id"] for i in find_running_instances(client)] == [1, 3]

    def test_select_running_instance_auto_picks_single(self):
        client = FakeClient(
            instances=[RUNNING],
            states=[RUNNING],
        )

        info = select_running_instance(client)

        assert info.id == 7

    def test_select_running_instance_requires_id_when_ambiguous(self):
        client = FakeClient(
            instances=[
                {"id": 1, "actual_status": "running"},
                {"id": 2, "actual_status": "running"},
            ]
        )

        with pytest.raises(VastHelperError, match="Multiple running instances"):
            select_running_instance(client)

    def test_select_running_instance_raises_when_none(self):
        client = FakeClient(instances=[])

        with pytest.raises(VastHelperError, match="No running instances"):
            select_running_instance(client)

    def test_attach_instance_returns_info(self):
        client = FakeClient(states=[RUNNING])

        info = attach_instance(client, 7)

        assert info.id == 7

    def test_attach_instance_can_wait_for_ssh(self, monkeypatch):
        client = FakeClient(states=[RUNNING])
        calls = []
        monkeypatch.setattr(
            "vast_helper.provision.wait_for_ssh",
            lambda ssh, **kwargs: calls.append(ssh),
        )

        attach_instance(client, 7, connect=True)

        assert len(calls) == 1


class TestVerifyConnection:
    def test_ok(self):
        client = FakeClient(auth={"email": "a@b.c", "balance": 4.5})

        check = verify_connection(client=client)

        assert check.api_ok is True
        assert check.user_email == "a@b.c"
        assert check.balance == 4.5

    def test_auth_error(self):
        client = FakeClient(auth=AuthError("nope"))

        check = verify_connection(client=client)

        assert check.api_ok is False
        assert check.user_email is None

    def test_network_error(self):
        client = FakeClient(auth=requests.ConnectionError("offline"))

        check = verify_connection(client=client)

        assert check.api_ok is False


def test_instance_status_prefers_actual_status():
    assert instance_status({"actual_status": "running", "cur_state": "loading"}) == "running"
    assert instance_status({"cur_state": "loading"}) == "loading"
    assert instance_status({}) == "unknown"
