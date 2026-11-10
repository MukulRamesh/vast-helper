"""Shared fakes for the offline test suite."""

from __future__ import annotations

import requests


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(
        self,
        status_code: int = 200,
        payload: object | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=self,  # type: ignore[arg-type]
            )


class FakeSession:
    """Records HTTP calls and returns queued responses in order."""

    def __init__(self, *responses: FakeResponse) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self._responses = list(responses)

    def _record(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if not self._responses:
            raise AssertionError(f"no response queued for {method} {url}")
        return self._responses.pop(0)

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        return self._record("GET", url, **kwargs)

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        return self._record("POST", url, **kwargs)

    def put(self, url: str, **kwargs: object) -> FakeResponse:
        return self._record("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: object) -> FakeResponse:
        return self._record("DELETE", url, **kwargs)

    def close(self) -> None:
        pass


class FakeClient:
    """Stand-in for :class:`vast_helper.VastClient`."""

    def __init__(
        self,
        *,
        offers: list[dict[str, object]] | None = None,
        created: dict[str, object] | None = None,
        states: list[dict[str, object]] | None = None,
        instances: list[dict[str, object]] | None = None,
        auth: dict[str, object] | None = None,
    ) -> None:
        self.offers = offers or []
        self.created = created if created is not None else {"instance_id": 1}
        self.states = states or []
        self.instances = instances or []
        self.auth = auth or {"email": "a@b.c", "balance": 10.0}
        self.destroyed: list[int] = []
        self.destroy_error: Exception | None = None
        self.created_with: dict[str, object] = {}

    def search_offers(self, criteria: object) -> list[dict[str, object]]:
        self.search_criteria = criteria
        return self.offers

    def create_instance(
        self,
        offer_id: int,
        *,
        template_hash: str,
        env: dict[str, str] | None = None,
    ) -> dict[str, object]:
        self.created_with = {
            "offer_id": offer_id,
            "template_hash": template_hash,
            "env": env,
        }
        return self.created

    def show_instance(self, instance_id: int) -> dict[str, object]:
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]

    def list_instances(self) -> list[dict[str, object]]:
        return self.instances

    def destroy_instance(self, instance_id: int) -> None:
        if self.destroy_error is not None:
            raise self.destroy_error
        self.destroyed.append(instance_id)

    def check_auth(self) -> dict[str, object]:
        if isinstance(self.auth, Exception):
            raise self.auth
        return self.auth
