"""M5 verification: strict-offline mode is enforced structurally.

The ladder never climbs past LOCAL_LARGE when offline, a cloud provider that is
present is never touched, and a full run that fails every local rung ends in
DEFER (never an exception) while opening zero non-localhost sockets.
"""

from __future__ import annotations

import socket

from local_infra.extraction.budget import Budget
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.providers.base import Stage
from tests._fakes import GOOD, SRC, FakeProvider, facts_result, transport_fail


def build(providers, *, offline, budget=None):
    return ExtractionLadder(
        {p.stage: p for p in providers}, offline=offline, budget=budget or Budget(1.0)
    )


async def test_offline_max_stage_is_large() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [])
    assert build([small], offline=True).max_stage() is Stage.LOCAL_LARGE


async def test_offline_caps_and_defers_never_reaching_cloud() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [transport_fail()])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [transport_fail()])
    large = FakeProvider(Stage.LOCAL_LARGE, [transport_fail()])
    # A cloud provider is present but must NEVER be called while offline.
    cloud = FakeProvider(Stage.CLOUD, [facts_result(GOOD)])

    res = await build([small, repair, large, cloud], offline=True).run(SRC)

    assert res.final_stage is Stage.DEFER
    assert res.facts == []
    assert Stage.CLOUD not in res.stages
    assert cloud.calls == []  # no object called by accident


async def test_offline_run_opens_no_nonlocal_sockets(monkeypatch) -> None:
    real_connect = socket.socket.connect

    def guard(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"offline run attempted a non-local socket to {host!r}")
        return real_connect(self, address, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", guard)

    small = FakeProvider(Stage.LOCAL_SMALL, [transport_fail()])
    large = FakeProvider(Stage.LOCAL_LARGE, [transport_fail()])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [transport_fail()])
    res = await build([small, repair, large], offline=True).run(SRC)
    assert res.final_stage is Stage.DEFER  # completed without touching the network
