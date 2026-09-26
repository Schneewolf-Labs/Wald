"""Both servers refuse to listen beyond loopback while identity is only a claim."""

from __future__ import annotations

import sys

import pytest

from wald.config import get_settings
from wald.guards import is_loopback, serving_problem


def _settings(**update):
    return get_settings().model_copy(update=update)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.1.1", "localhost", "::1", "[::1]"])
def test_loopback_hosts(host):
    assert is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "wald.internal"])
def test_reachable_hosts(host):
    # A hostname may resolve anywhere, so it counts as reachable.
    assert not is_loopback(host)


def test_loopback_without_auth_is_fine():
    assert serving_problem(_settings(require_auth=False), "127.0.0.1") is None


def test_a_wide_bind_without_auth_refuses():
    problem = serving_problem(_settings(require_auth=False), "0.0.0.0")
    assert problem and "WALD_REQUIRE_AUTH" in problem and "WALD_ALLOW_UNAUTHENTICATED" in problem


def test_a_wide_bind_with_auth_is_fine():
    assert serving_problem(_settings(require_auth=True), "0.0.0.0") is None


def test_the_override_permits_a_wide_bind():
    settings = _settings(require_auth=False, allow_unauthenticated=True)
    assert serving_problem(settings, "0.0.0.0") is None


def test_stdio_has_no_host_to_check():
    assert serving_problem(_settings(require_auth=False), None) is None


def test_authz_without_auth_refuses_even_on_loopback():
    settings = _settings(require_auth=False, enforce_authz=True)
    assert "WALD_ENFORCE_AUTHZ" in serving_problem(settings, "127.0.0.1")


def test_wald_api_refuses_before_touching_the_database(monkeypatch):
    from wald import main

    monkeypatch.setattr(main, "get_settings", lambda: _settings(host="0.0.0.0", require_auth=False))
    with pytest.raises(SystemExit, match="wald-api: refusing to listen on 0.0.0.0"):
        main.run()


def test_wald_mcp_refuses_a_wide_http_bind(monkeypatch):
    from wald.mcp import server

    monkeypatch.setattr(server, "_settings", _settings(mcp_host="0.0.0.0", require_auth=False))
    monkeypatch.setattr(sys, "argv", ["wald-mcp", "--transport", "streamable-http"])
    with pytest.raises(SystemExit, match="wald-mcp: refusing to listen on 0.0.0.0"):
        server.run()
