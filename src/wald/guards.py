"""Startup checks shared by `wald-api` and `wald-mcp`.

Both servers take a caller's identity on trust while WALD_REQUIRE_AUTH is off, which is
fine on loopback and not fine anywhere else. The defaults keep them on loopback, but a
default only protects someone who never changes it: set WALD_HOST=0.0.0.0 to reach the
hub from another machine, forget the auth switch, and the result is a hub that serves
every page and inbox to the network and lets anyone send as any agent. So that
combination refuses to start instead of relying on someone noticing.

WALD_ALLOW_UNAUTHENTICATED is the way out for setups where a wide bind is not wide
exposure -- a container binding 0.0.0.0 inside its own network namespace with the port
published only on the host's loopback, say. It is named for what it permits.
"""

from __future__ import annotations

import ipaddress

from wald.config import Settings


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        # Any other hostname may resolve to anything; treat it as reachable.
        return False


def serving_problem(settings: Settings, host: str | None) -> str | None:
    """Why this configuration must not start, or None. `host` is None for stdio."""
    if settings.enforce_authz and not settings.require_auth:
        # Grants attach to identities the token layer has proved; enforcing them against
        # claimed ones would look locked and not be.
        return (
            "WALD_ENFORCE_AUTHZ requires WALD_REQUIRE_AUTH -- authorization without "
            "authentication would enforce grants against unverified identities"
        )
    if (
        host is not None
        and not is_loopback(host)
        and not settings.require_auth
        and not settings.allow_unauthenticated
    ):
        return (
            f"refusing to listen on {host} with WALD_REQUIRE_AUTH off: anyone who can reach "
            "it could act as any agent. Turn on WALD_REQUIRE_AUTH (and issue tokens with "
            "wald-token), bind to 127.0.0.1, or set WALD_ALLOW_UNAUTHENTICATED=true if "
            "something else already keeps the port private"
        )
    return None
