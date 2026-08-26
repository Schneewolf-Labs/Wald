"""Authorization: what an authenticated agent may do.

Authentication (services/tokens.py) settles *who* is calling; until now nothing said
*what* they may do, so every agent with a token could read every resource and write any
wiki page. Grants close that gap.

A grant is a string, ``pillar:action[:selector]``:

    wiki:read:engineering      read wiki pages in the "engineering" space
    wiki:write:runbooks        write pages in "runbooks"
    resource:read:merlina      read one resource's connection + auth details
    resource:read:*            read the whole resource directory
    wiki:read                  selector defaults to *

Strings rather than a permission table because grants live where agents are declared --
the ``grants`` list in an agent's TOML file -- and a reviewer diffing that file should be
able to read the policy without joining anything. The grammar is validated strictly
(unknown pillars and actions are rejected, same reasoning as the seed loader rejecting
unknown fields): a typo that parsed would be a permission silently not granted, discovered
as a mystery denial much later.

Enforcement is **opt-in** (`WALD_ENFORCE_AUTHZ`) and requires authentication to be on,
because grants attach to a *proven* identity -- enforcing them against a claimed one would
be a lock on a door with no wall. With enforcement off, `ALLOW_ALL` keeps every surface
behaving exactly as before, so an upgraded hub does not strand its fleet.

Scope note: grants govern the knowledge pillars (wiki, resources). The agent registry
stays readable by every caller -- discovery is what it is *for* -- and A2A mailboxes are
already bound to the verified identity, which is a stronger statement than a grant.
"""

from __future__ import annotations

from dataclasses import dataclass

ALL = "*"

_ALLOWED = {
    ("wiki", "read"),
    ("wiki", "write"),
    ("resource", "read"),
}


@dataclass(frozen=True)
class Grant:
    pillar: str
    action: str
    selector: str


def parse_grant(text: str) -> Grant:
    """Parse ``pillar:action[:selector]``, rejecting anything outside the grammar."""
    parts = text.strip().split(":")
    if len(parts) == 2:
        parts.append(ALL)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"malformed grant '{text}': expected pillar:action[:selector]")
    pillar, action, selector = parts
    if (pillar, action) not in _ALLOWED:
        allowed = ", ".join(sorted(f"{p}:{a}" for p, a in _ALLOWED))
        raise ValueError(f"unknown grant '{pillar}:{action}' (known: {allowed})")
    return Grant(pillar, action, selector)


class Grants:
    """An agent's parsed grant set, with wildcard matching."""

    def __init__(self, strings: list[str] | None = None, *, allow_all: bool = False):
        self._allow_all = allow_all
        self._grants = [parse_grant(s) for s in strings or []]

    def allows(self, pillar: str, action: str, selector: str) -> bool:
        if self._allow_all:
            return True
        return any(
            g.pillar == pillar and g.action == action and g.selector in (ALL, selector)
            for g in self._grants
        )

    @property
    def unrestricted(self) -> bool:
        return self._allow_all


# What every caller gets while enforcement is off: the pre-authz behavior, verbatim.
ALLOW_ALL = Grants(allow_all=True)
