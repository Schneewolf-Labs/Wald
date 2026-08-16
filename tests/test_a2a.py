"""Agent registry and the A2A mailbox.

The mailbox has to survive a crashing recipient. An agent that fetches its inbox and dies
before acting must see the message again -- losing queued work silently is worse than
delivering it twice, so delivery and acknowledgement are separate states.

The other property is that an agent can only acknowledge its own mail. Without that, any
caller can make a message vanish from a queue it never read.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from wald.models import Agent
from wald.services import a2a


@pytest.fixture
def pair(session):
    alice, _ = a2a.register(session, {"slug": "alice", "name": "Alice", "capabilities": ["plan"]})
    bob, _ = a2a.register(session, {"slug": "bob", "name": "Bob", "capabilities": ["build"]})
    session.flush()
    return alice, bob


def test_register_creates_then_updates_in_place(session):
    agent, created = a2a.register(session, {"slug": "kira", "name": "Kira", "capabilities": ["a"]})
    session.flush()
    assert created

    again, created_again = a2a.register(
        session, {"slug": "kira", "name": "Kira", "capabilities": ["a", "b"], "owner": "nb"}
    )
    session.flush()

    # A restarting agent re-announcing itself must update its row, not collide with it.
    assert not created_again
    assert again.id == agent.id
    assert again.capabilities == ["a", "b"]
    assert again.owner == "nb"
    assert len(session.scalars(select(Agent).where(Agent.slug == "kira")).all()) == 1


def test_discovery_by_capability(session, pair):
    alice, bob = pair
    found = a2a.find_agents_by_capability(session, "build")
    assert [a.slug for a in found] == ["bob"]
    assert a2a.find_agents_by_capability(session, "nonexistent") == []


def test_inactive_agents_are_not_discoverable(session, pair):
    alice, bob = pair
    bob.status = "retired"
    session.flush()
    assert a2a.find_agents_by_capability(session, "build") == []


def test_message_round_trip(session, pair):
    alice, bob = pair
    msg = a2a.send_message(session, from_agent=alice, to_agent=bob, content="build the thing")
    session.flush()

    received = a2a.inbox(session, bob, unread_only=True)
    assert [m.id for m in received] == [msg.id]
    assert a2a.inbox(session, alice, unread_only=True) == []  # not a broadcast


def test_delivery_is_not_acknowledgement(session, pair):
    alice, bob = pair
    a2a.send_message(session, from_agent=alice, to_agent=bob, content="work")
    session.flush()

    first = a2a.inbox(session, bob, unread_only=True)
    a2a.mark_delivered(first)
    session.flush()

    # The recipient crashed here without acting. The work must still be waiting.
    second = a2a.inbox(session, bob, unread_only=True)
    assert [m.id for m in second] == [m.id for m in first]
    assert second[0].status == "delivered"


def test_acknowledgement_clears_the_unread_queue(session, pair):
    alice, bob = pair
    msg = a2a.send_message(session, from_agent=alice, to_agent=bob, content="work")
    session.flush()

    marked = a2a.mark_read(session, bob, [msg.id])
    session.flush()

    assert marked == 1
    assert a2a.inbox(session, bob, unread_only=True) == []
    assert len(a2a.inbox(session, bob, unread_only=False)) == 1  # still in the record


def test_acknowledging_twice_is_not_double_counted(session, pair):
    alice, bob = pair
    msg = a2a.send_message(session, from_agent=alice, to_agent=bob, content="work")
    session.flush()
    a2a.mark_read(session, bob, [msg.id])
    session.flush()
    assert a2a.mark_read(session, bob, [msg.id]) == 0


def test_an_agent_cannot_acknowledge_another_agents_mail(session, pair):
    alice, bob = pair
    msg = a2a.send_message(session, from_agent=alice, to_agent=bob, content="for bob only")
    session.flush()

    # Alice tries to ack a message addressed to Bob.
    assert a2a.mark_read(session, alice, [msg.id]) == 0
    assert len(a2a.inbox(session, bob, unread_only=True)) == 1


def test_thread_reads_oldest_first(session, pair):
    alice, bob = pair
    first = a2a.send_message(session, from_agent=alice, to_agent=bob, content="one")
    session.flush()
    second = a2a.send_message(
        session,
        from_agent=bob,
        to_agent=alice,
        content="two",
        role="response",
        thread_id=first.thread_id,
    )
    session.flush()

    convo = a2a.thread(session, first.thread_id)
    assert [m.content for m in convo] == ["one", "two"]
    assert convo[0].id == first.id and convo[1].id == second.id


def test_resolve_accepts_slug_or_uuid(session, pair):
    alice, _ = pair
    assert a2a.resolve_agent(session, "alice").id == alice.id
    assert a2a.resolve_agent(session, str(alice.id)).id == alice.id
    assert a2a.resolve_agent(session, "not-a-real-agent") is None
