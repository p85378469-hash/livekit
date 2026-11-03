# ============================================================
# tests/test_turn_detection.py
# Phase 1 exit check: Turn Detection
#
# Test cases from master plan:
# ☐ Short pause does not trigger premature response
# ☐ Partial text does not commit a budget or fire an LLM call
# ☐ Telugu amount split by pause (ఒక కోటి / ఇరవై లక్షలు)
# ☐ Turn ID advances correctly on TURN_COMPLETE
# ============================================================

from __future__ import annotations

import asyncio
import time
import pytest
import pytest_asyncio

from core.session import ConversationSession
from core.turn_detector import TurnDetector
from telemetry.latency import LatencyTracker


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def session() -> ConversationSession:
    return ConversationSession(session_id="test-session-01")


@pytest.fixture
def latency(session) -> LatencyTracker:
    t = LatencyTracker(session_id=session.session_id)
    t.new_turn(0)
    return t


@pytest.fixture
def completed_turns(request) -> list:
    """Collects turn completions for assertions."""
    return []


def make_turn_detector(session, latency, turns_completed):
    """Helper to build a TurnDetector with a capturing callback."""
    async def on_turn_complete(text, start_ms, end_ms):
        turns_completed.append({
            "text": text,
            "turn_id": session.current_turn_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

    return TurnDetector(
        session=session,
        latency=latency,
        on_turn_complete=on_turn_complete,
    )


# ── Test 1: Partial does not trigger a response ──────────────────────

@pytest.mark.asyncio
async def test_partial_does_not_trigger_turn(session, latency):
    """
    Phase 1 exit check: partial transcripts must never trigger the pipeline.
    """
    turns_completed = []
    detector = make_turn_detector(session, latency, turns_completed)

    # Simulate a series of partial updates
    await detector.on_partial("నేను")
    await detector.on_partial("నేను హైదరాబాద్")
    await detector.on_partial("నేను హైదరాబాద్ లో")

    # Assert: no turns should have been completed yet
    assert len(turns_completed) == 0, (
        f"Partial transcript triggered a turn completion: {turns_completed}"
    )

    # Assert: turn ID has NOT advanced
    assert session.current_turn_id == 0, (
        f"Turn ID advanced on partial: got {session.current_turn_id}"
    )


# ── Test 2: Final triggers exactly one TURN_COMPLETE ────────────────

@pytest.mark.asyncio
async def test_final_triggers_turn_complete(session, latency):
    """
    A Soniox final transcript event must trigger exactly one TURN_COMPLETE.
    """
    turns_completed = []
    detector = make_turn_detector(session, latency, turns_completed)

    await detector.on_partial("నేను హైదరాబాద్ లో")
    await detector.on_final("నేను హైదరాబాద్ లో ఇల్లు కొనాలని ఉంది")

    assert len(turns_completed) == 1, (
        f"Expected 1 turn completion, got {len(turns_completed)}"
    )
    assert turns_completed[0]["text"] == "నేను హైదరాబాద్ లో ఇల్లు కొనాలని ఉంది"
    assert session.current_turn_id == 1


# ── Test 3: Telugu amount split — continuation window ───────────────

@pytest.mark.asyncio
async def test_telugu_amount_split_waits_for_continuation(session, latency):
    """
    'ఒక కోటి' should NOT be finalized immediately — it looks incomplete.
    The detector should wait for continuation.
    If continuation arrives ('ఇరవై లక్షలు'), the full text should be the turn.

    This is the critical ₹1 crore vs ₹1.2 crore edge case.
    """
    turns_completed = []
    detector = make_turn_detector(session, latency, turns_completed)

    # First Soniox final — looks like incomplete amount
    final_task = asyncio.create_task(detector.on_final("ఒక కోటి"))

    # Simulate a brief pause (shorter than CONTINUATION_WAIT_MS)
    await asyncio.sleep(0.3)

    # No turn should have fired yet (still in wait window)
    assert len(turns_completed) == 0, (
        "Amount split: turned completed too early on 'ఒక కోటి'"
    )

    # Continuation arrives — caller continues speaking
    await detector.on_partial("ఒక కోటి ఇరవై లక్షలు")
    # Cancel event wakes the wait, but final for the full text comes separately
    await detector.on_final("ఒక కోటి ఇరవై లక్షలు")

    await final_task  # Ensure the first final's continuation wait finishes

    # Allow event loop to process
    await asyncio.sleep(0.1)

    # Exactly one turn should have completed
    assert len(turns_completed) == 1, (
        f"Expected 1 turn, got {len(turns_completed)}: {turns_completed}"
    )
    # The completed turn should contain the full amount
    assert "ఇరవై లక్షలు" in turns_completed[0]["text"], (
        f"Full amount not in completed turn: {turns_completed[0]['text']}"
    )


# ── Test 4: Continuation timeout fires if no new audio ──────────────

@pytest.mark.asyncio
async def test_amount_continuation_timeout(session, latency):
    """
    If 'ఒక కోటి' is said and no continuation arrives within
    CONTINUATION_WAIT_MS, the turn should finalize with the available text.
    """
    turns_completed = []
    detector = make_turn_detector(session, latency, turns_completed)
    detector.CONTINUATION_WAIT_MS = 200  # Speed up for testing

    await detector.on_final("ఒక కోటి")

    # Wait longer than the continuation window
    await asyncio.sleep(0.5)

    assert len(turns_completed) == 1, (
        f"Expected timeout finalization, got {len(turns_completed)} turns"
    )
    assert "ఒక కోటి" in turns_completed[0]["text"]


# ── Test 5: Turn ID advances atomically ─────────────────────────────

@pytest.mark.asyncio
async def test_turn_id_advances_before_pipeline(session, latency):
    """
    The turn_id must be advanced BEFORE the pipeline callback is invoked.
    This ensures any stale response tagged with the previous turn_id
    is immediately invalid when the new turn starts.
    """
    observed_turn_ids = []

    async def on_turn_complete(text, start_ms, end_ms):
        observed_turn_ids.append(session.current_turn_id)

    detector = TurnDetector(
        session=session,
        latency=latency,
        on_turn_complete=on_turn_complete,
    )

    initial_id = session.current_turn_id  # Should be 0
    await detector.on_final("విజయవాడ లో 3 బెడ్‌రూమ్ ఫ్లాట్ కావాలి")

    assert len(observed_turn_ids) == 1
    assert observed_turn_ids[0] > initial_id, (
        f"Turn ID should have advanced before pipeline callback. "
        f"initial={initial_id}, in_callback={observed_turn_ids[0]}"
    )
