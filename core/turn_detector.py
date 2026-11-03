# ==============================================================================
# core/turn_detector.py — Turn & Pause Detection Gate
#
# IN SIMPLE WORDS:
# How does the agent know when the customer has stopped talking and it's time to answer?
# That is called "Turn Detection".
#
# The biggest challenge in Telugu:
# People often say numbers with small natural pauses, like:
# "ఒక కోటి…" (1 Crore) -> [takes a 0.5 sec breath] -> "ఇరవై లక్షలు" (20 Lakhs).
# If the agent jumps in too fast during the breath, it will wrongly think the budget
# is ₹1 Crore instead of ₹1.2 Crore!
#
# This file solves that problem:
# 1. Partial updates (words spoken mid-sentence) are only buffered, NEVER acted on.
# 2. When a sentence ends with an incomplete amount word like "కోటి" or "లక్ష",
#    it waits a brief moment (1.2 seconds) to see if more numbers follow!
# 3. Only when the customer has truly finished does it signal TURN_COMPLETE.
# ==============================================================================

from __future__ import annotations

import asyncio
import time
from typing import Callable, Awaitable, Optional
import structlog

from core.session import ConversationSession
from telemetry.latency import LatencyTracker, Stage

log = structlog.get_logger(__name__)

# Minimum duration (in milliseconds) a turn must last (400 ms = 0.4 seconds)
MIN_TURN_DURATION_MS: float = 400.0

# If a caller's sentence ends with one of these words, they might be in the middle
# of saying an amount like "1 crore ... 20 lakhs". We watch for these endings!
_INCOMPLETE_AMOUNT_ENDINGS = (
    "కోటి",     # crore
    "లక్ష",     # lakh
    "వేల",     # thousand
    "crore",
    "lakh",
)


class TurnDetector:
    """
    The gatekeeper between what the Speech Recognizer (Soniox) hears
    and when the agent decides to trigger an answer.
    """

    # How long to wait (in milliseconds) if we suspect the user paused mid-number
    CONTINUATION_WAIT_MS: float = 1200.0

    def __init__(
        self,
        session: ConversationSession,
        latency: LatencyTracker,
        on_turn_complete: Callable[[str, float, float], Awaitable[None]],
    ) -> None:
        """
        session: Active call notebook.
        latency: Latency stopwatch.
        on_turn_complete: Function to run once the caller finishes speaking.
        """
        self._session = session
        self._latency = latency
        self._on_turn_complete = on_turn_complete

        # Temporary workspace for words being heard right now
        self._provisional_text: str = ""
        self._turn_start_ms: Optional[float] = None
        self._awaiting_continuation: bool = False
        self._continuation_task: Optional[asyncio.Task] = None

    # --------------------------------------------------------------------------
    # Live Events from Soniox Speech-to-Text
    # --------------------------------------------------------------------------

    async def on_partial(self, partial_text: str) -> None:
        """
        Runs while the caller is actively speaking (e.g. "నేను...", "నేను హైదరాబాద్...").
        This NEVER fires an AI response — it only updates our temporary screen buffer.
        """
        if self._turn_start_ms is None:
            # Mark the exact millisecond the user started speaking this turn
            self._turn_start_ms = time.time() * 1000
            log.debug("turn.speech_started", turn_id=self._session.current_turn_id)

        self._provisional_text = partial_text

        # If we were previously waiting to see if a paused number would continue,
        # and new words just arrived, cancel the timer because the user is indeed talking!
        if self._awaiting_continuation and self._continuation_task:
            self._continuation_task.cancel()
            self._awaiting_continuation = False

        log.debug(
            "turn.partial",
            turn_id=self._session.current_turn_id,
            text_preview=partial_text[:60],
        )

    async def on_final(self, final_text: str) -> None:
        """
        Runs when Soniox detects that the user finished an utterance.
        """
        turn_end_ms = time.time() * 1000
        turn_start_ms = self._turn_start_ms or turn_end_ms
        duration_ms = turn_end_ms - turn_start_ms

        # Record T0: End of user speech timestamp
        self._latency.record(Stage.TURN_END, turn_end_ms)

        log.info(
            "turn.final_received",
            turn_id=self._session.current_turn_id,
            text=final_text,
            duration_ms=round(duration_ms, 1),
        )

        # Check: Was it too short to be real speech? (e.g. a micro-click)
        if duration_ms < MIN_TURN_DURATION_MS:
            log.warning(
                "turn.final_too_short",
                duration_ms=round(duration_ms, 1),
                threshold_ms=MIN_TURN_DURATION_MS,
                text=final_text,
            )

        # Check: Did the caller say an amount that sounds half-finished?
        # e.g., "ఒక కోటి" (1 Crore) followed by silence
        if self._looks_like_incomplete_amount(final_text):
            log.info(
                "turn.suspected_amount_split",
                text=final_text,
                waiting_ms=self.CONTINUATION_WAIT_MS,
            )
            # Wait 1.2s to see if they say "...ఇరవై లక్షలు" (20 Lakhs)
            await self._wait_for_continuation(final_text, turn_start_ms, turn_end_ms)
            return

        # Everything looks solid — signal that the turn is complete!
        await self._emit_turn_complete(final_text, turn_start_ms, turn_end_ms)

    # --------------------------------------------------------------------------
    # Helper Checks
    # --------------------------------------------------------------------------

    def _looks_like_incomplete_amount(self, text: str) -> bool:
        """Checks if the sentence ends with words like 'Crore' or 'Lakh'."""
        stripped = text.strip()
        for ending in _INCOMPLETE_AMOUNT_ENDINGS:
            if stripped.endswith(ending):
                return True
        return False

    async def _wait_for_continuation(
        self,
        current_text: str,
        turn_start_ms: float,
        turn_end_ms: float,
    ) -> None:
        """
        Pauses for 1.2 seconds.
        If the user speaks more words, on_partial cancels this timer.
        If they remain silent for the full 1.2 seconds, we accept what they said.
        """
        self._awaiting_continuation = True

        async def _do_wait() -> None:
            await asyncio.sleep(self.CONTINUATION_WAIT_MS / 1000.0)
            # Time is up and caller didn't say anything more — proceed with current text
            self._awaiting_continuation = False
            log.info(
                "turn.continuation_timeout",
                text=current_text,
                waited_ms=self.CONTINUATION_WAIT_MS,
            )
            await self._emit_turn_complete(
                current_text,
                turn_start_ms,
                time.time() * 1000,
            )

        self._continuation_task = asyncio.create_task(_do_wait())

        try:
            await self._continuation_task
        except asyncio.CancelledError:
            # Caller spoke again during the pause — timer cancelled safely!
            log.info("turn.continuation_extended", text=current_text)

    async def _emit_turn_complete(
        self,
        final_text: str,
        turn_start_ms: float,
        turn_end_ms: float,
    ) -> None:
        """
        Increases the session turn ID and notifies the main pipeline that
        it's time to process the caller's text and reply.
        """
        # Advance turn ID BEFORE triggering pipeline
        # This instantly invalidates any old audio from previous turns
        self._session.advance_turn()

        log.info(
            "turn.complete",
            turn_id=self._session.current_turn_id,
            text=final_text,
            duration_ms=round(turn_end_ms - turn_start_ms, 1),
        )

        # Clear temporary buffers for the next turn
        self._provisional_text = ""
        self._turn_start_ms = None
        self._awaiting_continuation = False

        # Call the pipeline callback (on_turn_complete in agent.py)
        await self._on_turn_complete(final_text, turn_start_ms, turn_end_ms)
