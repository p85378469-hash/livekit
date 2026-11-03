# ==============================================================================
# core/playback.py — The Audio Player & Speaker Controller
#
# IN SIMPLE WORDS:
# This file is like the "Speaker system" of the agent.
# Its job is to take audio sound clips (either from Cartesia Text-to-Speech or
# from the greeting.wav file) and stream them to the caller's earphones/speakers.
#
# Crucial Safety Rule (Stale Chunk Guard):
# Voice audio arrives in small pieces (chunks) over the network.
# If the caller interrupts and says something new, the old answer is CANCELLED.
# Before playing each piece of audio, this player checks:
# "Does this audio belong to the current conversation turn?"
# If the audio belongs to an old turn, it throws it away immediately!
# This guarantees the agent never speaks an outdated answer!
# ==============================================================================

from __future__ import annotations

import asyncio
import time
from enum import Enum, auto
from typing import AsyncIterator, Callable, Awaitable, Optional
import structlog

from telemetry.latency import LatencyTracker, Stage

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------------------
# PlaybackState — What the speaker is doing right now
# ------------------------------------------------------------------------------
class PlaybackState(Enum):
    IDLE = auto()        # Silent, not playing anything
    PLAYING = auto()     # Currently sending voice audio to caller
    CANCELLED = auto()   # Stopped because caller interrupted


# ------------------------------------------------------------------------------
# PlaybackController — Audio Playback Manager
# ------------------------------------------------------------------------------
class PlaybackController:
    """
    Coordinates all audio played to the user.
    Enforces turn ID tags so only fresh, relevant audio can reach the caller's ears.
    """

    def __init__(
        self,
        latency: LatencyTracker,
        send_audio_chunk_fn: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """
        latency: Stopwatch tracker.
        send_audio_chunk_fn: LiveKit audio send function.
        """
        self._latency = latency
        self._send_audio_chunk = send_audio_chunk_fn

        # Current speaker state
        self._state: PlaybackState = PlaybackState.IDLE
        self._active_turn_id: int = -1
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._playback_task: Optional[asyncio.Task] = None

        # Filler words ("ఒక్క నిమిషం...") tracking (stubbed for Phase 1)
        self._filler_active: bool = False
        self._filler_cancel: asyncio.Event = asyncio.Event()

    @property
    def agent_is_speaking(self) -> bool:
        """Returns True if the agent is actively talking right now."""
        return self._state == PlaybackState.PLAYING

    @property
    def filler_active(self) -> bool:
        """Returns True if a filler phrase is playing."""
        return self._filler_active

    # --------------------------------------------------------------------------
    # Main Streaming Method
    # --------------------------------------------------------------------------
    async def play_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        turn_id: int,
    ) -> None:
        """
        Takes an incoming stream of voice audio chunks and plays them one by one.
        
        audio_stream: An async stream yielding small audio bytes.
        turn_id: The conversation turn number these bytes belong to.
        """
        # Stop any audio that might currently be playing before starting the new one
        await self.stop()

        self._active_turn_id = turn_id
        self._cancel_event.clear()
        self._state = PlaybackState.PLAYING

        first_chunk = True

        async def _stream_loop() -> None:
            nonlocal first_chunk
            try:
                async for chunk in audio_stream:
                    # Check 1: Has this turn expired? (e.g. caller interrupted)
                    if self._is_stale(turn_id):
                        log.warning(
                            "playback.stale_stream_aborted",
                            chunk_turn_id=turn_id,
                            active_turn_id=self._active_turn_id,
                        )
                        return

                    # Check 2: Was stop() called?
                    if self._cancel_event.is_set():
                        log.info("playback.cancelled_mid_stream", turn_id=turn_id)
                        return

                    # Record T6: The exact millisecond the caller starts hearing sound!
                    if first_chunk:
                        self._latency.record(Stage.PLAYBACK_FIRST_AUDIO, time.time() * 1000)
                        log.info("playback.first_audio_chunk", turn_id=turn_id)
                        first_chunk = False

                    # Send this audio chunk to the user's headphones
                    await self._send_audio_chunk(chunk)

                log.info("playback.stream_complete", turn_id=turn_id)

            except asyncio.CancelledError:
                log.info("playback.task_cancelled", turn_id=turn_id)
            finally:
                if self._state == PlaybackState.PLAYING:
                    self._state = PlaybackState.IDLE

        self._playback_task = asyncio.create_task(_stream_loop())
        try:
            await self._playback_task
        except asyncio.CancelledError:
            pass

    # --------------------------------------------------------------------------
    # Stopping and Cancelling Audio
    # --------------------------------------------------------------------------
    async def stop(self) -> None:
        """
        Immediately cuts off audio playback without waiting.
        Called the moment the user interrupts.
        """
        if self._state == PlaybackState.PLAYING:
            self._cancel_event.set()
            self._state = PlaybackState.CANCELLED

            if self._playback_task and not self._playback_task.done():
                self._playback_task.cancel()
                try:
                    await self._playback_task
                except (asyncio.CancelledError, Exception):
                    pass

            log.info("playback.stopped", was_playing_turn=self._active_turn_id)

        self._state = PlaybackState.IDLE

    async def cancel_generation_for_turn(self, stale_turn_id: int) -> None:
        """Marks any pending audio chunks for this turn as stale."""
        if self._active_turn_id == stale_turn_id:
            await self.stop()
        log.info("playback.generation_cancelled_for_turn", stale_turn_id=stale_turn_id)

    # --------------------------------------------------------------------------
    # Filler Words (Optional short phrases while thinking)
    # --------------------------------------------------------------------------
    async def play_filler(
        self,
        filler_audio_fn: Callable[[], Awaitable[bytes]],
        turn_id: int,
    ) -> None:
        """Plays a brief filler audio phrase (kept off by default in Phase 1)."""
        log.info(
            "playback.filler_requested_but_deferred",
            turn_id=turn_id,
            note="Fillers are OFF in Phase 1 per master plan",
        )

    async def cancel_filler(self) -> None:
        """Instantly stops any active filler phrase."""
        self._filler_cancel.set()
        self._filler_active = False
        log.info("playback.filler_cancelled")

    # --------------------------------------------------------------------------
    # Welcome Greeting Player
    # --------------------------------------------------------------------------
    async def play_greeting(
        self,
        wav_chunks: AsyncIterator[bytes],
        turn_id: int,
    ) -> None:
        """
        Plays the pre-recorded Telugu welcome audio (greeting.wav).
        Uses NO AI and NO TTS — 100% instant sound playback!
        """
        log.info("playback.greeting_start", turn_id=turn_id)
        await self.play_stream(wav_chunks, turn_id)
        log.info("playback.greeting_end", turn_id=turn_id)

    # --------------------------------------------------------------------------
    # Safety Check: Is Audio Chunk Stale?
    # --------------------------------------------------------------------------
    def _is_stale(self, chunk_turn_id: int) -> bool:
        """Returns True if this piece of sound belongs to an older, dead turn."""
        if self._state in (PlaybackState.IDLE, PlaybackState.CANCELLED):
            return True
        return chunk_turn_id != self._active_turn_id

    @property
    def stats(self) -> dict:
        """Returns a snapshot of player status for logging."""
        return {
            "state": self._state.name,
            "active_turn_id": self._active_turn_id,
            "filler_active": self._filler_active,
        }
