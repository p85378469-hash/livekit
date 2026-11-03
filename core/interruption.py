# ==============================================================================
# core/interruption.py — Handles Caller Interruptions (Barge-In)
#
# IN SIMPLE WORDS:
# What happens if the voice agent is in the middle of talking, and the human caller
# starts speaking?
# - If the caller says something real (like "Wait, stop" or "No, I want Hyderabad"):
#   The agent must shut up IMMEDIATELY! It must stop talking within 200 milliseconds.
# - But if the caller just makes a natural noise (like clearing throat, coughing,
#   or saying "hmm" or "ha" while listening):
#   The agent should NOT stop talking!
#
# This file decides whether an interruption is genuine or just background noise,
# and if genuine, it executes a clean shutdown of the agent's current speech.
# ==============================================================================

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable, Awaitable, Optional
import structlog

from core.session import ConversationSession
from telemetry.latency import LatencyTracker, Stage

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------------------
# SignalType — Category of sound detected while agent is speaking
# ------------------------------------------------------------------------------
class SignalType(Enum):
    """
    Different types of sounds the microphone might pick up while the agent speaks:
    - GENUINE_SPEECH: Real words like "Wait", "No", "Listen" -> STOP SPEAKING!
    - BACKCHANNEL: Small listening noises like "ha", "aa", "hmm", "ok" -> KEEP TALKING.
    - HESITATION: Caller starting to think "umm...", "aa..." -> KEEP TALKING.
    - BACKGROUND_NOISE: TV, traffic horn, cough, clicks -> KEEP TALKING.
    - UNCLEAR: Audio too blurry to understand -> KEEP TALKING.
    - NONE: Agent is not speaking anyway.
    """
    GENUINE_SPEECH = auto()       # Real words meant to interrupt
    BACKCHANNEL = auto()          # "ha", "aa", "hmm" (caller acknowledging)
    HESITATION = auto()           # "umm", "uh" (filler sounds)
    BACKGROUND_NOISE = auto()     # Dog barking, keyboard click, cough
    UNCLEAR = auto()              # Murmuring or unclear audio
    NONE = auto()                 # Normal situation (agent was not speaking)


# ------------------------------------------------------------------------------
# InterruptionHandler — The Traffic Cop for Interruptions
# ------------------------------------------------------------------------------
class InterruptionHandler:
    """
    Watches incoming microphone sound while the agent's speakers are playing.
    If real human speech is detected, it halts playback immediately.
    """

    # Settings:
    BACKCHANNEL_MAX_WORDS = 2       # Words like "ha", "ok" are usually 1 or 2 words
    NOISE_MAX_DURATION_MS = 300     # Sounds shorter than 0.3 seconds are usually noise
    GENUINE_MIN_WORDS = 1           # At least 1 recognizable word to be considered speech

    def __init__(
        self,
        session: ConversationSession,
        latency: LatencyTracker,
        stop_playback_fn: Callable[[], Awaitable[None]],
        cancel_generation_fn: Callable[[int], Awaitable[None]],
    ) -> None:
        """
        session: The call notebook.
        latency: Latency stopwatch.
        stop_playback_fn: Function to cut the speaker audio immediately.
        cancel_generation_fn: Function to tell OpenRouter/Groq to cancel in-flight text.
        """
        self._session = session
        self._latency = latency
        self._stop_playback = stop_playback_fn
        self._cancel_generation = cancel_generation_fn

        # Counters for analytics
        self._interruption_count: int = 0
        self._false_trigger_count: int = 0

    # --------------------------------------------------------------------------
    # Main Decision: Did the user interrupt?
    # --------------------------------------------------------------------------
    async def handle_incoming_audio(
        self,
        audio_text: Optional[str],
        audio_duration_ms: float,
        agent_is_speaking: bool,
    ) -> SignalType:
        """
        Called when sound is heard while the agent is speaking.
        
        audio_text: The words heard so far (or None).
        audio_duration_ms: How long the sound lasted in milliseconds.
        agent_is_speaking: True if agent audio is currently playing.
        """
        # If the agent is silent, there is nothing to interrupt!
        if not agent_is_speaking:
            log.debug("interruption.agent_not_speaking_normal_turn")
            return SignalType.NONE

        # Classify what the sound is
        signal = self._classify(audio_text, audio_duration_ms)

        log.info(
            "interruption.signal_classified",
            signal=signal.name,
            audio_preview=audio_text[:40] if audio_text else "",
            duration_ms=round(audio_duration_ms, 1),
            agent_turn_id=self._session.current_turn_id,
        )

        if signal == SignalType.GENUINE_SPEECH:
            # Caller wants to talk! Stop the agent right now!
            await self._execute_recovery_sequence(audio_text or "")
        elif signal in (SignalType.BACKGROUND_NOISE, SignalType.BACKCHANNEL, SignalType.HESITATION):
            # Just background noise or an "ok/hmm" — do not stop speaking!
            self._false_trigger_count += 1
            log.info(
                "interruption.false_trigger_logged",
                signal=signal.name,
                total_false_triggers=self._false_trigger_count,
            )
        elif signal == SignalType.UNCLEAR:
            log.warning(
                "interruption.unclear_signal",
                audio_preview=audio_text[:40] if audio_text else "",
            )

        return signal

    # --------------------------------------------------------------------------
    # Classifier Helper: Sorts sound into SignalType
    # --------------------------------------------------------------------------
    def _classify(
        self,
        audio_text: Optional[str],
        audio_duration_ms: float,
    ) -> SignalType:
        """
        Decides if words are an interruption vs background noise vs listening sounds.
        """
        if not audio_text or audio_text.strip() == "":
            if audio_duration_ms < self.NOISE_MAX_DURATION_MS:
                return SignalType.BACKGROUND_NOISE
            return SignalType.UNCLEAR

        words = audio_text.strip().split()
        word_count = len(words)
        lower_text = audio_text.strip().lower()

        # Common Telugu and English conversational listening sounds:
        # e.g., "హా", "సరే", "hmm", "ok"
        backchannel_tokens = {
            "ha", "haa", "aa", "aaa", "hmm", "hm",
            "అవును", "సరే", "ok", "okay", "yes", "yeah",
        }
        # Hesitation sounds:
        hesitation_tokens = {
            "umm", "uh", "uhh", "err", "em",
            "అం", "ఆ", "ఉం",
        }

        # If it's 1-2 words and matches an acknowledgment, caller is just listening
        if word_count <= self.BACKCHANNEL_MAX_WORDS:
            for token in backchannel_tokens:
                if token in lower_text:
                    return SignalType.BACKCHANNEL

        # Check hesitation
        for token in hesitation_tokens:
            if lower_text.startswith(token):
                return SignalType.HESITATION

        # If there are real words, caller is talking to us
        if word_count >= self.GENUINE_MIN_WORDS and audio_text.strip():
            return SignalType.GENUINE_SPEECH

        return SignalType.UNCLEAR

    # --------------------------------------------------------------------------
    # The Recovery Sequence: Cleanly Stop Agent Playback
    # --------------------------------------------------------------------------
    async def _execute_recovery_sequence(self, incoming_text: str) -> None:
        """
        Halts the agent cleanly when interrupted:
        1. Cuts speaker audio instantly.
        2. Marks the current answer as 'expired/stale'.
        3. Tells OpenRouter/Groq to cancel any remaining words being generated.
        4. Leaves the call memory clean so the agent can listen to the new request.
        """
        self._interruption_count += 1
        interrupted_turn_id = self._session.current_turn_id
        detection_time_ms = time.time() * 1000

        log.info(
            "interruption.recovery_start",
            step=1,
            interrupted_turn_id=interrupted_turn_id,
            incoming_preview=incoming_text[:60],
        )

        # Step 1: Mute speaker playback immediately
        await self._stop_playback()
        audio_stop_ms = time.time() * 1000
        self._latency.record(Stage.INTERRUPTION_AUDIO_STOPPED, audio_stop_ms)
        log.info(
            "interruption.audio_stopped",
            detection_to_stop_ms=round(audio_stop_ms - detection_time_ms, 1),
        )

        # Step 2: Cancel any in-progress AI word generation in the background
        await self._cancel_generation(interrupted_turn_id)
        self._latency.record(Stage.INTERRUPTION_GENERATION_CANCELLED, time.time() * 1000)

        log.info(
            "interruption.recovery",
            step=6,
            action="state_protected",
            current_stage=self._session.stage,
        )

    # --------------------------------------------------------------------------
    # Filler Word Protection
    # --------------------------------------------------------------------------
    async def handle_filler_interrupted(
        self,
        filler_active: bool,
        cancel_filler_fn: Callable[[], Awaitable[None]],
    ) -> None:
        """If caller interrupts while a filler phrase is playing, stop it immediately."""
        if filler_active:
            log.info("interruption.filler_interrupted")
            await cancel_filler_fn()

    async def handle_response_ready_during_filler(
        self,
        filler_active: bool,
        cancel_filler_fn: Callable[[], Awaitable[None]],
    ) -> None:
        """If real answer is ready while filler is playing, stop filler before real answer speaks."""
        if filler_active:
            log.info("interruption.response_ready_cancel_filler")
            await cancel_filler_fn()

    # --------------------------------------------------------------------------
    # Stale Audio Guard (Prevents Ghost Audio)
    # --------------------------------------------------------------------------
    def is_chunk_current(self, chunk_turn_id: int) -> bool:
        """
        Checks if an audio chunk belongs to the CURRENT turn.
        If an audio packet arrives from 2 seconds ago, this rejects it so it doesn't play!
        """
        current = self._session.is_turn_current(chunk_turn_id)
        if not current:
            log.warning(
                "interruption.stale_chunk_discarded",
                chunk_turn_id=chunk_turn_id,
                current_turn_id=self._session.current_turn_id,
            )
        return current

    @property
    def stats(self) -> dict:
        """Returns statistics on how many interruptions were handled."""
        return {
            "total_interruptions": self._interruption_count,
            "false_triggers": self._false_trigger_count,
        }
