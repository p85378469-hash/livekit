# ==============================================================================
# providers/stt.py — Soniox Telugu Speech-to-Text (STT) Service
#
# IN SIMPLE WORDS:
# This file is the "Ears" of the voice agent.
# It takes the live sound from the caller's microphone and turns it into written text.
#
# Why Soniox?
# Soniox has world-class accuracy for Indian languages, especially Telugu mixed with
# English (Tanglish), like: "Hyderabad lo 3BHK flat budget 1 crore".
#
# How it works:
# 1. While the user speaks: It emits INTERIM_TRANSCRIPT (partial words).
# 2. When the user pauses: It emits FINAL_TRANSCRIPT (complete confirmed sentence).
# ==============================================================================

from __future__ import annotations

import os
import time
from typing import Callable, Awaitable
import structlog

from livekit.agents import stt as lk_stt
import livekit.agents as _agents

# Compatibility shims for livekit plugins with livekit-agents 0.12.x
if not hasattr(_agents, "LanguageCode"):
    _agents.LanguageCode = str

if hasattr(lk_stt, "STTCapabilities"):
    _orig_stt_cap_init = lk_stt.STTCapabilities.__init__
    def _patched_stt_cap_init(self, streaming: bool = False, interim_results: bool = False, *args, **kwargs):
        return _orig_stt_cap_init(self, streaming=streaming, interim_results=interim_results)
    lk_stt.STTCapabilities.__init__ = _patched_stt_cap_init

from livekit.plugins import soniox

from telemetry.latency import LatencyTracker, Stage

log = structlog.get_logger(__name__)


class SonioxSTT:
    """
    Connects Soniox Telugu speech recognition to LiveKit.
    """

    def __init__(
        self,
        latency: LatencyTracker,
        on_partial: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
    ) -> None:
        """
        latency: Stopwatch to record when speech-to-text finishes.
        on_partial: Callback for in-progress words.
        on_final: Callback for completed sentences.
        """
        self._latency = latency
        self._on_partial = on_partial
        self._on_final = on_final

        # Retrieve the secret Soniox key from .env
        api_key = os.environ.get("SONIOX_API_KEY")
        if not api_key:
            raise EnvironmentError("SONIOX_API_KEY not set in environment")

        base_url = os.environ.get("SONIOX_BASE_URL", "wss://stt-rt.in.soniox.com/transcribe-websocket")
        # Initialize Soniox with Telugu ('te') and English ('en') language hints and endpointing
        self._stt = soniox.STT(
            api_key=api_key,
            base_url=base_url,
            params=soniox.stt.STTOptions(
                model="stt-rt-v5",
                language_hints=["te", "en"],  # Telugu primary, English secondary
                max_endpoint_delay_ms=2000,   # Soniox server-side endpointing window (500-3000ms)
            ),
        )

        log.info(
            "stt.initialized",
            provider="soniox",
            model="soniox-telephony",
            languages=["te", "en"],
        )

    def get_livekit_stt(self) -> soniox.STT:
        """Returns the configured Soniox STT plugin object for LiveKit."""
        return self._stt

    async def handle_stt_event(self, event: lk_stt.SpeechEvent) -> None:
        """
        Routes incoming speech events:
        - INTERIM_TRANSCRIPT: Words being spoken mid-sentence -> send to on_partial
        - FINAL_TRANSCRIPT: Complete sentence recognized -> record T1 time & send to on_final
        """
        event_type = event.type
        text = event.alternatives[0].text if event.alternatives else ""

        if event_type == lk_stt.SpeechEventType.INTERIM_TRANSCRIPT:
            # User is in the middle of talking
            log.debug("stt.partial", text_preview=text[:60])
            await self._on_partial(text)

        elif event_type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
            # User finished talking! Record T1 timestamp (millisecond STT finished)
            self._latency.record(Stage.STT_FINAL, time.time() * 1000)
            log.info("stt.final", text=text)
            await self._on_final(text)

        elif event_type == lk_stt.SpeechEventType.START_OF_SPEECH:
            log.debug("stt.speech_started")

        elif event_type == lk_stt.SpeechEventType.END_OF_SPEECH:
            log.debug("stt.speech_ended")
