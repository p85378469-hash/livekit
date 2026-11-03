# ==============================================================================
# providers/tts.py — Cartesia Telugu Text-to-Speech (TTS) Voice Synthesizer
#
# IN SIMPLE WORDS:
# This file is the "Voice / Vocal Cords" of the voice agent.
# It takes written Telugu text and converts it into realistic spoken human audio!
#
# Why Cartesia?
# Cartesia's "sonic-multilingual" model produces ultra-realistic Telugu pronunciation
# with world-record low latency (starts speaking in ~100-200 milliseconds!).
#
# Key Features:
# 1. Concurrent Streaming: As soon as the first sentence finishes generating from
#    the AI, this file turns it into sound without waiting for the full reply!
# 2. Interruption Safety: If the user interrupts, it instantly stops generating
#    further sound bytes.
# ==============================================================================

from __future__ import annotations

import os
import time
from typing import AsyncIterator
import structlog

from livekit.plugins import cartesia

from telemetry.latency import LatencyTracker, Stage

log = structlog.get_logger(__name__)

# Cartesia voice ID: The unique voice avatar code (configured in .env)
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "YOUR_CARTESIA_VOICE_ID_HERE")
CARTESIA_MODEL    = "sonic-3.5"            # Cartesia Sonic 3.5 with Telugu ('te') support


class CartesiaTTS:
    """
    Wraps Cartesia streaming Text-to-Speech for Telugu.
    """

    def __init__(self, latency: LatencyTracker) -> None:
        """
        Reads CARTESIA_API_KEY from environment and configures the voice model.
        """
        self._latency = latency

        api_key = os.environ.get("CARTESIA_API_KEY")
        if not api_key:
            raise EnvironmentError("CARTESIA_API_KEY not set in environment")

        # Configure Cartesia with Telugu language and 16kHz standard audio
        self._tts = cartesia.TTS(
            api_key=api_key,
            model=CARTESIA_MODEL,
            voice=CARTESIA_VOICE_ID,
            language="te",       # Telugu language code
            encoding="pcm_s16le",
            sample_rate=16000,   # 16000 samples/sec (CD quality telephone voice)
        )

        log.info(
            "tts.initialized",
            provider="cartesia",
            model=CARTESIA_MODEL,
            voice_id=CARTESIA_VOICE_ID,
        )

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        turn_id: int,
        is_turn_current_fn,
    ) -> AsyncIterator[bytes]:
        """
        Converts streaming text sentences into streaming voice sound bytes.
        
        text_stream: Sentences coming from OpenRouter / Groq.
        turn_id: Current turn number.
        is_turn_current_fn: Checks if the caller interrupted.
        """
        first_audio_recorded = False
        log.info("tts.synthesis_stream_start", turn_id=turn_id)

        async for text_chunk in text_stream:
            # Check: Did the caller interrupt while we were preparing audio?
            if not is_turn_current_fn():
                log.warning(
                    "tts.stale_turn_aborted",
                    turn_id=turn_id,
                    text_chunk_preview=text_chunk[:40],
                )
                return

            if not text_chunk.strip():
                continue

            log.debug(
                "tts.synthesizing_chunk",
                chunk_preview=text_chunk[:40],
                turn_id=turn_id,
            )

            try:
                # Ask Cartesia to synthesize this sentence into audio
                async for synthesis_event in self._tts.synthesize(text_chunk):
                    # Check again inside the audio chunk loop
                    if not is_turn_current_fn():
                        log.warning("tts.stale_inner_loop", turn_id=turn_id)
                        return

                    audio_bytes = synthesis_event.frame.data

                    # Record T4: The exact millisecond the first sound byte was ready
                    if not first_audio_recorded:
                        self._latency.record(Stage.TTS_FIRST_AUDIO, time.time() * 1000)
                        first_audio_recorded = True
                        log.info("tts.first_audio_chunk_ready", turn_id=turn_id)

                    yield audio_bytes

            except Exception as e:
                log.error(
                    "tts.synthesis_error",
                    error=str(e),
                    chunk=text_chunk[:40],
                    turn_id=turn_id,
                )
                raise

        log.info("tts.synthesis_stream_complete", turn_id=turn_id)

    async def synthesize_direct(
        self,
        text: str,
        turn_id: int,
    ) -> AsyncIterator[bytes]:
        """
        Converts a single fixed text string directly into audio bytes.
        Used for fixed template questions and confirmations.
        """
        log.info(
            "tts.direct_synthesis",
            text_preview=text[:60],
            turn_id=turn_id,
        )
        first_recorded = False

        try:
            async for synthesis_event in self._tts.synthesize(text):
                audio_bytes = synthesis_event.frame.data
                if not first_recorded:
                    self._latency.record(Stage.TTS_FIRST_AUDIO, time.time() * 1000)
                    first_recorded = True
                yield audio_bytes
        except Exception as e:
            log.error("tts.direct_synthesis_error", error=str(e), text=text[:60])
            raise
