# ==============================================================================
# audio/greeting.py — Pre-recorded Telugu Welcome Audio Player
#
# IN SIMPLE WORDS:
# When a customer picks up the phone or joins the WebRTC room, we do NOT want them
# to wait 2 seconds for an AI to generate a greeting.
# Instead, we play a crisp, warm, pre-recorded audio file (greeting.wav) IMMEDIATELY!
#
# Crucial Rules:
# 1. Zero AI & Zero TTS for the greeting: Just raw sound file playback.
# 2. Audio Format: Must be 16kHz, 1 channel (mono), 16-bit PCM WAV (LiveKit standard).
# 3. Interruption Check: If the caller starts talking during the greeting, the sound
#    stops instantly so the agent can listen!
# ==============================================================================

from __future__ import annotations

import asyncio
import os
import struct
import wave
from pathlib import Path
from typing import AsyncIterator
import structlog

log = structlog.get_logger(__name__)

# Where the greeting audio file is saved
# Checks assets/greeting.wav first, then falls back to root greeting.wav if present
_env_path = os.environ.get("GREETING_WAV_PATH")
if _env_path:
    GREETING_WAV_PATH = Path(_env_path)
elif Path("assets/greeting.wav").exists():
    GREETING_WAV_PATH = Path("assets/greeting.wav")
else:
    GREETING_WAV_PATH = Path("greeting.wav")

# Audio frame size sent to LiveKit:
# 20ms frames at 16,000 samples per second, 16-bit mono = 640 bytes per chunk
WAV_CHUNK_BYTES = 640


class GreetingPlayer:
    """
    Validates and streams the pre-recorded Telugu welcome sound file.
    """

    def __init__(self) -> None:
        self._verified = False
        self._wav_info: dict = {}

    def verify_wav(self) -> None:
        """
        Runs when the agent first boots up.
        Checks that greeting.wav exists and has the exact required audio specs.
        """
        if not GREETING_WAV_PATH.exists():
            log.error(
                "greeting.wav_not_found",
                path=str(GREETING_WAV_PATH),
                hint="Place a 16kHz mono 16-bit PCM WAV file at this path",
            )
            raise FileNotFoundError(
                f"Greeting WAV not found: {GREETING_WAV_PATH}. "
                "Place a verified Telugu greeting WAV at this path."
            )

        with wave.open(str(GREETING_WAV_PATH), "rb") as wf:
            framerate   = wf.getframerate()
            channels    = wf.getnchannels()
            sampwidth   = wf.getsampwidth()
            n_frames    = wf.getnframes()
            duration_ms = (n_frames / framerate) * 1000

            self._wav_info = {
                "path": str(GREETING_WAV_PATH),
                "sample_rate_hz": framerate,
                "channels": channels,
                "bit_depth": sampwidth * 8,
                "duration_ms": round(duration_ms, 1),
            }

        # Check technical requirements:
        errors = []
        if framerate != 16000:
            errors.append(f"sample_rate must be 16000, got {framerate}")
        if channels != 1:
            errors.append(f"channels must be 1 (mono), got {channels}")
        if sampwidth != 2:
            errors.append(f"bit_depth must be 16-bit (sampwidth=2), got {sampwidth*8}-bit")

        if errors:
            log.error("greeting.wav_format_invalid", errors=errors, **self._wav_info)
            raise ValueError(
                f"Greeting WAV format errors: {'; '.join(errors)}. "
                "Re-export the file as 16kHz mono 16-bit PCM WAV."
            )

        self._verified = True
        log.info("greeting.wav_verified", **self._wav_info)

    async def stream_greeting(self, is_current_fn) -> AsyncIterator[bytes]:
        """
        Reads the WAV file and yields small 20-millisecond audio chunks to play.
        
        is_current_fn: Function that checks if the caller interrupted during the greeting.
        """
        if not self._verified:
            self.verify_wav()

        log.info("greeting.playback_start", wav_info=self._wav_info)

        with wave.open(str(GREETING_WAV_PATH), "rb") as wf:
            while True:
                # If caller interrupted, stop reading the file immediately!
                if not is_current_fn():
                    log.info("greeting.interrupted_by_caller")
                    return

                chunk = wf.readframes(WAV_CHUNK_BYTES // 2)
                if not chunk:
                    break  # Reached end of audio file

                yield chunk
                await asyncio.sleep(0)  # Brief yield to let other tasks run

        log.info("greeting.playback_complete")
