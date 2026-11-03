# ==============================================================================
# telemetry/latency.py — Response Time Stopwatch & Telemetry
#
# IN SIMPLE WORDS:
# For voice agents, speed is everything. If the agent takes 3 seconds to reply,
# the phone call feels awkward and robotic.
#
# This file acts as a precision "digital stopwatch" that clicks at every step:
#   T0 (TURN_END): When the caller stops speaking.
#   T1 (STT_FINAL): When Soniox finishes translating voice to text.
#   T2 (PYTHON_COMPLETE): When our code finishes extracting facts & picking a question.
#   T3 (LLM_FIRST_TOKEN): When OpenRouter / Groq sends its first response word.
#   T4 (TTS_FIRST_AUDIO): When Cartesia finishes converting that word to sound.
#   T5 (PLAYBACK_FIRST_AUDIO): When the sound actually hits the caller's ears!
#
# Total End-to-End Latency = Time from T0 to T5 (Goal: Under 1000 milliseconds / 1 second!).
# ==============================================================================

from __future__ import annotations

import time
from collections import defaultdict
from enum import Enum, auto
from typing import Optional
import structlog

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------------------
# Stage — Checkpoints along the call pipeline
# ------------------------------------------------------------------------------
class Stage(Enum):
    # Pipeline milestones
    TURN_END = auto()                   # T0: Caller stopped talking
    STT_FINAL = auto()                  # T1: Speech-to-text text confirmed
    PYTHON_COMPLETE = auto()            # T2: Fact extraction & logic decision done
    LLM_FIRST_TOKEN = auto()            # T3: First word from OpenRouter / Groq
    TTS_FIRST_AUDIO = auto()            # T4: First audio packet from Cartesia
    PLAYBACK_FIRST_AUDIO = auto()       # T5: Caller begins hearing audio in speaker
    PLAYBACK_COMPLETE = auto()          # T6: Agent finished entire response

    # Interruption milestones
    INTERRUPTION_DETECTED = auto()      # The moment caller starts talking over agent
    INTERRUPTION_AUDIO_STOPPED = auto() # The moment speaker audio cuts off
    INTERRUPTION_GENERATION_CANCELLED = auto() # The moment background AI generation aborts


# ------------------------------------------------------------------------------
# TurnLatency — Stopwatch record for one question-and-answer turn
# ------------------------------------------------------------------------------
class TurnLatency:
    """
    Keeps all timestamp recordings for a single conversation exchange.
    """

    def __init__(self, turn_id: int, session_id: str) -> None:
        self.turn_id = turn_id
        self.session_id = session_id
        self._timestamps: dict[Stage, float] = {}

    def record(self, stage: Stage, timestamp_ms: Optional[float] = None) -> None:
        """Saves current clock time (in milliseconds) for a given stage."""
        ts = timestamp_ms if timestamp_ms is not None else time.time() * 1000
        self._timestamps[stage] = ts
        log.debug(
            "latency.stage_recorded",
            stage=stage.name,
            ts_ms=round(ts, 1),
            turn_id=self.turn_id,
        )

    def get(self, stage: Stage) -> Optional[float]:
        """Returns the recorded timestamp for a specific stage."""
        return self._timestamps.get(stage)

    def interval_ms(self, from_stage: Stage, to_stage: Stage) -> Optional[float]:
        """Calculates the time elapsed (in milliseconds) between two checkpoints."""
        t0 = self._timestamps.get(from_stage)
        t1 = self._timestamps.get(to_stage)
        if t0 is None or t1 is None:
            return None
        return round(t1 - t0, 2)

    def to_report(self) -> dict:
        """
        Creates a clean summary showing how many milliseconds each part took:
        - How long Speech-to-Text took
        - How long Python logic took
        - How long the AI took to generate words
        - How long the Voice Synthesizer took
        - Total round-trip delay heard by the customer!
        """
        report: dict = {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "stages_ms": {s.name: round(ts, 1) for s, ts in self._timestamps.items()},
            "intervals_ms": {},
        }

        intervals = [
            ("turn_end__stt_final",         Stage.TURN_END,         Stage.STT_FINAL),
            ("turn_end__python_complete",   Stage.TURN_END,         Stage.PYTHON_COMPLETE),
            ("python__llm_first_token",     Stage.PYTHON_COMPLETE,  Stage.LLM_FIRST_TOKEN),
            ("llm_first__tts_first_audio",  Stage.LLM_FIRST_TOKEN,  Stage.TTS_FIRST_AUDIO),
            ("tts_first__playback_start",   Stage.TTS_FIRST_AUDIO,  Stage.PLAYBACK_FIRST_AUDIO),
            ("e2e_turn_end__first_audio",   Stage.TURN_END,         Stage.PLAYBACK_FIRST_AUDIO),
            ("intr_detect__audio_stopped",  Stage.INTERRUPTION_DETECTED, Stage.INTERRUPTION_AUDIO_STOPPED),
            ("intr_audio_stop__gen_cancel", Stage.INTERRUPTION_AUDIO_STOPPED, Stage.INTERRUPTION_GENERATION_CANCELLED),
        ]

        for key, from_s, to_s in intervals:
            val = self.interval_ms(from_s, to_s)
            if val is not None:
                report["intervals_ms"][key] = val

        return report


# ------------------------------------------------------------------------------
# LatencyTracker — Master Call Stopwatch
# ------------------------------------------------------------------------------
class LatencyTracker:
    """
    Tracks and averages performance across all turns in a phone call.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._turns: dict[int, TurnLatency] = {}
        self._current_turn_id: int = -1

    def new_turn(self, turn_id: int) -> TurnLatency:
        """Starts a fresh stopwatch record for a new turn."""
        self._current_turn_id = turn_id
        record = TurnLatency(turn_id, self._session_id)
        self._turns[turn_id] = record
        log.debug("latency.new_turn", turn_id=turn_id)
        return record

    def record(self, stage: Stage, timestamp_ms: Optional[float] = None) -> None:
        """Clicks the stopwatch at a milestone for the active turn."""
        if self._current_turn_id < 0:
            log.warning("latency.record_before_turn_start", stage=stage.name)
            return
        turn = self._turns.get(self._current_turn_id)
        if turn is None:
            turn = self.new_turn(self._current_turn_id)
        turn.record(stage, timestamp_ms)

    def get_turn(self, turn_id: int) -> Optional[TurnLatency]:
        """Returns the timing record for a specific turn."""
        return self._turns.get(turn_id)

    def log_turn_report(self, turn_id: Optional[int] = None) -> None:
        """Prints the millisecond breakdown to terminal logs."""
        tid = turn_id if turn_id is not None else self._current_turn_id
        turn = self._turns.get(tid)
        if turn:
            report = turn.to_report()
            log.info("latency.turn_report", **report)
        else:
            log.warning("latency.no_turn_record", turn_id=tid)

    def session_summary(self) -> dict:
        """Averages latency numbers across the whole call when it ends."""
        total_e2e: list[float] = []
        for turn in self._turns.values():
            e2e = turn.interval_ms(Stage.TURN_END, Stage.PLAYBACK_FIRST_AUDIO)
            if e2e is not None:
                total_e2e.append(e2e)

        avg_e2e = sum(total_e2e) / len(total_e2e) if total_e2e else None
        return {
            "total_turns": len(self._turns),
            "measured_turns": len(total_e2e),
            "avg_e2e_ms": round(avg_e2e, 1) if avg_e2e is not None else None,
            "min_e2e_ms": min(total_e2e) if total_e2e else None,
            "max_e2e_ms": max(total_e2e) if total_e2e else None,
        }
