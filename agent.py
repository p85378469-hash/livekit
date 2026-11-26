# ==============================================================================
# agent.py — Telugu Real-Estate Voice Agent (LiveKit Agents 1.x Architecture)
#
# IN SIMPLE WORDS:
# This file is the "Brain and Director" of the voice agent.
# It uses the modern LiveKit Agents 1.x AgentSession architecture:
#
# 1. Welcomes caller with a clear Telugu greeting.
# 2. Listens to caller using Soniox (Telugu Speech-To-Text).
# 3. Preserves each finalized transcript and lets ordinary turns go directly
#    to the conversational LLM. Only explicit opt-out requests use a deterministic
#    compliance response.
# 4. Speaks replies in Telugu using Cartesia TTS (Sonic 3.5).
# ==============================================================================

from __future__ import annotations

from typing import Any

import asyncio
from collections import deque
import logging
import os
import time
import uuid
from dotenv import load_dotenv
import structlog

# Read secret API keys from the .env file
load_dotenv()

import livekit.agents as agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    StopResponse,
    WorkerOptions,
    cli,
    llm as lk_llm,
)
from livekit.plugins import cartesia, openai, silero, soniox  # soniox kept for later

# Internal modules (turn tracking, stable prompt, latency)
from core.session import ConversationSession
from pipeline.context import ContextBuilder
from telemetry.latency import LatencyTracker, Stage

# Configure structured logging for clean terminal output
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

log = structlog.get_logger(__name__)

# Pre-recorded / Default welcome greeting in Telugu
GREETING_TEXT = "నమస్కారం అండి! నేను మీ రియల్ ఎస్టేట్ అసిస్టెంట్ ని. మీకు ఎలాంటి ప్రాపర్టీ కావాలి?"


class TeluguRealEstateAgent(Agent):
    """
    LiveKit 1.x voice agent. Ordinary finalized user turns are passed directly
    to the configured LLM without Python extraction or conversational routing.
    """

    def __init__(
        self,
        session: ConversationSession,
        latency: LatencyTracker,
        context_builder: ContextBuilder,
        session_id: str,
        agent_session: AgentSession,
    ) -> None:
        system_instructions = context_builder.build_system_prompt()
        super().__init__(instructions=system_instructions)

        self._session = session
        self._latency = latency
        self._context_builder = context_builder
        self._session_id = session_id
        self._agent_session = agent_session
        self._processed_message_ids: deque[str] = deque(maxlen=128)
        self._active_response_turn_id: int = 0

    def invalidate_active_turn(self) -> None:
        """Invalidates in-flight response generation on genuine caller interruption."""
        log.info("pipeline.turn_invalidated", previous_active_turn=self._active_response_turn_id)
        self._active_response_turn_id = -1

    async def _handle_opt_out_farewell(self, farewell_handle: Any) -> None:
        """Waits for farewell speech playout to finish completely before disconnecting."""
        try:
            if hasattr(farewell_handle, "wait_for_playout"):
                await farewell_handle.wait_for_playout()
            elif hasattr(farewell_handle, "__await__"):
                await farewell_handle
        except Exception as e:
            log.warning("agent.opt_out_playout_error", error=str(e))
        try:
            room = getattr(self._agent_session, "room", None)
            if room:
                await room.disconnect()
        except Exception as e:
            log.warning("agent.opt_out_disconnect_error", error=str(e))

    @staticmethod
    def _is_explicit_opt_out(text: str) -> bool:
        """Minimal deterministic safeguard for clear stop/opt-out requests."""
        lower = text.lower()
        telugu_phrases = ("కాల్ చేయవద్దు", "మళ్ళీ కాల్ చేయకండి", "ఇకపై కాల్ చేయవద్దు", "నాకు ఆసక్తి లేదు")
        english_phrases = ("stop calling", "do not call", "don't call", "unsubscribe", "remove me", "not interested")
        return any(token in lower for token in telugu_phrases + english_phrases)

    async def on_user_turn_completed(
        self,
        turn_ctx: lk_llm.ChatContext,
        new_message: lk_llm.ChatMessage,
    ) -> None:
        """
        Preserve the finalized transcript and let LiveKit route every normal
        conversational turn directly to the configured LLM.
        No input normalization, fact extraction, qualification routing, or
        dynamic fact-context injection occurs here.
        """
        raw_text = new_message.text_content or ""
        if not raw_text.strip():
            return

        # Ignore duplicate delivery of the same finalized transcript event.
        msg_id = getattr(new_message, "id", None)
        if not msg_id:
            msg_id = f"{getattr(new_message, 'created_at', 0.0)}_{hash(raw_text)}"

        if msg_id in self._processed_message_ids:
            log.warning("pipeline.duplicate_message_ignored", message_id=msg_id)
            return
        self._processed_message_ids.append(msg_id)

        current_turn_id = self._session.advance_turn()
        self._active_response_turn_id = current_turn_id
        if self._latency.get_turn(current_turn_id) is None:
            self._latency.new_turn(current_turn_id)
        else:
            self._latency._current_turn_id = current_turn_id

        now_ms = time.time() * 1000
        self._session.commit_turn(
            raw_transcript=raw_text,
            normalized_transcript=raw_text,
            timestamp_start_ms=now_ms - 1000,
            timestamp_end_ms=now_ms,
        )

        log.info(
            "pipeline.turn_start",
            turn_id=current_turn_id,
            text=raw_text,
            message_id=msg_id,
        )

        # Explicit opt-out remains a deterministic safety/compliance exception.
        # All ordinary conversation is delegated directly to the LLM.
        if self._is_explicit_opt_out(raw_text):
            farewell = "అర్థమైంది. మళ్ళీ మీకు కాల్ చేయము. ధన్యవాదాలు. నమస్కారం."
            log.info("agent.session_ended_opt_out", session_id=self._session_id)
            self._latency.log_turn_report(turn_id=current_turn_id)
            farewell_handle = self._agent_session.say(farewell)
            asyncio.create_task(self._handle_opt_out_farewell(farewell_handle))
            raise StopResponse()

        if (
            not self._session.is_turn_current(current_turn_id)
            or self._active_response_turn_id != current_turn_id
        ):
            log.warning(
                "pipeline.stale_turn_discarded",
                turn_id=current_turn_id,
                active_turn_id=self._session.current_turn_id,
                response_turn_id=self._active_response_turn_id,
            )
            try:
                self._agent_session.interrupt(force=True)
            except Exception:
                pass
            raise StopResponse()

        log.info(
            "pipeline.llm_response_delegated",
            turn_id=current_turn_id,
            route="direct",
            transcript_chars=len(raw_text),
        )


def validate_environment() -> None:
    """Strict Phase 1 startup validation of credentials and configuration."""
    stt_provider_choice = os.environ.get("STT_PROVIDER", "").strip().lower()
    if stt_provider_choice and stt_provider_choice not in ("soniox", "whisper"):
        raise ValueError(f"Invalid STT_PROVIDER='{stt_provider_choice}'. Allowed values are 'soniox' or 'whisper'.")

    soniox_key = os.environ.get("SONIOX_API_KEY", "").strip()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    cartesia_key = os.environ.get("CARTESIA_API_KEY", "").strip()

    if not cartesia_key:
        raise ValueError("CARTESIA_API_KEY is required for voice synthesis.")

    if not groq_key:
        raise ValueError("GROQ_API_KEY is required for the LLM.")

    if stt_provider_choice == "whisper":
        if not groq_key:
            raise ValueError("GROQ_API_KEY is required when STT_PROVIDER=whisper is configured.")
    else:
        # Default or explicit 'soniox' requires SONIOX_API_KEY (no silent fallback to Whisper)
        if not soniox_key:
            raise ValueError(
                "SONIOX_API_KEY is required for Soniox STT. "
                "Configure SONIOX_API_KEY, or explicitly set STT_PROVIDER=whisper to use Whisper fallback."
            )


async def entrypoint(ctx: JobContext) -> None:
    """
    Main job entrypoint triggered when a call arrives.
    """
    validate_environment()
    session_id = str(uuid.uuid4())[:8]
    log.info("agent.session_start", session_id=session_id)

    # 1. Connect to the LiveKit Room
    await ctx.connect()

    # 2. Initialize tracking & pipeline state
    session = ConversationSession(session_id=session_id)
    latency = LatencyTracker(session_id=session_id)
    latency.new_turn(turn_id=0)

    context_builder = ContextBuilder()

    # 3. Initialize Provider Plugins (LiveKit 1.x)
    # STT Provider Selection (Soniox as primary Telugu engine; Whisper when explicitly configured)
    stt_provider_choice = os.environ.get("STT_PROVIDER", "").strip().lower()
    soniox_key = os.environ.get("SONIOX_API_KEY", "").strip()

    if stt_provider_choice == "whisper":
        log.info(
            "stt.provider_selected",
            provider="openai_whisper",
            model="openai/whisper-large-v3",
            reason="STT_PROVIDER=whisper explicitly requested",
        )
        stt_provider = openai.STT(
            model="whisper-large-v3",
            language="te",
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    elif soniox_key:
        soniox_base_url = os.environ.get("SONIOX_BASE_URL", "wss://stt-rt.in.soniox.com/transcribe-websocket")
        log.info("stt.provider_selected", provider="soniox", model="stt-rt-v5", base_url=soniox_base_url)
        stt_provider = soniox.STT(
            api_key=soniox_key,
            base_url=soniox_base_url,
            params=soniox.stt.STTOptions(
                model="stt-rt-v5",
                language_hints=["te", "en"],
                max_endpoint_delay_ms=2000,
            ),
        )
    else:
        raise ValueError("No valid STT provider configured. Set SONIOX_API_KEY or explicit STT_PROVIDER=whisper.")

    # LLM: Groq GPT-OSS 20B via the pinned LiveKit OpenAI-compatible plugin.
    # Groq prompt caching is automatic; keep the stable system prompt unchanged.
    llm_provider = openai.LLM.with_groq(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=float(os.environ.get("GROQ_TEMPERATURE", "0.6")),
        max_tokens=int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "256")),
    )

    # TTS: Cartesia Sonic 3.5 Telugu voice
    tts_provider = cartesia.TTS(
        api_key=os.environ.get("CARTESIA_API_KEY"),
        model="sonic-3.5",
        voice=os.environ.get("CARTESIA_VOICE_ID", "26bfa541-7cb2-49b2-a2a2-c07cfec33e9d"),
        language="te",
        word_timestamps=False,
    )

    # VAD: Silero Voice Activity Detector
    vad_provider = silero.VAD.load()

    # 4. Create LiveKit 1.x AgentSession with Phase 1 endpointing configuration
    session_voice = AgentSession(
        stt=stt_provider,
        llm=llm_provider,
        tts=tts_provider,
        vad=vad_provider,
        allow_interruptions=True,
        min_interruption_duration=0.5,
        min_interruption_words=1,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )

    # 5. Create Telugu Agent instance
    agent = TeluguRealEstateAgent(
        session=session,
        latency=latency,
        context_builder=context_builder,
        session_id=session_id,
        agent_session=session_voice,
    )

    # 4b. Wire Phase 1 documented LiveKit session events (Transcript, Latency, & State)
    @session_voice.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        """
        Explicit Phase 1 Transcript Event Handling:
        - Associates transcripts with the pending turn lifecycle (session.current_turn_id + 1).
        - Logs partial transcripts (provisional, never commits facts or replies).
        - Logs final transcript segments and records STT_FINAL latency on the pending turn.
        """
        pending_turn_id = session.current_turn_id + 1
        if latency.get_turn(pending_turn_id) is None:
            latency.new_turn(pending_turn_id)

        if not ev.is_final:
            log.info("transcript.partial", text=ev.transcript, language=str(ev.language), turn_id=pending_turn_id)
        else:
            log.info("transcript.final_segment", text=ev.transcript, language=str(ev.language), turn_id=pending_turn_id)
            ts_ms = (ev.created_at * 1000) if hasattr(ev, "created_at") and ev.created_at else time.time() * 1000
            latency.record(Stage.STT_FINAL, ts_ms)

    @session_voice.on("user_state_changed")
    def on_user_state_changed(ev):
        """Detect when caller stops speaking (T0) or barges in (interruption)."""
        log.info("agent.user_state_changed", old_state=ev.old_state, new_state=ev.new_state)
        ts_ms = (ev.created_at * 1000) if hasattr(ev, "created_at") and ev.created_at else time.time() * 1000
        pending_turn_id = session.current_turn_id + 1

        if ev.old_state == "speaking" and ev.new_state != "speaking":
            if latency.get_turn(pending_turn_id) is None:
                latency.new_turn(pending_turn_id)
            latency.record(Stage.TURN_END, ts_ms)
        elif ev.new_state == "speaking" and session_voice.agent_state in ("speaking", "thinking"):
            log.info("interruption.detected", turn_id=session.current_turn_id, agent_state=session_voice.agent_state)
            if latency.get_turn(pending_turn_id) is None:
                latency.new_turn(pending_turn_id)
            latency.record(Stage.INTERRUPTION_DETECTED, ts_ms)
            agent.invalidate_active_turn()
            # Explicitly halt in-flight playback / generation and cancel queued speech
            try:
                session_voice.interrupt(force=True)
            except Exception as e:
                log.warning("interruption.interrupt_call_failed", error=str(e))

    @session_voice.on("agent_state_changed")
    def on_agent_state_changed(ev):
        """
        Detect when agent begins speaking audio to caller (T5) or stops speaking.
        Note: PLAYBACK_FIRST_AUDIO marks the SDK playback transition event.
        """
        log.info("agent.agent_state_changed", old_state=ev.old_state, new_state=ev.new_state)
        ts_ms = (ev.created_at * 1000) if hasattr(ev, "created_at") and ev.created_at else time.time() * 1000
        if ev.new_state == "speaking":
            latency.record(Stage.PLAYBACK_FIRST_AUDIO, ts_ms)
        elif ev.old_state == "speaking" and ev.new_state != "speaking":
            # Agent finished speaking its turn response
            latency.record(Stage.PLAYBACK_COMPLETE, ts_ms)
            if agent._active_response_turn_id > 0:
                latency.log_turn_report(turn_id=agent._active_response_turn_id)

    @session_voice.on("metrics_collected")
    def on_metrics_collected(ev):
        """Capture LLM and TTS latency milestones (T3 and T4) using true event timestamps."""
        m = getattr(ev, "metrics", ev)
        from livekit.agents.metrics import LLMMetrics, TTSMetrics
        if isinstance(m, LLMMetrics):
            first_token_ts_ms = (m.timestamp + m.ttft) * 1000 if m.ttft > 0 else m.timestamp * 1000
            latency.record(Stage.LLM_FIRST_TOKEN, first_token_ts_ms)
            log.info(
                "latency.llm_metrics",
                ttft_ms=round(m.ttft * 1000, 1) if m.ttft > 0 else None,
                duration_ms=round(m.duration * 1000, 1) if m.duration else None,
                total_tokens=m.total_tokens,
            )
        elif isinstance(m, TTSMetrics):
            first_audio_ts_ms = (m.timestamp + m.ttfb) * 1000 if m.ttfb > 0 else m.timestamp * 1000
            latency.record(Stage.TTS_FIRST_AUDIO, first_audio_ts_ms)
            log.info(
                "latency.tts_metrics",
                ttfb_ms=round(m.ttfb * 1000, 1) if m.ttfb > 0 else None,
                duration_ms=round(m.duration * 1000, 1) if m.duration else None,
                characters=m.characters_count,
            )

    # 4c. Setup disconnect and close event handlers BEFORE greeting to guard against early disconnect
    stop_event = asyncio.Event()

    @session_voice.on("close")
    def on_session_close(ev=None):
        log.info("agent.session_close_event", ev=str(ev))
        stop_event.set()

    @ctx.room.on("disconnected")
    def on_disconnected():
        log.info("agent.room_disconnected")
        stop_event.set()

    # 6. Start the session in the LiveKit room
    await session_voice.start(agent, room=ctx.room)
    log.info("agent.pipeline_started", session_id=session_id)

    # 7. Greet the user in Telugu and await playout completion (with disconnect guard)
    log.info("agent.greeting_start")
    greeting_handle = session_voice.say(GREETING_TEXT)
    # SpeechHandle.wait_for_playout() provides a clean coroutine for asyncio.create_task
    if hasattr(greeting_handle, "wait_for_playout"):
        playout_coro = greeting_handle.wait_for_playout()
    else:
        async def _await_greeting():
            await greeting_handle
        playout_coro = _await_greeting()

    playout_task = asyncio.create_task(playout_coro)
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait([playout_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    log.info("agent.greeting_complete")

    # 8. Keep session active until caller disconnects or session closes
    await stop_event.wait()

    summary = latency.session_summary()
    log.info("agent.session_ended", session_id=session_id, latency_summary=summary)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint)
    )
