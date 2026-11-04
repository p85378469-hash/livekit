# ==============================================================================
# agent.py — Telugu Real-Estate Voice Agent (LiveKit Agents 1.x Architecture)
#
# IN SIMPLE WORDS:
# This file is the "Brain and Director" of the voice agent.
# It uses the modern LiveKit Agents 1.x AgentSession architecture:
#
# 1. Welcomes caller with a clear Telugu greeting.
# 2. Listens to caller using Soniox (Telugu Speech-To-Text).
# 3. Intercepts speech before AI generation in on_user_turn_completed():
#    - Advances session turn ID
#    - Preserves exact raw Telugu transcript
#    - Normalizes numbers (Lakhs, Crores, BHK)
#    - Updates new_message content so OpenRouter receives normalized text
#    - Injects collected real-estate facts into system instructions
#    - Distinguishes customer questions (OpenRouter LLM) vs simple qualification (StopResponse)
# 4. Speaks replies in realistic Telugu using Cartesia TTS (Sonic 3.5).
# ==============================================================================

from __future__ import annotations

import asyncio
from collections import deque
import logging
import os
import time
import uuid
from typing import Optional
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

# Internal Phase 1 modules (Normalizer, Policy, Session, Latency)
from core.session import ConversationSession
from pipeline.normalizer import Normalizer
from pipeline.policy import PolicySelector, ResponseMode
from pipeline.context import ContextBuilder, retain_latest_user_turn
from pipeline.jev_location import JevLocationClient, build_jev_location_state
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
    LiveKit 1.x Voice Agent subclass.
    Overrides `on_user_turn_completed` to run Telugu normalization, turn-tracking,
    and policy selection BEFORE OpenRouter LLM responds!
    """

    def __init__(
        self,
        session: ConversationSession,
        latency: LatencyTracker,
        normalizer: Normalizer,
        policy: PolicySelector,
        context_builder: ContextBuilder,
        session_id: str,
        agent_session: AgentSession,
        jev_location_client: Optional[JevLocationClient] = None,
    ) -> None:
        system_instructions = context_builder.build_system_prompt()
        super().__init__(instructions=system_instructions)

        self._session = session
        self._latency = latency
        self._normalizer = normalizer
        self._policy = policy
        self._context_builder = context_builder
        self._session_id = session_id
        self._agent_session = agent_session
        self._jev_location_client = jev_location_client
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

    async def on_user_turn_completed(
        self,
        turn_ctx: lk_llm.ChatContext,
        new_message: lk_llm.ChatMessage,
    ) -> None:
        """
        Called when caller finishes speaking, BEFORE LLM starts generating!
        1. Checks for duplicate message events via verified event identifier with bounded memory
        2. Advances Turn ID exactly once
        3. Preserves raw transcript without modification and normalizes numbers (Lakhs, Crores, BHK)
        4. Updates new_message with normalized text so LLM understands properly
        5. Extracts facts (Budget, Location, BHK)
        6. Injects current facts into system prompt for LLM context without swallowing errors
        7. Enforces turn-version protection against stale responses
        8. Selects response mode:
           - PYTHON_DIRECT: speaks direct answer & stops normal LLM via StopResponse()
           - LLM_GENERATE: lets OpenRouter LLM answer customer queries naturally
        """
        raw_text = new_message.text_content or ""
        if not raw_text.strip():
            return

        # Phase 1: Reliable message ID duplicate detection (allows repeated sentences in different turns)
        msg_id = getattr(new_message, "id", None)
        if not msg_id:
            # Deterministic content+time fallback replacing unstable object memory address id(new_message)
            content_str = new_message.text_content or ""
            msg_id = f"{getattr(new_message, 'created_at', 0.0)}_{hash(content_str)}"

        if msg_id in self._processed_message_ids:
            log.warning("pipeline.duplicate_message_ignored", message_id=msg_id)
            return
        self._processed_message_ids.append(msg_id)

        # Advance the session turn ID exactly once per confirmed turn
        current_turn_id = self._session.advance_turn()
        self._active_response_turn_id = current_turn_id
        active_playback_turn_id["value"] = current_turn_id
        # Ensure latency tracker is tracking current_turn_id (capturing T0/T1 already recorded on pending turn)
        if self._latency.get_turn(current_turn_id) is None:
            self._latency.new_turn(current_turn_id)
        else:
            self._latency._current_turn_id = current_turn_id

        log.info("pipeline.turn_start", turn_id=current_turn_id, text=raw_text, message_id=msg_id)

        # 1. Clean and normalize (Telugu words, numbers, lakhs, crores)
        # Preserve original raw transcript exactly as received
        raw_transcript, normalized_transcript = self._normalizer.process(raw_text)
        now_ms = time.time() * 1000
        self._session.commit_turn(
            raw_transcript=raw_transcript,
            normalized_transcript=normalized_transcript,
            timestamp_start_ms=now_ms - 1000,
            timestamp_end_ms=now_ms,
        )

        # Update new_message content with normalized Telugu text
        new_message.content = [normalized_transcript]

        # 2. Extract deterministic facts. Location is special: Jev is consulted
        # only while location is the pending qualification field.
        extraction = self._normalizer.extract(normalized_transcript)
        jev_intent = None

        if self._session.pending_field == "location":
            jev_result = None
            if self._jev_location_client is not None:
                jev_result = await self._jev_location_client.classify(
                    build_jev_location_state(raw_transcript)
                )

            # Advance only when Jev confirms a location AND the existing normalizer
            # can produce a storable canonical value. Otherwise let the main LLM
            # handle the caller's question/unclear response naturally.
            if jev_result is not None and jev_result.status == "provided" and extraction.location:
                log.info(
                    "pipeline.location_answer_confirmed",
                    location=extraction.location,
                    jev_confidence=jev_result.confidence,
                    model=jev_result.model,
                )
            else:
                extraction.location = None
                jev_intent = "question"
                log.info(
                    "pipeline.location_not_confirmed",
                    jev_status=jev_result.status if jev_result is not None else "unavailable_or_uncertain",
                    action="delegate_to_main_llm",
                )

        # Python remains the policy/state owner. Jev only gates location presence.
        decision = self._policy.select(self._session, extraction, intent_override=jev_intent)
        self._latency.record(Stage.PYTHON_COMPLETE)

        log.info(
            "pipeline.policy_decision",
            mode=decision.mode.name,
            stage=decision.next_stage,
            turn_id=current_turn_id,
        )

        if decision.next_stage:
            # Policy decisions use canonical fact keys; session.stage retains legacy display names.
            stage_name = "budget" if decision.next_stage == "budget_rupees" else decision.next_stage
            self._session.update_stage(stage_name, self._session.next_missing_field)

        # Phase 1 Turn-version check: Discard response if turn advanced/invalidated while processing
        if not self._session.is_turn_current(current_turn_id) or self._active_response_turn_id != current_turn_id:
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

        # 4. Handle customer opt-out ("not interested" / "stop calling")
        if self._session.facts.opted_out or decision.next_stage == "ended":
            opt_out_reply = decision.direct_text or "ధన్యవాదాలు. నమస్కారం."
            log.info("agent.session_ended_opt_out", session_id=self._session_id)
            self._latency.log_turn_report(turn_id=current_turn_id)
            farewell_handle = self._agent_session.say(opt_out_reply)
            asyncio.create_task(self._handle_opt_out_farewell(farewell_handle))
            raise StopResponse()

        # 5. Handle Direct Response (qualification question - no AI delay)
        if decision.mode == ResponseMode.PYTHON_DIRECT and decision.direct_text:
            log.info("pipeline.direct_response", text_preview=decision.direct_text[:60], turn_id=current_turn_id)
            self._latency.log_turn_report(turn_id=current_turn_id)
            self._agent_session.say(decision.direct_text)
            raise StopResponse()

        # 6. LLM Conversational Mode:
        # Caller asked a genuine question or discussion!
        # Update system instructions with up-to-date collected facts so LLM is fully informed.
        updated_system_prompt = self._context_builder.build_system_prompt(self._session)
        if decision.llm_prompt:
            updated_system_prompt += "\n\nTurn-specific response policy:\n" + decision.llm_prompt
        try:
            # Verified LiveKit Agent async method
            await self.update_instructions(updated_system_prompt)
            system_item_found = False
            for item in turn_ctx.items:
                if getattr(item, "role", None) == "system":
                    item.content = [updated_system_prompt]
                    system_item_found = True
                    break
            if not system_item_found:
                turn_ctx.items.insert(0, lk_llm.ChatMessage(role="system", content=[updated_system_prompt]))
        except Exception as e:
            log.error("context.instructions_update_error", error=str(e), exc_info=True)
            raise

        # Facts live in the system prompt; send no earlier chat history to the LLM.
        # Keep system/developer instructions and exactly the current caller message.
        before_items = len(turn_ctx.items)
        turn_ctx.items = retain_latest_user_turn(turn_ctx.items, new_message)
        log.info(
            "context.latest_turn_only",
            turn_id=current_turn_id,
            items_before=before_items,
            items_after=len(turn_ctx.items),
            retained_conversation_items=1,
        )
        active_playback_turn_id["value"] = current_turn_id
        log.info("pipeline.llm_response_delegated", turn_id=current_turn_id)


def validate_environment() -> None:
    """Strict Phase 1 startup validation of credentials and configuration."""
    stt_provider_choice = os.environ.get("STT_PROVIDER", "").strip().lower()
    if stt_provider_choice and stt_provider_choice not in ("soniox", "whisper"):
        raise ValueError(f"Invalid STT_PROVIDER='{stt_provider_choice}'. Allowed values are 'soniox' or 'whisper'.")

    soniox_key = os.environ.get("SONIOX_API_KEY", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    cartesia_key = os.environ.get("CARTESIA_API_KEY", "").strip()

    if not cartesia_key:
        raise ValueError("CARTESIA_API_KEY is required for voice synthesis.")

    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY is required for the LLM.")

    if stt_provider_choice == "whisper":
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is required when STT_PROVIDER=whisper is configured.")
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

    normalizer = Normalizer()
    policy = PolicySelector()
    context_builder = ContextBuilder()
    jev_location_client = JevLocationClient()
    log.info(
        "jev.location_gate_configuration",
        enabled=jev_location_client.enabled,
        model=jev_location_client.model,
        timeout_seconds=jev_location_client.timeout_seconds,
    )

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
            model="openai/whisper-large-v3",
            language="te",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
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

    # LLM: OpenRouter via LiveKit OpenAI plugin
    llm_provider = openai.LLM.with_openrouter(
        model=os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct"),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        temperature=0.3,
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
        normalizer=normalizer,
        policy=policy,
        context_builder=context_builder,
        session_id=session_id,
        agent_session=session_voice,
        jev_location_client=jev_location_client,
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
            turn_end_monotonic[pending_turn_id] = time.perf_counter()
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
            turn_id = active_playback_turn_id["value"]
            started = turn_end_monotonic.pop(turn_id, None)
            if started is not None:
                log.info(
                    "latency.overall_turn",
                    turn_id=turn_id,
                    caller_turn_end_to_first_playback_ms=round((time.perf_counter() - started) * 1000, 1),
                )
        elif ev.old_state == "speaking" and ev.new_state != "speaking":
            # Agent finished speaking its turn response
            latency.record(Stage.PLAYBACK_COMPLETE, ts_ms)
            if agent._active_response_turn_id > 0:
                latency.log_turn_report(turn_id=agent._active_response_turn_id)

    # Provider-level metrics are the authoritative per-request LLM/TTS measurements.
    # LiveKit documents metrics_collected on each plugin instance (not the deprecated
    # session-level metrics event).
    def on_llm_metrics_collected(ev):
        m = getattr(ev, "metrics", ev)
        from livekit.agents.metrics import LLMMetrics
        if not isinstance(m, LLMMetrics):
            log.debug("latency.llm_metrics_unhandled", metrics_type=type(m).__name__)
            return
        log.info(
            "latency.llm_request_metrics",
            request_id=m.request_id,
            speech_id=m.speech_id,
            metric_timestamp=m.timestamp,
            ttft_ms=round(m.ttft * 1000, 1) if m.ttft >= 0 else None,
            duration_ms=round(m.duration * 1000, 1),
            tokens_per_second=round(m.tokens_per_second, 2),
            prompt_tokens=m.prompt_tokens,
            prompt_cached_tokens=m.prompt_cached_tokens,
            completion_tokens=m.completion_tokens,
            total_tokens=m.total_tokens,
            cancelled=m.cancelled,
            label=m.label,
            model=getattr(getattr(m, "metadata", None), "model_name", None),
        )

    def on_tts_metrics_collected(ev):
        m = getattr(ev, "metrics", ev)
        from livekit.agents.metrics import TTSMetrics
        if not isinstance(m, TTSMetrics):
            log.debug("latency.tts_metrics_unhandled", metrics_type=type(m).__name__)
            return
        log.info(
            "latency.tts_request_metrics",
            request_id=m.request_id,
            speech_id=m.speech_id,
            ttfb_ms=round(m.ttfb * 1000, 1) if m.ttfb >= 0 else None,
            duration_ms=round(m.duration * 1000, 1),
            audio_duration_ms=round(m.audio_duration * 1000, 1),
            characters=m.characters_count,
            cancelled=m.cancelled,
            streamed=m.streamed,
            label=m.label,
        )

    llm_provider.on("metrics_collected", on_llm_metrics_collected)
    tts_provider.on("metrics_collected", on_tts_metrics_collected)

    # Overall caller-turn-end -> first agent playback-start latency.
    # Use monotonic time and explicit turn IDs; never subtract wall-clock timestamps.
    turn_end_monotonic: dict[int, float] = {}
    active_playback_turn_id = {"value": 0}

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
