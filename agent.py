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
#    - Updates new_message content so Groq receives normalized text
#    - Injects collected real-estate facts into system instructions
#    - Distinguishes customer questions (Groq LLM) vs simple qualification (StopResponse)
# 4. Speaks replies in realistic Telugu using Cartesia TTS (Sonic 3.5).
# ==============================================================================

from __future__ import annotations

import asyncio
from collections import deque
import logging
import os
import sys
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
    PreemptiveGenerationOptions,
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
from pipeline.context import ContextBuilder
from telemetry.latency import LatencyTracker, Stage
from telemetry.call_logger import CallLogger

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
_DYNAMIC_CONTEXT_MARKER = "Current validated lead context:"


class TeluguRealEstateAgent(Agent):
    """
    LiveKit 1.x Voice Agent subclass.
    Overrides `on_user_turn_completed` to run Telugu normalization, turn-tracking,
    and policy selection BEFORE Groq LLM responds!
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
        call_logger: Optional[CallLogger] = None,
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
        self._call_logger = call_logger
        self._processed_message_ids: deque[str] = deque(maxlen=128)
        self._active_response_turn_id: int = 0

    def invalidate_active_turn(self) -> None:
        """Invalidates in-flight response generation on genuine caller interruption."""
        log.info("pipeline.turn_invalidated", previous_active_turn=self._active_response_turn_id)
        self._active_response_turn_id = -1

    async def sync_rolling_chat_ctx(self, max_exchanges: int = 2) -> None:
        """
        Keeps `self.chat_ctx` trimmed to:
          1. Stable SYSTEM_PROMPT (index 0, for Groq prefix caching)
          2. Only the last `max_exchanges` (user+assistant) turns
          3. Latest `dynamic_context` system message at the end

        Updating `self.chat_ctx` between turns ensures LiveKit's `preemptive_generation`
        runs with up-to-date facts and is NOT invalidated inside `on_user_turn_completed`.
        """
        try:
            current_items = list(self.chat_ctx.items)
        except Exception:
            return

        dialogue_items = [
            item
            for item in current_items
            if getattr(item, "role", None) in ("user", "assistant")
        ]
        # Keep only the last max_exchanges (user + assistant pairs = 2 * max_exchanges)
        trimmed_dialogue = dialogue_items[-(max_exchanges * 2) :] if max_exchanges > 0 else []

        # Preserve LiveKit's single stable instructions system message at index 0 if present,
        # otherwise insert SYSTEM_PROMPT once.
        stable_sys = next(
            (
                item
                for item in current_items
                if getattr(item, "role", None) == "system"
                and _DYNAMIC_CONTEXT_MARKER not in (getattr(item, "text_content", None) or "")
            ),
            None,
        )
        if stable_sys is None:
            stable_sys = lk_llm.ChatMessage(
                role="system", content=[self._context_builder.build_system_prompt()]
            )

        new_ctx = lk_llm.ChatContext()
        new_ctx.items.append(stable_sys)
        new_ctx.items.extend(trimmed_dialogue)

        # Ensure next_missing_field is populated before Turn 1 (starts as None on greeting)
        if self._session.next_missing_field is None and self._session.stage not in ("handoff", "ended"):
            self._session.next_missing_field = self._policy._next_missing_field(self._session.facts)

        dynamic_context = self._context_builder.build_dynamic_context(self._session)
        if dynamic_context:
            new_ctx.items.append(lk_llm.ChatMessage(role="system", content=[dynamic_context]))

        await self.update_chat_ctx(new_ctx)
        log.info(
            "pipeline.rolling_ctx_synced",
            total_items=len(new_ctx.items),
            dialogue_msgs=len(trimmed_dialogue),
            stage=self._session.stage,
            next_missing=self._session.next_missing_field,
        )

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
           - LLM_GENERATE: lets Groq LLM answer customer queries naturally
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

        # Preserve original new_message.content when preemptive generation is enabled,
        # because mutating new_message.content alters raw_text_content and invalidates
        # LiveKit's preemptive LLM generation.
        if not os.environ.get("PREEMPTIVE_GENERATION", "true").lower() in ("true", "1", "yes"):
            new_message.content = [normalized_transcript]

        # 2. Extract facts (Location, Budget, BHK)
        extraction = self._normalizer.extract(normalized_transcript)

        # 3. Policy selector: Decide whether AI is needed or fixed answer
        decision = self._policy.select(self._session, extraction)
        self._latency.record(Stage.PYTHON_COMPLETE)

        log.info(
            "pipeline.policy_decision",
            mode=decision.mode.name,
            stage=decision.next_stage,
            turn_id=current_turn_id,
        )

        next_missing = self._policy._next_missing_field(self._session.facts)
        if decision.next_stage:
            self._session.update_stage(decision.next_stage, next_missing)

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
            if self._call_logger:
                self._call_logger.set_status("OPTED_OUT")
                self._call_logger.record_turn_response(current_turn_id, opt_out_reply, "PYTHON_DIRECT")
            self._latency.log_turn_report(turn_id=current_turn_id)
            farewell_handle = self._agent_session.say(opt_out_reply)
            asyncio.create_task(self._handle_opt_out_farewell(farewell_handle))
            raise StopResponse()

        # 5. Handle Direct Response (qualification question)
        # When ALWAYS_USE_LLM is enabled (default in .env), every turn is generated dynamically by Groq LLM!
        always_use_llm = os.environ.get("ALWAYS_USE_LLM", "true").lower() in ("true", "1", "yes")
        is_testing = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
        if is_testing and os.environ.get("FORCE_LLM_TEST", "").lower() not in ("true", "1", "yes"):
            always_use_llm = False

        if not always_use_llm and decision.mode == ResponseMode.PYTHON_DIRECT and decision.direct_text:
            log.info("pipeline.direct_response", text_preview=decision.direct_text[:60], turn_id=current_turn_id)
            if self._call_logger:
                self._call_logger.record_turn_response(current_turn_id, decision.direct_text, "PYTHON_DIRECT")
            self._latency.log_turn_report(turn_id=current_turn_id)
            self._agent_session.say(decision.direct_text)
            raise StopResponse()

        # 6. LLM Conversational Mode:
        # Keep the initial system prompt byte-for-byte stable for Groq prefix caching.
        # If `sync_rolling_chat_ctx()` already populated dynamic_context between turns,
        # avoid mutating `turn_ctx` here so LiveKit's preemptive generation is preserved!
        if self._call_logger:
            self._call_logger.record_turn_response(current_turn_id, "[LLM Generation]", "LLM_GENERATE")

        stable_system_prompt = self._context_builder.build_system_prompt()
        has_dynamic_context = any(
            getattr(item, "role", None) == "system"
            and _DYNAMIC_CONTEXT_MARKER in (getattr(item, "text_content", None) or "")
            for item in turn_ctx.items
        )

        dynamic_context = ""
        if not has_dynamic_context:
            system_item_found = any(getattr(item, "role", None) == "system" for item in turn_ctx.items)
            if not system_item_found:
                turn_ctx.items.insert(0, lk_llm.ChatMessage(role="system", content=[stable_system_prompt]))

            dynamic_context = self._context_builder.build_dynamic_context(self._session)
            if dynamic_context:
                turn_ctx.items.append(lk_llm.ChatMessage(role="system", content=[dynamic_context]))

        if self._call_logger:
            self._call_logger.record_turn_streaming_metrics(
                current_turn_id,
                preemptive_ctx_preserved=has_dynamic_context,
            )

        log.info(
            "pipeline.llm_response_delegated",
            turn_id=current_turn_id,
            stable_prompt_chars=len(stable_system_prompt),
            dynamic_context_chars=len(dynamic_context),
            preemptive_ctx_preserved=has_dynamic_context,
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
    call_logger = CallLogger(session_id=session_id, room_name=getattr(ctx.room, "name", None))

    normalizer = Normalizer()
    policy = PolicySelector()
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
        soniox_endpoint_ms = max(500, min(3000, int(os.environ.get("SONIOX_ENDPOINT_DELAY_MS", "500"))))
        stt_provider = soniox.STT(
            api_key=soniox_key,
            base_url=soniox_base_url,
            params=soniox.stt.STTOptions(
                model="stt-rt-v5",
                language_hints=["te", "en"],
                max_endpoint_delay_ms=soniox_endpoint_ms,
            ),
        )
    else:
        raise ValueError("No valid STT provider configured. Set SONIOX_API_KEY or explicit STT_PROVIDER=whisper.")

    # LLM: Groq GPT-OSS 20B via the pinned LiveKit OpenAI-compatible plugin.
    # Groq prompt caching is automatic; keep the stable system prompt unchanged.
    llm_kwargs: dict[str, Any] = {
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        "api_key": os.environ.get("GROQ_API_KEY"),
        "base_url": "https://api.groq.com/openai/v1",
        "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.6")),
        "max_completion_tokens": int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "512")),
    }
    reasoning_effort = os.environ.get("GROQ_REASONING_EFFORT", "low").strip()
    if reasoning_effort:
        llm_kwargs["reasoning_effort"] = reasoning_effort
    llm_provider = openai.LLM(**llm_kwargs)

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

    # 4. Create LiveKit 1.x AgentSession with low-latency endpointing & preemptive generation
    min_endpointing = float(os.environ.get("MIN_ENDPOINTING_DELAY", "0.2"))
    max_endpointing = float(os.environ.get("MAX_ENDPOINTING_DELAY", "0.4"))
    preemptive_enabled = os.environ.get("PREEMPTIVE_GENERATION", "true").lower() in ("true", "1", "yes")
    preemptive_tts_enabled = os.environ.get("PREEMPTIVE_TTS", "false").lower() in ("true", "1", "yes")
    preemptive_opts = PreemptiveGenerationOptions(
        enabled=preemptive_enabled,
        preemptive_tts=preemptive_tts_enabled,
        max_speech_duration=10.0,
        max_retries=3,
    )
    session_voice = AgentSession(
        stt=stt_provider,
        llm=llm_provider,
        tts=tts_provider,
        vad=vad_provider,
        allow_interruptions=True,
        min_interruption_duration=0.5,
        min_interruption_words=1,
        min_endpointing_delay=min_endpointing,
        max_endpointing_delay=max_endpointing,
        preemptive_generation=preemptive_opts,
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
        call_logger=call_logger,
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

    # Track pending speech_ids created before on_user_turn_completed (preemptive hits)
    preemptive_speech_ids: set[str] = set()

    @session_voice.on("speech_created")
    def on_speech_created(ev):
        """Logs when a reply SpeechHandle is created (tracks preemptive vs normal generation)."""
        handle = getattr(ev, "speech_handle", None)
        speech_id = getattr(handle, "id", None)
        source = getattr(ev, "source", None)
        pending_turn_id = session.current_turn_id + 1
        # A generate_reply created BEFORE on_user_turn_completed advances _active_response_turn_id
        # to pending_turn_id means preemptive generation kicked off early!
        # (Note: LiveKit 1.8.10 sets user_initiated=True on all _generate_reply calls.)
        is_preemptive_early = (
            source == "generate_reply"
            and agent._active_response_turn_id < pending_turn_id
        )
        if is_preemptive_early and speech_id:
            preemptive_speech_ids.add(speech_id)
        log.info(
            "streaming.speech_created",
            source=source,
            speech_id=speech_id,
            user_initiated=getattr(ev, "user_initiated", None),
            pending_turn_id=pending_turn_id,
            preemptive_early=is_preemptive_early,
        )

    @session_voice.on("metrics_collected")
    def on_metrics_collected(ev):
        """Capture EOU, LLM, and TTS streaming/preemptive metrics and persist to call_logger."""
        m = getattr(ev, "metrics", ev)
        from livekit.agents.metrics import EOUMetrics, LLMMetrics, TTSMetrics

        active_tid = agent._active_response_turn_id if agent._active_response_turn_id > 0 else session.current_turn_id + 1
        m_speech_id = getattr(m, "speech_id", None)
        if m_speech_id and m_speech_id in preemptive_speech_ids and agent._active_response_turn_id <= session.current_turn_id:
            active_tid = session.current_turn_id + 1

        if isinstance(m, EOUMetrics):
            eou_ms = round(m.end_of_utterance_delay * 1000, 1)
            stt_ms = round(m.transcription_delay * 1000, 1)
            hook_ms = round(m.on_user_turn_completed_delay * 1000, 1)
            speech_id = getattr(m, "speech_id", None)
            preemptive_hit = bool(speech_id and speech_id in preemptive_speech_ids)
            log.info(
                "latency.eou_metrics",
                turn_id=active_tid,
                eou_delay_ms=eou_ms,
                stt_delay_ms=stt_ms,
                hook_delay_ms=hook_ms,
                speech_id=speech_id,
                preemptive_hit=preemptive_hit,
            )
            call_logger.record_turn_streaming_metrics(
                active_tid,
                eou_delay_ms=eou_ms,
                stt_delay_ms=stt_ms,
                hook_delay_ms=hook_ms,
                preemptive_hit=preemptive_hit,
            )
        elif isinstance(m, LLMMetrics):
            first_token_ts_ms = (m.timestamp + m.ttft) * 1000 if m.ttft > 0 else m.timestamp * 1000
            latency.record(Stage.LLM_FIRST_TOKEN, first_token_ts_ms)
            ttft_ms = round(m.ttft * 1000, 1) if m.ttft >= 0 else None
            dur_ms = round(m.duration * 1000, 1) if m.duration else None
            cached_toks = getattr(m, "prompt_cached_tokens", 0) or 0
            reasoning_toks = getattr(m, "reasoning_tokens", 0) or 0
            cancelled = bool(getattr(m, "cancelled", False))
            speech_id = getattr(m, "speech_id", None)
            log.info(
                "latency.llm_metrics",
                turn_id=active_tid,
                ttft_ms=ttft_ms,
                duration_ms=dur_ms,
                prompt_tokens=m.prompt_tokens,
                cached_tokens=cached_toks,
                reasoning_tokens=reasoning_toks,
                completion_tokens=m.completion_tokens,
                cancelled=cancelled,
                speech_id=speech_id,
            )
            if cancelled:
                call_logger.increment_turn_streaming_counter(active_tid, "preemptive_cancelled_count")
            else:
                call_logger.record_turn_streaming_metrics(
                    active_tid,
                    llm_ttft_ms=ttft_ms,
                    llm_duration_ms=dur_ms,
                    llm_prompt_tokens=m.prompt_tokens,
                    llm_cached_tokens=cached_toks,
                    llm_reasoning_tokens=reasoning_toks,
                    llm_completion_tokens=m.completion_tokens,
                )
        elif isinstance(m, TTSMetrics):
            first_audio_ts_ms = (m.timestamp + m.ttfb) * 1000 if m.ttfb > 0 else m.timestamp * 1000
            latency.record(Stage.TTS_FIRST_AUDIO, first_audio_ts_ms)
            ttfb_ms = round(m.ttfb * 1000, 1) if m.ttfb >= 0 else None
            dur_ms = round(m.duration * 1000, 1) if m.duration else None
            cancelled = bool(getattr(m, "cancelled", False))
            log.info(
                "latency.tts_metrics",
                turn_id=active_tid,
                ttfb_ms=ttfb_ms,
                duration_ms=dur_ms,
                characters=m.characters_count,
                cancelled=cancelled,
                speech_id=getattr(m, "speech_id", None),
            )
            if not cancelled:
                call_logger.record_turn_streaming_metrics(
                    active_tid,
                    tts_ttfb_ms=ttfb_ms,
                    tts_duration_ms=dur_ms,
                    tts_characters=m.characters_count,
                )

    @session_voice.on("conversation_item_added")
    def on_conversation_item_added(ev):
        """Captures LLM-generated speech in real time for call logging and syncs rolling context."""
        item = getattr(ev, "item", None)
        if item and getattr(item, "role", None) == "assistant":
            text = getattr(item, "text_content", "")
            if text and agent._active_response_turn_id > 0:
                log.info("agent.llm_reply_generated", turn_id=agent._active_response_turn_id, reply_preview=text[:60])
                call_logger.record_turn_response(agent._active_response_turn_id, text, "LLM_GENERATE")
            asyncio.create_task(agent.sync_rolling_chat_ctx())

    # 4c. Setup disconnect and close event handlers BEFORE greeting to guard against early disconnect
    stop_event = asyncio.Event()

    def _save_call_record(status: Optional[str] = None) -> None:
        try:
            saved_info = call_logger.save_call(
                session=session,
                latency=latency,
                status=status,
                agent_history=getattr(session_voice, "history", None),
            )
            log.info(
                "call_logger.completed",
                session_id=session_id,
                json_file=saved_info.get("json_path"),
                txt_file=saved_info.get("txt_path"),
            )
        except Exception as e:
            log.warning("call_logger.save_error", error=str(e))

    @session_voice.on("close")
    def on_session_close(ev=None):
        log.info("agent.session_close_event", ev=str(ev))
        _save_call_record(status="COMPLETED")
        stop_event.set()

    @ctx.room.on("disconnected")
    def on_disconnected():
        log.info("agent.room_disconnected")
        _save_call_record(status="DISCONNECTED")
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
    await agent.sync_rolling_chat_ctx()

    try:
        # 8. Keep session active until caller disconnects or session closes
        await stop_event.wait()
    finally:
        summary = latency.session_summary()
        log.info("agent.session_ended", session_id=session_id, latency_summary=summary)
        _save_call_record(status="COMPLETED")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint)
    )
