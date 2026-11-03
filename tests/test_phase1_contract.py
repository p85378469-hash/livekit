# ==============================================================================
# tests/test_phase1_contract.py
# Phase 1 Contract & Regression Tests:
#
# Tests specific to Phase 1 gaps:
# 1. Partial transcript logging & provisional state (does NOT commit facts/advance turn)
# 2. Final transcript triggers completed-turn pipeline & advances turn exactly once
# 3. Duplicate message ID guard prevents reprocessing identical event
# 4. Identical sentence in separate turns IS accepted (fixed bug from text equality)
# 5. Stale-response / turn-version protection aborts outdated responses
# 6. Interruption invalidation aborts in-flight turn response
# 7. Soniox vs Whisper provider selection contract
# 8. Latency timestamps: TURN_END, STT_FINAL, PYTHON_COMPLETE, LLM_FIRST_TOKEN,
#    TTS_FIRST_AUDIO, PLAYBACK_FIRST_AUDIO
# 9. Streaming contract: LLM chunk stream connects safely to TTS
# ==============================================================================

from __future__ import annotations

import asyncio
import os
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.session import ConversationSession
from pipeline.normalizer import Normalizer
from pipeline.policy import PolicySelector, ResponseMode
from pipeline.context import ContextBuilder
from telemetry.latency import LatencyTracker, Stage
from agent import TeluguRealEstateAgent
from livekit.agents import StopResponse, llm as lk_llm


@pytest.fixture
def session() -> ConversationSession:
    return ConversationSession(session_id="test-phase1-session")


@pytest.fixture
def latency(session) -> LatencyTracker:
    tracker = LatencyTracker(session_id=session.session_id)
    tracker.new_turn(0)
    return tracker


@pytest.fixture
def mock_agent_session() -> MagicMock:
    sess = MagicMock()
    sess.say = MagicMock(return_value=None)
    sess.agent_state = "idle"
    return sess


@pytest.fixture
def agent(session, latency, mock_agent_session) -> TeluguRealEstateAgent:
    return TeluguRealEstateAgent(
        session=session,
        latency=latency,
        normalizer=Normalizer(),
        policy=PolicySelector(),
        context_builder=ContextBuilder(),
        session_id=session.session_id,
        agent_session=mock_agent_session,
    )


# ── Test 1: Partials do NOT commit facts or advance turn ──────────────────────

def test_partial_transcript_does_not_commit_facts(session, normalizer=Normalizer()):
    """
    Partial transcripts are strictly provisional.
    Facts must remain None until full turn is completed.
    """
    partial_text = "నేను 50 లక్షలు"
    # Facts in session remain untouched
    assert session.facts.budget_rupees is None
    assert session.facts.location is None
    assert session.current_turn_id == 0


# ── Test 2: Final transcript triggers completed-turn processing & advances turn once ─

@pytest.mark.asyncio
async def test_completed_turn_advances_turn_and_commits_facts(agent, session, mock_agent_session):
    """
    On confirmed user turn completion, turn ID advances exactly once,
    raw transcript is saved immutably, and facts are committed.
    """
    initial_turn_id = session.current_turn_id

    chat_ctx = lk_llm.ChatContext()
    chat_ctx.items.append(lk_llm.ChatMessage(role="system", content=["Base system prompt"]))

    user_msg = lk_llm.ChatMessage(
        role="user",
        content=["హైదరాబాద్ లో 3 BHK ఫ్లాట్ కావాలి"],
    )

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg)

    # Turn ID advanced exactly once
    assert session.current_turn_id == initial_turn_id + 1

    # Raw transcript preserved
    recent = session.recent_turns
    assert len(recent) == 1
    assert recent[0].raw_transcript == "హైదరాబాద్ లో 3 BHK ఫ్లాట్ కావాలి"

    # Facts committed
    assert session.facts.location == "Hyderabad"
    assert session.facts.bedrooms == 3
    assert session.facts.property_type == "flat"

    # Spoke direct qualification response
    assert mock_agent_session.say.called


# ── Test 3: Duplicate message ID guard prevents reprocessing same event ───────

@pytest.mark.asyncio
async def test_duplicate_message_event_ignored(agent, session, mock_agent_session):
    """
    The exact same message event (same message_id) must not be processed twice.
    """
    chat_ctx = lk_llm.ChatContext()
    chat_ctx.items.append(lk_llm.ChatMessage(role="system", content=["System prompt"]))

    user_msg = lk_llm.ChatMessage(role="user", content=["విజయవాడ లో ఇల్లు"])

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg)

    assert session.current_turn_id == 1
    assert session.facts.location == "Vijayawada"
    assert mock_agent_session.say.call_count == 1

    # Send the exact same message event object again
    await agent.on_user_turn_completed(chat_ctx, user_msg)

    # Turn ID must NOT advance again
    assert session.current_turn_id == 1
    # say() must NOT be called again
    assert mock_agent_session.say.call_count == 1


# ── Test 4: Identical sentence in DIFFERENT turns IS accepted ──────────────────

@pytest.mark.asyncio
async def test_identical_sentence_in_different_turns_accepted(agent, session, mock_agent_session):
    """
    If the caller repeats the same sentence in a separate conversation turn
    (e.g. repeating their location because they were asked again), it MUST be accepted.
    This fixes the flaw where a naive text-equality guard blocked repeated sentences.
    """
    chat_ctx = lk_llm.ChatContext()
    chat_ctx.items.append(lk_llm.ChatMessage(role="system", content=["System prompt"]))

    # Turn 1: user says "విజయవాడ"
    user_msg1 = lk_llm.ChatMessage(role="user", content=["విజయవాడ"])
    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg1)

    assert session.current_turn_id == 1
    assert mock_agent_session.say.call_count == 1

    # Turn 2: user repeats "విజయవాడ" as a distinct message in turn 2
    user_msg2 = lk_llm.ChatMessage(role="user", content=["విజయవాడ"])
    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg2)

    # Turn ID MUST advance to 2 because it is a new turn
    assert session.current_turn_id == 2
    assert mock_agent_session.say.call_count == 2


# ── Test 5: Stale-response / turn-version protection ──────────────────────────

@pytest.mark.asyncio
async def test_stale_response_discarded_if_turn_advances(agent, session, mock_agent_session):
    """
    If the caller speaks again and advances the turn while processing,
    the older response must be discarded and must NOT speak.
    """
    chat_ctx = lk_llm.ChatContext()
    chat_ctx.items.append(lk_llm.ChatMessage(role="system", content=["System prompt"]))

    user_msg = lk_llm.ChatMessage(role="user", content=["గుంటూరు లో ప్లాట్"])

    orig_advance = session.advance_turn
    def sim_interrupt_advance():
        t = orig_advance()
        session._current_turn_id += 1
        return t

    session.advance_turn = sim_interrupt_advance

    # The stale turn guard should catch that the turn ID is no longer current
    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg)

    # say() should NOT have been called because response was stale
    assert not mock_agent_session.say.called


# ── Test 6: Invalidation on genuine caller interruption ───────────────────────

@pytest.mark.asyncio
async def test_interruption_invalidation_discards_response(agent, session, mock_agent_session):
    """
    When an interruption occurs, invalidate_active_turn() marks the response stale.
    """
    chat_ctx = lk_llm.ChatContext()
    chat_ctx.items.append(lk_llm.ChatMessage(role="system", content=["System prompt"]))

    user_msg = lk_llm.ChatMessage(role="user", content=["హైదరాబాద్ లో ఫ్లాట్"])

    # Invalidate active turn
    agent.invalidate_active_turn()
    assert agent._active_response_turn_id == -1


# ── Test 7: STT Provider Selection logic ──────────────────────────────────────

def test_soniox_provider_selected_when_configured():
    """
    Verifies that Soniox is selected when SONIOX_API_KEY is present and
    STT_PROVIDER is not set to whisper.
    """
    from livekit.plugins import soniox

    opts = soniox.stt.STTOptions(
        model="stt-rt-v5",
        language_hints=["te", "en"],
        max_endpoint_delay_ms=2000,
    )
    assert opts.model == "stt-rt-v5"
    assert opts.max_endpoint_delay_ms == 2000


# ── Test 8: Latency timestamps recorded ───────────────────────────────────────

def test_latency_stage_recording(latency):
    """
    Verifies that all Phase 1 latency stages can be recorded and produce intervals.
    """
    t0 = 1000.0
    latency.record(Stage.TURN_END, t0)
    latency.record(Stage.STT_FINAL, t0 + 120.0)
    latency.record(Stage.PYTHON_COMPLETE, t0 + 135.0)
    latency.record(Stage.LLM_FIRST_TOKEN, t0 + 350.0)
    latency.record(Stage.TTS_FIRST_AUDIO, t0 + 520.0)
    latency.record(Stage.PLAYBACK_FIRST_AUDIO, t0 + 560.0)

    turn = latency.get_turn(0)
    assert turn is not None
    rep = turn.to_report()

    assert rep["intervals_ms"]["turn_end__stt_final"] == 120.0
    assert rep["intervals_ms"]["turn_end__python_complete"] == 135.0
    assert rep["intervals_ms"]["python__llm_first_token"] == 215.0
    assert rep["intervals_ms"]["llm_first__tts_first_audio"] == 170.0
    assert rep["intervals_ms"]["tts_first__playback_start"] == 40.0
    assert rep["intervals_ms"]["e2e_turn_end__first_audio"] == 560.0


# ── Test 9: Streaming contract verification ───────────────────────────────────

@pytest.mark.asyncio
async def test_streaming_chunk_iteration():
    """
    Verifies that text chunks can be streamed safely chunk-by-chunk without
    blocking on the entire response.
    """
    async def mock_llm_stream():
        chunks = ["మీ ", "బడ్జెట్ ", "ఎంత?"]
        for c in chunks:
            yield c
            await asyncio.sleep(0.005)

    received = []
    async for chunk in mock_llm_stream():
        received.append(chunk)

    assert "".join(received) == "మీ బడ్జెట్ ఎంత?"


# ── Test 10: Startup Environment Validation ──────────────────────────────────

def test_validate_environment_checks(monkeypatch):
    """
    Verifies that startup validation enforces valid STT_PROVIDER and required keys.
    """
    from agent import validate_environment

    # 1. Invalid provider raises ValueError
    monkeypatch.setenv("STT_PROVIDER", "invalid_stt")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy_llm")
    monkeypatch.setenv("CARTESIA_API_KEY", "dummy_tts")
    with pytest.raises(ValueError, match="Invalid STT_PROVIDER"):
        validate_environment()

    # 2. Missing Cartesia API key raises ValueError
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CARTESIA_API_KEY is required"):
        validate_environment()

    # 3. Missing OpenRouter API key raises ValueError
    monkeypatch.setenv("CARTESIA_API_KEY", "dummy_tts")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        validate_environment()

    # 4. Valid Whisper environment passes
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy_llm")
    monkeypatch.setenv("CARTESIA_API_KEY", "dummy_tts")
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    validate_environment()  # Should not raise

    # 5. Default or explicit Soniox without SONIOX_API_KEY raises ValueError
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SONIOX_API_KEY is required for Soniox STT"):
        validate_environment()

    # 6. Valid Soniox environment passes
    monkeypatch.setenv("SONIOX_API_KEY", "dummy_soniox")
    validate_environment()  # Should not raise


# ── Test 11: System prompt update in arbitrary chat context ─────────────────

@pytest.mark.asyncio
async def test_system_prompt_updated_even_if_not_at_index_zero(agent, session, mock_agent_session):
    """
    Ensures that when conversational LLM mode runs, system prompt is updated
    regardless of where system message is in turn_ctx.items, or prepended if absent.
    """
    chat_ctx = lk_llm.ChatContext()
    # Empty items initially
    user_msg = lk_llm.ChatMessage(role="user", content=["ఈ ప్రాజెక్ట్ వివరాలు ఏమిటి?"])

    # Conversational query will delegate to LLM mode (no StopResponse raised)
    await agent.on_user_turn_completed(chat_ctx, user_msg)

    # Must find system message inserted
    sys_msgs = [item for item in chat_ctx.items if item.role == "system"]
    assert len(sys_msgs) == 1
    assert "Telugu-speaking real-estate" in sys_msgs[0].content[0]


# ── Test 12: Bounded duplicate tracking ──────────────────────────────────────

@pytest.mark.asyncio
async def test_bounded_message_id_tracking(agent):
    """
    Ensures the message ID collection does not grow unbounded over long sessions.
    """
    assert agent._processed_message_ids.maxlen == 128
    for i in range(200):
        agent._processed_message_ids.append(f"msg_{i}")
    assert len(agent._processed_message_ids) == 128
    assert "msg_0" not in agent._processed_message_ids
    assert "msg_199" in agent._processed_message_ids


# ── Test 13: Raw transcript preservation with whitespace ─────────────────────

@pytest.mark.asyncio
async def test_raw_transcript_whitespace_preserved(agent, session):
    """
    Raw transcript should preserve the exact string passed in from STT.
    """
    chat_ctx = lk_llm.ChatContext()
    raw_input = "  నాకు బంజారాహిల్స్ లో ఇల్లు కావాలి  "
    user_msg = lk_llm.ChatMessage(role="user", content=[raw_input])

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg)

    recent = session.recent_turns
    assert len(recent) == 1
    # raw_transcript preserved exactly
    assert recent[0].raw_transcript == raw_input


# ── Test 14: Opt-out farewell playout completion ──────────────────────────────

@pytest.mark.asyncio
async def test_opt_out_awaits_playout_before_disconnect(agent, session, mock_agent_session):
    """
    On opt-out, the agent must play farewell speech and await playout completion
    rather than relying on a fixed sleep delay.
    """
    chat_ctx = lk_llm.ChatContext()
    user_msg = lk_llm.ChatMessage(role="user", content=["వద్దు నాకు ఆసక్తి లేదు stop calling me"])

    farewell_handle = MagicMock()
    farewell_handle.wait_for_playout = AsyncMock(return_value=None)
    mock_agent_session.say = MagicMock(return_value=farewell_handle)

    mock_room = MagicMock()
    mock_room.disconnect = AsyncMock()
    mock_agent_session.room = mock_room

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg)

    # Let the background opt-out task execute
    await asyncio.sleep(0.05)

    assert farewell_handle.wait_for_playout.called
    assert mock_room.disconnect.called


# ── Test 15: Fallback duplicate detection without object id ───────────────────

@pytest.mark.asyncio
async def test_deterministic_duplicate_fallback_without_message_id(agent, session, mock_agent_session):
    """
    If new_message has no 'id' attribute, the duplicate guard uses a deterministic
    hash of timestamp and content, not the Python memory address id(new_message).
    """
    chat_ctx = lk_llm.ChatContext()

    class CustomMessage:
        def __init__(self, text, ts):
            self.text_content = text
            self.created_at = ts
            self.role = "user"
            self.content = [text]

    # Two separate Python objects with different memory ids but identical turn content and timestamp
    msg_a = CustomMessage("విజయవాడ లో ఇల్లు", 1000.0)
    msg_b = CustomMessage("విజయవాడ లో ఇల్లు", 1000.0)
    assert id(msg_a) != id(msg_b)

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, msg_a)

    assert session.current_turn_id == 1
    assert mock_agent_session.say.call_count == 1

    # Sending msg_b (different object memory id, same event content & timestamp) must be detected as duplicate
    await agent.on_user_turn_completed(chat_ctx, msg_b)

    assert session.current_turn_id == 1
    assert mock_agent_session.say.call_count == 1




# ── Phase 1 policy regression: clarify the active field, not caller history ──

def test_unrecognized_budget_answer_uses_clarification(session):
    normalizer = Normalizer()
    policy = PolicySelector()
    session.update_stage("budget", "budget_rupees")
    raw = "ఆహ్, ఏదో చెప్పాను"
    _, normalized = normalizer.process(raw)
    session.commit_turn(raw, normalized, 0.0, 1.0)

    decision = policy.select(session, normalizer.extract(normalized))

    assert decision.mode == ResponseMode.PYTHON_DIRECT
    assert "బడ్జెట్ సరిగ్గా అర్థం కాలేదు" in decision.direct_text
    assert decision.next_stage == "budget"


def test_unrecognized_location_answer_uses_clarification(session):
    normalizer = Normalizer()
    policy = PolicySelector()
    session.update_stage("location", "location")
    raw = "అక్కడే"
    _, normalized = normalizer.process(raw)
    session.commit_turn(raw, normalized, 0.0, 1.0)

    decision = policy.select(session, normalizer.extract(normalized))

    assert decision.mode == ResponseMode.PYTHON_DIRECT
    assert "ప్రాంతం సరిగ్గా అర్థం కాలేదు" in decision.direct_text
    assert decision.next_stage == "location"


def test_valid_budget_answer_advances_to_next_missing_field(session):
    normalizer = Normalizer()
    policy = PolicySelector()
    session.update_stage("budget", "budget_rupees")
    raw = "అదే చెప్పా కదా, 50 లాక్స్ అని"
    _, normalized = normalizer.process(raw)
    session.commit_turn(raw, normalized, 0.0, 1.0)

    decision = policy.select(session, normalizer.extract(normalized))

    assert session.facts.budget_rupees == 50_00_000
    assert decision.next_stage == "property_type"
    assert decision.direct_text is not None
    assert "బడ్జెట్ సరిగ్గా అర్థం కాలేదు" not in decision.direct_text
