# ==============================================================================
# tests/test_phase1_contract.py
# Regression tests for direct conversational LLM routing.
# ==============================================================================

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.session import ConversationSession
from pipeline.context import ContextBuilder
from telemetry.latency import LatencyTracker, Stage
from agent import TeluguRealEstateAgent, validate_environment
from livekit.agents import StopResponse, llm as lk_llm


@pytest.fixture
def session() -> ConversationSession:
    return ConversationSession(session_id="test-direct-llm-session")


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
        context_builder=ContextBuilder(),
        session_id=session.session_id,
        agent_session=mock_agent_session,
    )


@pytest.mark.asyncio
async def test_normal_customer_turn_is_delegated_directly_to_llm(agent, session, mock_agent_session):
    chat_ctx = lk_llm.ChatContext()
    raw_text = "నాకు విజయవాడలో 70 లక్షల బడ్జెట్‌తో 3BHK ఫ్లాట్ కావాలి"
    user_msg = lk_llm.ChatMessage(role="user", content=[raw_text])

    result = await agent.on_user_turn_completed(chat_ctx, user_msg)

    assert result is None
    assert session.current_turn_id == 1
    assert len(session.recent_turns) == 1
    assert session.recent_turns[0].raw_transcript == raw_text
    assert session.recent_turns[0].normalized_transcript == raw_text
    assert user_msg.text_content == raw_text
    assert mock_agent_session.say.call_count == 0
    assert chat_ctx.items == []


@pytest.mark.asyncio
async def test_duplicate_message_event_is_ignored(agent, session, mock_agent_session):
    chat_ctx = lk_llm.ChatContext()
    user_msg = lk_llm.ChatMessage(role="user", content=["విజయవాడలో ఇల్లు కావాలి"])

    await agent.on_user_turn_completed(chat_ctx, user_msg)
    await agent.on_user_turn_completed(chat_ctx, user_msg)

    assert session.current_turn_id == 1
    assert len(session.recent_turns) == 1
    assert mock_agent_session.say.call_count == 0


@pytest.mark.asyncio
async def test_distinct_turns_are_both_delegated(agent, session):
    chat_ctx = lk_llm.ChatContext()
    first = lk_llm.ChatMessage(role="user", content=["విజయవాడ"])
    second = lk_llm.ChatMessage(role="user", content=["విజయవాడ"])

    await agent.on_user_turn_completed(chat_ctx, first)
    await agent.on_user_turn_completed(chat_ctx, second)

    assert session.current_turn_id == 2
    assert len(session.recent_turns) == 2


@pytest.mark.asyncio
async def test_stale_turn_is_discarded(agent, session, mock_agent_session):
    chat_ctx = lk_llm.ChatContext()
    user_msg = lk_llm.ChatMessage(role="user", content=["గుంటూరులో ఇల్లు కావాలి"])

    original_advance = session.advance_turn

    def simulate_newer_turn():
        turn_id = original_advance()
        session._current_turn_id += 1
        return turn_id

    session.advance_turn = simulate_newer_turn

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg)

    assert mock_agent_session.say.call_count == 0


@pytest.mark.asyncio
async def test_interruption_invalidates_active_turn(agent):
    agent.invalidate_active_turn()
    assert agent._active_response_turn_id == -1


@pytest.mark.asyncio
async def test_explicit_opt_out_uses_deterministic_compliance_response(agent, mock_agent_session):
    chat_ctx = lk_llm.ChatContext()
    user_msg = lk_llm.ChatMessage(role="user", content=["దయచేసి మళ్ళీ నాకు కాల్ చేయకండి"])

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(chat_ctx, user_msg)

    assert mock_agent_session.say.call_count == 1
    assert "మళ్ళీ మీకు కాల్ చేయము" in mock_agent_session.say.call_args.args[0]


def test_context_is_a_stable_conversational_prompt():
    prompt = ContextBuilder().build_system_prompt()

    assert "natural" in prompt.lower()
    assert "Do not follow a rigid question sequence." in prompt
    assert "Current validated lead context" not in prompt
    assert "Collect exactly these facts in order" not in prompt


def test_latency_stage_recording(latency):
    t0 = 1000.0
    latency.record(Stage.TURN_END, t0)
    latency.record(Stage.STT_FINAL, t0 + 120.0)
    latency.record(Stage.LLM_FIRST_TOKEN, t0 + 350.0)
    latency.record(Stage.TTS_FIRST_AUDIO, t0 + 520.0)
    latency.record(Stage.PLAYBACK_FIRST_AUDIO, t0 + 560.0)

    turn = latency.get_turn(0)
    assert turn is not None
    report = turn.to_report()
    assert report["intervals_ms"]["turn_end__stt_final"] == 120.0
    assert report["intervals_ms"]["llm_first__tts_first_audio"] == 170.0
    assert report["intervals_ms"]["tts_first__playback_start"] == 40.0
    assert report["intervals_ms"]["e2e_turn_end__first_audio"] == 560.0


def test_validate_environment_checks(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "invalid_stt")
    monkeypatch.setenv("GROQ_API_KEY", "dummy_llm")
    monkeypatch.setenv("CARTESIA_API_KEY", "dummy_tts")
    with pytest.raises(ValueError, match="Invalid STT_PROVIDER"):
        validate_environment()

    monkeypatch.setenv("STT_PROVIDER", "whisper")
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CARTESIA_API_KEY is required"):
        validate_environment()

    monkeypatch.setenv("CARTESIA_API_KEY", "dummy_tts")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
        validate_environment()

    monkeypatch.setenv("GROQ_API_KEY", "dummy_llm")
    monkeypatch.setenv("CARTESIA_API_KEY", "dummy_tts")
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    validate_environment()

    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SONIOX_API_KEY is required for Soniox STT"):
        validate_environment()

    monkeypatch.setenv("SONIOX_API_KEY", "dummy_soniox")
    validate_environment()
