import pytest

from core.session import ConversationSession
from pipeline.jev_decision import JevDecisionClient, build_jev_state
from pipeline.normalizer import Normalizer
from pipeline.policy import PolicySelector, ResponseMode


def _choice_response(intent, confidence=0.93):
    return {
        "model": "typesafe/jev-1.14",
        "answers": {
            "caller_intent": {
                "type": "choice",
                "choice": intent,
                "confidence": confidence,
                "probabilities": {intent: confidence},
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 0, "cost": 0.00001},
    }


@pytest.mark.asyncio
async def test_jev_classifies_with_typed_choice_and_context(monkeypatch):
    monkeypatch.setenv("JEV_DECISION_ENABLED", "true")
    client = JevDecisionClient(api_key="test-key")
    captured = {}

    def fake_post(payload):
        captured.update(payload)
        return _choice_response("question")

    monkeypatch.setattr(client, "_post_decision", fake_post)
    session = ConversationSession("jev-test")
    session.pending_field = "budget_rupees"
    session.commit_turn("మీ దగ్గర 2BHK ఉందా?", "మీ దగ్గర 2BHK ఉందా?", 0, 1)

    result = await client.classify(build_jev_state(session, "మీ దగ్గర 2BHK ఉందా?"))

    assert result is not None
    assert result.intent == "question"
    assert result.confidence == pytest.approx(0.93)
    assert captured["model"] == "typesafe/jev-1.14"
    assert captured["questions"]["caller_intent"]["type"] == "choice"
    assert captured["state"]["pending_qualification_field"] == "budget_rupees"


@pytest.mark.asyncio
async def test_jev_disabled_by_default(monkeypatch):
    monkeypatch.delenv("JEV_DECISION_ENABLED", raising=False)
    client = JevDecisionClient(api_key="test-key")
    assert client.enabled is False
    assert await client.classify({"latest_caller_utterance": "hello"}) is None


@pytest.mark.asyncio
async def test_jev_low_confidence_uses_fallback(monkeypatch):
    monkeypatch.setenv("JEV_DECISION_ENABLED", "true")
    client = JevDecisionClient(api_key="test-key", min_confidence=0.8)
    monkeypatch.setattr(client, "_post_decision", lambda payload: _choice_response("question", 0.51))
    assert await client.classify({"latest_caller_utterance": "ఏమిటి?"}) is None



@pytest.mark.asyncio
async def test_jev_missing_confidence_uses_fallback(monkeypatch):
    monkeypatch.setenv("JEV_DECISION_ENABLED", "true")
    client = JevDecisionClient(api_key="test-key")
    malformed = _choice_response("question")
    del malformed["answers"]["caller_intent"]["confidence"]
    monkeypatch.setattr(client, "_post_decision", lambda payload: malformed)
    assert await client.classify({"latest_caller_utterance": "ఏమిటి?"}) is None

@pytest.mark.asyncio
async def test_jev_malformed_response_uses_fallback(monkeypatch):
    monkeypatch.setenv("JEV_DECISION_ENABLED", "true")
    client = JevDecisionClient(api_key="test-key")
    monkeypatch.setattr(client, "_post_decision", lambda payload: {"answers": {}})
    assert await client.classify({"latest_caller_utterance": "ఏమిటి?"}) is None


def _run_policy_turn(session, text, jev_intent):
    normalizer = Normalizer()
    _, normalized = normalizer.process(text)
    session.advance_turn()
    session.commit_turn(text, normalized, 0, 1)
    extraction = normalizer.extract(normalized)
    return PolicySelector().select(session, extraction, intent_override=jev_intent)


def _session_waiting_for_budget():
    session = ConversationSession("jev-policy-test")
    session.update_fact("location", "Hyderabad")
    session.pending_field = "budget_rupees"
    session.update_stage("budget", "budget_rupees")
    return session


def test_jev_question_routes_to_generative_llm_even_without_keyword_marker():
    session = _session_waiting_for_budget()
    decision = _run_policy_turn(session, "మీరు కొంచెం వివరించగలరా", "question")
    assert decision.mode is ResponseMode.LLM_GENERATE
    assert decision.intent == "customer_question"
    assert session.pending_field == "budget_rupees"


def test_jev_doubt_without_question_routes_to_llm_and_preserves_pending_slot():
    session = _session_waiting_for_budget()
    decision = _run_policy_turn(session, "నాకు ఒక సందేహం ఉంది", "doubt_without_question")
    assert decision.mode is ResponseMode.LLM_GENERATE
    assert decision.intent == "customer_doubt"
    assert session.pending_field == "budget_rupees"
    assert "invite them to explain" in decision.llm_prompt


def test_jev_defer_skips_only_current_pending_field():
    session = _session_waiting_for_budget()
    decision = _run_policy_turn(session, "ఇప్పుడు చెప్పలేను", "defer")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "customer_deferred"
    assert "budget_rupees" in session.deferred_fields
    assert session.pending_field == "property_type"


def test_jev_refusal_does_not_end_the_call():
    session = _session_waiting_for_budget()
    decision = _run_policy_turn(session, "అది చెప్పడం ఇష్టం లేదు", "refusal")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "customer_refused"
    assert session.facts.opted_out is False
    assert "budget_rupees" in session.refused_fields


def test_jev_global_not_interested_ends_call():
    session = _session_waiting_for_budget()
    decision = _run_policy_turn(session, "ఇక నాకు అవసరం లేదు", "not_interested")
    assert decision.next_stage == "ended"
    assert decision.intent == "customer_not_interested"


def test_jev_repeat_repeats_the_pending_question():
    session = _session_waiting_for_budget()
    decision = _run_policy_turn(session, "మళ్ళీ అడగండి", "repeat")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "repeat_question"
    assert decision.next_stage == "budget_rupees"
