from core.session import ConversationSession
from pipeline.normalizer import Normalizer
from pipeline.policy import PolicySelector, ResponseMode


def run_turn(session, text):
    session.advance_turn()
    session.commit_turn(
        raw_transcript=text,
        normalized_transcript=text,
        timestamp_start_ms=0,
        timestamp_end_ms=1,
    )
    normalizer = Normalizer()
    extraction = normalizer.extract(normalizer.process(text)[1])
    return PolicySelector().select(session, extraction)


def session_waiting_for_budget():
    session = ConversationSession("policy-test")
    session.update_fact("location", "Vijayawada")
    session.update_stage("budget", "budget")
    session.pending_field = "budget_rupees"
    return session


def test_budget_answer_advances_to_next_missing_field():
    session = session_waiting_for_budget()
    decision = run_turn(session, "50 లాక్స్")
    assert session.facts.budget_rupees == 5_000_000
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.next_stage == "property_type"
    assert session.pending_field == "property_type"


def test_customer_question_is_sent_to_llm_without_losing_pending_slot():
    session = session_waiting_for_budget()
    decision = run_turn(session, "మీ బడ్జెట్ ఎందుకు అడుగుతున్నారు?")
    assert decision.mode is ResponseMode.LLM_GENERATE
    assert decision.intent == "customer_question"
    assert session.pending_field == "budget_rupees"
    assert session.facts.budget_rupees is None


def test_property_availability_question_is_sent_to_llm():
    session = session_waiting_for_budget()
    decision = run_turn(session, "మాకు ఎలాంటి ప్రాపర్టీస్ ఉన్నాయి వీటిలో?")
    assert decision.mode is ResponseMode.LLM_GENERATE
    assert decision.intent == "customer_question"
    assert session.pending_field == "budget_rupees"


def test_undecided_budget_is_deferred_and_not_reasked_immediately():
    session = session_waiting_for_budget()
    decision = run_turn(session, "ఇంకా నిర్ణయించుకోలేదు")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "customer_undecided"
    assert "budget_rupees" in session.deferred_fields
    assert session.pending_field == "property_type"


def test_bare_no_refuses_current_field_but_does_not_end_call():
    session = session_waiting_for_budget()
    decision = run_turn(session, "వద్దు")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "customer_refused"
    assert session.facts.opted_out is False
    assert "budget_rupees" in session.refused_fields
    assert decision.next_stage == "property_type"


def test_explicit_stop_call_request_ends_call():
    session = session_waiting_for_budget()
    decision = run_turn(session, "నాకు మళ్ళీ కాల్ చేయకండి")
    assert decision.intent == "global_opt_out"
    assert decision.next_stage == "ended"
    assert session.facts.opted_out is True


def test_budget_and_question_in_same_turn_save_fact_then_answer_question():
    session = session_waiting_for_budget()
    decision = run_turn(session, "50 lakhs, do you have flats?")
    assert session.facts.budget_rupees == 5_000_000
    assert decision.mode is ResponseMode.LLM_GENERATE
    assert session.pending_field == "property_type"


def test_unclear_answer_clarifies_the_pending_field():
    session = session_waiting_for_budget()
    decision = run_turn(session, "హ్మ్")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "clarify_pending_field"
    assert decision.next_stage == "budget_rupees"


def test_repeat_request_repeats_pending_question():
    session = session_waiting_for_budget()
    decision = run_turn(session, "మళ్ళీ చెప్పండి")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "repeat_question"
    assert decision.next_stage == "budget_rupees"



def test_doubt_without_question_invites_customer_to_explain_and_preserves_pending_slot():
    session = session_waiting_for_budget()
    decision = run_turn(session, "నాకు ఒక డౌట్ ఉంది")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "customer_has_doubt"
    assert "సందేహం" in decision.direct_text
    assert session.pending_field == "budget_rupees"


def test_english_doubt_without_question_invites_customer_to_explain():
    session = session_waiting_for_budget()
    decision = run_turn(session, "I have a doubt")
    assert decision.mode is ResponseMode.PYTHON_DIRECT
    assert decision.intent == "customer_has_doubt"
    assert session.pending_field == "budget_rupees"


def test_availability_question_does_not_save_property_type_as_preference():
    session = session_waiting_for_budget()
    decision = run_turn(session, "50 lakhs, do you have flats?")
    assert session.facts.budget_rupees == 5_000_000
    assert session.facts.property_type is None
    assert decision.mode is ResponseMode.LLM_GENERATE
    assert session.pending_field == "property_type"
