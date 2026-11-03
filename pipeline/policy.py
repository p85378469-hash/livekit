from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import structlog

from core.session import AgentFacts, ConversationSession

log = structlog.get_logger(__name__)


class ResponseMode(Enum):
    PYTHON_DIRECT = auto()
    LLM_GENERATE = auto()


@dataclass
class PolicyDecision:
    mode: ResponseMode
    direct_text: Optional[str] = None
    llm_prompt: Optional[str] = None
    next_stage: Optional[str] = None
    intent: str = "continue_qualification"


_Q_LOCATION = "మీరు ఏ ప్రాంతంలో ఇల్లు వెతుకుతున్నారు? ఉదాహరణకు విజయవాడ, గుంటూరు లేదా హైదరాబాద్?"
_Q_BUDGET = "మీ బడ్జెట్ ఎంత? ఉదాహరణకు 50 లక్షలు లేదా 1 కోటి."
_Q_PROPERTY_TYPE = "మీకు ఎలాంటి ప్రాపర్టీ కావాలి — ఫ్లాట్, విల్లా, ఇండిపెండెంట్ హౌస్ లేదా ప్లాట్?"
_Q_BEDROOMS = "మీకు ఎన్ని బెడ్‌రూమ్‌లు కావాలి? 2BHK లేదా 3BHK?"
_Q_TIMELINE = "మీరు ఎప్పటిలోపు ప్రాపర్టీ కొనాలనుకుంటున్నారు?"
_Q_CONSENT = "మా ఏజెంట్ మీకు మళ్ళీ కాల్ చేసి వివరాలు అందించవచ్చా?"
_Q_CLARIFY = {
    "location": "క్షమించండి, మీరు చెప్పిన ప్రాంతం సరిగ్గా అర్థం కాలేదు. మరోసారి చెప్పగలరా?",
    "budget_rupees": "క్షమించండి, మీ బడ్జెట్ సరిగ్గా అర్థం కాలేదు. మరోసారి చెప్పగలరా?",
    "property_type": "మీకు ఫ్లాట్, విల్లా లేదా ప్లాట్ — వీటిలో ఏది కావాలి?",
    "bedrooms": "మీకు ఎన్ని బెడ్‌రూమ్‌లు కావాలో మరోసారి చెప్పగలరా?",
    "timeline": "మీరు ఎప్పటిలోపు కొనాలనుకుంటున్నారో చెప్పగలరా?",
    "consent": "మా ఏజెంట్ మిమ్మల్ని సంప్రదించడానికి అంగీకరిస్తారా?",
}
_QUESTIONS = {
    "location": _Q_LOCATION,
    "budget_rupees": _Q_BUDGET,
    "property_type": _Q_PROPERTY_TYPE,
    "bedrooms": _Q_BEDROOMS,
    "timeline": _Q_TIMELINE,
    "consent": _Q_CONSENT,
}
_GOODBYE_OPT_OUT = "అర్థమైంది. ఇకపై మిమ్మల్ని సంప్రదించము. మీ సమయానికి ధన్యవాదాలు. నమస్కారం."
_GOODBYE_NOT_INTERESTED = "అర్థమైంది. మీ సమయానికి ధన్యవాదాలు. నమస్కారం."
_CONFIRM_HANDOFF = "మీరు పంచుకున్న వివరాలకు ధన్యవాదాలు. నమస్కారం."

# A bare 'వద్దు' can refuse one question; only explicit contact-stop language ends a call.
_OPT_OUT_RE = re.compile(
    r"\b(?:stop\s+calling|do\s+not\s+call|don't\s+call|unsubscribe|remove\s+me)\b"
    r"|కాల్\s*చేయకండి|మళ్ళీ\s*కాల్\s*చేయకండి|సంప్రదించకండి",
    re.IGNORECASE,
)
_QUESTION_MARKERS = (
    "ఏమిటి", "ఏంటి", "ఏది", "ఎక్కడ", "ఎలా", "ఎందుకు", "ఉన్నాయా", "కావాలా",
    "what", "which", "where", "how", "why", "who", "do you", "is there",
    "are there", "can you", "could you", "ప్రశ్న", "ఎలాంటి", "ఏ రకమైన",
    "ఎంత", "ఎప్పుడు", "ఏమైనా", "మీ దగ్గర", "details", "వివరాలు",
)
_DOUBT_MARKERS = (
    "నాకు ఒక డౌట్ ఉంది", "నాకు డౌట్ ఉంది", "ఒక డౌట్ ఉంది", "డౌట్ ఉంది",
    "నాకు ఒక సందేహం ఉంది", "నాకు సందేహం ఉంది", "ఒక సందేహం ఉంది", "సందేహం ఉంది",
    "నాకు అర్థం కాలేదు", "అర్థం కాలేదు", "వివరంగా చెప్పండి", "మరోసారి వివరించండి",
    "i have a doubt", "i have a question", "i don't understand", "i do not understand",
    "can you clarify", "please explain", "explain that",
)
_UNDECIDED_MARKERS = (
    "ఇంకా నిర్ణయించుకోలేదు", "నిర్ణయం తీసుకోలేదు", "నిర్ణయించలేదు", "decide కాలేదు", "తెలియదు", "not decided",
    "haven't decided", "have not decided", "not sure", "undecided", "budget flexible", "flexible", "ఏదైనా అయినా పర్వాలేదు",
)
_DEFER_MARKERS = (
    "తర్వాత చెప్తాను", "తర్వాత చెప్తా", "తర్వాత మాట్లాడదాం", "తర్వాత",
    "later", "tell you later", "another time",
)
_REFUSAL_MARKERS = (
    "చెప్పను", "చెప్పడం ఇష్టం లేదు", "షేర్ చేయను", "పంచుకోను",
    "వద్దు", "don't want to share", "prefer not to say", "won't share",
)
_REPEAT_MARKERS = (
    "మళ్ళీ చెప్పండి", "మరోసారి చెప్పండి", "repeat", "say that again", "పునరావృతం",
)


class PolicySelector:
    """Select safe deterministic actions or delegate conversational turns to the LLM."""

    FACT_COLLECTION_ORDER = [
        "location", "budget_rupees", "property_type", "bedrooms", "timeline", "consent"
    ]

    def select(self, session: ConversationSession, extraction) -> PolicyDecision:
        self._ensure_policy_state(session)
        user_text = session.recent_turns[-1].normalized_transcript.strip() if session.recent_turns else ""
        lower = user_text.lower()
        pending = session.pending_field

        if _OPT_OUT_RE.search(user_text):
            session.update_fact("opted_out", True, raw_text=user_text)
            session.update_stage("ended")
            return PolicyDecision(ResponseMode.PYTHON_DIRECT, _GOODBYE_OPT_OUT, next_stage="ended", intent="global_opt_out")

        if any(marker in lower for marker in ("not interested", "ఆసక్తి లేదు", "ఇష్టం లేదు")):
            session.update_stage("ended")
            return PolicyDecision(ResponseMode.PYTHON_DIRECT, _GOODBYE_NOT_INTERESTED, next_stage="ended", intent="customer_not_interested")

        self._apply_extraction(session, extraction, user_text)
        next_field = self._next_missing_field(session.facts, session)

        # If the caller signals confusion but has not stated the question, invite them
        # to explain while preserving the active qualification slot.
        if any(marker in lower for marker in _DOUBT_MARKERS) and not self._looks_like_question(lower):
            # If the same utterance also supplied the pending fact, advance the slot;
            # otherwise keep asking for the caller's clarification.
            doubt_pending = next_field if self._has_extracted_fact(extraction, pending) else pending
            session.pending_field = doubt_pending
            session.next_missing_field = doubt_pending
            return PolicyDecision(
                ResponseMode.PYTHON_DIRECT,
                "తప్పకుండా, మీకు ఏ విషయం గురించి సందేహం ఉందో చెప్పండి. నేను వివరిస్తాను.",
                next_stage=doubt_pending or session.stage,
                intent="customer_has_doubt",
            )

        # Save facts first, but answer a simultaneous customer question before continuing.
        if self._looks_like_question(lower):
            session.pending_field = next_field
            session.next_missing_field = next_field
            return PolicyDecision(
                ResponseMode.LLM_GENERATE,
                llm_prompt=self._build_llm_prompt(session, user_text),
                next_stage=next_field or session.stage,
                intent="customer_question",
            )

        if any(marker in lower for marker in _REPEAT_MARKERS):
            field = pending or next_field
            if field:
                session.pending_field = field
                return PolicyDecision(
                    ResponseMode.PYTHON_DIRECT, _QUESTIONS.get(field, _Q_CLARIFY.get(field)),
                    next_stage=field, intent="repeat_question",
                )

        if self._has_extracted_fact(extraction, pending):
            return self._ask_next_or_finish(
                session, self._next_missing_field(session.facts, session), intent="fact_answered"
            )

        if pending and any(marker in lower for marker in _UNDECIDED_MARKERS):
            session.deferred_fields.add(pending)
            session.pending_field = None
            return self._acknowledge_and_continue(
                session, self._next_missing_field(session.facts, session), "customer_undecided"
            )

        if pending and any(marker in lower for marker in _DEFER_MARKERS):
            session.deferred_fields.add(pending)
            session.pending_field = None
            return self._acknowledge_and_continue(
                session, self._next_missing_field(session.facts, session), "customer_deferred"
            )

        if pending and any(marker in lower for marker in _REFUSAL_MARKERS):
            session.refused_fields.add(pending)
            session.pending_field = None
            return self._acknowledge_and_continue(
                session, self._next_missing_field(session.facts, session), "customer_refused"
            )

        if pending:
            return PolicyDecision(
                ResponseMode.PYTHON_DIRECT,
                _Q_CLARIFY.get(pending, "క్షమించండి, మరోసారి చెప్పగలరా?"),
                next_stage=pending, intent="clarify_pending_field",
            )

        return self._ask_next_or_finish(
            session, self._next_missing_field(session.facts, session), intent="start_or_continue"
        )

    def _ensure_policy_state(self, session: ConversationSession) -> None:
        # Backward compatibility for older callers that set stage/next_missing_field.
        if getattr(session, "pending_field", None) is None:
            stage_aliases = {
                "location": "location", "budget": "budget_rupees",
                "property": "property_type", "property_type": "property_type",
                "bedrooms": "bedrooms", "timeline": "timeline", "consent": "consent",
            }
            candidate = getattr(session, "next_missing_field", None) or stage_aliases.get(session.stage)
            session.pending_field = candidate if candidate in self.FACT_COLLECTION_ORDER else None
        if not hasattr(session, "deferred_fields"):
            session.deferred_fields = set()
        if not hasattr(session, "refused_fields"):
            session.refused_fields = set()

    def _apply_extraction(self, session, extraction, raw_text: str) -> None:
        for name in ("location", "budget_rupees", "property_type", "bedrooms", "timeline"):
            value = getattr(extraction, name, None)
            if value is not None:
                session.update_fact(name, value, raw_text=raw_text)
        # A bare 'no' while asking for budget is not consent. Only interpret yes/no
        # as contact consent when the consent question is the active pending slot.
        if session.pending_field == "consent" and extraction.consent is not None:
            session.update_fact("consent", extraction.consent, raw_text=raw_text)

    def _has_extracted_fact(self, extraction, pending: Optional[str] = None) -> bool:
        names = ("location", "budget_rupees", "property_type", "bedrooms", "timeline")
        if pending == "consent":
            names += ("consent",)
        return any(getattr(extraction, name, None) is not None for name in names)

    def _next_missing_field(self, facts: AgentFacts, session) -> Optional[str]:
        skipped = session.deferred_fields | session.refused_fields
        for name in self.FACT_COLLECTION_ORDER:
            if getattr(facts, name, None) is None and name not in skipped:
                return name
        return None

    def _ask_next_or_finish(self, session, next_field: Optional[str], intent: str) -> PolicyDecision:
        if next_field is None:
            session.pending_field = None
            session.next_missing_field = None
            return PolicyDecision(
                ResponseMode.PYTHON_DIRECT, _CONFIRM_HANDOFF,
                next_stage="handoff", intent="qualification_complete",
            )
        session.pending_field = next_field
        session.next_missing_field = next_field
        return PolicyDecision(
            ResponseMode.PYTHON_DIRECT, _QUESTIONS[next_field],
            next_stage=next_field, intent=intent,
        )

    def _acknowledge_and_continue(self, session, next_field, intent) -> PolicyDecision:
        acknowledgements = {
            "customer_undecided": "పర్వాలేదు, మీరు నిర్ణయించుకున్న తర్వాత చెప్పవచ్చు.",
            "customer_deferred": "సరే, ఆ వివరాన్ని తర్వాత తెలుసుకుందాం.",
            "customer_refused": "సరే, ఆ సమాచారాన్ని పంచుకోవాల్సిన అవసరం లేదు.",
        }
        if next_field is None:
            return PolicyDecision(
                ResponseMode.PYTHON_DIRECT, acknowledgements[intent] + " " + _CONFIRM_HANDOFF,
                next_stage="handoff", intent=intent,
            )
        session.pending_field = next_field
        session.next_missing_field = next_field
        return PolicyDecision(
            ResponseMode.PYTHON_DIRECT, acknowledgements[intent] + " " + _QUESTIONS[next_field],
            next_stage=next_field, intent=intent,
        )

    def _looks_like_question(self, text: str) -> bool:
        return "?" in text or any(marker in text for marker in _QUESTION_MARKERS)

    def _build_llm_prompt(self, session, user_text: str) -> str:
        facts = session.to_context_dict()["facts"]
        known = "\n".join(f"- {key}: {value}" for key, value in facts.items() if value is not None)
        pending = session.pending_field or "none"
        return (
            "You are a Telugu-speaking real-estate lead qualification assistant. "
            "Answer the caller's latest question first, briefly and naturally in Telugu. "
            "Do not invent property listings, availability, prices, company policies, locations, or guarantees. "
            "No verified property inventory/database is connected in this Phase 1 agent; if asked for specific "
            "listings or live availability, say you cannot verify them and offer human-agent confirmation. "
            "Do not claim a callback, appointment, or booking is arranged. Do not change stored facts or mark "
            "qualification complete. After answering, if appropriate, gently return to the pending question. "
            "Ask at most one question.\n\n"
            f"Known customer facts:\n{known or '- none'}\n"
            f"Pending qualification field: {pending}"
        )
