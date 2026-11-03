# ==============================================================================
# pipeline/policy.py — The Decision Maker (What should the agent say next?)
#
# IN SIMPLE WORDS:
# When the customer answers a question, this file decides what to say in response:
# 1. Did the customer say "stop calling" / "not interested"?
#    -> Say polite goodbye and hang up.
# 2. Did we fail to understand the budget or location?
#    -> Politely ask them to clarify (e.g. "Sorry, could you repeat the budget?").
# 3. What piece of information are we still missing?
#    - Do we need Location? -> Ask: "మీరు ఏ ప్రాంతంలో ఇల్లు వెతుకుతున్నారు?"
#    - Do we need Budget?   -> Ask: "మీ బడ్జెట్ ఎంత? (50 లక్షలు, 1 కోటి?)"
#    - Do we need Property? -> Ask: "ఫ్లాట్ లేదా విల్లా?"
#    - Do we have everything? -> Say: "ధన్యవాదాలు! మా ఏజెంట్ మిమ్మల్ని సంప్రదిస్తారు."
#
# Fast Response Trick (PYTHON_DIRECT vs LLM_GENERATE):
# For standard step-by-step questions, we reply instantly with Python templates
# without waiting for OpenRouter/Groq!
# We only call the AI when the user asks an unexpected, open-ended question.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import structlog

from core.session import ConversationSession, AgentFacts

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------------------
# ResponseMode — How we create the answer
# ------------------------------------------------------------------------------
class ResponseMode(Enum):
    PYTHON_DIRECT = auto()   # Fixed template (zero delay, instant Telugu audio)
    LLM_GENERATE  = auto()   # Send to OpenRouter / Groq AI (for smart custom answers)


# ------------------------------------------------------------------------------
# PolicyDecision — The Result of the Decision Maker
# ------------------------------------------------------------------------------
@dataclass
class PolicyDecision:
    mode: ResponseMode                  # PYTHON_DIRECT or LLM_GENERATE
    direct_text: Optional[str] = None   # The Telugu text to speak (if PYTHON_DIRECT)
    llm_prompt: Optional[str] = None    # The prompt to send to the AI (if LLM_GENERATE)
    next_stage: Optional[str] = None    # The next step in the conversation


# ------------------------------------------------------------------------------
# Standard Telugu Question Scripts
# ------------------------------------------------------------------------------
_Q_LOCATION = (
    "మీరు ఏ ప్రాంతంలో ఇల్లు వెతుకుతున్నారు? "
    "ఉదాహరణకు — విజయవాడ, గుంటూరు, హైదరాబాద్?"
)
_Q_BUDGET = (
    "మీ బడ్జెట్ ఎంత? లక్షలలో చెప్పండి — "
    "ఉదాహరణకు '50 లక్షలు' లేదా '1 కోటి'."
)
_Q_PROPERTY_TYPE = (
    "మీకు ఏ రకమైన ఆస్తి కావాలి? "
    "ఫ్లాట్, విల్లా, ఇండిపెండెంట్ హౌస్, లేదా ప్లాట్?"
)
_Q_BEDROOMS = "మీకు ఎన్ని బెడ్‌రూమ్‌లు కావాలి? 2BHK, 3BHK?"
_Q_TIMELINE  = "మీరు ఎప్పటి లోపు కొనాలనుకుంటున్నారు?"
_Q_CONSENT   = (
    "మా ఏజెంట్ మీకు మళ్ళీ కాల్ చేసి వివరాలు అందించవచ్చా?"
)
_Q_CLARIFY_BUDGET = (
    "క్షమించండి — బడ్జెట్ సరిగ్గా అర్థం కాలేదు. "
    "మళ్ళీ చెప్పగలరా? ఉదాహరణకు '50 లక్షలు' లేదా '1.2 కోటి'."
)
_Q_CLARIFY_LOCATION = (
    "మీరు చెప్పిన ప్రాంతం సరిగ్గా అర్థం కాలేదు. "
    "దయచేసి మళ్ళీ చెప్పగలరా?"
)

_GOODBYE_OPT_OUT = (
    "అర్థమైంది. మీ సమయానికి ధన్యవాదాలు. మళ్ళీ అవసరమైతే తప్పకుండా సంప్రదించండి. నమస్కారం."
)
_CONFIRM_HANDOFF = (
    "మీ వివరాలు నమోదు చేసాం. మా ఏజెంట్ త్వరలో మీకు సంప్రదిస్తారు. ధన్యవాదాలు!"
)


# ------------------------------------------------------------------------------
# PolicySelector Class
# ------------------------------------------------------------------------------
class PolicySelector:
    """
    Evaluates current call state and selects the next question or response.
    """

    # The step-by-step order in which we collect information from the customer
    FACT_COLLECTION_ORDER = [
        "location",
        "budget_rupees",
        "property_type",
        "bedrooms",
        "timeline",
        "consent",
    ]

    def select(
        self,
        session: ConversationSession,
        extraction,
    ) -> PolicyDecision:
        """
        Calculates the best next move for this turn.
        
        session: Current call state and notepad.
        extraction: Facts extracted from the caller's last sentence.
        """
        facts = session.facts

        # Rule 1: Opt-Out Check (Caller said "Stop calling me" or "Not interested")
        if extraction.opted_out or facts.opted_out:
            log.info("policy.opted_out")
            return PolicyDecision(
                mode=ResponseMode.PYTHON_DIRECT,
                direct_text=_GOODBYE_OPT_OUT,
                next_stage="ended",
            )

        # Rule 2: Save any newly mentioned facts to session
        self._apply_extraction(session, extraction)

        # Rule 2b: Question Detection (Did the caller ask an open-ended question?)
        # If the caller is asking something (e.g., "Which areas are good?", "Do you have flats?"),
        # we delegate to OpenRouter LLM to answer naturally rather than rigidly pushing the script.
        user_query = ""
        if session.recent_turns:
            user_query = session.recent_turns[-1].normalized_transcript.lower()

        is_question = (
            "?" in user_query
            or any(q in user_query for q in [
                "ఏమిటి", "ఏది", "ఎక్కడ", "ఎలా", "ఎందుకు", "ఉన్నాయా", "చెప్పండి",
                "what", "which", "where", "how", "why", "who", "tell me", "explain", "is there", "are there"
            ])
        )

        if is_question:
            log.info("policy.customer_question_detected", query=user_query[:50])
            return PolicyDecision(
                mode=ResponseMode.LLM_GENERATE,
                llm_prompt=self._build_llm_prompt(session),
                next_stage=session.stage,
            )

        # Rule 3: Clarify an unrecognized answer to the field currently being collected.
        # Recent turns contain caller transcripts, not agent speech; session.stage
        # is the reliable indicator of which qualification field is pending.
        if extraction.budget_rupees is None and session.stage == "budget":
            log.info("policy.clarify_budget")
            return PolicyDecision(
                mode=ResponseMode.PYTHON_DIRECT,
                direct_text=_Q_CLARIFY_BUDGET,
                next_stage="budget",
            )

        if extraction.location is None and session.stage == "location":
            log.info("policy.clarify_location")
            return PolicyDecision(
                mode=ResponseMode.PYTHON_DIRECT,
                direct_text=_Q_CLARIFY_LOCATION,
                next_stage="location",
            )

        # Rule 4: Find the next missing question in our checklist
        next_field = self._next_missing_field(facts)

        if next_field is None:
            # All 6 questions answered! Wrap up and thank the customer
            log.info("policy.all_facts_collected")
            return PolicyDecision(
                mode=ResponseMode.PYTHON_DIRECT,
                direct_text=_CONFIRM_HANDOFF,
                next_stage="handoff",
            )

        # Rule 5: Ask the next question using our fixed template (fastest path)
        question = self._question_for_field(next_field)
        if question:
            log.info("policy.next_question", field=next_field)
            return PolicyDecision(
                mode=ResponseMode.PYTHON_DIRECT,
                direct_text=question,
                next_stage=next_field,
            )

        # Rule 6: Fallback to AI (OpenRouter / Groq) if the conversation went off-script
        log.info("policy.llm_fallback", stage=session.stage)
        return PolicyDecision(
            mode=ResponseMode.LLM_GENERATE,
            llm_prompt=self._build_llm_prompt(session),
            next_stage=session.stage,
        )

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _apply_extraction(
        self,
        session: ConversationSession,
        extraction,
    ) -> None:
        """Saves newly extracted values to the session state."""
        if extraction.location is not None:
            session.update_fact("location", extraction.location)
        if extraction.budget_rupees is not None:
            session.update_fact("budget_rupees", extraction.budget_rupees)
        if extraction.property_type is not None:
            session.update_fact("property_type", extraction.property_type)
        if extraction.bedrooms is not None:
            session.update_fact("bedrooms", extraction.bedrooms)
        if extraction.consent is not None:
            session.update_fact("consent", extraction.consent)
        if extraction.opted_out:
            session.update_fact("opted_out", True)

    def _next_missing_field(self, facts: AgentFacts) -> Optional[str]:
        """Scans the checklist to find the first fact that is still None."""
        for field in self.FACT_COLLECTION_ORDER:
            val = getattr(facts, field, None)
            if val is None:
                return field
        return None

    def _question_for_field(self, field: str) -> Optional[str]:
        """Returns the Telugu question text for a given field."""
        mapping = {
            "location":       _Q_LOCATION,
            "budget_rupees":  _Q_BUDGET,
            "property_type":  _Q_PROPERTY_TYPE,
            "bedrooms":       _Q_BEDROOMS,
            "timeline":       _Q_TIMELINE,
            "consent":        _Q_CONSENT,
        }
        return mapping.get(field)

    def _build_llm_prompt(self, session: ConversationSession) -> str:
        """Creates the prompt telling the AI who it is and what to say next."""
        ctx = session.to_context_dict()
        facts_str = "\n".join(
            f"  {k}: {v}" for k, v in ctx["facts"].items() if v is not None
        )
        recent_str = "\n".join(
            f"  Turn {t['turn_id']}: {t['normalized']}"
            for t in ctx["recent_turns"][-3:]
        )

        prompt = f"""You are a Telugu-speaking real-estate lead qualification agent.
Your goal is to collect: location, budget, property type, bedrooms, timeline, and consent.
Always respond in Telugu. Be polite and concise. Never ask more than one question at a time.

Current facts collected:
{facts_str or "  (none yet)"}

Recent conversation:
{recent_str or "  (none)"}

Current stage: {ctx["stage"]}
Next missing field: {ctx["next_missing_field"] or "all collected"}

Respond with a single natural Telugu sentence to continue the conversation."""
        return prompt
