# ==============================================================================
# pipeline/context.py — AI Prompt & Conversation History Builder
#
# IN SIMPLE WORDS:
# Whenever we need to ask the AI (OpenRouter or Groq) to speak to the caller,
# the AI needs to know:
# 1. Who it is: A friendly Telugu real estate agent.
# 2. What it knows so far: "Customer already told us they want Vijayawada for ₹50 Lakhs".
# 3. What it still needs to find out: "We still need to ask how many bedrooms (BHK)".
# 4. The recent chat history: What the customer said in the last 2-3 turns.
#
# This file stitches all those pieces together into a clean prompt list
# (messages array: system instruction + user chat turns) before sending it to the AI.
# ==============================================================================

from __future__ import annotations

from typing import Any, Optional
import structlog

from core.session import ConversationSession

log = structlog.get_logger(__name__)

# ------------------------------------------------------------------------------
# The Agent's Persona & Rules (System Prompt)
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Telugu-speaking real-estate lead qualification agent for a real-estate company.

Your job:
1. Greet the caller warmly (greeting is handled separately — do not repeat it).
2. Collect exactly these facts in order:
   - Location (city or area in Andhra Pradesh/Telangana)
   - Budget (in lakhs or crores)
   - Property type (flat, villa, independent house, plot)
   - Number of bedrooms (BHK)
   - Purchase timeline
   - Consent to be contacted by an agent

Rules:
- Always respond in Telugu. You may use English words if the caller uses them.
- Ask only ONE question per response.
- Never guess or invent information the caller did not provide.
- If the caller's answer is unclear, ask for clarification politely.
- If the caller wants to end the call or says they're not interested, say goodbye gracefully.
- You may answer general questions about the qualification process.
- No verified property inventory is connected in this Phase 1 agent. Never invent listings, prices, availability, locations, or recommendations. If asked for specific properties or live availability, say you cannot verify them and offer human-agent confirmation.
- Answer the caller's question before returning to qualification. Do not repeatedly ask the pending question while a customer question remains unanswered.
- Never claim a callback, appointment, or booking is arranged unless the application confirms it.
- Never change structured customer facts or qualification status yourself; Python owns session state.
- Keep responses short — no more than 2 sentences.
"""


# ------------------------------------------------------------------------------
# ContextBuilder Class
# ------------------------------------------------------------------------------
class ContextBuilder:
    """
    Constructs the list of messages in OpenAI format (role: system / role: user)
    so OpenRouter / Groq knows the exact context before generating a response.
    """

    def build_system_prompt(self, session: Optional[ConversationSession] = None) -> str:
        """
        Builds the base system prompt. If session is provided, includes collected facts.
        """
        if session is not None:
            return self._build_system_with_facts(session)
        return SYSTEM_PROMPT

    def build_messages(self, session: ConversationSession) -> list[dict[str, str]]:
        """
        Builds the conversation history array for the AI call.
        
        session: The active call notepad containing facts and recent turns.
        """
        # Message 1: System prompt with personality, rules, and current facts
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_with_facts(session)},
        ]

        # Only the latest caller turn is sent; structured facts carry older information.
        latest_turns = session.recent_turns[-1:]
        for turn in latest_turns:
            messages.append({
                "role": "user",
                "content": turn.normalized_transcript,
            })

        log.debug(
            "context.built",
            turn_count=len(latest_turns),
            session_id=session.session_id,
        )
        return messages

    def _build_system_with_facts(self, session: ConversationSession) -> str:
        """
        Appends the customer's known facts and what question is next
        directly into the system prompt.
        """
        ctx = session.to_context_dict()
        facts = ctx["facts"]

        collected = {k: v for k, v in facts.items() if v is not None}
        skipped = set(getattr(session, "deferred_fields", set())) | set(getattr(session, "refused_fields", set()))
        missing = [k for k, v in facts.items() if v is None and k not in skipped]

        facts_section = ""
        if collected:
            facts_section = "\n\nFacts already collected:\n" + "\n".join(
                f"  {k}: {v}" for k, v in collected.items()
            )

        if missing:
            facts_section += "\n\nStill needed:\n" + "\n".join(
                f"  - {f}" for f in missing
            )

        stage_section = f"\n\nCurrent stage: {ctx['stage']}"
        pending = getattr(session, "pending_field", None)
        stage_section += f"\nPending qualification field: {pending or 'none'}"
        deferred = sorted(getattr(session, "deferred_fields", set()))
        refused = sorted(getattr(session, "refused_fields", set()))
        if deferred:
            stage_section += f"\nDeferred fields (do not immediately ask again): {deferred}"
        if refused:
            stage_section += f"\nFields caller declined (do not pressure): {refused}"
        if ctx["next_missing_field"]:
            stage_section += f"\nNext question to ask: {ctx['next_missing_field']}"

        return SYSTEM_PROMPT + facts_section + stage_section


def retain_latest_user_turn(items: list[Any], latest_message: Any) -> list[Any]:
    """Keep instruction messages and only the current caller message for the LLM."""
    instructions = [
        item for item in items
        if getattr(item, "role", None) in {"system", "developer"}
    ]
    return instructions + [latest_message]
