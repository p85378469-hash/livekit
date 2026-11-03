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

from typing import Any
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
- Do not give property listings, prices, or recommendations — you are only qualifying the lead.
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

        # Messages 2+: What the caller said in recent turns
        for turn in session.recent_turns:
            messages.append({
                "role": "user",
                "content": turn.normalized_transcript,
            })

        log.debug(
            "context.built",
            turn_count=len(session.recent_turns),
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
        missing = [k for k, v in facts.items() if v is None]

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
        if ctx["next_missing_field"]:
            stage_section += f"\nNext question to ask: {ctx['next_missing_field']}"

        return SYSTEM_PROMPT + facts_section + stage_section
