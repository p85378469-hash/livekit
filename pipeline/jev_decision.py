"""Experimental Jev conversation-intent classifier.

Jev is used only to classify a completed caller turn. It does not generate speech,
extract/store customer facts, or control session state. Any failure returns None so
the existing deterministic PolicySelector remains the fallback.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib.request import Request, urlopen

import structlog

log = structlog.get_logger(__name__)

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.14"

INTENT_CRITERIA = {
    "answer": "The caller is answering the currently pending qualification question or providing requested information.",
    "question": "The caller is asking a substantive question, including property availability, process, price, location, or asking for information.",
    "doubt_without_question": "The caller only signals confusion or says they have a doubt, but has not yet stated the actual question.",
    "correction": "The caller corrects, replaces, or updates something they said earlier.",
    "defer": "The caller wants to answer later, is undecided, or is not ready to provide this information now.",
    "refusal": "The caller declines to answer the current qualification question but does not ask to end all contact.",
    "repeat": "The caller asks the agent to repeat or say the previous question/information again.",
    "unclear": "The caller's turn is incomplete, ambiguous, unrelated, or cannot safely be assigned to another category.",
    "not_interested": "The caller clearly says they are not interested in the property enquiry or sales conversation.",
    "other": "The caller says something conversational that does not fit the other categories.",
}


@dataclass(frozen=True)
class JevDecision:
    intent: str
    confidence: Optional[float]
    probabilities: dict[str, float]
    model: str


class JevDecisionClient:
    """Small async wrapper around OpenRouter's Jev Decisions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.model = (model or os.getenv("JEV_MODEL") or DEFAULT_MODEL).strip()
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None
            else os.getenv("JEV_TIMEOUT_SECONDS", "1.5")
        )
        self.min_confidence = float(
            min_confidence if min_confidence is not None
            else os.getenv("JEV_MIN_CONFIDENCE", "0.55")
        )

    @property
    def enabled(self) -> bool:
        return os.getenv("JEV_DECISION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"} and bool(self.api_key)

    async def classify(self, state: dict[str, Any]) -> Optional[JevDecision]:
        """Classify one completed caller turn; return None on any failure/low confidence."""
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "caller_intent": {
                    "type": "choice",
                    "instructions": (
                        "Classify the caller's LATEST utterance in the context of the conversation. "
                        "Use the pending field to distinguish an answer from an unrelated number or phrase. "
                        "A question about available properties is a question, not a property preference. "
                        "A correction must be distinguished from a new answer. A bare doubt signal without "
                        "an actual question is doubt_without_question. A refusal of one field is not a global "
                        "request to stop contact. Choose unclear rather than guessing."
                    ),
                    "criteria": INTENT_CRITERIA,
                }
            },
        }

        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(self._post_decision, payload),
                timeout=self.timeout_seconds,
            )
            decision = self._parse_response(raw)
            if decision.confidence is not None and decision.confidence < self.min_confidence:
                log.warning(
                    "jev.low_confidence_fallback",
                    intent=decision.intent,
                    confidence=decision.confidence,
                    threshold=self.min_confidence,
                )
                return None
            log.info(
                "jev.decision",
                intent=decision.intent,
                confidence=decision.confidence,
                model=decision.model,
            )
            return decision
        except asyncio.TimeoutError:
            log.warning("jev.timeout_fallback", timeout_seconds=self.timeout_seconds)
        except Exception as exc:
            log.warning("jev.error_fallback", error=str(exc), model=self.model)
        return None

    def _post_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            DECISIONS_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> JevDecision:
        """Validate the typed answer strictly; malformed responses trigger fallback."""
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("Jev response is missing answers")
        answer = answers.get("caller_intent")
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Jev response has no caller_intent choice answer")

        intent = answer.get("choice")
        if intent not in INTENT_CRITERIA:
            raise ValueError(f"Jev returned an unsupported intent: {intent!r}")

        confidence_value = answer.get("confidence")
        if confidence_value is None:
            raise ValueError("Jev choice answer is missing confidence")
        confidence = float(confidence_value)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Jev confidence must be a finite probability between 0 and 1")
        raw_probabilities = answer.get("probabilities")
        if not isinstance(raw_probabilities, dict) or not raw_probabilities:
            raise ValueError("Jev choice answer is missing probabilities")
        if not isinstance(raw_probabilities, dict):
            raise ValueError("Jev probabilities must be an object")
        probabilities = {
            str(key): float(value)
            for key, value in raw_probabilities.items()
        }
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities.values()):
            raise ValueError("Jev probabilities must be finite values between 0 and 1")
        model = str(response.get("model") or DEFAULT_MODEL)
        return JevDecision(
            intent=intent,
            confidence=confidence,
            probabilities=probabilities,
            model=model,
        )


def build_jev_state(session: Any, raw_transcript: str) -> dict[str, Any]:
    """Build a compact, auditable context for Jev without asking it to extract facts."""
    facts = session.to_context_dict().get("facts", {})
    recent_turns = [
        {"role": "caller", "text": turn.raw_transcript}
        for turn in session.recent_turns[-4:-1]
    ]
    return {
        "task": "Choose the caller's conversational intent for a Telugu real-estate qualification call.",
        "latest_caller_utterance": raw_transcript,
        "pending_qualification_field": session.pending_field or session.next_missing_field,
        "current_stage": session.stage,
        "known_customer_facts": facts,
        "recent_caller_turns": recent_turns,
        "important_rules": [
            "Classify intent only. Do not extract or invent customer facts.",
            "A property availability question is not a property-type preference.",
            "A bare refusal of one question is not a global opt-out.",
            "Only explicit contact-stop language is a global opt-out; Python enforces it separately.",
            "When uncertain, choose unclear.",
        ],
    }
