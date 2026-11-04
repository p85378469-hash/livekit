"""Jev gate used only while the caller is answering the location question.

Jev classifies whether the latest caller turn actually supplies a location.
Python remains responsible for state, fact storage, and routing. If Jev is
unavailable, uncertain, or the location cannot be normalized, route to the
main conversational LLM rather than silently advancing qualification.
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
LOCATION_STATUSES = {"provided", "not_provided", "unclear"}


@dataclass(frozen=True)
class JevLocationResult:
    status: str
    confidence: float
    model: str


class JevLocationClient:
    """Async, timeout-bounded client for the OpenRouter Jev Decisions API."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = os.getenv("JEV_MODEL", DEFAULT_MODEL).strip()
        self.timeout_seconds = float(os.getenv("JEV_TIMEOUT_SECONDS", "1.5"))
        self.min_confidence = float(os.getenv("JEV_MIN_CONFIDENCE", "0.55"))

    @property
    def enabled(self) -> bool:
        return (
            os.getenv("JEV_LOCATION_ENABLED", "true").strip().lower()
            in {"1", "true", "yes", "on"}
            and bool(self.api_key)
        )

    async def classify(self, state: dict[str, Any]) -> Optional[JevLocationResult]:
        if not self.enabled:
            return None
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "location_status": {
                    "type": "choice",
                    "instructions": (
                        "Decide only whether the caller has actually PROVIDED a location "
                        "as their answer to the pending location question. 'provided' means "
                        "the caller states a city, town, locality, area, or landmark they "
                        "want. A location merely mentioned inside a question (for example, "
                        "'Do you have properties in Vijayawada?') is NOT a provided answer. "
                        "Questions about available areas, requests for examples, uncertainty, "
                        "or unrelated replies are not provided. If the meaning is ambiguous, "
                        "choose unclear. Classify using only the latest caller utterance; no prior context is supplied."
                    ),
                    "criteria": {
                        "provided": "Caller clearly states one or more desired locations.",
                        "not_provided": "Caller has not supplied a desired location; they ask a question, request information, or respond without a location.",
                        "unclear": "It is ambiguous whether the caller is giving a desired location.",
                    },
                }
            },
        }
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._post_decision, payload),
                timeout=self.timeout_seconds,
            )
            result = self._parse_response(response)
            if result.confidence < self.min_confidence:
                log.warning(
                    "jev.location_low_confidence",
                    status=result.status,
                    confidence=result.confidence,
                )
                return None
            log.info(
                "jev.location_decision",
                status=result.status,
                confidence=result.confidence,
                model=result.model,
            )
            return result
        except asyncio.TimeoutError:
            log.warning("jev.location_timeout", timeout_seconds=self.timeout_seconds)
        except Exception as exc:
            log.warning("jev.location_error_fallback", error=str(exc), model=self.model)
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
    def _parse_response(response: dict[str, Any]) -> JevLocationResult:
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("Jev response is missing answers")
        answer = answers.get("location_status")
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Jev response is missing location_status choice")
        status = answer.get("choice")
        if status not in LOCATION_STATUSES:
            raise ValueError(f"Unsupported Jev location status: {status!r}")
        confidence = float(answer.get("confidence"))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Jev confidence must be between 0 and 1")
        return JevLocationResult(
            status=status,
            confidence=confidence,
            model=str(response.get("model") or DEFAULT_MODEL),
        )


def build_jev_location_state(latest_transcript: str) -> dict[str, Any]:
    """Jev receives only the latest caller answer, never session facts or history."""
    return {"latest_caller_utterance": latest_transcript}
