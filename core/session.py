# ==============================================================================
# core/session.py — Memory Notebook for the Entire Call
#
# IN SIMPLE WORDS:
# Think of this file as the agent's "notepad" during the call.
# As the customer speaks, the agent writes down:
# - Where they want to buy (e.g., Vijayawada, Hyderabad).
# - How much they want to spend (e.g., ₹50 Lakhs, ₹1.5 Crores).
# - What type of property (e.g., 2BHK flat, Villa).
# - Any corrections (e.g. if the caller said "Vijayawada", but then changed to "Guntur").
# - The "Turn ID" counter: increases every time the user speaks, so we never play
#   an old/expired answer by mistake.
# ==============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
import structlog

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------------------
# CorrectionEntry — Records when the customer changes their mind
# ------------------------------------------------------------------------------
@dataclass
class CorrectionEntry:
    """
    Stores an audit log entry whenever a customer updates or corrects a previous answer.
    
    Example:
    If the caller first says "Vijayawada" and later says "No, Guntur", this stores:
    - field: "location"
    - old_value: "Vijayawada"
    - new_value: "Guntur"
    - raw_text: "Vijayawada kaadu, Guntur"
    """
    turn_id: int             # Which conversation turn this change occurred on
    field: str               # The name of the fact changed (e.g. "location", "budget")
    old_value: object        # The previous value before the correction
    new_value: object        # The updated value the customer gave
    raw_text: str            # The exact words the customer said when correcting
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)  # When it happened


# ------------------------------------------------------------------------------
# TurnRecord — Stores one spoken exchange
# ------------------------------------------------------------------------------
@dataclass
class TurnRecord:
    """
    Saves what the caller said during one turn of conversation.
    
    - raw_transcript: The original words exactly as heard (never modified).
    - normalized_transcript: A clean copy used for extracting numbers & locations.
    """
    turn_id: int                 # Sequential turn number (1, 2, 3...)
    raw_transcript: str          # Immutable original text (e.g. "నేను విజయవాడ లో 50 lakhs budget")
    normalized_transcript: str   # Cleaned text (e.g. "నేను విజయవాడ లో 50 lakhs budget")
    timestamp_start_ms: float    # Clock time when user started speaking
    timestamp_end_ms: float      # Clock time when user stopped speaking


# ------------------------------------------------------------------------------
# AgentFacts — The customer's real estate requirements
# ------------------------------------------------------------------------------
@dataclass
class AgentFacts:
    """
    The extracted lead qualification information collected from the customer.
    All values start as None until the customer mentions them.
    """
    location: Optional[str] = None           # City/Area name (e.g. "Vijayawada", "Guntur")
    budget_rupees: Optional[float] = None    # Customer budget in Indian Rupees (e.g. 5000000 for 50 Lakhs)
    property_type: Optional[str] = None      # Type: "flat", "villa", "plot", "independent_house"
    bedrooms: Optional[int] = None           # Number of bedrooms (e.g. 2 for 2BHK, 3 for 3BHK)
    timeline: Optional[str] = None           # When they plan to buy (e.g. "immediately", "3 months")
    preferences: list[str] = field(default_factory=list)  # Extra wishes (e.g. "near school", "gated community")
    consent: Optional[bool] = None           # Did the user agree to receive property details on WhatsApp?
    opted_out: bool = False                  # True if caller said "stop calling" or "not interested"


# ------------------------------------------------------------------------------
# ConversationSession — The Master Call State
# ------------------------------------------------------------------------------
class ConversationSession:
    """
    The single memory center for the entire call session.
    All changes to caller information or turn counters happen through this class.
    """

    # We only keep the last 6 turns in active prompt context to prevent LLM memory overload
    RECENT_TURNS_WINDOW = 6

    def __init__(self, session_id: str) -> None:
        """
        Sets up a fresh notepad for a new call.
        session_id: A unique code for this call.
        """
        self.session_id = session_id
        self.session_start_ms: float = time.time() * 1000

        # Turn ID: Starts at 0 (greeting) and increments every time the user speaks
        self._current_turn_id: int = 0

        # Holds customer facts (budget, location, etc.)
        self.facts = AgentFacts()

        # Audit trail of any corrections made by the customer
        self.corrections: list[CorrectionEntry] = []

        # Last 6 conversation turns (rolling window)
        self._recent_turns: list[TurnRecord] = []

        # Complete history of all turns in this call (for debugging)
        self._all_turns: list[TurnRecord] = []

        # Current question stage:
        # Flow: greeting -> location -> budget -> property -> confirm -> handoff
        self.stage: str = "greeting"
        self.next_missing_field: Optional[str] = None
        self.pending_field: Optional[str] = None
        self.deferred_fields: set[str] = set()
        self.refused_fields: set[str] = set()

        log.info("session.created", session_id=session_id)

    # --------------------------------------------------------------------------
    # Turn Number Management (Prevents Playing Old Audio)
    # --------------------------------------------------------------------------

    @property
    def current_turn_id(self) -> int:
        """Returns the current turn number."""
        return self._current_turn_id

    def advance_turn(self) -> int:
        """
        Increases the turn number by 1 when the user finishes speaking.
        Any background audio generation from an older turn immediately becomes invalid.
        """
        self._current_turn_id += 1
        log.debug("session.turn_advanced", turn_id=self._current_turn_id, session_id=self.session_id)
        return self._current_turn_id

    def is_turn_current(self, turn_id: int) -> bool:
        """
        Checks if an audio chunk or answer belongs to the CURRENT active turn.
        Returns False if the caller has already started a newer turn.
        """
        return turn_id == self._current_turn_id

    # --------------------------------------------------------------------------
    # Storing What the Caller Said
    # --------------------------------------------------------------------------

    def commit_turn(
        self,
        raw_transcript: str,
        normalized_transcript: str,
        timestamp_start_ms: float,
        timestamp_end_ms: float,
    ) -> TurnRecord:
        """
        Saves a completed turn into call memory.
        The raw transcript is permanently preserved as heard.
        """
        record = TurnRecord(
            turn_id=self._current_turn_id,
            raw_transcript=raw_transcript,
            normalized_transcript=normalized_transcript,
            timestamp_start_ms=timestamp_start_ms,
            timestamp_end_ms=timestamp_end_ms,
        )
        self._all_turns.append(record)

        # Keep rolling window to max 6 items
        self._recent_turns.append(record)
        if len(self._recent_turns) > self.RECENT_TURNS_WINDOW:
            self._recent_turns.pop(0)

        log.info(
            "session.turn_committed",
            turn_id=record.turn_id,
            raw_len=len(raw_transcript),
            session_id=self.session_id,
        )
        return record

    @property
    def recent_turns(self) -> list[TurnRecord]:
        """Returns the last few turns (safe to send to AI prompt without exceeding limits)."""
        return list(self._recent_turns)

    # --------------------------------------------------------------------------
    # Updating Facts & Recording Corrections
    # --------------------------------------------------------------------------

    def update_fact(self, field: str, new_value: object, raw_text: str = "") -> None:
        """
        Updates a customer requirement (e.g. location = 'Guntur').
        If this field already had a value, it logs a correction so we can see what changed.
        """
        old_value = getattr(self.facts, field, None)

        # If changing an existing value, record the correction
        if old_value is not None and old_value != new_value:
            correction = CorrectionEntry(
                turn_id=self._current_turn_id,
                field=field,
                old_value=old_value,
                new_value=new_value,
                raw_text=raw_text,
            )
            self.corrections.append(correction)
            log.info(
                "session.fact_corrected",
                field=field,
                old_value=old_value,
                new_value=new_value,
                turn_id=self._current_turn_id,
            )

        setattr(self.facts, field, new_value)
        log.debug("session.fact_updated", field=field, value=new_value)

    def update_stage(self, new_stage: str, next_missing: Optional[str] = None) -> None:
        """Moves the conversation forward (e.g. from 'location' to 'budget')."""
        log.info(
            "session.stage_changed",
            from_stage=self.stage,
            to_stage=new_stage,
            next_missing=next_missing,
        )
        self.stage = new_stage
        self.next_missing_field = next_missing

    # --------------------------------------------------------------------------
    # Formats memory for the AI Prompt
    # --------------------------------------------------------------------------

    def to_context_dict(self) -> dict:
        """Packages the customer facts and recent turns into a dictionary for the AI prompt."""
        return {
            "facts": {
                "location": self.facts.location,
                "budget_rupees": self.facts.budget_rupees,
                "property_type": self.facts.property_type,
                "bedrooms": self.facts.bedrooms,
                "timeline": self.facts.timeline,
                "preferences": self.facts.preferences,
                "consent": self.facts.consent,
                "opted_out": self.facts.opted_out,
            },
            "stage": self.stage,
            "next_missing_field": self.next_missing_field,
            "pending_field": self.pending_field,
            "deferred_fields": sorted(self.deferred_fields),
            "refused_fields": sorted(self.refused_fields),
            "recent_turns": [
                {
                    "turn_id": t.turn_id,
                    "raw": t.raw_transcript,
                    "normalized": t.normalized_transcript,
                }
                for t in self._recent_turns
            ],
        }

    def debug_dump(self) -> dict:
        """Creates a complete snapshot of everything in memory (used for tests and debugging)."""
        return {
            "session_id": self.session_id,
            "current_turn_id": self._current_turn_id,
            "stage": self.stage,
            "next_missing_field": self.next_missing_field,
            "pending_field": self.pending_field,
            "deferred_fields": sorted(self.deferred_fields),
            "refused_fields": sorted(self.refused_fields),
            "facts": self.facts.__dict__,
            "corrections": [c.__dict__ for c in self.corrections],
            "recent_turns_count": len(self._recent_turns),
            "total_turns": len(self._all_turns),
        }
