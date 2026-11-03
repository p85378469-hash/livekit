# ==============================================================================
# pipeline/normalizer.py — The Text Cleaner & Fact Extractor
#
# IN SIMPLE WORDS:
# When a customer speaks in Telugu (or mixed Telugu-English / Tanglish), their words
# come out like: "నేను విజయవాడ లో 50 lakhs budget ఉంది" or "2bhk flat in vizag".
#
# This file does two big jobs:
# 1. Two-Buffer Safety:
#    It keeps the exact original Telugu text untouched (Raw Buffer).
#    It makes a separate cleaned copy for extracting information (Normalized Copy).
# 2. Fact Extraction:
#    - Understands Budgets: Converts "50 lakhs", "25 lacs", "1.5 crore",
#      "ఒక కోటి" (1 crore), or "ఇరవై లక్షలు" (20 lakhs) into exact Indian Rupees (₹).
#    - Understands Locations: Maps Telugu or casual city names like "విజయవాడ"
#      or "hyd" or "vizag" to standardized official names ("Vijayawada", "Hyderabad", "Visakhapatnam").
#    - Understands BHK: Extracts 2BHK, 3BHK, etc.
#    - Understands Opt-Outs: Catches phrases like "వద్దు" (no) or "stop calling me".
# ==============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
import structlog

from pipeline.number_parser import parse_indian_amount

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------------------
# ExtractionResult — Container for Extracted Information
# ------------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """
    Stores all the details extracted from one spoken sentence.
    Any field that wasn't mentioned remains None (we never guess!).
    """
    budget_rupees: Optional[float] = None     # Price in rupees (e.g. 5000000.0)
    location: Optional[str] = None            # Canonical city name (e.g. "Vijayawada")
    property_type: Optional[str] = None       # "flat", "villa", "plot", etc.
    bedrooms: Optional[int] = None            # 2, 3, 4 bedrooms
    timeline: Optional[str] = None             # Canonical purchase timeline, e.g. "3_months"
    opted_out: Optional[bool] = None          # True if caller said "don't call" or "not interested"
    consent: Optional[bool] = None            # True if caller agreed to WhatsApp / contact
    needs_clarification: list[str] = field(default_factory=list)  # Any vague fields


# ------------------------------------------------------------------------------
# Location Aliases — Telugu Script & Common Abbreviations
# ------------------------------------------------------------------------------
# Maps canonical official English city names to their Telugu script & casual nicknames
_LOCATION_ALIASES: dict[str, list[str]] = {
    "Vijayawada": ["vijayawada", "vijayavada", "vijywada", "విజయవాడ"],
    "Guntur":     ["guntur", "gunthur", "గుంటూరు"],
    "Hyderabad":  ["hyderabad", "hyd", "hyderbad", "హైదరాబాద్"],
    "Amaravati":  ["amaravati", "amravati", "అమరావతి"],
    "Tirupati":   ["tirupati", "tirupathi", "తిరుపతి"],
    "Visakhapatnam": ["visakhapatnam", "vizag", "vishakhapatnam", "విశాఖపట్నం"],
    "Rajahmundry": ["rajahmundry", "rajamahendravaram", "రాజమహేంద్రవరం"],
    "Nellore":    ["nellore", "నెల్లూరు"],
    "Kurnool":    ["kurnool", "కర్నూలు"],
}

# Inverted dictionary so looking up "vizag" immediately returns "Visakhapatnam"
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _LOCATION_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


# ------------------------------------------------------------------------------
# Telugu Number Words & Indian Currency Multipliers
# ------------------------------------------------------------------------------
# Telugu spoken numbers (e.g. "ఒక" = 1, "రెండు" = 2, "ఐదు" = 5, "ఇరవై" = 20)
_TELUGU_ONES: dict[str, int] = {
    "ఒక": 1, "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4,
    "ఐదు": 5, "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9,
    "పది": 10, "పదిహేను": 15, "ఇరవై": 20, "ముప్పై": 30,
    "నలభై": 40, "యాభై": 50, "అరవై": 60, "డెబ్బై": 70,
    "ఎనభై": 80, "తొంభై": 90,
}

# Indian currency units and their multiplier values in Rupees
_TELUGU_MULTIPLIERS: dict[str, float] = {
    "వేల": 1_000,
    "వేలు": 1_000,
    "లక్ష": 1_00_000,
    "లక్షలు": 1_00_000,
    "లాక్స్": 1_00_000,
    "లాక్": 1_00_000,
    "కోటి": 1_00_00_000,
    "కోట్లు": 1_00_00_000,
    "lakh": 1_00_000,
    "lakhs": 1_00_000,
    "lac": 1_00_000,
    "lacs": 1_00_000,
    "crore": 1_00_00_000,
    "crores": 1_00_00_000,
    "thousand": 1_000,
}

# Regular expression pattern to detect English amounts (e.g. "50 lakhs", "1.5 crore", "25 lacs")
_AMOUNT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|crore|crores|thousand|lacs?)",
    re.IGNORECASE,
)

# Arabic numerals followed by Telugu-script currency units, including common
# STT transliterations such as "లాక్స్".
_DIGIT_TELUGU_AMOUNT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(లక్షలు|లక్ష|లాక్స్|లాక్|కోట్లు|కోటి|వేలు|వేల)"
)


# ------------------------------------------------------------------------------
# Property Types & Patterns
# ------------------------------------------------------------------------------
_PROPERTY_TYPES: dict[str, list[str]] = {
    "flat":             ["flat", "apartment", "ఫ్లాట్", "అపార్ట్‌మెంట్"],
    "villa":            ["villa", "విల్లా"],
    "independent_house": ["independent house", "individual house", "స్వతంత్ర ఇల్లు"],
    "plot":             ["plot", "land", "స్థలం", "ప్లాట్"],
    "gated_community":  ["gated community", "గేటెడ్ కమ్యూనిటీ"],
}
_PROP_ALIAS_MAP: dict[str, str] = {}
for canonical, aliases in _PROPERTY_TYPES.items():
    for a in aliases:
        _PROP_ALIAS_MAP[a.lower()] = canonical

# Regex pattern for bedroom count: e.g. "2bhk", "3 BHK", "4 bhk"
_BHK_PATTERN = re.compile(r"(\d)\s*bhk", re.IGNORECASE)

# Words indicating customer agreement or disagreement:
_YES_TOKENS = {"అవును", "సరే", "yes", "yeah", "ok", "okay", "sure", "ha", "haa"}
_NO_TOKENS  = {"వద్దు", "లేదు", "no", "nope", "not interested", "cancel"}
_OPT_OUT_TOKENS = {
    "stop", "not interested", "remove",
    "unsubscribe", "do not call", "don't call", "కాల్ చేయకండి", "మళ్ళీ కాల్ చేయకండి", "సంప్రదించకండి",
}


# ------------------------------------------------------------------------------
# Normalizer Class
# ------------------------------------------------------------------------------
class Normalizer:
    """
    Cleans transcript text and extracts real estate fields (budget, city, BHK, opt-out).
    """

    def process(self, stt_text: str) -> tuple[str, str]:
        """
        Takes raw spoken text from Soniox and returns two separate strings:
        (raw_text, normalized_text).
        
        raw_text: Original text, preserved forever for audit and legal compliance.
        normalized_text: Cleaned copy used for keyword & regex parsing.
        """
        raw_text = stt_text                     # Keep exact original untouched
        normalized_text = self._normalize(stt_text)

        log.debug(
            "normalizer.processed",
            raw_len=len(raw_text),
            normalized_len=len(normalized_text),
        )
        return raw_text, normalized_text

    def extract(self, normalized_text: str) -> ExtractionResult:
        """
        Extracts structured data (budget, city, bedrooms, opt-out) from the cleaned text.
        Only called AFTER the customer has finished speaking.
        """
        result = ExtractionResult()

        result.budget_rupees = self._extract_budget(normalized_text)
        result.location      = self._extract_location(normalized_text)
        result.property_type = self._extract_property_type(normalized_text)
        result.bedrooms      = self._extract_bedrooms(normalized_text)
        result.timeline      = self._extract_timeline(normalized_text)
        result.opted_out     = self._detect_opt_out(normalized_text)
        result.consent       = self._detect_consent(normalized_text)

        log.info(
            "normalizer.extraction_result",
            budget_rupees=result.budget_rupees,
            location=result.location,
            property_type=result.property_type,
            bedrooms=result.bedrooms,
            opted_out=result.opted_out,
        )
        return result

    # --------------------------------------------------------------------------
    # Private Cleaning & Extraction Helpers
    # --------------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """Strips leading/trailing spaces and collapses double spaces."""
        normalized = text.strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _extract_budget(self, text: str) -> Optional[float]:
        """Parse explicit Indian currency expressions into rupees."""
        amount = parse_indian_amount(text)
        return float(amount) if amount is not None else None

    def _extract_location(self, text: str) -> Optional[str]:
        """Checks if any recognized city name or Telugu spelling is in the sentence."""
        lower = text.lower()
        for alias, canonical in _ALIAS_TO_CANONICAL.items():
            if alias in lower:
                log.debug("normalizer.location_matched", alias=alias, canonical=canonical)
                return canonical
        return None

    def _extract_property_type(self, text: str) -> Optional[str]:
        """Checks for property words like 'flat', 'apartment', 'villa', or 'plot'."""
        lower = text.lower()
        for alias, canonical in _PROP_ALIAS_MAP.items():
            if alias in lower:
                return canonical
        return None

    def _extract_bedrooms(self, text: str) -> Optional[int]:
        """Checks for '2bhk', '3 BHK', '4bhk' and returns the number 2, 3, or 4."""
        match = _BHK_PATTERN.search(text)
        if match:
            return int(match.group(1))
        return None

    def _extract_timeline(self, text: str) -> Optional[str]:
        """Extract only explicit purchase-time expressions; do not infer vague intent."""
        lower = text.lower()
        if any(token in lower for token in ("వెంటనే", "ఇప్పుడే", "immediately", "right away", "as soon as possible")):
            return "immediate"
        telugu_months = {
            "ఒక నెలలో": 1, "నెలలోపు": 1, "రెండు నెలల్లో": 2,
            "మూడు నెలల్లో": 3, "ఆరు నెలల్లో": 6, "ఆరు నెలలలో": 6,
            "ఒక సంవత్సరంలో": 12, "సంవత్సరం లోపు": 12,
        }
        for phrase, months in telugu_months.items():
            if phrase in lower:
                return f"{months}_months"
        match = re.search(r"\\b(\\d{1,2})\\s*(months?|నెలల్లో|నెలలు|నెలలో)\\b?", lower)
        if match:
            months = int(match.group(1))
            if 1 <= months <= 36:
                return f"{months}_months"
        english_words = {"one": 1, "two": 2, "three": 3, "six": 6, "twelve": 12}
        for word, months in english_words.items():
            if re.search(rf"\\b{word}\\s+months?\\b", lower):
                return f"{months}_months"
        return None

    def _detect_opt_out(self, text: str) -> Optional[bool]:
        """Detects if caller said 'not interested', 'stop calling', or 'వద్దు'."""
        lower = text.lower()
        for token in _OPT_OUT_TOKENS:
            if token in lower:
                log.info("normalizer.opt_out_detected", matched_token=token)
                return True
        return None

    def _detect_consent(self, text: str) -> Optional[bool]:
        """Detects if caller agreed ('yes', 'ok', 'సరే') to proceed."""
        lower = text.lower()
        if self._detect_opt_out(text):
            return False
        for token in _YES_TOKENS:
            if token in lower:
                return True
        for token in _NO_TOKENS:
            if token in lower:
                return False
        return None
