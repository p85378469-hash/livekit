"""Deterministic parser for Indian real-estate budget expressions.

This module deliberately parses only explicit amount + currency-unit expressions.
It does not infer a budget from bare numbers or ambiguous ranges.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


_UNIT_VALUES = {
    # Indian numbering units
    "thousand": 1_000, "thousands": 1_000, "k": 1_000,
    "వేలు": 1_000, "వేల": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000,
    "lacs": 100_000, "లక్ష": 100_000,
    "లక్షలు": 100_000, "లాక్స్": 100_000, "లాక్": 100_000,
    "లక్షల": 100_000,
    "crore": 10_000_000, "crores": 10_000_000,
    "కోటి": 10_000_000, "కోట్లు": 10_000_000,
}

# Cardinal words used in ordinary English and Telugu speech. Values are additive
# below 100, so "ఇరవై ఐదు" and "twenty five" become 25.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
    "ఒక": 1, "ఒకటి": 1, "ఒక్క": 1, "ఒక్కటి": 1,
    "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5, "ఆరు": 6,
    "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9, "పది": 10,
    "పదకొండు": 11, "పన్నెండు": 12, "పదమూడు": 13, "పద్నాలుగు": 14,
    "పదిహేను": 15, "పదహారు": 16, "పదిహేడు": 17, "పద్దెనిమిది": 18,
    "పంతొమ్మిది": 19, "ఇరవై": 20, "ముప్పై": 30, "నలభై": 40,
    "యాభై": 50, "అరవై": 60, "డెబ్బై": 70, "ఎనభై": 80,
    "తొంభై": 90, "తొంభది": 90,
}
_HUNDRED_WORDS = {"hundred", "వంద", "వందలు"}
_CONNECTORS = {"and", "మరియు"}
_FRACTION_WORDS = {"half": Decimal("0.5"), "అర": Decimal("0.5")}
_TOKEN_RE = re.compile(r"\d+(?:,\d{2,3})*(?:\.\d+)?|[A-Za-z]+|[\u0C00-\u0C7F]+", re.UNICODE)
_RANGE_RE = re.compile(r"\d\s*(?:-|–|to|నుంచి)\s*\d", re.IGNORECASE)


def _parse_number_words(tokens: list[str]) -> Optional[int]:
    """Parse a number phrase such as 'fifty', 'ఇరవై ఐదు', or 'one hundred'."""
    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        word = token.lower()
        if word in _CONNECTORS:
            continue
        if word in _NUMBER_WORDS:
            current += _NUMBER_WORDS[word]
            saw_number = True
        elif word in _HUNDRED_WORDS:
            current = max(current, 1) * 100
            saw_number = True
        else:
            return None
    return total + current if saw_number else None


def parse_indian_amount(text: str) -> Optional[int]:
    """Return rupees for explicit Indian-unit amounts; None when not safely parsed.

    Supports digits and spoken English/Telugu cardinal words, compound units
    (for example, '2 crore 50 lakhs'), and decimal amounts.
    """
    if not text or _RANGE_RE.search(text):
        return None

    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return None

    total = Decimal(0)
    found_unit = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        lower = token.lower()

        if lower in _UNIT_VALUES:
            # A unit without a preceding number is not enough to infer an amount.
            i += 1
            continue

        # Common half-unit expressions, e.g. "half crore" / "అర కోటి".
        if lower in _FRACTION_WORDS:
            j = i + 1
            if j < len(tokens) and tokens[j].lower() in _UNIT_VALUES:
                total += _FRACTION_WORDS[lower] * _UNIT_VALUES[tokens[j].lower()]
                found_unit = True
                i = j + 1
                continue

        # Numeric amount immediately followed by a recognized unit.
        numeric = token.replace(",", "")
        try:
            value = Decimal(numeric)
            is_numeric = bool(re.fullmatch(r"\d+(?:,\d{2,3})*(?:\.\d+)?", token))
        except InvalidOperation:
            value, is_numeric = Decimal(0), False

        if is_numeric:
            j = i + 1
            if j < len(tokens) and tokens[j].lower() in _UNIT_VALUES:
                total += value * _UNIT_VALUES[tokens[j].lower()]
                found_unit = True
                i = j + 1
                continue
            i += 1
            continue

        # Collect a contiguous number-word phrase before its unit.
        j = i
        phrase: list[str] = []
        while j < len(tokens):
            word = tokens[j].lower()
            if word in _NUMBER_WORDS or word in _HUNDRED_WORDS or word in _CONNECTORS:
                phrase.append(word)
                j += 1
            else:
                break
        if phrase and j < len(tokens) and tokens[j].lower() in _UNIT_VALUES:
            number = _parse_number_words(phrase)
            if number is not None:
                total += Decimal(number) * _UNIT_VALUES[tokens[j].lower()]
                found_unit = True
                i = j + 1
                continue
        i += 1

    if not found_unit:
        return None
    return int(total)
