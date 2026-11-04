"""Create labelled conversational Telugu/Tanglish training examples from a small CSV.

CSV columns: telugu,romanized. These generated examples are TRAINING-ONLY and require
human review. This is augmentation, not an independent evaluation dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TEMPLATES = [
    ("నేను {place} లో ఇల్లు కొనాలనుకుంటున్నాను", "telugu"),
    ("{place} దగ్గర మంచి ఫ్లాట్ ఉందా", "telugu"),
    ("{place} లో 2BHK ఫ్లాట్ కావాలి", "telugu"),
    ("నాకు {place} లో ప్లాట్ కావాలి", "telugu"),
    ("Nenu {place} lo illu konali", "romanized"),
    ("{place} daggara manchi flat undha", "romanized"),
    ("{place} lo 2BHK flat kavali", "romanized"),
    ("I am looking for a flat in {place}", "romanized"),
]


def make_row(sentence: str, place: str) -> dict:
    tokens = sentence.split()
    place_tokens = place.split()
    matches = [
        i for i in range(len(tokens) - len(place_tokens) + 1)
        if tokens[i:i + len(place_tokens)] == place_tokens
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely locate inserted place {place!r} in {sentence!r}")
    start = matches[0]
    tags = [0] * len(tokens)
    tags[start] = 1
    for index in range(start + 1, start + len(place_tokens)):
        tags[index] = 2
    return {"tokens": tokens, "location_tags": tags, "source": "synthetic_template"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.input_csv, newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            for field, language in (("telugu", "telugu"), ("romanized", "romanized")):
                place = (item.get(field) or "").strip()
                if not place:
                    continue
                for template, template_language in TEMPLATES:
                    if template_language != language:
                        continue
                    sentence = template.format(place=place)
                    rows.append(make_row(sentence, place))

    if not rows:
        raise SystemExit("No rows generated. CSV needs telugu and/or romanized columns.")
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} training-only examples to {output}")
    print("Human-review every row before using it for training; never use it for validation/test.")


if __name__ == "__main__":
    main()
