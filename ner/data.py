"""Dataset loading and token-label alignment for Telugu location NER.

Naamapadam uses CoNLL-style ids:
0 O, 1 B-PER, 2 I-PER, 3 B-ORG, 4 I-ORG, 5 B-LOC, 6 I-LOC.
We collapse non-location entities to O because this project extracts locations only.
"""
from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset

DATASET_ID = "ai4bharat/naamapadam"
DATASET_CONFIG = "te"
BASE_LABELS = ["O", "B-LOC", "I-LOC"]
SOURCE_ID_TO_LOCATION_ID = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2}


def load_telugu_dataset() -> DatasetDict:
    data = load_dataset(DATASET_ID, DATASET_CONFIG)
    required = {"words", "ner"}
    for split in ("train", "validation", "test"):
        if split not in data:
            raise ValueError(f"Naamapadam Telugu dataset is missing split: {split}")
        if not required.issubset(data[split].column_names):
            raise ValueError(
                f"Unexpected columns in {split}: {data[split].column_names}; "
                f"expected {sorted(required)}"
            )
    return data


def convert_tags(tags: list[int]) -> list[int]:
    converted = []
    for tag in tags:
        tag = int(tag)
        if tag not in SOURCE_ID_TO_LOCATION_ID:
            raise ValueError(f"Unexpected Naamapadam tag id: {tag}")
        converted.append(SOURCE_ID_TO_LOCATION_ID[tag])
    return converted


def add_location_labels(example: dict[str, Any]) -> dict[str, Any]:
    words = example["words"]
    tags = example["ner"]
    if len(words) != len(tags):
        raise ValueError("Token/tag length mismatch in source dataset")
    return {"tokens": words, "location_tags": convert_tags(tags)}


def load_synthetic_jsonl(path: str):
    """Load manually reviewed synthetic rows: {tokens: [...], location_tags: [...]}."""
    from datasets import load_dataset as hf_load_dataset

    extra = hf_load_dataset("json", data_files=path, split="train")
    for row in extra:
        if len(row["tokens"]) != len(row["location_tags"]):
            raise ValueError("Synthetic row has different token and tag counts")
        if any(int(tag) not in (0, 1, 2) for tag in row["location_tags"]):
            raise ValueError("Synthetic tags must use O=0, B-LOC=1, I-LOC=2")
    keep = {"tokens", "location_tags"}
    drop = [column for column in extra.column_names if column not in keep]
    return extra.remove_columns(drop)
