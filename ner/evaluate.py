"""Evaluate a saved model on Naamapadam's held-out Telugu test split."""
from __future__ import annotations

import argparse
import json

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments

from ner.data import add_location_labels, load_telugu_dataset
from ner.train import ID2LABEL, tokenize_and_align


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best-model checkpoint")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    data = load_telugu_dataset()["test"].map(add_location_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def encode(batch):
        return tokenize_and_align(batch, tokenizer, args.max_length)

    encoded = data.map(encode, batched=True, remove_columns=data.column_names)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="artifacts/eval-temp", per_device_eval_batch_size=args.batch_size, report_to="none"),
        eval_dataset=encoded,
        tokenizer=tokenizer,
    )
    output = trainer.predict(encoded)
    pred_ids = np.argmax(output.predictions, axis=-1)
    gold_ids = output.label_ids
    pred_tags, gold_tags = [], []
    for pred_row, gold_row in zip(pred_ids, gold_ids):
        p, g = [], []
        for pi, gi in zip(pred_row, gold_row):
            if int(gi) == -100:
                continue
            p.append(ID2LABEL[int(pi)])
            g.append(ID2LABEL[int(gi)])
        pred_tags.append(p)
        gold_tags.append(g)
    metrics = {
        "test_location_precision": precision_score(gold_tags, pred_tags, zero_division=0),
        "test_location_recall": recall_score(gold_tags, pred_tags, zero_division=0),
        "test_location_f1": f1_score(gold_tags, pred_tags, zero_division=0),
        "test_sentences": len(data),
        "note": "This test split is projected-label data, not a real-estate voice-call benchmark.",
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
