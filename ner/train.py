"""Fine-tune a pretrained encoder for Telugu location-only NER.

Example:
  python -m ner.train --output-dir artifacts/telugu-location-ner
  python -m ner.train --output-dir artifacts/telugu-location-ner --resume-from-checkpoint artifacts/telugu-location-ner/checkpoint-500
  python -m ner.train --synthetic-jsonl data/reviewed_tanglish_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ner.data import BASE_LABELS, add_location_labels, load_synthetic_jsonl, load_telugu_dataset

DEFAULT_MODEL = "xlm-roberta-base"
ID2LABEL = {i: label for i, label in enumerate(BASE_LABELS)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default="artifacts/telugu-location-ner")
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--synthetic-jsonl", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bio_tags_to_strings(ids):
    return [ID2LABEL[int(i)] for i in ids]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_predictions, true_labels = [], []
    for pred_row, label_row in zip(predictions, labels):
        pred_tags, gold_tags = [], []
        for pred_id, gold_id in zip(pred_row, label_row):
            if int(gold_id) == -100:
                continue
            pred_tags.append(ID2LABEL[int(pred_id)])
            gold_tags.append(ID2LABEL[int(gold_id)])
        true_predictions.append(pred_tags)
        true_labels.append(gold_tags)
    return {
        "location_precision": precision_score(true_labels, true_predictions, zero_division=0),
        "location_recall": recall_score(true_labels, true_predictions, zero_division=0),
        "location_f1": f1_score(true_labels, true_predictions, zero_division=0),
    }


def tokenize_and_align(batch, tokenizer, max_length):
    encoded = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    aligned = []
    for batch_index, word_tags in enumerate(batch["location_tags"]):
        word_ids = encoded.word_ids(batch_index=batch_index)
        previous_word = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word:
                label_ids.append(int(word_tags[word_id]))
            else:
                # Label only the first subword to avoid overweighting split words.
                label_ids.append(-100)
            previous_word = word_id
        aligned.append(label_ids)
    encoded["labels"] = aligned
    return encoded


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_telugu_dataset()
    source_columns = raw["train"].column_names
    mapped = raw.map(add_location_labels, remove_columns=source_columns)
    train_data = mapped["train"]
    if args.synthetic_jsonl:
        synthetic = load_synthetic_jsonl(args.synthetic_jsonl)
        train_data = concatenate_datasets([train_data, synthetic])
        print(f"Added {len(synthetic)} reviewed synthetic training rows.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for word_ids alignment.")

    def encode(batch):
        return tokenize_and_align(batch, tokenizer, args.max_length)

    remove_columns = train_data.column_names
    train_encoded = train_data.map(encode, batched=True, remove_columns=remove_columns)
    val_encoded = mapped["validation"].map(
        encode, batched=True, remove_columns=mapped["validation"].column_names
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(BASE_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="location_f1",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        fp16=bool(__import__("torch").cuda.is_available()),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "best-model"))
    tokenizer.save_pretrained(str(output_dir / "best-model"))
    val_metrics = trainer.evaluate()
    metadata = {
        "base_model": args.model,
        "dataset": "ai4bharat/naamapadam",
        "dataset_config": "te",
        "task": "location-only token classification",
        "labels": ID2LABEL,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "train_rows_including_optional_synthetic": len(train_data),
        "validation_metrics": val_metrics,
        "train_metrics": train_result.metrics,
        "synthetic_jsonl": args.synthetic_jsonl,
        "warning": "Naamapadam labels are projected from English alignments; validate on real Telugu voice transcripts before production.",
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
