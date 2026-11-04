"""Local, no-LLM location extraction from a sentence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


class TeluguLocationNER:
    def __init__(self, model_path: str, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def extract(self, text: str) -> list[dict]:
        words = text.split()
        if not words:
            return []
        inputs = self.tokenizer(
            words, is_split_into_words=True, return_tensors="pt",
            truncation=True, max_length=160,
        )
        word_ids = inputs.word_ids(batch_index=0)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        word_predictions = []
        seen = set()
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            label_id = int(torch.argmax(probs[token_index]).item())
            word_predictions.append((word_id, self.id2label[label_id], float(probs[token_index, label_id])))
        entities = []
        current = []
        scores = []
        def flush():
            if current:
                entities.append({
                    "text": " ".join(current),
                    "confidence": round(sum(scores) / len(scores), 4),
                    "label": "LOC",
                })
                current.clear()
                scores.clear()
        for _, tag, score in word_predictions:
            if tag == "B-LOC":
                flush()
                current.append(words[len(word_predictions[:len(entities)])] if False else "")
                current[-1] = ""  # replaced below using the source word index
        # Build spans directly with original word indices.
        flush()
        entities = []
        active_words, active_scores = [], []
        for word_id, tag, score in word_predictions:
            if tag == "B-LOC":
                if active_words:
                    entities.append({"text": " ".join(active_words), "confidence": round(sum(active_scores)/len(active_scores), 4), "label": "LOC"})
                active_words, active_scores = [words[word_id]], [score]
            elif tag == "I-LOC" and active_words:
                active_words.append(words[word_id])
                active_scores.append(score)
            else:
                if active_words:
                    entities.append({"text": " ".join(active_words), "confidence": round(sum(active_scores)/len(active_scores), 4), "label": "LOC"})
                    active_words, active_scores = [], []
        if active_words:
            entities.append({"text": " ".join(active_words), "confidence": round(sum(active_scores)/len(active_scores), 4), "label": "LOC"})
        return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    ner = TeluguLocationNER(args.model)
    print(json.dumps(ner.extract(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
