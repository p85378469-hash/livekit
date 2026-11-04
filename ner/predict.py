"""Local, no-LLM location extraction from a sentence."""
from __future__ import annotations

import argparse
import json

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
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=160,
        )
        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        probabilities = torch.softmax(self.model(**model_inputs).logits[0], dim=-1)

        # Use only the first subword prediction for each original whitespace token.
        word_predictions = []
        seen_word_ids = set()
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            scores = probabilities[token_index]
            label_id = int(torch.argmax(scores).item())
            word_predictions.append(
                (word_id, self.id2label[label_id], float(scores[label_id].item()))
            )

        entities = []
        active_words: list[str] = []
        active_scores: list[float] = []

        def flush():
            if active_words:
                entities.append({
                    "text": " ".join(active_words),
                    "confidence": round(sum(active_scores) / len(active_scores), 4),
                    "label": "LOC",
                })
                active_words.clear()
                active_scores.clear()

        for word_id, tag, score in word_predictions:
            if tag == "B-LOC":
                flush()
                active_words.append(words[word_id])
                active_scores.append(score)
            elif tag == "I-LOC" and active_words:
                active_words.append(words[word_id])
                active_scores.append(score)
            else:
                flush()
        flush()
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
