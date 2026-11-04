# Telugu Location NER (experimental training pipeline)

This folder adds a separate, reproducible training workflow. It does **not** change the live voice-agent path or replace the current normalizer until the model has passed a real-call benchmark.

## Data source

The default dataset is [AI4Bharat Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam), Telugu subset `te`. It provides token-level NER annotations. The published dataset card reports 507,741 Telugu training rows, 2,700 validation rows, and 53 test rows. The annotations were projected from English entities using word alignment; therefore this is useful pretraining/fine-tuning data, **not proof of accuracy on Telugu voice calls**.

We collapse the source tags to a location-only task:
- `O` -> `O`
- `B-LOC` / `I-LOC` -> `B-LOC` / `I-LOC`
- Person and organisation tags -> `O`

The training code checks token/tag lengths and source tag IDs. The held-out test split is not used during training or checkpoint selection.

## Install (isolated environment)

From the repository root:

```powershell
python -m venv .venv-ner
.\.venv-ner\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ner/requirements.txt
```

## Fine-tune weights

```powershell
python -m ner.train --output-dir artifacts/telugu-location-ner --epochs 5 --batch-size 8
```

This downloads the dataset and pretrained `xlm-roberta-base` weights, computes token-classification loss, and updates the model weights using gradient descent through Hugging Face Trainer.

To continue training from an existing checkpoint rather than starting from the base model again:

```powershell
python -m ner.train --output-dir artifacts/telugu-location-ner --resume-from-checkpoint artifacts/telugu-location-ner/checkpoint-500
```

Keep checkpoints and metadata. Do not replace the best model unless validation improves and the independent test/voice benchmark remains acceptable.

## Evaluate

```powershell
python -m ner.evaluate --model artifacts/telugu-location-ner/best-model
```

The output reports entity-level location precision, recall and F1. The target of 0.80 is an acceptance target, **not a claimed result**. The public Naamapadam test set is small and projected-label; a separate, manually reviewed test set of anonymized Soniox transcripts is required before claiming 80% for the real-estate calling domain.

## Predict locally (no LLM API call)

```powershell
python -m ner.predict --model artifacts/telugu-location-ner/best-model --text "నేను విజయవాడలో ఇల్లు కొనాలనుకుంటున్నాను"
```

Output includes extracted spans and model confidence. Confidence is not calibrated probability. Production integration should abstain/ask a clarification for uncertain spans and must not treat a predicted proper noun as verified real-world geography.

## Tanglish / conversational data

The base dataset is Telugu-script text, not a representative Tanglish voice corpus. Do not fabricate a Tanglish score from Telugu-script results. Add reviewed conversational examples as JSONL with one record per sentence:

```json
{"tokens":["Nenu","Vijayawada","lo","illu","konali"],"location_tags":[0,1,0,0,0]}
{"tokens":["Guntur","daggara","manchi","location","kavali"],"location_tags":[1,0,0,0,0]}
```

Generate candidates from a curated location CSV if needed, but keep those generated examples in training only. Human-review them before use. Never put generated examples in validation/test. Split actual call transcripts by caller/conversation to prevent leakage and remove personal identifiers.

## Important limitations

- This is NER: it detects a location mention span. It does not geocode it, resolve aliases to canonical city names, or prove the place exists.
- Training from Wikipedia/books alone does not create labels. Text must be annotated or weakly labelled and quality-checked.
- The model has not been trained by adding these files. Training requires running the commands in an environment with Python dependencies, internet access to Hugging Face, and sufficient compute.
- No production accuracy claim is made until evaluation is run on an independently annotated test set of real conversational Telugu/Tanglish.
