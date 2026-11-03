# Telugu Voice Agent — Phase 1

> LiveKit + Soniox STT + Groq LLM + Cartesia TTS  
> Real-estate lead qualification in Telugu

---

## Project Structure

```
telugu-voice-agent/
├── agent.py                    ← Main entry point (LiveKit agent)
├── requirements.txt            ← Pinned dependencies (DO NOT upgrade without re-verifying)
├── .env.example                ← Copy to .env and fill API keys
├── assets/
│   └── greeting.wav            ← Prerecorded Telugu greeting (16kHz mono 16-bit PCM)
├── core/
│   ├── session.py              ← Session state: facts, turn_id, transcripts
│   ├── turn_detector.py        ← Soniox STT endpointing → turn gate
│   ├── interruption.py         ← 9-step interruption recovery sequence
│   └── playback.py             ← Audio playback with stale-chunk guard
├── pipeline/
│   ├── normalizer.py           ← Two-buffer normalization + fact extraction
│   ├── policy.py               ← Response policy (Python vs LLM decision)
│   └── context.py              ← LLM context builder (rolling history + facts)
├── providers/
│   ├── stt.py                  ← Soniox STT wrapper
│   ├── llm.py                  ← Groq streaming LLM wrapper
│   └── tts.py                  ← Cartesia streaming TTS wrapper
├── audio/
│   └── greeting.py             ← WAV greeting player (separate from TTS)
├── telemetry/
│   └── latency.py              ← T0–T6 per-stage latency tracker
└── tests/
    ├── test_turn_detection.py  ← Phase 1 exit check: turn gate
    ├── test_interruption.py    ← Phase 1 exit check: interruption
    └── test_normalization.py   ← Phase 1 exit check: normalization
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Prepare the greeting WAV

```bash
mkdir assets
# Convert your greeting audio to the required format:
ffmpeg -i your_greeting.mp3 -ar 16000 -ac 1 -sample_fmt s16 assets/greeting.wav
```

### 4. Run Phase 1 tests

```bash
pytest tests/ -v
```

All 7 test checklist items from the master plan must pass before proceeding.

### 5. Run the agent

```bash
python agent.py start
```

---

## Phase 1 Exit Checklist

Before moving to Phase 2, all of these must pass:

- [ ] Short pause does NOT trigger premature response
- [ ] Partial text does NOT commit a budget or fire an LLM call
- [ ] Raw and normalized text are separately inspectable in logs
- [ ] A genuine caller interruption stops stale audible output
- [ ] Noise/backchannel false interruption is logged (not acted upon)
- [ ] Per-stage latency timestamps (T0–T6) collected in every call
- [ ] Exact package versions + event names are pinned (see `requirements.txt`)

---

## Key Design Rules (from Master Plan)

| Rule | Enforced In |
|---|---|
| Partials never trigger pipeline | `turn_detector.py` |
| Raw transcript is immutable | `normalizer.py` → `session.commit_turn()` |
| Turn ID advances BEFORE pipeline starts | `turn_detector._emit_turn_complete()` |
| Audio stops BEFORE LLM cancelled | `interruption.py` steps 3→4→5 |
| WAV greeting uses NO LLM or TTS | `greeting.py` + `agent.py` |
| Fillers are OFF until timing verified | `playback.play_filler()` (stubbed) |
| All facts committed only after TURN_COMPLETE | `pipeline/policy.py` |

---

## What Is Deferred to Phase 2

- Telugu-aware pause classification (advanced turn tuning)
- Backchannel/noise/filler ML classifier
- Full Telugu number parser (word-based amounts beyond basic patterns)
- Date parsing, negation, confidence scoring
- WAV playback hardening and dynamic fillers
- CRM integration and context compaction
- Streaming optimization (chunk size, buffering)

---

## Pinned SDK Versions

| Package | Version | Notes |
|---|---|---|
| `livekit-agents` | 0.12.16 | Event names verified |
| `livekit-plugins-soniox` | 0.12.5 | `SpeechEventType.FINAL_TRANSCRIPT` confirmed |
| `livekit-plugins-cartesia` | 0.12.5 | `.synthesize()` streaming confirmed |
| `soniox` | 2.1.0 | Telugu + English code-switching |
| `groq` | 0.11.0 | OpenAI-compatible streaming |
| `cartesia` | 1.4.1 | `sonic-multilingual` model |
