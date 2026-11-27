# ==============================================================================
# tests/test_call_logger.py — Tests for CallLogger & Log Export
# ==============================================================================

import json
from pathlib import Path
import pytest

from core.session import ConversationSession
from telemetry.call_logger import CallLogger, format_rupees
from telemetry.latency import LatencyTracker, Stage
from scripts.view_call_logs import load_all_calls


def test_format_rupees():
    assert format_rupees(None) == "Not specified"
    assert "50.00 Lakhs" in format_rupees(5000000)
    assert "1.50 Crores" in format_rupees(15000000)
    assert "75,000" in format_rupees(75000)


def test_call_logger_creates_all_files(tmp_path: Path):
    session_id = "test_call_123"
    logger = CallLogger(session_id=session_id, logs_dir=str(tmp_path))

    session = ConversationSession(session_id=session_id)
    session.update_fact("location", "Vijayawada")
    session.update_fact("budget_rupees", 5000000.0)
    session.update_fact("bedrooms", 2)
    session.update_fact("property_type", "flat")

    tid = session.advance_turn()
    session.commit_turn(
        raw_transcript="నాకు విజయవాడ లో 50 లక్షలు బడ్జెట్",
        normalized_transcript="నాకు విజయవాడ లో 5000000 బడ్జెట్",
        timestamp_start_ms=1000.0,
        timestamp_end_ms=2000.0,
    )

    latency = LatencyTracker(session_id=session_id)
    latency.new_turn(tid)
    latency.record(Stage.TURN_END, 2000.0)
    latency.record(Stage.PLAYBACK_FIRST_AUDIO, 2800.0)

    logger.record_turn_response(tid, "విజయవాడ లో 50 లక్షల బడ్జెట్ లో ఫ్లాట్స్ వివరాలు...", "PYTHON_DIRECT")

    res = logger.save_call(session, latency, status="COMPLETED")
    assert res["status"] == "saved"

    # Verify JSON file
    json_path = Path(res["json_path"])
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["session_id"] == session_id
    assert data["status"] == "COMPLETED"
    assert data["lead_facts"]["location"] == "Vijayawada"
    assert data["lead_facts"]["budget_rupees"] == 5000000.0
    assert data["lead_facts"]["qualification_status"] == "FULLY_QUALIFIED"
    assert len(data["turns"]) == 1
    assert data["turns"][0]["caller_raw_transcript"] == "నాకు విజయవాడ లో 50 లక్షలు బడ్జెట్"
    assert data["turns"][0]["e2e_latency_ms"] == 800.0

    # Verify TXT file
    txt_path = Path(res["txt_path"])
    assert txt_path.exists()
    with open(txt_path, "r", encoding="utf-8") as f:
        txt = f.read()
    assert "Vijayawada" in txt
    assert "FULLY_QUALIFIED" in txt
    assert "TELUGU VOICE AGENT — CALL RECORD" in txt

    # Verify JSONL index
    jsonl_path = Path(res["jsonl_path"])
    assert jsonl_path.exists()
    calls = load_all_calls(tmp_path)
    assert len(calls) == 1
    assert calls[0]["session_id"] == session_id
    assert calls[0]["location"] == "Vijayawada"
