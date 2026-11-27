#!/usr/bin/env python3
# ==============================================================================
# scripts/test_call_simulation.py — Offline Telugu Call Simulator & Tester
#
# IN SIMPLE WORDS:
# This script lets you test the voice agent's entire brain without needing a live
# phone call or microphone!
#
# Features:
# 1. Automated qualification test: Simulates a Telugu customer looking for a flat in Vijayawada.
# 2. Interactive mode (--interactive): Lets you type Telugu phrases live and see
#    normalization, fact extraction, Groq LLM answers, and call log persistence.
# 3. Saves complete call logs to call_logs/ directory.
# ==============================================================================

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from core.session import ConversationSession
from pipeline.context import ContextBuilder
from pipeline.normalizer import Normalizer
from pipeline.policy import PolicySelector, ResponseMode
from telemetry.call_logger import CallLogger
from telemetry.latency import LatencyTracker, Stage


def simulate_automated_call() -> None:
    """Runs a complete 3-turn qualification test in Telugu and saves the call log."""
    session_id = f"sim_{int(time.time()) % 100000:05d}"
    print("\n" + "=" * 78)
    print(f" STARTING SIMULATED TELUGU VOICE CALL [Session: {session_id}]")
    print("=" * 78)

    session = ConversationSession(session_id=session_id)
    latency = LatencyTracker(session_id=session_id)
    normalizer = Normalizer()
    policy = PolicySelector()
    context_builder = ContextBuilder()
    call_logger = CallLogger(session_id=session_id)

    # Greeting
    greeting_telugu = "నమస్కారం అండి! నేను మీ రియల్ ఎస్టేట్ అసిస్టెంట్ ని. మీకు ఎలాంటి ప్రాపర్టీ కావాలి?"
    print(f"\n[Agent Greeting]: {greeting_telugu}\n")

    # Scripted caller turns
    test_turns = [
        "నాకు విజయవాడ లో ప్రాపర్టీ కావాలి అండి",
        "నా బడ్జెట్ 50 లక్షలు, 2BHK ఫ్లాట్ చూస్తున్నాను",
        "అవును వివరాలు వాట్సాప్ లో పంపండి",
    ]

    for user_speech in test_turns:
        turn_id = session.advance_turn()
        latency.new_turn(turn_id)

        t0 = time.time() * 1000
        latency.record(Stage.TURN_END, t0)

        # STT simulated finish
        time.sleep(0.05)
        t1 = time.time() * 1000
        latency.record(Stage.STT_FINAL, t1)

        # Normalization and Fact Extraction
        raw_text, normalized_text = normalizer.process(user_speech)
        session.commit_turn(
            raw_transcript=raw_text,
            normalized_transcript=normalized_text,
            timestamp_start_ms=t0 - 1500,
            timestamp_end_ms=t0,
        )

        extraction = normalizer.extract(normalized_text)
        decision = policy.select(session, extraction)

        t2 = time.time() * 1000
        latency.record(Stage.PYTHON_COMPLETE, t2)

        if decision.next_stage:
            session.update_stage(decision.next_stage, session.next_missing_field)

        # Simulated playback / response
        latency.record(Stage.LLM_FIRST_TOKEN, t2 + 180)
        latency.record(Stage.TTS_FIRST_AUDIO, t2 + 250)
        latency.record(Stage.PLAYBACK_FIRST_AUDIO, t2 + 300)

        agent_reply = decision.direct_text or "ధన్యవాదాలు! మీ ప్రాపర్టీ వివరాలను సరిచూస్తున్నాను."
        mode_str = decision.mode.name

        call_logger.record_turn_response(turn_id, agent_reply, mode_str)

        print(f"--- Turn #{turn_id} ---")
        print(f"  Caller Said (Raw)        : {raw_text}")
        if normalized_text != raw_text:
            print(f"  Caller Said (Normalized) : {normalized_text}")
        print(f"  Facts Extracted          : {session.facts}")
        print(f"  Agent Decision Mode      : {mode_str}")
        print(f"  Agent Spoke              : {agent_reply}")
        latency_e2e = latency.get_turn(turn_id).interval_ms(Stage.TURN_END, Stage.PLAYBACK_FIRST_AUDIO)
        print(f"  Simulated E2E Latency    : {latency_e2e} ms\n")

    # Save Call Log
    saved_info = call_logger.save_call(session, latency, status="COMPLETED")
    print("=" * 78)
    print(" [✓] Call Completed & Log Saved Successfully!")
    print(f" JSON Log       : {saved_info['json_path']}")
    print(f" Text Transcript: {saved_info['txt_path']}")
    print(f" Master Index   : {saved_info['jsonl_path']}")
    print("=" * 78 + "\n")


def interactive_mode() -> None:
    """Lets the user type inputs interactively to test real Telugu extraction."""
    session_id = f"cli_{int(time.time()) % 100000:05d}"
    print("\n" + "=" * 78)
    print(f" INTERACTIVE TELUGU CALL TEST [Session: {session_id}]")
    print(" Type Telugu or English sentences. Type 'exit' or 'quit' to end the call.")
    print("=" * 78)

    session = ConversationSession(session_id=session_id)
    latency = LatencyTracker(session_id=session_id)
    normalizer = Normalizer()
    policy = PolicySelector()
    call_logger = CallLogger(session_id=session_id)

    print("\n[Agent Greeting]: నమస్కారం అండి! నేను మీ రియల్ ఎస్టేట్ అసిస్టెంట్ ని. మీకు ఎలాంటి ప్రాపర్టీ కావాలి?\n")

    while True:
        try:
            user_input = input("Caller > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        turn_id = session.advance_turn()
        latency.new_turn(turn_id)
        t0 = time.time() * 1000
        latency.record(Stage.TURN_END, t0)
        latency.record(Stage.STT_FINAL, t0 + 20)

        raw_text, norm_text = normalizer.process(user_input)
        session.commit_turn(raw_text, norm_text, t0 - 1000, t0)

        extraction = normalizer.extract(norm_text)
        decision = policy.select(session, extraction)
        latency.record(Stage.PYTHON_COMPLETE, t0 + 40)
        latency.record(Stage.PLAYBACK_FIRST_AUDIO, t0 + 120)

        if decision.next_stage:
            session.update_stage(decision.next_stage, session.next_missing_field)

        reply = decision.direct_text or "మీ సమాచారం నమోదు చేయబడింది."
        call_logger.record_turn_response(turn_id, reply, decision.mode.name)

        print(f"\n  [Normalized] : {norm_text}")
        print(f"  [Facts]      : Location={session.facts.location}, Budget={session.facts.budget_rupees}, BHK={session.facts.bedrooms}")
        print(f"  [Agent Reply]: {reply}\n")

        if session.facts.opted_out:
            print("  Caller opted out. Ending call.")
            call_logger.set_status("OPTED_OUT")
            break

    saved_info = call_logger.save_call(session, latency)
    print(f"\n[✓] Call saved: {saved_info['txt_path']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and simulate Telugu Voice Agent calls.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run interactive terminal conversation")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        simulate_automated_call()


if __name__ == "__main__":
    main()
