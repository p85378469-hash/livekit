# ==============================================================================
# telemetry/call_logger.py — Complete Call Logging & Persistence System
#
# IN SIMPLE WORDS:
# This module captures and permanently stores call logs for every conversation.
# For each call, it saves:
#   1. Structured JSON file: call_logs/call_<session_id>_<timestamp>.json
#   2. Human-readable text transcript: call_logs/call_<session_id>_<timestamp>.txt
#   3. Master index line in: call_logs/call_logs.jsonl
#
# It records caller raw/normalized text, agent replies, extracted facts
# (budget, location, BHK, timeline), corrections, and stage latency (T0-T6).
# ==============================================================================

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import structlog

from core.session import ConversationSession
from telemetry.latency import LatencyTracker, Stage

log = structlog.get_logger(__name__)

DEFAULT_CALL_LOGS_DIR = os.environ.get("CALL_LOGS_DIR", "call_logs")


def format_rupees(amount: Optional[float]) -> str:
    """Formats numeric budget into human-friendly Indian currency string."""
    if amount is None:
        return "Not specified"
    if amount >= 10_000_000:
        cr = amount / 10_000_000
        return f"₹{cr:.2f} Crores (₹{amount:,.0f})"
    if amount >= 100_000:
        lakhs = amount / 100_000
        return f"₹{lakhs:.2f} Lakhs (₹{amount:,.0f})"
    return f"₹{amount:,.0f}"


class CallLogger:
    """
    Manages structured call logging for telephone/WebRTC voice conversations.
    Thread-safe and failsafe: logging errors never crash active voice calls.
    """

    def __init__(
        self,
        session_id: str,
        logs_dir: str = DEFAULT_CALL_LOGS_DIR,
        room_name: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.logs_dir = Path(logs_dir)
        self.room_name = room_name or session_id
        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        self.start_ms = time.time() * 1000
        self.end_time: Optional[datetime.datetime] = None
        self.end_ms: Optional[float] = None
        self.status: str = "IN_PROGRESS"
        self._turn_agent_replies: dict[int, dict[str, str]] = {}
        self._turn_streaming_metrics: dict[int, dict[str, Any]] = {}
        self._saved = False

        # Ensure directory exists
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning("call_logger.mkdir_failed", error=str(e), dir=str(self.logs_dir))

    def set_status(self, status: str) -> None:
        """Sets the call exit status (e.g. COMPLETED, OPTED_OUT, DISCONNECTED, ERROR)."""
        self.status = status

    def record_turn_response(
        self,
        turn_id: int,
        agent_reply: str,
        mode: str = "PYTHON_DIRECT",
    ) -> None:
        """Records what the agent replied for a specific turn."""
        self._turn_agent_replies[turn_id] = {
            "reply": agent_reply,
            "mode": mode,
        }

    def record_turn_streaming_metrics(self, turn_id: int, **metrics: Any) -> None:
        """Merges streaming, preemptive, EOU, LLM, and TTS metrics for a turn."""
        if turn_id <= 0:
            return
        entry = self._turn_streaming_metrics.setdefault(turn_id, {})
        for k, v in metrics.items():
            if v is not None:
                entry[k] = v

    def increment_turn_streaming_counter(self, turn_id: int, key: str, amount: int = 1) -> None:
        """Increments a per-turn streaming counter (e.g. preemptive_cancelled_count)."""
        if turn_id <= 0:
            return
        entry = self._turn_streaming_metrics.setdefault(turn_id, {})
        entry[key] = int(entry.get(key, 0)) + amount

    def _determine_qualification(self, session: ConversationSession) -> tuple[str, str]:
        """Calculates lead qualification status based on extracted facts."""
        facts = session.facts
        if facts.opted_out:
            return "OPTED_OUT", "Caller declined or opted out"

        has_location = bool(facts.location)
        has_budget = facts.budget_rupees is not None and facts.budget_rupees > 0
        has_bhk = bool(facts.bedrooms or facts.property_type)

        if has_location and has_budget and has_bhk:
            return "FULLY_QUALIFIED", "Location, Budget, and Property Type captured"
        if has_location and has_budget:
            return "QUALIFIED", "Location and Budget captured"
        if has_location or has_budget or has_bhk:
            return "PARTIALLY_QUALIFIED", "Some requirements collected"
        return "UNQUALIFIED", "No critical requirements collected"

    def _build_call_data(
        self,
        session: ConversationSession,
        latency: LatencyTracker,
        agent_history: Any = None,
    ) -> dict[str, Any]:
        """Constructs the comprehensive dictionary of all call metrics and conversation data."""
        if not self.end_time:
            self.end_time = datetime.datetime.now(datetime.timezone.utc)
            self.end_ms = time.time() * 1000

        duration_sec = max(0.0, round((self.end_ms - self.start_ms) / 1000.0, 2))
        qualification_code, qualification_desc = self._determine_qualification(session)

        # Merge assistant responses from AgentSession history if available
        history_replies: list[str] = []
        if agent_history and hasattr(agent_history, "messages"):
            try:
                for msg in agent_history.messages:
                    if getattr(msg, "role", None) == "assistant":
                        txt = getattr(msg, "text_content", None) or ""
                        if txt.strip():
                            history_replies.append(txt.strip())
            except Exception as e:
                log.debug("call_logger.history_parse_error", error=str(e))

        # Build turns list
        turns_data: list[dict[str, Any]] = []
        for idx, turn in enumerate(session._all_turns):
            tid = turn.turn_id
            turn_latency = latency.get_turn(tid)
            latency_report = turn_latency.to_report() if turn_latency else {}

            agent_info = self._turn_agent_replies.get(tid, {})
            agent_text = agent_info.get("reply", "")
            response_mode = agent_info.get("mode", "UNKNOWN")

            # Fallback to history messages if agent_text is blank or generic placeholder
            if (not agent_text or agent_text == "[LLM Generation]") and idx < len(history_replies):
                agent_text = history_replies[idx]
                if response_mode == "UNKNOWN":
                    response_mode = "LLM_GENERATE"

            turns_data.append({
                "turn_id": tid,
                "timestamp_start_ms": turn.timestamp_start_ms,
                "timestamp_end_ms": turn.timestamp_end_ms,
                "duration_ms": round(max(0.0, turn.timestamp_end_ms - turn.timestamp_start_ms), 1),
                "caller_raw_transcript": turn.raw_transcript,
                "caller_normalized_transcript": turn.normalized_transcript,
                "agent_response": agent_text,
                "response_mode": response_mode,
                "latency_breakdown": latency_report.get("intervals_ms", {}),
                "e2e_latency_ms": latency_report.get("intervals_ms", {}).get("e2e_turn_end__first_audio"),
                "streaming_metrics": self._turn_streaming_metrics.get(tid, {}),
            })

        # Build corrections list
        corrections_data = [
            {
                "turn_id": c.turn_id,
                "field": c.field,
                "old_value": c.old_value,
                "new_value": c.new_value,
                "raw_text": c.raw_text,
                "timestamp_ms": c.timestamp_ms,
            }
            for c in session.corrections
        ]

        facts = session.facts
        facts_data = {
            "location": facts.location,
            "budget_rupees": facts.budget_rupees,
            "budget_formatted": format_rupees(facts.budget_rupees),
            "property_type": facts.property_type,
            "bedrooms": facts.bedrooms,
            "timeline": facts.timeline,
            "preferences": facts.preferences,
            "consent": facts.consent,
            "opted_out": facts.opted_out,
            "qualification_status": qualification_code,
            "qualification_description": qualification_desc,
        }

        latency_summary = latency.session_summary()

        # Aggregate live-streaming & preemptive metrics across turns
        all_sm = [t.get("streaming_metrics", {}) for t in turns_data]
        llm_ttfts = [sm["llm_ttft_ms"] for sm in all_sm if sm.get("llm_ttft_ms") is not None]
        tts_ttfbs = [sm["tts_ttfb_ms"] for sm in all_sm if sm.get("tts_ttfb_ms") is not None]
        eou_delays = [sm["eou_delay_ms"] for sm in all_sm if sm.get("eou_delay_ms") is not None]
        preemptive_hits = sum(1 for sm in all_sm if sm.get("preemptive_hit") is True)
        preemptive_cancels = sum(int(sm.get("preemptive_cancelled_count", 0)) for sm in all_sm)
        total_prompt_toks = sum(int(sm.get("llm_prompt_tokens", 0)) for sm in all_sm)
        total_cached_toks = sum(int(sm.get("llm_cached_tokens", 0)) for sm in all_sm)
        streaming_summary = {
            "preemptive_hits": preemptive_hits,
            "total_turns": len(turns_data),
            "preemptive_cancelled_attempts": preemptive_cancels,
            "avg_llm_ttft_ms": round(sum(llm_ttfts) / len(llm_ttfts), 1) if llm_ttfts else None,
            "avg_tts_ttfb_ms": round(sum(tts_ttfbs) / len(tts_ttfbs), 1) if tts_ttfbs else None,
            "avg_eou_delay_ms": round(sum(eou_delays) / len(eou_delays), 1) if eou_delays else None,
            "total_prompt_tokens": total_prompt_toks,
            "total_cached_tokens": total_cached_toks,
        }

        call_data: dict[str, Any] = {
            "session_id": self.session_id,
            "room_name": self.room_name,
            "status": self.status,
            "started_at": self.start_time.isoformat(),
            "ended_at": self.end_time.isoformat(),
            "duration_seconds": duration_sec,
            "total_turns": len(turns_data),
            "final_stage": session.stage,
            "lead_facts": facts_data,
            "corrections": corrections_data,
            "latency_summary": latency_summary,
            "streaming_summary": streaming_summary,
            "turns": turns_data,
        }
        return call_data

    def _generate_readable_transcript(self, call_data: dict[str, Any]) -> str:
        """Generates a clean, readable text report suitable for viewing in terminal or Notepad."""
        facts = call_data["lead_facts"]
        lat = call_data["latency_summary"]

        bhk_str = f"{facts['bedrooms']} BHK" if facts.get("bedrooms") else "Not specified"
        lines = [
            "=" * 78,
            f"TELUGU VOICE AGENT — CALL RECORD: {call_data['session_id']}",
            "=" * 78,
            f"Status           : {call_data['status']}",
            f"Started At       : {call_data['started_at']}",
            f"Ended At         : {call_data['ended_at']}",
            f"Duration         : {call_data['duration_seconds']}s",
            f"Total Turns      : {call_data['total_turns']}",
            f"Final Stage      : {call_data['final_stage']}",
            f"Lead Status      : {facts['qualification_status']} ({facts['qualification_description']})",
            "-" * 78,
            "CUSTOMER REQUIREMENTS EXTRACTED:",
            f"  • Location      : {facts['location'] or 'Not specified'}",
            f"  • Budget        : {facts['budget_formatted']}",
            f"  • Property Type : {facts['property_type'] or 'Not specified'}",
            f"  • Bedrooms      : {bhk_str}",
            f"  • Timeline      : {facts['timeline'] or 'Not specified'}",
            f"  • Preferences   : {', '.join(facts['preferences']) if facts['preferences'] else 'None'}",
            f"  • WhatsApp OK   : {'Yes' if facts['consent'] is True else ('No' if facts['consent'] is False else 'Not asked')}",
            f"  • Opted Out     : {'YES' if facts['opted_out'] else 'No'}",
            "-" * 78,
            "LATENCY PERFORMANCE (Phase 1 Target: <1000ms):",
            f"  • Avg E2E Delay : {lat.get('avg_e2e_ms', 'N/A')} ms",
            f"  • Min E2E Delay : {lat.get('min_e2e_ms', 'N/A')} ms",
            f"  • Max E2E Delay : {lat.get('max_e2e_ms', 'N/A')} ms",
        ]

        ss = call_data.get("streaming_summary") or {}
        if ss:
            lines.extend([
                "-" * 78,
                "LIVE-STREAMING & PREEMPTIVE METRICS:",
                f"  • Preemptive Hits      : {ss.get('preemptive_hits', 0)} / {ss.get('total_turns', 0)} turns (cancelled: {ss.get('preemptive_cancelled_attempts', 0)})",
                f"  • Avg EOU Wait         : {ss.get('avg_eou_delay_ms', 'N/A')} ms",
                f"  • Avg Groq LLM TTFT    : {ss.get('avg_llm_ttft_ms', 'N/A')} ms",
                f"  • Avg Cartesia TTFB    : {ss.get('avg_tts_ttfb_ms', 'N/A')} ms",
                f"  • Prompt Cache Tokens  : {ss.get('total_cached_tokens', 0)} / {ss.get('total_prompt_tokens', 0)} cached",
            ])

        if call_data.get("corrections"):
            lines.append("-" * 78)
            lines.append("FACT CORRECTIONS AUDIT TRAIL:")
            for c in call_data["corrections"]:
                lines.append(
                    f"  • Turn {c['turn_id']}: {c['field']} changed from '{c['old_value']}' "
                    f"to '{c['new_value']}' (caller said: \"{c['raw_text']}\")"
                )

        lines.append("-" * 78)
        lines.append("CONVERSATION DIALOGUE:")
        lines.append("  [Assistant - Greeting]:")
        lines.append("    నమస్కారం అండి! నేను మీ రియల్ ఎస్టేట్ అసిస్టెంట్ ని. మీకు ఎలాంటి ప్రాపర్టీ కావాలి?")
        lines.append("")

        for t in call_data.get("turns", []):
            e2e_str = f"{t['e2e_latency_ms']}ms" if t.get("e2e_latency_ms") is not None else "N/A"
            lines.append(f"  --- Turn #{t['turn_id']} (E2E Latency: {e2e_str}) ---")
            lines.append(f"    Caller (Raw)        : {t['caller_raw_transcript']}")
            if t['caller_normalized_transcript'] != t['caller_raw_transcript']:
                lines.append(f"    Caller (Normalized) : {t['caller_normalized_transcript']}")
            lines.append(f"    Agent [{t['response_mode']}] : {t['agent_response'] or '(Audio playout)'}")
            if t.get("latency_breakdown"):
                lb = t["latency_breakdown"]
                metrics = [f"{k}={v}ms" for k, v in lb.items() if v is not None]
                if metrics:
                    lines.append(f"    Timing Metrics      : {', '.join(metrics)}")
            if t.get("streaming_metrics"):
                sm = t["streaming_metrics"]
                sm_parts = [f"{k}={v}" for k, v in sm.items() if v is not None]
                if sm_parts:
                    lines.append(f"    Streaming Metrics   : {', '.join(sm_parts)}")
            lines.append("")

        lines.append("=" * 78)
        return "\n".join(lines)

    def save_call(
        self,
        session: ConversationSession,
        latency: LatencyTracker,
        status: Optional[str] = None,
        agent_history: Any = None,
    ) -> dict[str, Any]:
        """
        Saves the complete call records to:
          - call_logs/call_<session_id>_<timestamp>.json
          - call_logs/call_<session_id>_<timestamp>.txt
          - Appends one summary line to call_logs/call_logs.jsonl
        """
        if self._saved:
            return {"status": "already_saved", "session_id": self.session_id}

        if status:
            self.status = status
        elif self.status == "IN_PROGRESS":
            self.status = "COMPLETED"

        self.end_time = datetime.datetime.now(datetime.timezone.utc)
        self.end_ms = time.time() * 1000

        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        call_data = self._build_call_data(session, latency, agent_history)
        timestamp_slug = self.start_time.strftime("%Y%m%d_%H%M%S")
        file_prefix = f"call_{self.session_id}_{timestamp_slug}"

        json_path = self.logs_dir / f"{file_prefix}.json"
        txt_path = self.logs_dir / f"{file_prefix}.txt"
        jsonl_path = self.logs_dir / "call_logs.jsonl"

        # 1. Write structured JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(call_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("call_logger.write_json_failed", error=str(e), path=str(json_path))

        # 2. Write human-readable text transcript
        try:
            transcript_text = self._generate_readable_transcript(call_data)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
        except Exception as e:
            log.warning("call_logger.write_txt_failed", error=str(e), path=str(txt_path))

        # 3. Append to JSONL master index
        try:
            index_entry = {
                "session_id": call_data["session_id"],
                "room_name": call_data["room_name"],
                "status": call_data["status"],
                "started_at": call_data["started_at"],
                "ended_at": call_data["ended_at"],
                "duration_seconds": call_data["duration_seconds"],
                "total_turns": call_data["total_turns"],
                "qualification": call_data["lead_facts"]["qualification_status"],
                "location": call_data["lead_facts"]["location"],
                "budget_rupees": call_data["lead_facts"]["budget_rupees"],
                "budget_formatted": call_data["lead_facts"]["budget_formatted"],
                "property_type": call_data["lead_facts"]["property_type"],
                "bedrooms": call_data["lead_facts"]["bedrooms"],
                "avg_e2e_ms": call_data["latency_summary"].get("avg_e2e_ms"),
                "json_file": json_path.name,
                "txt_file": txt_path.name,
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("call_logger.write_jsonl_failed", error=str(e), path=str(jsonl_path))

        self._saved = True
        log.info(
            "call_logger.saved",
            session_id=self.session_id,
            duration=call_data["duration_seconds"],
            turns=call_data["total_turns"],
            qualification=call_data["lead_facts"]["qualification_status"],
            json_file=str(json_path),
            txt_file=str(txt_path),
        )

        return {
            "status": "saved",
            "session_id": self.session_id,
            "json_path": str(json_path),
            "txt_path": str(txt_path),
            "jsonl_path": str(jsonl_path),
            "data": call_data,
        }
