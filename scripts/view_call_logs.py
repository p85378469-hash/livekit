#!/usr/bin/env python3
# ==============================================================================
# scripts/view_call_logs.py — CLI Tool to View, Filter, and Export Call Logs
#
# USAGE EXAMPLES:
#   python scripts/view_call_logs.py                  # Show overview table of all calls
#   python scripts/view_call_logs.py --latest         # View full transcript of most recent call
#   python scripts/view_call_logs.py --id 7c9c02b5    # View full transcript for specific call ID
#   python scripts/view_call_logs.py --stats          # Show lead qualification & latency metrics
#   python scripts/view_call_logs.py --export-csv     # Export summary of all calls to CSV
# ==============================================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_all_calls(logs_dir: Path) -> list[dict]:
    """Loads all call index records from call_logs.jsonl or fallback to individual JSON files."""
    jsonl_path = logs_dir / "call_logs.jsonl"
    calls: list[dict] = []

    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            calls.append(json.loads(line))
                        except Exception:
                            pass
            return calls
        except Exception:
            pass

    # Fallback to scanning individual json files
    for p in sorted(logs_dir.glob("call_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                facts = d.get("lead_facts", {})
                lat = d.get("latency_summary", {})
                calls.append({
                    "session_id": d.get("session_id"),
                    "room_name": d.get("room_name"),
                    "status": d.get("status"),
                    "started_at": d.get("started_at"),
                    "ended_at": d.get("ended_at"),
                    "duration_seconds": d.get("duration_seconds", 0),
                    "total_turns": d.get("total_turns", 0),
                    "qualification": facts.get("qualification_status", "UNKNOWN"),
                    "location": facts.get("location"),
                    "budget_rupees": facts.get("budget_rupees"),
                    "budget_formatted": facts.get("budget_formatted", "N/A"),
                    "property_type": facts.get("property_type"),
                    "bedrooms": facts.get("bedrooms"),
                    "avg_e2e_ms": lat.get("avg_e2e_ms"),
                    "json_file": p.name,
                    "txt_file": p.with_suffix(".txt").name,
                })
        except Exception:
            pass

    return calls


def display_overview_table(calls: list[dict], logs_dir: Path) -> None:
    """Prints a styled ASCII summary table of all recorded calls."""
    if not calls:
        print("\n" + "=" * 78)
        print(" [!] No call logs found in:", logs_dir.resolve())
        print(" When a call completes or is simulated, logs are automatically saved here.")
        print(" Run the agent or simulation to generate your first call log:")
        print("   python scripts/test_call_simulation.py")
        print("=" * 78 + "\n")
        return

    # Calculate statistics
    total_calls = len(calls)
    qualified = sum(1 for c in calls if c.get("qualification") in ("FULLY_QUALIFIED", "QUALIFIED"))
    opt_outs = sum(1 for c in calls if c.get("qualification") == "OPTED_OUT")
    avg_dur = round(sum(c.get("duration_seconds", 0) for c in calls) / total_calls, 1)

    latencies = [c["avg_e2e_ms"] for c in calls if c.get("avg_e2e_ms") is not None]
    avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else None

    print("\n" + "=" * 88)
    print(f"               TELUGU VOICE AGENT — CALL LOGS REPOSITORY ({total_calls} Total Calls)")
    print("=" * 88)
    print(f" Summary: {qualified} Qualified Leads | {opt_outs} Opt-Outs | Avg Duration: {avg_dur}s | "
          f"Avg E2E Latency: {avg_lat if avg_lat else 'N/A'} ms")
    print("-" * 88)

    header = f"{'SESSION ID':<10} | {'DATE & TIME (UTC)':<19} | {'DUR':<5} | {'TURNS':<5} | {'STATUS':<15} | {'LOCATION':<12} | {'BUDGET':<15}"
    print(header)
    print("-" * 88)

    for c in calls:
        sid = str(c.get("session_id", ""))[:8]
        started = str(c.get("started_at", "")).replace("T", " ")[:19]
        dur = f"{round(float(c.get('duration_seconds', 0)), 1)}s"
        turns = str(c.get("total_turns", 0))
        qual = str(c.get("qualification", ""))
        loc = str(c.get("location") or "-")[:12]
        budget = str(c.get("budget_formatted") or "-")[:15]

        print(f"{sid:<10} | {started:<19} | {dur:<5} | {turns:<5} | {qual:<15} | {loc:<12} | {budget:<15}")

    print("-" * 88)
    print(" QUICK COMMANDS:")
    print("   • Read latest call transcript : python scripts/view_call_logs.py --latest")
    print("   • Read specific call          : python scripts/view_call_logs.py --id <SESSION_ID>")
    print("   • Export to Excel CSV         : python scripts/view_call_logs.py --export-csv")
    print("   • Run simulation test call    : python scripts/test_call_simulation.py")
    print("=" * 88 + "\n")


def display_call_detail(session_id: str, logs_dir: Path) -> None:
    """Displays the full text transcript of a specific call."""
    # Find matching txt file
    matches = list(logs_dir.glob(f"call_{session_id}*.txt"))
    if not matches:
        # Search inside json files in case of prefix match
        for j in logs_dir.glob("call_*.json"):
            if session_id.lower() in j.name.lower():
                txt_cand = j.with_suffix(".txt")
                if txt_cand.exists():
                    matches.append(txt_cand)
                    break

    if not matches:
        print(f"\n[!] Error: No call log found matching session ID: '{session_id}' in {logs_dir}\n")
        return

    txt_file = matches[0]
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            print("\n" + f.read() + "\n")
    except Exception as e:
        print(f"[!] Error reading {txt_file}: {e}")


def display_latest_call(logs_dir: Path) -> None:
    """Displays the most recently saved call transcript."""
    txt_files = sorted(logs_dir.glob("call_*.txt"), key=os.path.getmtime)
    if not txt_files:
        print(f"\n[!] No call logs found in {logs_dir}\n")
        return
    latest_txt = txt_files[-1]
    try:
        with open(latest_txt, "r", encoding="utf-8") as f:
            print("\n" + f.read() + "\n")
    except Exception as e:
        print(f"[!] Error reading {latest_txt}: {e}")


def export_to_csv(calls: list[dict], output_path: Path) -> None:
    """Exports all calls summary data to CSV for analysis or Excel import."""
    if not calls:
        print("[!] No calls to export.")
        return

    fieldnames = [
        "session_id", "status", "started_at", "ended_at", "duration_seconds",
        "total_turns", "qualification", "location", "budget_rupees",
        "budget_formatted", "property_type", "bedrooms", "avg_e2e_ms",
        "json_file", "txt_file"
    ]

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for c in calls:
                writer.writerow(c)
        print(f"\n[✓] Successfully exported {len(calls)} call records to: {output_path.resolve()}\n")
    except Exception as e:
        print(f"[!] Failed to export CSV: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View and inspect Telugu Voice Agent call logs.")
    parser.add_argument(
        "--dir",
        default=os.environ.get("CALL_LOGS_DIR", "call_logs"),
        help="Directory where call logs are saved (default: call_logs)",
    )
    parser.add_argument("--latest", action="store_true", help="Display full transcript of the most recent call")
    parser.add_argument("--id", dest="session_id", help="Display full transcript for the specified session ID")
    parser.add_argument("--list", action="store_true", help="Display list table of all recorded calls")
    parser.add_argument("--json", action="store_true", help="Output raw JSON data for the requested call")
    parser.add_argument(
        "--export-csv",
        nargs="?",
        const="call_logs/exported_calls.csv",
        help="Export all call logs to CSV (default: call_logs/exported_calls.csv)",
    )

    args = parser.parse_args()
    logs_dir = Path(args.dir)

    calls = load_all_calls(logs_dir)

    if args.latest:
        display_latest_call(logs_dir)
    elif args.session_id:
        if args.json:
            for j in logs_dir.glob(f"call_{args.session_id}*.json"):
                with open(j, "r", encoding="utf-8") as f:
                    print(f.read())
                return
        display_call_detail(args.session_id, logs_dir)
    elif args.export_csv:
        csv_file = Path(args.export_csv)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        export_to_csv(calls, csv_file)
    else:
        # Default view: overview table
        display_overview_table(calls, logs_dir)


if __name__ == "__main__":
    main()
