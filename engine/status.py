#!/usr/bin/env python3
"""
Shows the progress of the running knowledge extraction.

Usage:
    python3 engine/status.py

Reads the checkpoint file and the log, and estimates the remaining time
from the progress made since the last time you ran this command.
"""

import json
import sys
import time
from pathlib import Path

CHECKPOINT = Path(__file__).parent / "knowledge_base.partial.json"
LOG_FILE = Path("/tmp/extraction_run.log")
STATE_FILE = Path(__file__).parent / ".status_state"
TOTAL_PAGES = 328


def main():
    if not CHECKPOINT.exists():
        print("No extraction running (no checkpoint file yet).")
        print("Start it with:  python3 engine/extract_knowledge.py")
        sys.exit(0)

    partial = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    done = len(partial.get("processed_pages", []))
    rules = len(partial.get("rules", []))
    failed = partial.get("failed_pages", [])
    remaining = TOTAL_PAGES - done

    print("=" * 46)
    print("📚 Samudrik Shastra knowledge extraction")
    print("=" * 46)
    print(f"Pages:   {done}/{TOTAL_PAGES}  ({100 * done // TOTAL_PAGES}%)")
    bar_width = 30
    filled = int(bar_width * done / TOTAL_PAGES)
    print(f"         [{'#' * filled}{'-' * (bar_width - filled)}]")
    print(f"Rules:   {rules}")
    print(f"Failed:  {failed if failed else 'none'}")

    # ETA: compare with the state saved on the previous run of this command.
    now = time.time()
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            pages_gained = done - state.get("pages", 0)
            seconds_elapsed = now - state.get("time", now)
            if pages_gained > 0 and seconds_elapsed > 10:
                rate = seconds_elapsed / pages_gained
                eta_minutes = int(remaining * rate / 60)
                print(f"Rate:    {rate:.0f}s per page")
                print(f"ETA:     ~{eta_minutes // 60}h {eta_minutes % 60}m")
        except Exception:
            pass
    STATE_FILE.write_text(json.dumps({"time": now, "pages": done}), encoding="utf-8")

    if LOG_FILE.exists():
        lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
        print("\nLast log lines:")
        for line in lines[-3:]:
            print(f"  {line}")

    print("\nTip: run this command again later to get the rate and ETA.")
    print("     Or watch live output:  tail -f /tmp/extraction_run.log")


if __name__ == "__main__":
    main()
