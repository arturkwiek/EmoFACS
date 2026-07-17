#!/usr/bin/env python
# scripts/log_factors.py
"""
Record daily external factors that may influence mood — for later
correlation with the logged emotion measurements.

Usage
-----
Interactive (asks for each value, Enter skips):

    python scripts/log_factors.py

Non-interactive via flags (only given values are updated):

    python scripts/log_factors.py --sleep 7.5 --stress 2 --caffeine 3 \
        --exercise 40 --mood 4 --note "long walk, sunny day"

    python scripts/log_factors.py --date 2026-07-16 --sleep 6
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.emolog.logger import connect, default_db_path

_FIELDS = [
    # (column, flag, prompt, cast)
    ("sleep_hours",   "--sleep",    "Sleep hours last night (e.g. 7.5)", float),
    ("sleep_quality", "--sleepq",   "Sleep quality 1-5",                 int),
    ("caffeine",      "--caffeine", "Caffeine servings today",           int),
    ("alcohol",       "--alcohol",  "Alcohol units yesterday",           int),
    ("exercise_min",  "--exercise", "Exercise minutes today",            int),
    ("stress",        "--stress",   "Stress level 1-5",                  int),
    ("mood_self",     "--mood",     "Subjective mood 1-5",               int),
    ("note",          "--note",     "Free-form note",                    str),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log daily mood factors.")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Day to record (YYYY-MM-DD, default: today).")
    parser.add_argument("--db", default=None,
                        help="Path to the emolog SQLite database "
                             "(default: <repo>/data/emolog.db).")
    for col, flag, prompt, cast in _FIELDS:
        parser.add_argument(flag, dest=col, type=cast, default=None,
                            help=prompt + ".")
    return parser.parse_args()


def interactive_fill(values: dict) -> dict:
    print("Enter today's factors (press Enter to skip a question).")
    for col, _flag, prompt, cast in _FIELDS:
        if values.get(col) is not None:
            continue  # already provided via flag
        raw = input(f"  {prompt}: ").strip()
        if not raw:
            continue
        try:
            values[col] = cast(raw)
        except ValueError:
            print(f"    [skipped — could not parse {raw!r}]")
    return values


def main() -> None:
    args = parse_args()
    values = {col: getattr(args, col) for col, *_ in _FIELDS}

    # If no flag was given at all, fall back to the interactive wizard.
    if all(v is None for v in values.values()):
        values = interactive_fill(values)

    provided = {k: v for k, v in values.items() if v is not None}
    if not provided:
        print("Nothing to save.")
        return

    conn = connect(args.db)
    conn.execute("INSERT OR IGNORE INTO daily_factors (date) VALUES (?)",
                 (args.date,))
    sets = ", ".join(f"{col} = ?" for col in provided)
    conn.execute(
        f"UPDATE daily_factors SET {sets}, updated_at = ? WHERE date = ?",
        (*provided.values(),
         datetime.now().isoformat(timespec="seconds"), args.date),
    )
    conn.commit()
    conn.close()

    db_display = args.db or default_db_path()
    print(f"Saved {len(provided)} value(s) for {args.date} → {db_display}")


if __name__ == "__main__":
    main()
