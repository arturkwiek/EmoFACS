#!/usr/bin/env python
# scripts/analyze_history.py
"""
Day-by-day analysis of logged emotion measurements, optionally correlated
with the manually recorded daily factors (sleep, caffeine, stress, ...).

Usage
-----
    python scripts/analyze_history.py                 # full history
    python scripts/analyze_history.py --days 30       # last 30 days
    python scripts/analyze_history.py --plot out.png  # custom plot path

Outputs
-------
- A per-day summary table printed to the console.
- Correlations between daily mood metrics and recorded factors
  (needs at least 3 overlapping days).
- A PNG chart (default: <repo>/data/mood_report.png).

Metrics
-------
valence / arousal   daily mean of the circumplex coordinates
mood_index          mean(Happy) − mean(Sad + Angry + Fearful + Disgusted)
                    → positive = good day, negative = rough day
volatility          std of valence within the day (emotional stability)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.emolog.logger import connect, default_db_path

_NEGATIVE = ["sad", "angry", "fearful", "disgusted"]
_EMOTIONS = ["happy", "neutral", "surprised", "sad", "angry",
             "fearful", "disgusted"]
_FACTORS = ["sleep_hours", "sleep_quality", "caffeine", "alcohol",
            "exercise_min", "stress", "mood_self"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse emotion history.")
    parser.add_argument("--days", type=int, default=None,
                        help="Only include the last N days (default: all).")
    parser.add_argument("--db", default=None,
                        help="Path to the emolog SQLite database "
                             "(default: <repo>/data/emolog.db).")
    parser.add_argument("--plot", default=None,
                        help="Output path for the PNG chart "
                             "(default: <repo>/data/mood_report.png).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation.")
    return parser.parse_args()


def load_daily(conn, days: int | None) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM measurements", conn)
    if df.empty:
        return df
    if days:
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        df = df[df["date"] >= cutoff]
    if df.empty:
        return df

    grouped = df.groupby("date")
    daily = pd.DataFrame({
        "n": grouped.size(),
        "valence": grouped["valence"].mean(),
        "arousal": grouped["arousal"].mean(),
        "volatility": grouped["valence"].std(),
    })
    for emo in _EMOTIONS:
        daily[emo] = grouped[emo].mean()
    daily["mood_index"] = daily["happy"] - daily[_NEGATIVE].sum(axis=1)
    daily["dominant"] = daily[_EMOTIONS].idxmax(axis=1)
    return daily.reset_index()


def load_factors(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM daily_factors", conn)


def print_summary(daily: pd.DataFrame) -> None:
    print(f"\n=== Daily summary ({len(daily)} day(s)) ===\n")
    cols = ["date", "n", "valence", "arousal", "volatility",
            "mood_index", "dominant"]
    view = daily[cols].copy()
    for c in ["valence", "arousal", "volatility", "mood_index"]:
        view[c] = view[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "  —  ")
    print(view.to_string(index=False))


def print_correlations(daily: pd.DataFrame, factors: pd.DataFrame) -> None:
    if factors.empty:
        print("\n[correlations] No daily factors recorded yet — "
              "use scripts/log_factors.py to start collecting them.")
        return

    merged = daily.merge(factors, on="date", how="inner")
    metrics = ["valence", "mood_index", "volatility", "arousal"]
    print("\n=== Correlations (daily metrics vs factors) ===\n")

    any_printed = False
    for factor in _FACTORS:
        if factor not in merged or merged[factor].notna().sum() < 3:
            continue
        for metric in metrics:
            sub = merged[[metric, factor]].dropna()
            if len(sub) < 3 or sub[factor].nunique() < 2:
                continue
            r = sub[metric].corr(sub[factor])
            if pd.isna(r):
                continue
            flag = " ***" if abs(r) >= 0.7 else ("  **" if abs(r) >= 0.5 else "")
            print(f"  {metric:<11} vs {factor:<13} r = {r:+.2f} "
                  f"(n={len(sub)}){flag}")
            any_printed = True
    if not any_printed:
        print("  Not enough overlapping days yet (need ≥ 3 with both "
              "measurements and a recorded factor).")
    else:
        print("\n  ** |r| ≥ 0.5   *** |r| ≥ 0.7   "
              "(small n → treat as hints, not conclusions)")


def make_plot(daily: pd.DataFrame, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = pd.to_datetime(daily["date"])
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, len(daily) * 0.6), 8), sharex=True)

    # Panel 1 — valence / arousal / mood index over days
    ax1.axhline(0, color="#999", lw=0.8)
    ax1.plot(x, daily["valence"], "o-", color="#2a9d8f", label="valence")
    ax1.plot(x, daily["arousal"], "s--", color="#e9c46a", label="arousal")
    ax1.plot(x, daily["mood_index"], "^-", color="#264653", label="mood index")
    vol = daily["volatility"].fillna(0)
    ax1.fill_between(x, daily["valence"] - vol, daily["valence"] + vol,
                     color="#2a9d8f", alpha=0.15, label="±volatility")
    ax1.set_ylabel("value")
    ax1.set_title("Mood over time")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2 — stacked emotion composition per day
    colors = {"happy": "#3cb44b", "neutral": "#b4b4b4", "surprised": "#ffd8b1",
              "sad": "#4363d8", "angry": "#e6194b", "fearful": "#911eb4",
              "disgusted": "#808000"}
    bottom = np.zeros(len(daily))
    for emo in _EMOTIONS:
        vals = daily[emo].fillna(0).to_numpy()
        ax2.bar(x, vals, bottom=bottom, width=0.6,
                color=colors[emo], label=emo)
        bottom += vals
    ax2.set_ylabel("mean probability")
    ax2.set_title("Daily emotion composition")
    ax2.legend(loc="upper left", fontsize=8, ncol=4)
    ax2.grid(alpha=0.3, axis="y")

    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n[plot] Saved chart → {out_path}")


def main() -> None:
    args = parse_args()
    conn = connect(args.db)

    daily = load_daily(conn, args.days)
    if daily.empty:
        print("No measurements found yet. Run scripts/run_on_webcam.py "
              "(logging is on by default) and come back.")
        conn.close()
        return

    factors = load_factors(conn)
    conn.close()

    print_summary(daily)
    print_correlations(daily, factors)

    if not args.no_plot:
        default_plot = os.path.join(
            os.path.dirname(default_db_path()), "mood_report.png")
        make_plot(daily, args.plot or default_plot)


if __name__ == "__main__":
    main()
