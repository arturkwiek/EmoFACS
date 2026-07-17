# src/emolog/logger.py
"""
Persistent emotion logging to SQLite.

Every pipeline measurement (V/A + 7 emotion probabilities + raw AUs) is
stored so that mood can be analysed over time — day by day — and later
correlated with external factors (sleep, caffeine, exercise, stress, ...)
recorded via scripts/log_factors.py.

Schema
------
sessions        one row per webcam run (started_at, ended_at, source)
measurements    one row per successful detection, linked to a session
daily_factors   one row per calendar day, filled in manually

The database lives in <repo>/data/emolog.db by default. Raw AUs are kept
as a JSON array so they can feed future training of the AU→V/A regressor.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

# Display names used across the project → measurement column names.
_EMOTION_COLUMNS = {
    "Angry": "angry",
    "Disgusted": "disgusted",
    "Fearful": "fearful",
    "Happy": "happy",
    "Sad": "sad",
    "Surprised": "surprised",
    "Neutral": "neutral",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    source      TEXT,
    va_trained  INTEGER NOT NULL DEFAULT 0,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS measurements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    ts          TEXT NOT NULL,
    date        TEXT NOT NULL,              -- YYYY-MM-DD, for fast grouping
    valence     REAL,
    arousal     REAL,
    va_trained  INTEGER NOT NULL DEFAULT 0,
    angry       REAL,
    disgusted   REAL,
    fearful     REAL,
    happy       REAL,
    sad         REAL,
    surprised   REAL,
    neutral     REAL,
    dominant    TEXT,
    aus_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_measurements_date ON measurements(date);
CREATE INDEX IF NOT EXISTS idx_measurements_session ON measurements(session_id);

CREATE TABLE IF NOT EXISTS daily_factors (
    date          TEXT PRIMARY KEY,         -- YYYY-MM-DD
    sleep_hours   REAL,
    sleep_quality INTEGER,                  -- 1-5
    caffeine      INTEGER,                  -- cups / servings
    alcohol       INTEGER,                  -- units
    exercise_min  INTEGER,
    stress        INTEGER,                  -- 1-5
    mood_self     INTEGER,                  -- 1-5 subjective self-rating
    note          TEXT,
    updated_at    TEXT
);
"""


def default_db_path() -> str:
    """<repo>/data/emolog.db, regardless of the current working directory."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    return os.path.join(repo_root, "data", "emolog.db")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open (and initialise, if needed) the emolog database."""
    path = db_path or default_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(_SCHEMA)
    return conn


class EmotionLogger:
    """
    Logs pipeline results for one session.

    Usage
    -----
        logger = EmotionLogger(source="webcam:0", va_trained=False)
        ...
        logger.log(result)      # result = pipeline.run_on_frame(frame)
        ...
        logger.close()

    Results containing an "error" key are silently ignored, so the call
    site does not need to filter.
    """

    def __init__(
        self,
        db_path: str | None = None,
        source: str = "webcam",
        va_trained: bool = False,
    ) -> None:
        self.conn = connect(db_path)
        self.va_trained = va_trained
        cur = self.conn.execute(
            "INSERT INTO sessions (started_at, source, va_trained) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), source, int(va_trained)),
        )
        self.session_id = cur.lastrowid
        self.count = 0
        self.conn.commit()

    def log(self, result: dict, ts: datetime | None = None) -> None:
        """Store one pipeline result. No-op for error results."""
        if not result or "error" in result:
            return

        now = ts or datetime.now()
        emotions = result.get("emotions") or {}
        cols = {name: emotions.get(display)
                for display, name in _EMOTION_COLUMNS.items()}
        dominant = max(emotions, key=emotions.get) if emotions else None

        self.conn.execute(
            """INSERT INTO measurements
               (session_id, ts, date, valence, arousal, va_trained,
                angry, disgusted, fearful, happy, sad, surprised, neutral,
                dominant, aus_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.session_id,
                now.isoformat(timespec="seconds"),
                now.strftime("%Y-%m-%d"),
                result.get("valence"),
                result.get("arousal"),
                int(result.get("va_trained", self.va_trained)),
                cols["angry"], cols["disgusted"], cols["fearful"],
                cols["happy"], cols["sad"], cols["surprised"], cols["neutral"],
                dominant,
                json.dumps(result.get("raw_aus")),
            ),
        )
        self.count += 1
        self.conn.commit()

    def close(self, notes: str | None = None) -> None:
        """Finalise the session row and close the connection."""
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, notes = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), notes, self.session_id),
        )
        self.conn.commit()
        self.conn.close()
