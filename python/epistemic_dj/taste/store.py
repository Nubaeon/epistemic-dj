"""Lightweight, standalone SQLite store for taste findings/patterns.

Deliberately NOT empirica.SessionDatabase (user decision, 2026-07-19) --
this keeps human taste data separate from this session's own AI-epistemic
tracking, while borrowing the conceptual pattern (findings -> distilled
patterns, confidence decay). Intended future path: plug into a dedicated
'epistemic-dj-lab' Empirica practice once there's an actual module/skills
built around it -- not before (see docs/dev/architecture.md).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from epistemic_dj.models import (
    MusicVectors,
    TasteFinding,
    TastePattern,
    TastePatternType,
    TasteProfile,
    UserTasteVectors,
)

DEFAULT_DB_PATH = Path(__file__).parent / "taste.db"

MIN_SIGNAL_FOR_VECTORS = 3
DEFAULT_DECAY_FACTOR = 0.7
DEFAULT_DECAY_FLOOR = 0.3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS taste_findings (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    impact REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS taste_patterns (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    vectors TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_findings_user ON taste_findings(user_id);
CREATE INDEX IF NOT EXISTS idx_patterns_user ON taste_patterns(user_id);
"""


class PatternNotFoundError(KeyError):
    pass


class TasteStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def log_finding(self, user_id: str, content: str, impact: float = 0.5) -> TasteFinding:
        finding = TasteFinding(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            impact=impact,
            created_at=datetime.now(UTC),
        )
        self._conn.execute(
            "INSERT INTO taste_findings (id, user_id, content, impact, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (finding.id, finding.user_id, finding.content, finding.impact,
             finding.created_at.isoformat()),
        )
        self._conn.commit()
        return finding

    def log_pattern(
        self,
        user_id: str,
        content: str,
        pattern_type: TastePatternType,
        confidence: float,
        vectors: MusicVectors | None = None,
    ) -> TastePattern:
        now = datetime.now(UTC)
        pattern = TastePattern(
            id=str(uuid.uuid4()),
            user_id=user_id,
            pattern_type=pattern_type,
            content=content,
            confidence=confidence,
            vectors=vectors,
            created_at=now,
            updated_at=now,
        )
        self._conn.execute(
            "INSERT INTO taste_patterns "
            "(id, user_id, pattern_type, content, confidence, vectors, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pattern.id, pattern.user_id, pattern.pattern_type.value, pattern.content,
                pattern.confidence,
                pattern.vectors.model_dump_json() if pattern.vectors else None,
                pattern.created_at.isoformat(), pattern.updated_at.isoformat(),
            ),
        )
        self._conn.commit()
        return pattern

    def decay_pattern(
        self,
        pattern_id: str,
        factor: float = DEFAULT_DECAY_FACTOR,
        floor: float = DEFAULT_DECAY_FLOOR,
    ) -> TastePattern:
        """Explicitly decay a pattern's confidence -- called when the interviewing
        Claude judges a new finding contradicts it. Not automatic (see module
        docstring): no semantic-similarity infra exists yet to detect
        contradictions on its own.
        """
        row = self._conn.execute(
            "SELECT confidence FROM taste_patterns WHERE id = ?", (pattern_id,)
        ).fetchone()
        if row is None:
            raise PatternNotFoundError(pattern_id)
        new_confidence = max(floor, row[0] * factor)
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE taste_patterns SET confidence = ?, updated_at = ? WHERE id = ?",
            (new_confidence, now, pattern_id),
        )
        self._conn.commit()
        return self._get_pattern(pattern_id)

    def _get_pattern(self, pattern_id: str) -> TastePattern:
        row = self._conn.execute(
            "SELECT id, user_id, pattern_type, content, confidence, vectors, "
            "created_at, updated_at FROM taste_patterns WHERE id = ?",
            (pattern_id,),
        ).fetchone()
        if row is None:
            raise PatternNotFoundError(pattern_id)
        return _row_to_pattern(row)

    def get_findings(self, user_id: str) -> list[TasteFinding]:
        rows = self._conn.execute(
            "SELECT id, user_id, content, impact, created_at FROM taste_findings "
            "WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        ).fetchall()
        return [
            TasteFinding(
                id=r[0], user_id=r[1], content=r[2], impact=r[3],
                created_at=datetime.fromisoformat(r[4]),
            )
            for r in rows
        ]

    def get_patterns(self, user_id: str) -> list[TastePattern]:
        rows = self._conn.execute(
            "SELECT id, user_id, pattern_type, content, confidence, vectors, "
            "created_at, updated_at FROM taste_patterns WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        ).fetchall()
        return [_row_to_pattern(r) for r in rows]

    def get_profile(self, user_id: str) -> TasteProfile:
        findings = self.get_findings(user_id)
        patterns = self.get_patterns(user_id)
        return TasteProfile(
            user_id=user_id,
            findings=findings,
            patterns=patterns,
            vectors=_heuristic_vectors(findings, patterns),
        )


def _heuristic_vectors(
    findings: list[TasteFinding], patterns: list[TastePattern]
) -> UserTasteVectors | None:
    """Sprint 2 MVP: derived from interview signal VOLUME, not real behavioral
    telemetry (skip/replay/collect), which doesn't exist yet. Returns None
    below MIN_SIGNAL_FOR_VECTORS rather than fabricating precision from
    almost nothing.
    """
    total_signal = len(findings) + len(patterns)
    if total_signal < MIN_SIGNAL_FOR_VECTORS:
        return None

    avg_confidence = (
        sum(p.confidence for p in patterns) / len(patterns) if patterns else 0.5
    )
    density = min(1.0, total_signal / 10)

    return UserTasteVectors(
        know=min(1.0, len(patterns) / 5),
        do=0.3,  # no catalog matched against this profile yet
        context=0.4,  # onboarding interview only, no session-context signal yet
        clarity=avg_confidence,
        coherence=avg_confidence,  # no contradiction-detection yet -- approximated via confidence
        signal=min(1.0, len(findings) / 10),
        density=density,
        state=0.3,  # no ongoing-session awareness yet
        change=0.0,  # first snapshot, nothing to compare against
        completion=density,
        impact=0.5,
        engagement=min(1.0, total_signal / 8),
        uncertainty=1.0 - avg_confidence,
    )


def _row_to_pattern(row: tuple) -> TastePattern:
    return TastePattern(
        id=row[0],
        user_id=row[1],
        pattern_type=TastePatternType(row[2]),
        content=row[3],
        confidence=row[4],
        vectors=MusicVectors.model_validate_json(row[5]) if row[5] else None,
        created_at=datetime.fromisoformat(row[6]),
        updated_at=datetime.fromisoformat(row[7]),
    )
