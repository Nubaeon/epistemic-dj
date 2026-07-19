"""Standalone SQLite store for the track-prediction calibration loop.

Deliberately NOT routed through the empirica CLI/package (David's
product-positioning decision, 2026-07-19, docs/dev/track-calibration-loop.md)
-- epistemic-dj must not require Empirica at runtime for its own core
knowledge graph, mirroring the same call already made for TasteStore
(taste/store.py). Same conceptual shape (predict -> measure -> resolve, a
confidence-scoreable forecast) as Empirica's assumption/resolve-artifacts
primitive, reimplemented standalone.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from epistemic_dj.models import BrierResult, MusicVectors, TrackPrediction

DEFAULT_DB_PATH = Path(__file__).parent / "calibration.db"

DEFAULT_TOLERANCE = 0.2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_predictions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    track_ref TEXT NOT NULL,
    track_name TEXT NOT NULL,
    term TEXT NOT NULL,
    predicted_kinetic_energy REAL NOT NULL,
    predicted_vectors TEXT,
    confidence REAL NOT NULL,
    taste_similarity REAL,
    practitioner_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    measured_vectors TEXT,
    verified INTEGER,
    delta REAL,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_term ON track_predictions(term);
CREATE INDEX IF NOT EXISTS idx_predictions_practitioner
    ON track_predictions(practitioner_id);
"""


class PredictionNotFoundError(KeyError):
    pass


class CalibrationStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        # WAL mode from the start -- cheap insurance for Phase C (parallel
        # practitioners writing to the same store), not proven necessary yet
        # at single-practitioner volume but no reason not to have it.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def log_prediction(
        self,
        source: str,
        track_ref: str,
        track_name: str,
        term: str,
        predicted_kinetic_energy: float,
        confidence: float,
        practitioner_id: str = "default",
        predicted_vectors: MusicVectors | None = None,
        taste_similarity: float | None = None,
    ) -> TrackPrediction:
        prediction = TrackPrediction(
            id=str(uuid.uuid4()),
            source=source,
            track_ref=track_ref,
            track_name=track_name,
            term=term,
            predicted_kinetic_energy=predicted_kinetic_energy,
            predicted_vectors=predicted_vectors,
            confidence=confidence,
            taste_similarity=taste_similarity,
            practitioner_id=practitioner_id,
            created_at=datetime.now(UTC),
        )
        self._conn.execute(
            "INSERT INTO track_predictions "
            "(id, source, track_ref, track_name, term, predicted_kinetic_energy, "
            "predicted_vectors, confidence, taste_similarity, practitioner_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                prediction.id, prediction.source, prediction.track_ref,
                prediction.track_name, prediction.term, prediction.predicted_kinetic_energy,
                prediction.predicted_vectors.model_dump_json()
                if prediction.predicted_vectors else None,
                prediction.confidence, prediction.taste_similarity,
                prediction.practitioner_id, prediction.created_at.isoformat(),
            ),
        )
        self._conn.commit()
        return prediction

    def resolve_prediction(
        self,
        prediction_id: str,
        measured_vectors: MusicVectors,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> TrackPrediction:
        row = self._conn.execute(
            "SELECT predicted_kinetic_energy FROM track_predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        if row is None:
            raise PredictionNotFoundError(prediction_id)
        if measured_vectors.kinetic_energy is None:
            raise ValueError("measured_vectors.kinetic_energy is None -- cannot resolve.")

        delta = abs(row[0] - measured_vectors.kinetic_energy)
        verified = delta <= tolerance
        resolved_at = datetime.now(UTC)
        self._conn.execute(
            "UPDATE track_predictions SET measured_vectors = ?, verified = ?, "
            "delta = ?, resolved_at = ? WHERE id = ?",
            (
                measured_vectors.model_dump_json(), int(verified), delta,
                resolved_at.isoformat(), prediction_id,
            ),
        )
        self._conn.commit()
        return self.get_prediction(prediction_id)

    def get_prediction(self, prediction_id: str) -> TrackPrediction:
        row = self._conn.execute(
            "SELECT id, source, track_ref, track_name, term, predicted_kinetic_energy, "
            "predicted_vectors, confidence, taste_similarity, practitioner_id, created_at, "
            "measured_vectors, verified, delta, resolved_at "
            "FROM track_predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        if row is None:
            raise PredictionNotFoundError(prediction_id)
        return _row_to_prediction(row)

    def get_predictions(
        self,
        source: str | None = None,
        term: str | None = None,
        practitioner_id: str | None = None,
        resolved_only: bool = False,
    ) -> list[TrackPrediction]:
        query = (
            "SELECT id, source, track_ref, track_name, term, predicted_kinetic_energy, "
            "predicted_vectors, confidence, taste_similarity, practitioner_id, created_at, "
            "measured_vectors, verified, delta, resolved_at FROM track_predictions"
        )
        clauses = []
        params: list[str] = []
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if term is not None:
            clauses.append("term = ?")
            params.append(term)
        if practitioner_id is not None:
            clauses.append("practitioner_id = ?")
            params.append(practitioner_id)
        if resolved_only:
            clauses.append("resolved_at IS NOT NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at"

        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_prediction(r) for r in rows]

    def brier_score(
        self,
        term_prefix: str | None = None,
        practitioner_id: str | None = None,
    ) -> BrierResult:
        query = (
            "SELECT confidence, verified FROM track_predictions "
            "WHERE resolved_at IS NOT NULL"
        )
        params: list[str] = []
        if term_prefix is not None:
            query += " AND term LIKE ?"
            params.append(f"{term_prefix}%")
        if practitioner_id is not None:
            query += " AND practitioner_id = ?"
            params.append(practitioner_id)

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return BrierResult(brier_score=None, n=0)

        squared_errors = [(confidence - float(verified)) ** 2 for confidence, verified in rows]
        return BrierResult(brier_score=sum(squared_errors) / len(squared_errors), n=len(rows))


def _row_to_prediction(row: tuple) -> TrackPrediction:
    return TrackPrediction(
        id=row[0],
        source=row[1],
        track_ref=row[2],
        track_name=row[3],
        term=row[4],
        predicted_kinetic_energy=row[5],
        predicted_vectors=MusicVectors.model_validate_json(row[6]) if row[6] else None,
        confidence=row[7],
        taste_similarity=row[8],
        practitioner_id=row[9],
        created_at=datetime.fromisoformat(row[10]),
        measured_vectors=MusicVectors.model_validate_json(row[11]) if row[11] else None,
        verified=bool(row[12]) if row[12] is not None else None,
        delta=row[13],
        resolved_at=datetime.fromisoformat(row[14]) if row[14] else None,
    )
