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

from epistemic_dj.calibration.bayesian_belief import (
    BETA_PRIOR_ALPHA,
    BETA_PRIOR_BETA,
    DEFAULT_OBSERVATION_VARIANCE,
    beta_belief,
    beta_update,
    update_belief,
)
from epistemic_dj.models import Belief, BrierResult, MusicVectors, TrackPrediction

DEFAULT_DB_PATH = Path(__file__).parent / "calibration.db"

DEFAULT_TOLERANCE = 0.2

# Uninformative-ish priors for the two beliefs tracked here (see
# bayesian_belief.py). TERM_BIAS starts at "no known bias" (mean=0).
# MARGIN_SCALE starts at 0.5 -- the ORIGINAL assumption baked into the old
# confidence=margin*2 formula -- so evidence pulls it toward the true,
# much smaller observed scale rather than starting from nothing.
TERM_BIAS_PRIOR_MEAN = 0.0
TERM_BIAS_PRIOR_VARIANCE = 0.1
MARGIN_SCALE_PRIOR_MEAN = 0.5
MARGIN_SCALE_PRIOR_VARIANCE = 0.1
GLOBAL_BELIEF_KEY = "__global__"

# Confidence hit-rate buckets: "weak"/"strong" signal, split at the current
# margin_scale.mean (see mcp_server.calibration_predict_from_tags). Each
# bucket tracks its OWN Beta-Binomial belief -- confidence should reflect
# how often THIS bucket's predictions actually verify, not an assumption
# that bigger margin = more accurate.
HIT_RATE_BUCKET_WEAK = "weak"
HIT_RATE_BUCKET_STRONG = "strong"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_predictions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    track_ref TEXT NOT NULL,
    track_name TEXT NOT NULL,
    term TEXT NOT NULL,
    quantity TEXT NOT NULL DEFAULT 'kinetic_energy',
    predicted_value REAL NOT NULL,
    predicted_vectors TEXT,
    confidence REAL NOT NULL,
    taste_similarity REAL,
    practitioner_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    measured_vectors TEXT,
    measured_value REAL,
    verified INTEGER,
    delta REAL,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_term ON track_predictions(term);
CREATE INDEX IF NOT EXISTS idx_predictions_practitioner
    ON track_predictions(practitioner_id);

CREATE TABLE IF NOT EXISTS beliefs (
    belief_type TEXT NOT NULL,
    key TEXT NOT NULL,
    mean REAL NOT NULL,
    variance REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (belief_type, key)
);

CREATE TABLE IF NOT EXISTS hit_rate_beliefs (
    bucket TEXT PRIMARY KEY,
    alpha REAL NOT NULL,
    beta REAL NOT NULL,
    updated_at TEXT NOT NULL
);
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
        # Real migration, not a "delete the local db" workaround -- this
        # store now holds real accumulated prediction history worth
        # preserving. ALTER TABLE ADD COLUMN is safe on SQLite (existing
        # rows get NULL, which resolve_prediction treats as "no hit-rate
        # feedback for this legacy row").
        existing_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(track_predictions)")
        }
        if "confidence_bucket" not in existing_columns:
            self._conn.execute("ALTER TABLE track_predictions ADD COLUMN confidence_bucket TEXT")
            existing_columns.add("confidence_bucket")
        # quantity/predicted_value/measured_value: generalizes this store
        # beyond kinetic_energy (mixing-engine roadmap, decision d55de6e8).
        # Existing rows are ALL kinetic_energy predictions -- the rename
        # preserves their values exactly; DEFAULT 'kinetic_energy' backfills
        # quantity for them. Safe: this is a single-practitioner local
        # SQLite file, not a shared/prod migration.
        has_old_predicted_col = (
            "predicted_kinetic_energy" in existing_columns
            and "predicted_value" not in existing_columns
        )
        if has_old_predicted_col:
            self._conn.execute(
                "ALTER TABLE track_predictions "
                "RENAME COLUMN predicted_kinetic_energy TO predicted_value"
            )
            existing_columns.discard("predicted_kinetic_energy")
            existing_columns.add("predicted_value")
        if "quantity" not in existing_columns:
            self._conn.execute(
                "ALTER TABLE track_predictions "
                "ADD COLUMN quantity TEXT NOT NULL DEFAULT 'kinetic_energy'"
            )
            existing_columns.add("quantity")
        if "measured_value" not in existing_columns:
            self._conn.execute("ALTER TABLE track_predictions ADD COLUMN measured_value REAL")
            existing_columns.add("measured_value")
            # Backfill from measured_vectors for already-resolved kinetic_energy
            # rows so measured_value is uniformly queryable going forward.
            self._conn.execute(
                "UPDATE track_predictions SET measured_value = "
                "json_extract(measured_vectors, '$.kinetic_energy.value') "
                "WHERE measured_vectors IS NOT NULL AND measured_value IS NULL"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_quantity ON track_predictions(quantity)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def log_prediction(
        self,
        source: str,
        track_ref: str,
        track_name: str,
        term: str,
        predicted_value: float,
        confidence: float,
        practitioner_id: str = "default",
        predicted_vectors: MusicVectors | None = None,
        taste_similarity: float | None = None,
        confidence_bucket: str | None = None,
        quantity: str = "kinetic_energy",
    ) -> TrackPrediction:
        prediction = TrackPrediction(
            id=str(uuid.uuid4()),
            source=source,
            track_ref=track_ref,
            track_name=track_name,
            term=term,
            quantity=quantity,
            predicted_value=predicted_value,
            predicted_vectors=predicted_vectors,
            confidence=confidence,
            taste_similarity=taste_similarity,
            practitioner_id=practitioner_id,
            created_at=datetime.now(UTC),
        )
        self._conn.execute(
            "INSERT INTO track_predictions "
            "(id, source, track_ref, track_name, term, quantity, predicted_value, "
            "predicted_vectors, confidence, taste_similarity, practitioner_id, created_at, "
            "confidence_bucket) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                prediction.id, prediction.source, prediction.track_ref,
                prediction.track_name, prediction.term, prediction.quantity,
                prediction.predicted_value,
                prediction.predicted_vectors.model_dump_json()
                if prediction.predicted_vectors else None,
                prediction.confidence, prediction.taste_similarity,
                prediction.practitioner_id, prediction.created_at.isoformat(),
                confidence_bucket,
            ),
        )
        self._conn.commit()
        return prediction

    def resolve_prediction(
        self,
        prediction_id: str,
        measured_vectors: MusicVectors | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
        *,
        measured_value: float | None = None,
    ) -> TrackPrediction:
        """Resolves against either measured_vectors (the original kinetic_energy
        path -- measured_value is extracted as vectors.kinetic_energy.value) or
        a generic measured_value directly (new quantities, e.g. tempo_bpm,
        that don't route through MusicVectors at all). Exactly one must be given.
        """
        if (measured_vectors is None) == (measured_value is None):
            raise ValueError("Supply exactly one of measured_vectors or measured_value.")

        row = self._conn.execute(
            "SELECT predicted_value, term, confidence_bucket, quantity "
            "FROM track_predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        if row is None:
            raise PredictionNotFoundError(prediction_id)
        predicted_value, term, confidence_bucket, quantity = row

        if measured_vectors is not None:
            resolved_measured_value = measured_vectors.kinetic_energy.value
        else:
            resolved_measured_value = measured_value
        delta = abs(predicted_value - resolved_measured_value)
        verified = delta <= tolerance
        resolved_at = datetime.now(UTC)
        self._conn.execute(
            "UPDATE track_predictions SET measured_vectors = ?, measured_value = ?, "
            "verified = ?, delta = ?, resolved_at = ? WHERE id = ?",
            (
                measured_vectors.model_dump_json() if measured_vectors is not None else None,
                resolved_measured_value, int(verified), delta,
                resolved_at.isoformat(), prediction_id,
            ),
        )
        self._conn.commit()

        # Closed-loop bias correction: this resolution's signed residual
        # becomes evidence for the term's bias belief, informing the NEXT
        # prediction logged for this term (see log_prediction's
        # apply_term_bias_correction). Signed (not abs) so direction is
        # preserved -- consistently over- or under-predicting a term should
        # converge the belief toward that bias.
        #
        # Key is quantity-namespaced for anything other than kinetic_energy
        # (bare `term` stays unprefixed there, preserving the ~170 rows of
        # accumulated term_bias history from before `quantity` existed).
        # New quantities get their own namespace so e.g. a tempo_bpm delta
        # (scale: tens of BPM) never gets Gaussian-averaged into the same
        # belief as a kinetic_energy delta (scale: 0-1) under the same key.
        belief_key = term if quantity == "kinetic_energy" else f"{quantity}:{term}"
        self._update_belief("term_bias", belief_key, predicted_value - resolved_measured_value)

        # Closed-loop confidence calibration: this resolution's actual
        # verified/not outcome becomes evidence for the bucket's hit-rate
        # belief -- confidence for the NEXT prediction in that bucket
        # reflects real accuracy, not an assumption. Legacy rows (predicted
        # before this feature existed) have confidence_bucket=NULL and are
        # skipped rather than misattributed to a bucket they were never
        # assigned to.
        if confidence_bucket is not None:
            self._update_hit_rate(confidence_bucket, verified)

        return self.get_prediction(prediction_id)

    def _get_belief(
        self, belief_type: str, key: str, prior_mean: float, prior_variance: float
    ) -> Belief:
        row = self._conn.execute(
            "SELECT mean, variance, evidence_count FROM beliefs "
            "WHERE belief_type = ? AND key = ?",
            (belief_type, key),
        ).fetchone()
        if row is None:
            return Belief(mean=prior_mean, variance=prior_variance, evidence_count=0)
        return Belief(mean=row[0], variance=row[1], evidence_count=row[2])

    def _update_belief(
        self,
        belief_type: str,
        key: str,
        observation: float,
        obs_variance: float = DEFAULT_OBSERVATION_VARIANCE,
    ) -> Belief:
        prior_mean, prior_variance = self._belief_priors(belief_type)
        current = self._get_belief(belief_type, key, prior_mean, prior_variance)
        updated = update_belief(
            current.mean, current.variance, current.evidence_count, observation, obs_variance
        )
        self._conn.execute(
            "INSERT INTO beliefs (belief_type, key, mean, variance, evidence_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(belief_type, key) DO UPDATE SET "
            "mean = excluded.mean, variance = excluded.variance, "
            "evidence_count = excluded.evidence_count, updated_at = excluded.updated_at",
            (
                belief_type, key, updated.mean, updated.variance, updated.evidence_count,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()
        return updated

    @staticmethod
    def _belief_priors(belief_type: str) -> tuple[float, float]:
        if belief_type == "term_bias":
            return TERM_BIAS_PRIOR_MEAN, TERM_BIAS_PRIOR_VARIANCE
        if belief_type == "margin_scale":
            return MARGIN_SCALE_PRIOR_MEAN, MARGIN_SCALE_PRIOR_VARIANCE
        raise ValueError(f"Unknown belief_type '{belief_type}'.")

    def get_term_bias(self, term: str) -> Belief:
        """Current belief about how far off (signed) predictions for this
        term have run historically. Subtract .mean from a raw prediction to
        bias-correct it -- see mcp_server.calibration_predict_from_tags.
        """
        return self._get_belief("term_bias", term, TERM_BIAS_PRIOR_MEAN, TERM_BIAS_PRIOR_VARIANCE)

    def get_margin_scale(self) -> Belief:
        """Current belief about the typical observed anchor-margin scale
        (global, not per-term -- too little data per term to slice this
        further). Starts at 0.5 (the original, wrong assumption) and is
        pulled toward the true smaller scale as margin observations
        accumulate via update_margin_scale.
        """
        return self._get_belief(
            "margin_scale", GLOBAL_BELIEF_KEY, MARGIN_SCALE_PRIOR_MEAN, MARGIN_SCALE_PRIOR_VARIANCE
        )

    def update_margin_scale(self, raw_margin: float) -> Belief:
        return self._update_belief("margin_scale", GLOBAL_BELIEF_KEY, raw_margin)

    def get_hit_rate(self, bucket: str) -> Belief:
        """Current belief about P(verified) for predictions in this
        margin-strength bucket ("weak" | "strong") -- this IS the
        confidence value calibration_predict_from_tags should report,
        replacing the old margin*scale rescale that measured signal
        strength rather than actual accuracy.
        """
        row = self._conn.execute(
            "SELECT alpha, beta FROM hit_rate_beliefs WHERE bucket = ?", (bucket,)
        ).fetchone()
        alpha, beta = row if row is not None else (BETA_PRIOR_ALPHA, BETA_PRIOR_BETA)
        return beta_belief(alpha, beta)

    def _update_hit_rate(self, bucket: str, verified: bool) -> Belief:
        row = self._conn.execute(
            "SELECT alpha, beta FROM hit_rate_beliefs WHERE bucket = ?", (bucket,)
        ).fetchone()
        alpha, beta = row if row is not None else (BETA_PRIOR_ALPHA, BETA_PRIOR_BETA)
        new_alpha, new_beta = beta_update(alpha, beta, int(verified))
        self._conn.execute(
            "INSERT INTO hit_rate_beliefs (bucket, alpha, beta, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(bucket) DO UPDATE SET "
            "alpha = excluded.alpha, beta = excluded.beta, updated_at = excluded.updated_at",
            (bucket, new_alpha, new_beta, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()
        return beta_belief(new_alpha, new_beta)

    def get_prediction(self, prediction_id: str) -> TrackPrediction:
        row = self._conn.execute(
            "SELECT id, source, track_ref, track_name, term, quantity, predicted_value, "
            "predicted_vectors, confidence, taste_similarity, practitioner_id, created_at, "
            "measured_vectors, measured_value, verified, delta, resolved_at "
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
        quantity: str | None = None,
    ) -> list[TrackPrediction]:
        query = (
            "SELECT id, source, track_ref, track_name, term, quantity, predicted_value, "
            "predicted_vectors, confidence, taste_similarity, practitioner_id, created_at, "
            "measured_vectors, measured_value, verified, delta, resolved_at FROM track_predictions"
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
        if quantity is not None:
            clauses.append("quantity = ?")
            params.append(quantity)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at"

        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_prediction(r) for r in rows]

    def delete_unresolved_predictions(
        self,
        source: str | None = None,
        term: str | None = None,
        confidence_bucket_is_null: bool = False,
    ) -> int:
        """Delete stale unresolved predictions -- e.g. orphans left by an
        interrupted batch script (confirmed real occurrence twice this
        session: a killed Bash timeout, then an accidental script import
        that ran the full batch). Always scoped to resolved_at IS NULL --
        never touches resolved predictions, which are real historical data
        regardless of how the confidence was originally computed.
        confidence_bucket_is_null narrows to predictions logged before the
        confidence_bucket feature existed (or via calibration_predict
        without a bucket) -- useful for clearing a specific rejected
        approach's orphans without touching genuine unresolved predictions
        from a newer approach that simply failed to resolve (e.g. a
        transient download failure) and are worth retrying.
        """
        clauses = ["resolved_at IS NULL"]
        params: list[str] = []
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if term is not None:
            clauses.append("term = ?")
            params.append(term)
        if confidence_bucket_is_null:
            clauses.append("confidence_bucket IS NULL")
        query = "DELETE FROM track_predictions WHERE " + " AND ".join(clauses)
        cursor = self._conn.execute(query, params)
        self._conn.commit()
        return cursor.rowcount

    def brier_score(
        self,
        term_prefix: str | None = None,
        practitioner_id: str | None = None,
        quantity: str | None = None,
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
        if quantity is not None:
            query += " AND quantity = ?"
            params.append(quantity)

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
        quantity=row[5],
        predicted_value=row[6],
        predicted_vectors=MusicVectors.model_validate_json(row[7]) if row[7] else None,
        confidence=row[8],
        taste_similarity=row[9],
        practitioner_id=row[10],
        created_at=datetime.fromisoformat(row[11]),
        measured_vectors=MusicVectors.model_validate_json(row[12]) if row[12] else None,
        measured_value=row[13],
        verified=bool(row[14]) if row[14] is not None else None,
        delta=row[15],
        resolved_at=datetime.fromisoformat(row[16]) if row[16] else None,
    )
