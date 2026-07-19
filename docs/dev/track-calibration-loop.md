# Track-Prediction Calibration Loop — Spec

Status: **Standalone architecture (v2). Ready to build.**
Origin: David's observation that anti-patterns (dead-ends/mistakes) are the
strongest calibration signal in engineering work, and that music-taste
prediction might generate that signal *faster and cheaper* than most
engineering domains — every track is a fast, cheap, atomic, immediately
falsifiable prediction, unlike incident-postmortem-timescale feedback in most
critical-software work.

**Revision note (2026-07-19)**: v1 of this spec routed the predict/measure/
resolve loop through the Empirica CLI (`assumption-log` /
`resolve-artifacts` / `calibration-report`). David corrected that: if
epistemic-dj is going to be a standalone product, its core knowledge graph
can't require the `empirica` package/CLI at runtime — that reads as "just
another Empirica product" rather than its own thing, even if the underlying
design pattern (predict → measure → resolve → confidence-decayed pattern) is
the same one Empirica already uses. Confirmed via `grep` that `epistemic_dj/`
has zero existing `empirica` package imports today, matching the standalone
`TasteStore` decision already made earlier — this extends that same
principle to the new calibration graph rather than introducing a new one.
**This does not change how *I* (the practitioner building this) use
Empirica for my own dev-process discipline** — PREFLIGHT/CHECK/POSTFLIGHT
stays exactly as-is; that's orthogonal to what ships in the product.

## The core idea

Right now `audio_analyze_track` (see `architecture.md`) only *measures*. It
never *predicts* first, so a real audio result is just a data point, not a
calibration signal — there's nothing to confirm or refute.

The fix: before running real audio analysis on a candidate track, state a
prediction (from title/tags/genre-text alone — no audio) with a stated
confidence. After measurement, resolve that prediction as confirmed or
refuted against what was actually measured. Do this enough times across
enough genre-terms/platforms and the resolved-prediction graph *is* the
"which genre-tags are trustworthy" knowledge base — owned entirely by
epistemic-dj's own store.

## Standalone `CalibrationStore` — mirrors `TasteStore`, doesn't depend on it

Same conceptual shape Empirica's `assumption` → `resolve-artifacts(verified)`
gives, reimplemented as its own SQLite-backed store in
`epistemic_dj/calibration/store.py`, following `TasteStore`'s exact
established pattern (`epistemic_dj/taste/store.py`: plain `sqlite3`, a
`_SCHEMA` string, pydantic models for I/O, no external DB dependency).

```python
class TrackPrediction(BaseModel):
    id: str
    source: str              # "bandcamp" | "youtube"
    track_ref: str            # e.g. "artist_id:track_id" or a video id
    track_name: str
    term: str                 # search term / genre tag this candidate came from
    predicted_kinetic_energy: Scalar
    predicted_vectors: MusicVectors | None = None   # fuller prediction, optional
    confidence: Scalar        # stated P(confirmed)
    practitioner_id: str      # who made the call -- see Phase C
    created_at: datetime
    # resolution -- nullable until measured
    measured_vectors: MusicVectors | None = None
    verified: bool | None = None
    delta: Scalar | None = None
    resolved_at: datetime | None = None
```

Store methods (mirroring `TasteStore.log_finding` / `.decay_pattern` shape):

- `log_prediction(source, track_ref, track_name, term, predicted_kinetic_energy, confidence, practitioner_id="default", predicted_vectors=None) -> TrackPrediction`
- `resolve_prediction(prediction_id, measured_vectors: MusicVectors, tolerance=0.2) -> TrackPrediction` —
  computes `delta = abs(predicted_kinetic_energy - measured_vectors.kinetic_energy)`,
  `verified = delta <= tolerance`, persists both, stamps `resolved_at`.
- `get_predictions(source=None, term=None, practitioner_id=None, resolved_only=False) -> list[TrackPrediction]`
- `brier_score(term_prefix=None, practitioner_id=None) -> BrierResult` —
  `mean((confidence - float(verified))²)` over resolved rows matching the
  filters, plus `n`. This is epistemic-dj's own Brier computation — it does
  NOT call `calibration-report`. (v1 spec conflated "Brier score" with
  Empirica's existing command; that command scores the *practitioner's*
  general self-assessment, a different signal — see below.)

The `practitioner_id` column costs nothing now and directly serves Phase C
(explicitly requested, not speculative) — default it to a plain string
identifying whoever's running the loop (e.g. a session or agent label), not
hardcoded to one value.

## Phase A — per-track transaction (single practitioner, Bandcamp + YouTube)

**One transaction per track** (not one transaction per step — the three
steps below are *tasks* within a single PREFLIGHT/POSTFLIGHT window, matching
David's "goal is a track, tasks are the things to be done for that track").
The transaction discipline is still mine (the practitioner's) — the
*product* data (the prediction/measurement/resolution itself) goes to
`CalibrationStore`, not to an Empirica artifact.

```
Goal: "Track: <title> [<term>, <source>]"
  Task 1: predict  — read title/tags/url, form an actual belief about
                      predicted_kinetic_energy + P(confirmed), call
                      CalibrationStore.log_prediction(...).
  Task 2: measure   — audio_analyze_track() (Bandcamp) or the new YouTube
                      measure() path.
  Task 3: compare   — CalibrationStore.resolve_prediction(...); if my own
                      PREFLIGHT/POSTFLIGHT for this transaction turns up
                      something reusable about MY predictive judgment
                      (not the track), that's still fair game for a real
                      empirica finding/dead-end -- the distinction is
                      "product data" (CalibrationStore) vs. "my own
                      epistemic-transaction learning" (Empirica), not
                      "never touch Empirica during this work."
```

**PREFLIGHT**: `work_type=data`. `task_context` names the track + term +
source. Standard 13-vector self-assessment as normal — this is a *separate*
signal from the track-prediction confidence (see below); don't conflate
PREFLIGHT `know`/`uncertainty` with `predicted_kinetic_energy`'s confidence.

The prediction is a real judgment call from title/tags/genre-text — not a
rule-based lookup. The point is calibrating the *practitioner's* predictive
judgment (vectors are beliefs), so this has to be an actual read-and-decide
step, not a canned heuristic standing in for one.

### Primary confirm/refute dimension

Genre-tags most directly claim tempo/energy, so v1 scores confirm/refute
against **`kinetic_energy` only**, tolerance `|predicted − measured| ≤ 0.2`
→ confirmed. The other three derivable dims (`cognitive_load`,
`groove_consistency`, `textural_density`) are still predicted and stored
(richer signal, and a plain RMSE across all four is worth computing once
there's enough data), but don't gate the binary verified/refuted call — a
single, well-defined outcome per track is what makes confidence-vs-verified
actually Brier-scoreable, and folding four dims into one confirm/refute call
would just make the threshold arbitrary in a different way. This tolerance
is a starting heuristic, not derived from data — expect to revise once
~30-50 tracks exist to check it against.

### Two distinct calibration signals — still don't conflate them

1. **General practitioner Brier score** (`empirica calibration-report
   --ai-id epistemic-dj --brier`) — scores *my* general 13-vector PREFLIGHT
   self-assessment against grounded evidence. Running many per-track
   transactions sharpens this as a side effect (more transactions = more
   n_predictions), but it's about my own epistemic self-calibration, not
   genre-tag prediction accuracy.
2. **Product Brier score** (`CalibrationStore.brier_score()`) — the new
   thing this spec builds, scoring genre/title-text prediction accuracy
   against real audio measurement. Fully owned by epistemic-dj, computed
   directly from `TrackPrediction` rows, no Empirica dependency.

## Phase B — YouTube Music as a second source (folded into near-term scope)

David's framing: discovery is platform-agnostic (the music is the same
everywhere), measurement is platform-dependent (quality/downloadability
differ). Split the two capabilities explicitly rather than assuming one
adapter does both — mirrors the existing `bandcamp/client.py` +
`bandcamp/adapter.py` split:

- **`discover()`** — search, metadata only. Bandcamp: `client.search()`
  (already built). YouTube Music: `ytmusicapi` (public search, no login
  required).
- **`measure()`** — needs a real downloadable/streamable audio source.
  Bandcamp: `get_track().streaming_url["mp3-128"]` (already built). YouTube:
  `yt-dlp` audio extraction.

**YouTube extraction: approved (David, 2026-07-19) under fair-use personal/
research use.** Formal YouTube relationship (partnership/API terms) is
explicitly deferred to an eventual production conversation, not blocking
this build — David's framing: if this drives YouTube usage/subscriptions,
that's additive for them, not adversarial, so a future formal conversation
is plausible rather than a hard legal blocker. Noted here so the record is
honest about the current basis (fair use, personal/research scope) versus
what a shipped-product-at-scale posture would need later.

`epistemic_dj/youtube/client.py` + `epistemic_dj/youtube/adapter.py`,
mirroring the Bandcamp module shape: a thin wrapper managing `ytmusicapi`
search + `yt-dlp` extraction, mapping results to the same source-agnostic
`Track`/`AudioFeatures` shapes so `audio_features_to_vectors()` and the
`CalibrationStore` loop are source-agnostic — the loop shouldn't need to
know which platform a candidate came from beyond the `source` field.

## Phase C — parallel practitioners on the shared store (design-level)

Multiple subagents (or multiple Claude Code sessions) each run the Phase A
loop, each with a `practitioner_id`, all writing to the *same*
`CalibrationStore` SQLite file. Compare individual calibration via
`CalibrationStore.brier_score(practitioner_id=...)` — a first-class filter
on the store itself now (unlike v1's plan to slice Empirica's
`calibration-report` after the fact), because we own the schema.

**Open question, not yet verified**: whether plain `sqlite3` (as `TasteStore`
already uses, no WAL mode configured) is safe under real concurrent writes
from multiple parallel practitioners. `TasteStore`'s existing single-writer
usage hasn't tested this. Before Phase C runs more than one or two
simultaneous practitioners: either confirm SQLite's default locking is
sufficient at this write volume, or turn on WAL mode
(`PRAGMA journal_mode=WAL`) on `CalibrationStore.__init__` — cheap
insurance, worth doing regardless once Phase C is real rather than assuming
the default is fine because it hasn't broken yet in single-practitioner use.

## Build order

**A** (standalone `CalibrationStore` + models, single practitioner, Bandcamp
first since the pipeline already exists, ~8-10 tracks to prove the
mechanism) → **B** (YouTube Music `discover()`+`measure()`, folded in once A
proves the loop mechanics work — not gated on A being "done," just on the
loop shape being validated) → **C** (parallel practitioners, after adding
WAL mode and deciding on `practitioner_id` assignment for concurrent runs).

## Operational concerns for any loop, not phase-specific

- **Rate limiting**: no measured read yet on what request rate Bandcamp (or
  YouTube) tolerates before throttling/blocking. Cap batch size and add
  backoff on errors rather than firing an unbounded loop — an unannounced
  IP block would itself be worth a dead-end entry, but better to not cause
  one.
- **Candidate source for a given loop run**: can keep reusing arbitrary
  genre-adjacent search terms (as in the earlier discovery test), but
  deriving terms from David's actual taste-profile `mcp_query_arrays` would
  make the calibration signal directly relevant to real curation rather
  than generic genre coverage. Not built yet — the taste-profile data
  currently lives in conversation/git notes only, per the earlier decision
  to defer full `TasteStore` integration until a working module exists to
  justify it.
