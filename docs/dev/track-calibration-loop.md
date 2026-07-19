# Track-Prediction Calibration Loop — Spec

Status: **Phase A specced for build. Phase B/C design-level, not yet built.**
Origin: David's observation that anti-patterns (dead-ends/mistakes) are the
strongest calibration signal in engineering work, and that music-taste
prediction might generate that signal *faster and cheaper* than most
engineering domains — every track is a fast, cheap, atomic, immediately
falsifiable prediction, unlike incident-postmortem-timescale feedback in most
critical-software work.

## The core idea

Right now `audio_analyze_track` (see `architecture.md`) only *measures*. It
never *predicts* first, so a real audio result is just a data point, not a
calibration signal — there's nothing to confirm or refute.

The fix: before running real audio analysis on a candidate track, state a
prediction (from title/tags/genre-text alone — no audio) as an Empirica
**assumption**, with a stated confidence. After measurement, **resolve** that
assumption as confirmed or refuted against what was actually measured. Do
this enough times across enough genre-terms and the resolved-assumption graph
*is* the "which genre-tags are trustworthy" knowledge base — filed under
Empirica's existing artifact substrate, not a new parallel store (same call
made for the `TastePattern`/`TasteFinding` design in `architecture.md` — see
"The taste engine reuses Empirica's artifact substrate").

## Why reuse `assumption` → `resolve-artifacts`, not a new schema

Checked against the CLI rather than assumed: `resolve-artifacts` already
supports `type: assumption` resolution with a `verified: true/false` field
(`empirica resolve-artifacts --schema`). That is *exactly* the predict/verify
primitive this loop needs — a stated confidence at prediction time, a binary
verified/refuted outcome at resolution time. No new storage layer required.

## Phase A — per-track transaction

**One transaction per track** (not one transaction per step — the three
steps below are *tasks* within a single PREFLIGHT/POSTFLIGHT window, matching
David's "goal is a track, tasks are the things to be done for that track").

```
Goal: "Track: <title> [<term>]"
  Task 1: predict  — read title/tags/url, form an actual belief about the
                      derivable MusicVectors dims (kinetic_energy first —
                      see "Primary confirm/refute dimension" below), log as
                      an assumption with a stated P(confirmed).
  Task 2: measure   — audio_analyze_track(artist_id, track_id)
  Task 3: compare   — diff predicted vs. measured, resolve the assumption
                      (verified: true/false), log a finding (confirmed) or
                      dead_end (refuted) capturing what was learned.
```

**PREFLIGHT**: `work_type=data` (not `code` — no git/pytest signal is
relevant to a measurement transaction; `data` down-weights those the same
way `research`/`docs` do for their domains). `task_context` names the track
+ term. Standard 13-vector self-assessment as normal — this is a *separate*
signal from the track-prediction confidence (see "Two distinct calibration
signals" below); don't conflate PREFLIGHT `know`/`uncertainty` with the
predicted-vector confidence.

**Task 1 — predict** (noetic, before CHECK):

```bash
empirica assumption-log \
  --assumption "Track '<title>' (term=<genre-term>): predicted kinetic_energy≈<X>, P(confirmed)=<Y>" \
  --confidence <Y> \
  --domain "genre:<term>"
```

The prediction is a real judgment call from title/tags/genre-text — not a
rule-based lookup. The point is calibrating the *practitioner's* predictive
judgment (vectors are beliefs), so this has to be an actual read-and-decide
step, not a canned heuristic standing in for one.

**Task 2 — measure** (praxic, after CHECK): call `audio_analyze_track`,
already built and tested (commit `27d325c`).

**Task 3 — compare + resolve**:

```bash
empirica resolve-artifacts - <<'EOF'
{"resolutions": [{"type": "assumption", "id": "<assumption-uuid>",
                   "resolution": "measured kinetic_energy=<M> (predicted <X>, delta=<D>)",
                   "verified": <true|false>}]}
EOF
```

Then `finding-log` (confirmed — capture the reusable pattern, e.g. "'power
breaks'-titled tracks reliably predict high kinetic_energy") or `deadend-log`
(refuted — approach = "assumed <term> title predicts <profile>", why_failed
= the measured mismatch) via `log-artifacts` so the finding/dead-end is wired
to the resolved assumption (`relation: evidence`).

**POSTFLIGHT**: report actual completion/change for this one track's
prediction-verify cycle. Batch multiple tracks as multiple transactions
within one session, not one giant transaction — keeps each POSTFLIGHT's
delta interpretable per-track rather than averaged into mush.

### Primary confirm/refute dimension

Genre-tags most directly claim tempo/energy, so v1 scores confirm/refute
against **`kinetic_energy` only**, tolerance `|predicted − measured| ≤ 0.2`
→ confirmed. The other three derivable dims (`cognitive_load`,
`groove_consistency`, `textural_density`) are still predicted and logged
(richer signal in the finding/dead-end body, and a plain RMSE across all
four is worth computing once there's enough data), but don't gate the
binary verified/refuted call — a single, well-defined outcome per track is
what makes the confidence-vs-verified pair actually Brier-scoreable, and
folding four dims into one confirm/refute call would just make the
threshold arbitrary in a different way. This tolerance is a starting
heuristic, not derived from data — expect to revise once ~30-50 tracks
exist to check it against.

### Two distinct calibration signals — don't conflate them

1. **General practitioner Brier score** (`calibration-report --ai-id
   epistemic-dj --brier`) — already live today (verified: real output,
   n_predictions=50 as of this write-up). Scores the practitioner's own
   13-vector PREFLIGHT self-assessment against grounded evidence. Running
   many per-track transactions sharpens this *as a side effect* (more
   transactions = more n_predictions) but it is NOT measuring genre-tag
   prediction accuracy — it's measuring "is Claude well-calibrated about
   its own general epistemic state."
2. **Domain-specific track-prediction calibration** (assumption confidence
   vs. `verified`) — the new signal this spec builds. **Open question, not
   yet verified**: does `calibration-report --brier` already ingest
   resolved-assumption confidence/verified pairs as an evidence source, or
   does genre-tag-prediction Brier scoring need a separate aggregation
   (e.g., pull all resolved assumptions with `domain` matching `genre:*`
   via `project-search`/`investigate` and compute `mean((confidence -
   verified)²)` directly)? Check before claiming an automatic combined
   score — don't assume the general command already covers this just
   because both use the word "Brier."

## Phase B — YouTube Music as a second source (design-level, not built)

David's framing: discovery is platform-agnostic (the music is the same
everywhere), measurement is platform-dependent (quality/downloadability
differ). Split the two capabilities explicitly rather than assuming one
adapter does both:

- **`discover()`** — search, metadata only. Bandcamp: `client.search()`
  (already built). YouTube Music: `ytmusicapi` (public search doesn't
  require login).
- **`measure()`** — needs a real downloadable/streamable audio source.
  Bandcamp: `get_track().streaming_url["mp3-128"]` (already built). YouTube:
  `yt-dlp` audio extraction — same "unofficial API access" category as
  `bandcamp_async_api` already is, but a materially different ToS risk
  profile (YouTube's terms are more actively hostile to extraction than
  Bandcamp's). Not equivalent to the torrent option already ruled out, but
  worth a named, explicit go/no-go before building — not something to
  quietly bundle into "just another adapter."

No code proposed here yet — this is scoped, not built. Revisit once Phase A
has produced enough resolved assumptions to know the loop mechanics actually
work end-to-end.

## Phase C — parallel practitioners on the shared practice (design-level)

Multiple subagents (or multiple Claude Code sessions) each run the Phase A
loop against the same project (`epistemic-dj`), each getting its own
`session_id` automatically. All findings/dead-ends/resolved-assumptions land
in the same practice graph — exactly the practice/practitioner split already
established in `architecture.md`'s "Dual calibration" section, applied to
this new object type (tracks) instead of code.

**Verified**: `calibration-report --ai-id <id>` scopes to the *practice*
(project basename), not to an individual parallel instance within it — so
out of the box this gives one pooled score across every practitioner writing
to `epistemic-dj`, not automatic per-practitioner comparison.

**Not yet verified — named as an open question, not assumed**:
per-practitioner comparison is derivable by slicing on `session_id` (each
parallel run gets a distinct one), but that slicing isn't a first-class
report view today — would need a small script pulling resolved assumptions
grouped by `session_id` rather than relying on `calibration-report` doing it
automatically. Also unverified: whether concurrent writes from multiple
parallel agents to the same SQLite-backed `sessions.db` are safe as-is
(WAL mode? contention under real concurrency?) — check before running
Phase C with more than one or two simultaneous practitioners, don't assume
it's fine because it hasn't broken yet in this session's single-practitioner
usage.

## Build order

**A** (this spec, single practitioner, Bandcamp only, ~8-10 tracks to prove
the mechanism) → **B** (YouTube Music, after an explicit go on the ToS
question) → **C** (parallel practitioners, after verifying SQLite
concurrency safety and deciding whether per-session Brier slicing needs
tooling or a one-off script is enough).

## Operational concerns for any loop, not phase-specific

- **Rate limiting**: no measured read yet on what request rate Bandcamp (or
  YouTube) tolerates before throttling/blocking. Cap batch size and add
  backoff on errors rather than firing an unbounded loop — an unannounced
  IP block would itself be worth a dead-end entry, but better to not cause
  one.
- **Candidate source for a given loop run**: Phase A can keep reusing
  arbitrary genre-adjacent search terms (as in the earlier discovery test),
  but deriving terms from David's actual taste-profile `mcp_query_arrays`
  would make the calibration signal directly relevant to real curation
  rather than generic genre coverage. Not built yet — the taste-profile
  data currently lives in conversation/git notes only, per the earlier
  decision to defer full `TasteStore` integration until a working module
  exists to justify it.
