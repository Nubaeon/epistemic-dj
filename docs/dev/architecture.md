# Architecture

This is a curated snapshot of the design decisions logged in Empirica
(`empirica project-search --task "epistemic-dj architecture"` for the full,
connected artifact graph — findings/decisions/edges). This doc is not the
source of truth; it's a periodically-refreshed export so anyone without CLI
access (or a future Claude session starting cold) can get oriented fast.

## Repo shape: two languages, not a rewrite

`src/` (JS/ESM) is existing, working code — the MCP server that turns
epistemic vectors into Strudel patterns (`generate_pattern`, `generate_mood`,
etc.). It stays exactly as-is.

`python/` is new. Bandcamp integration, stem separation, and taste profiling
are fundamentally ML/Python-native (Demucs, PyTorch-based separation models —
there's no serious non-Python alternative). Rather than porting the working
JS layer to unify languages, we register a **second MCP server**
(`python/epistemic_dj/mcp_server.py`) alongside the first. Claude Code (or
any MCP client) talks to both.

Rust was considered for the eventual binary and explicitly deferred: Python
first for iteration speed while the taste-engine design is still unsettled;
port a specific hot path later only if profiling shows a real need (e.g. a
dependency-free single-binary distribution), not a wholesale rewrite.

## Two vector spaces — don't conflate them

**`MusicVectors`** (object-level, per-track): `kinetic_energy`, `valence`,
`vocal_density`, `structural_repetition`, `cognitive_load` (foundation
tier); `novelty`, `familiarity_fit`, `production_rawness` (discovery-tuning
tier, weighted by mode); `groove_consistency`, `textural_density`,
`harmonic_tension` (situational tier — DJ/creative-seed modes).
`structural_repetition` and `cognitive_load` are the most load-bearing for
the focus-session thesis specifically.

**`UserTasteVectors`** (meta-epistemic, per-user): Empirica's existing 13
universal vectors, reused unchanged, re-anchored to the taste domain —
`know` = understanding of this user's taste, `signal` = quality of
behavioral evidence, `uncertainty` = gates further active interview, etc.
These are genuinely domain-agnostic; no new vector set needed here.

See `python/epistemic_dj/models.py` for the pydantic definitions.

## Dual calibration: practice vs. practitioner

Reuses Empirica's own practice/practitioner split directly:

- **Track A (practice-level)** — the user's persistent taste profile
  (findings/patterns/anti-patterns/`UserTasteVectors`). Accumulates over
  time, independent of which AI is serving the user.
- **Track B (practitioner-level)** — how well the *current* AI executes
  curation/mixing/creation from that profile
  (`PractitionerExecutionVectors`, same 13-vector shape, scored as a
  distinct work_type e.g. `curate`/`mix`). Execution skill can vary by AI
  model even against an identical profile.

## The taste engine reuses Empirica's artifact substrate

Not a bespoke black-box LLM taste model. Behavioral signal becomes
`finding-log` entries; distilled cross-session patterns become
`lesson`-shaped taste patterns (and anti-patterns) with the same
decay-with-floor confidence mechanism; open questions become
`unknown-log` entries that drive active onboarding interview questions;
curation choices become `decision-log` entries with rationale attached.
This is what makes "why did you play me this" answerable — the chain is
real artifacts, not an opaque embedding.

Existing Empirica users get a bootstrap advantage here: their EWM
workflow-protocol data (cognitive style, domain expertise, autonomy
posture) is a prior the onboarding interview can build on rather than
re-derive from zero.

## Mesh as a shared substrate for music + epistemic knowledge

A `Mixtape` (see `models.py`) is a shareable artifact subgraph — curated
tracks plus the decisions/findings that justified each pick. It moves
between users via Empirica's *existing* mesh sharing primitives
(`--visibility shared/public`, `source-add`, `cortex_collab`) — no new
social/sharing infrastructure. Each epistemic-dj user is modeled as their
own practice, so sharing a mixtape is structurally identical to one AI
practice sharing a lesson with another today. The receiving side gets the
*why*, so it can evaluate a shared mixtape against its own calibration
before accepting/surfacing tracks, rather than just inheriting a flat
playlist.

## Source-agnostic by design

Bandcamp is ingestion adapter #1, not a foundation dependency. `Track.source`
is a string, not an enum tied to one platform — a future SoundCloud or
local-file adapter slots in the same way. This matters for the mesh-sharing
case too: a shared mixtape shouldn't assume everyone bought from the same
platform.

## Build-vs-integrate research summary

Full research (6 areas, sourced, dated 2026-07-18) is logged as findings in
Empirica. Headline calls:

| Area | Call |
|---|---|
| Bandcamp access | Integrate `bandcamp_async_api` or `bandcamp-fetch` (cookie-auth) — no official API covers personal collections. **Verify the lossless-download path early**; the private collection-sync API is lossy MP3-V0 only. |
| Stem separation | Use `ZFTurbo/Music-Source-Separation-Training` (unified surface, MIT) instead of hand-picking bare Demucs — Demucs is no longer SOTA. |
| `.stem.mp4` | Don't chase strict NI/Traktor byte-compatibility unless hardware interop is a confirmed need — the ecosystem (stemgen, Mixxx) is already drifting to simpler lossless containers. |
| Consumer stem-remix apps | **Don't build one.** Real-time GPU stem separation is now standard across every major DJ platform (2025-2026). Differentiation lives entirely in curation/economics/UX, not separation tech. |
| Taste profiling | Genuinely close to greenfield for the Bandcamp-specific, artifact-driven angle. Watch for the documented LLM-taste-profile bias risk (genre/origin skew) — the artifact-substrate approach sidesteps it by construction (interpretable by design, not bolted on). |
| Direct-to-artist economics | The Web3/NFT "bypass streaming" thesis failed outright 2021-2026. Anchor to plain Bandcamp-purchase economics, not any ownership/token framing. |

## Dev-time audio sources (beyond Bandcamp, to avoid real cost while iterating)

Piracy/torrent sources were considered and explicitly rejected — not worth
it even for dev, and there are legitimate options that are arguably better
for testing anyway:

- **MUSDB18 sample set** (via the `musdb` package, `musdb.DB(download=True)`)
  — the *lightweight* ~10MB 7-second-excerpt mode MUSDB18 ships for its own
  quick-eval/prototyping use, auto-downloaded, wired into
  `python/tests/conftest.py` as the `musdb_sample` fixture. Comes with
  ground-truth stems (vocals/drums/bass/other) per clip, so it validates
  separation-pipeline *correctness*, not just "did it run." Already used in
  `tests/test_separation_fixtures.py`. Note: pulled in `stempeg` as a
  transitive dependency — the same library flagged in the research pass for
  `.stem.mp4` I/O, a nice confirmation of that finding.
- **Full MUSDB18HQ** (real separation-quality benchmarking, 150 full
  tracks) — a separate, deliberate download from Zenodo under its own
  license terms. Not auto-fetched by tests; wire in explicitly if/when
  real quality benchmarking (not just pipeline correctness) is needed.
- **Bandcamp $0 "name your price" tracks** — real catalog, real
  OAuth/auth/ingestion path, exercises the actual Bandcamp integration
  code (which MUSDB18 can't, since it's not on Bandcamp), zero cost.
- **Free Music Archive, Jamendo, ccMixter** — CC-licensed catalogs, no
  cost, no legal ambiguity, useful for volume if the taste-model needs a
  broader test corpus than what's on Bandcamp's free tier.

## Track-prediction calibration loop

Spec'd separately: `docs/dev/track-calibration-loop.md`. The idea: treat
title/tag-based audio-feature prediction as a genuine forecast (a stated
confidence), resolve it against real `audio_analyze_track` measurement
(confirmed/refuted), and let the resolved-prediction graph become the
"which genre-tags are trustworthy" knowledge base. **Standalone by design**
(2026-07-19 product-positioning decision): this lives in epistemic-dj's own
`CalibrationStore` (SQLite, mirrors `TasteStore`'s pattern), not routed
through the Empirica CLI/package — the shipped product must not require
Empirica at runtime, same principle already applied to `TasteStore`. My own
dev-process PREFLIGHT/CHECK/POSTFLIGHT discipline while building this is
unaffected; that's orthogonal to what ships. Phase A (Bandcamp + YouTube
Music, single practitioner) is specced for build; Phase C (parallel
practitioners on the shared store) is design-level with named open
questions.

## Deferred (real direction, not current scope)

AI-driven control of prosumer MIDI hardware (controllers, mixers,
stream-deck-class devices, embedded boxes) — the user chooses material and
makes engineering-level calls, the AI drives execution. Flagged so it isn't
lost, explicitly out of scope for Sprints 1-3.

## Toolchain

`uv` for Python dependency management, `ruff` for lint+format, `pyright`
for type checking, `pytest` for tests. `pydantic` v2 for all schemas
(MCP tool I/O, taste vectors, tracks, mixtapes). Run from `python/`:

```bash
uv sync              # install
uv run pytest -q     # test
uv run ruff check .  # lint
uv run pyright       # typecheck
uv run epistemic-dj-mcp   # run the MCP server
```

**Gotcha**: if your shell has an unrelated `VIRTUAL_ENV` set (e.g. a tmux
session venv), `uv run` can silently use *that* interpreter instead of
`python/.venv`, despite printing a warning that claims it's being ignored.
Symptom: `ModuleNotFoundError` for a dependency that's clearly in
`pyproject.toml`, or a `pyright`/`pytest` result that doesn't match what
you'd expect from the code. Fix: `unset VIRTUAL_ENV` before `uv run`
commands, or run `echo $VIRTUAL_ENV` first to check.
