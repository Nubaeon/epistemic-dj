# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/) — this project is
alpha, pre-1.0, so expect breaking changes between minor versions without
a deprecation period.

## [0.1.0] — 2026-08-03 (first tagged release)

**Status: alpha, developers only.** No packaging, no installer, no
stability guarantees. Run from source.

### Added — JS side (pre-existing, first tag)
- Epistemic State → Sound: 13 Empirica vectors → Strudel live-coding
  patterns via MCP tools (`generate_pattern`, `generate_mood`,
  `explain_mapping`, `crossfade_pattern`)
- Web UI with interactive sliders + embedded Strudel REPL

### Added — Python side
- **Source integration**: real Bandcamp collection access (cookie auth)
  and YouTube Music search/playlists/subscriptions (browser-header auth) —
  no official personal-collection API exists for either, so both use the
  same unofficial-ecosystem trust model
- **Real audio analysis**: tempo, energy (`kinetic_energy`), and mood
  (`valence`, DEAM-trained regression) extracted from actual downloaded
  audio via multi-checkpoint sampling (beginning/middle/end, more
  checkpoints for long material) — never from track titles or metadata
- **Calibration loop**: a standalone `CalibrationStore` (no Empirica
  runtime dependency) implementing predict → measure → resolve →
  Brier-score, generalized beyond a single quantity (`kinetic_energy`,
  `tempo_bpm`, `tempo_compatibility_pct`) with Bayesian bias-correction
  and self-calibrating confidence buckets
- **Taste profiling**: findings/patterns/anti-patterns as real artifacts
  (not an opaque embedding), exportable taste profiles, shareable
  mixtapes
- **Mashup rendering** (new this cycle): pitch-preserving time-stretch
  (librosa phase vocoder) to beatmatch two tracks, full-track overlay,
  and a genuine alignment-quality metric (cross-correlation of
  onset-strength envelopes) — not a guess. `render_mashup` renders a
  naive version, then auto-corrects using its own alignment signal and
  renders a second version, so before/after is directly A/B-listenable

### Known gaps (tracked, not silently missing)
- Stem separation (Demucs/`stemgen` chosen, not wired in yet) — mashups
  are full-track overlay for now, not selective vocals/instrumental
- No export/upload pipeline yet (YouTube first, Bandcamp last — no
  confirmed public Bandcamp upload API)
- Bandcamp lossless audio download path unconfirmed (public streams are
  MP3-128 only)
- Auto-alignment is a single correction step, not an iterative optimizer
  — measurably helps, doesn't always fully converge
- Onboarding-interview taste profile builder not built; taste profiles
  currently accumulate from ad-hoc findings/patterns only
