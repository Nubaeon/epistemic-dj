# 🎧 Epistemic DJ

**TL;DR:** Most music tools are one of two things — an app that just plays
you stuff (Spotify), or a full production studio you need years to learn
(a DAW). Nothing in between for people who want to *shape* their
listening without becoming a producer. epistemic-dj is that middle
ground: it learns what you actually like — not by watching what you
skip, but by asking you and by really listening to your tracks — and
keeps a record of *why*, so "why did you play me this" has a real
answer instead of a shrug. With that, it can find you more of what
you'd genuinely want, and build actual mashups: match the tempo, line
up the beats, even pull the vocals off one track and lay them over
another track's instrumental. Every step is checked against the real
audio, never a genre-tag guess standing in for actually listening. Full
story: [`docs/human/overview.md`](docs/human/overview.md).

> **Status: alpha, developers only.** This is a working local MCP server you
> run from source and drive via Claude — not a packaged app, no installer,
> no stability guarantees between commits. Expect rough edges. See
> [`docs/human/setup.md`](docs/human/setup.md) for the real setup process.

**Music taste profiling & AI-driven mashup generation system.**

Analyze, curate, and create music grounded in learned epistemic profiles.
epistemic-dj builds taste models from real music analysis and listener
preferences, then uses a calibrated AI (predict → measure → resolve,
Brier-scored — the same discipline Empirica uses on itself) to curate and
render actual mashups: not a lookup table, and never a metadata/genre
guess standing in for listening to the track.

**Three core workflows:**

1. **Epistemic → Sound** (original, JS side): Transform your cognitive
   state into music. High uncertainty? Dissonant, chaotic patterns. Deep
   focus? Clean, driving beats. Celebrating a win? Full-on euphoric
   build-ups.

2. **Taste Profiling & Curation** (Python side): Build epistemic profiles
   of musical taste from your real Bandcamp collection and YouTube
   library/playlists, then:
   - Analyze real audio (tempo, energy, valence — never metadata alone)
   - Curate tracks matching your taste profile, with the *why* attached
   - Calibrate every prediction against real measurement, Brier-scored,
     so confidence means something

3. **Mashup Rendering** (Python side, new): Beatmatch and overlay real
   tracks into an actual rendered mashup — tempo-matched via
   pitch-preserving time-stretch, alignment-scored via real audio
   cross-correlation (not a guess), auto-corrected against its own
   measurement. Offline, calibrated composition — not a real-time
   DJ-booth tool (see [`docs/human/overview.md`](docs/human/overview.md)
   for that distinction). Stem separation (vocals/instrumental overlay)
   is next; full-track overlay works today.

## Features

### Epistemic State → Sound (JS)
- **MCP Tools** for Claude Code integration
- **Pattern Generator** - 13 epistemic vectors → Strudel live coding patterns
- **Mood Presets** - focus, energize, reflect, debug, celebrate
- **Web UI** - Interactive sliders + embedded Strudel REPL

### Music Taste Profiling & Curation (Python)
- **Bandcamp + YouTube integration** - real collection/library ingestion,
  cookie/header auth (no official personal-collection API exists for either)
- **Real audio analysis** - tempo, energy (`kinetic_energy`), mood
  (`valence`) fit via a DEAM-trained regression, never metadata guessing
- **Calibration loop** - every prediction (energy, tempo, tempo
  compatibility) is logged, resolved against real measurement, and
  Brier-scored — self-correcting confidence, not a static number
- **Taste profiling** - findings/patterns/anti-patterns as real Empirica-
  style artifacts, so "why did you play me this" has an actual answer

### Mashup Rendering (Python, new)
- **Beatmatching** - pitch-preserving time-stretch (librosa phase vocoder)
  to a real measured target tempo, octave-aware (half/double-time)
  compatibility scoring
- **Real renders** - full-track overlay today, written as actual audio
  files (`epistemic-dj/renders/`)
- **Alignment scoring** - genuine cross-correlation of onset-strength
  envelopes measures how well two tracks' beats actually line up, not a
  guess — and the render auto-corrects using its own signal
- **Next**: stem separation (Demucs/`stemgen`) for selective overlay
  (e.g. vocals-over-instrumental) instead of two full mixes competing

### Generative Composition (upcoming, JS)
- **Strudel Integration** - Algorithmic music composition via live-coding
- **Profile-Driven Generation** - Generate original music grounded in taste models
- **Cross-Fade Patterns** - Smooth transitions between epistemic states

## Quick Start

```bash
# JS side: Epistemic State -> Sound
npm install
node src/generator/epistemic-to-strudel.js --mood celebrate   # pattern generator
node src/mcp/server.js                                        # MCP server
npx serve src/web                                              # web UI

# Python side: taste profiling, calibration, mashup rendering
cd python
uv sync
uv run epistemic-dj-mcp
```

See [`docs/human/setup.md`](docs/human/setup.md) for connecting your real
Bandcamp/YouTube accounts (both need a one-time manual credential step —
there's no OAuth flow for either).

## MCP Tools (JS side)

### `generate_pattern`
Generate a Strudel pattern from epistemic vectors.

```json
{
  "vectors": {
    "know": 0.7,
    "uncertainty": 0.3,
    "engagement": 0.8,
    "clarity": 0.6,
    "coherence": 0.7,
    "signal": 0.5,
    "completion": 0.4
  },
  "mode": "full"
}
```

### `generate_mood`
Generate a pattern for a mood preset.

```json
{
  "mood": "focus"
}
```

### `explain_mapping`
Understand how vectors map to music.

### `crossfade_pattern`
Generate transition patterns between states.

## MCP Tools (Python side)

The full tool list is large (Bandcamp/YouTube search+ingestion, taste
findings/patterns/mixtapes, audio analysis, calibration, rendering) — see
`python/epistemic_dj/mcp_server.py` for the authoritative, documented list.
Highlights:

- `bandcamp_get_collection` / `youtube_get_playlist_tracks` — real
  source ingestion
- `audio_analyze_track` — real tempo/energy/valence from actual audio
- `calibration_predict_tempo` / `calibration_resolve` /
  `calibration_brier` — the predict → measure → resolve → score loop
- `calibration_predict_tempo_compatibility` /
  `calibration_resolve_tempo_compatibility` — pairwise mixability,
  audio-grounded on both ends
- `render_mashup` — real time-stretched, beat-aligned overlay render,
  writes actual `.wav` output

## Epistemic → Musical Mappings (JS side)

| Vector | Musical Parameter |
|--------|-------------------|
| **know** | Scale consonance (pentatonic → diminished) |
| **uncertainty** | Pattern degradation, probability |
| **engagement** | Tempo (60-140 BPM), drum intensity |
| **clarity** | Filter cutoff (dark → bright) |
| **coherence** | Rhythmic stability |
| **signal** | Note density, hi-hat patterns |
| **state** | Reverb/room size |
| **change** | Pattern variation (jux, rev) |
| **completion** | Build-up intensity |
| **impact** | Overall volume |

## Claude Code Integration

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "epistemic-dj": {
      "command": "node",
      "args": ["/path/to/epistemic-dj/src/mcp/server.js"]
    }
  }
}
```

Then in Claude:
```
Generate a pattern for my current epistemic state:
- know: 0.6 (decent understanding)
- uncertainty: 0.4 (some unknowns)
- engagement: 0.8 (highly focused)
```

## Roadmap

### Epistemic State → Sound (JS, stable)
- [x] Pattern generator (vectors → Strudel)
- [x] MCP server with tools
- [x] Web UI with sliders
- [ ] WebSocket bridge for real-time updates
- [ ] Empirica integration for automatic state tracking

### Taste Profiling & Curation (Python, in progress)
- [x] Bandcamp + YouTube source integration (real ingestion, not mocked)
- [x] Real audio analysis (tempo/energy/valence from actual audio)
- [x] Calibration loop (predict → measure → resolve → Brier score),
      generalized beyond a single quantity
- [ ] Full onboarding-interview taste profile builder
- [ ] LLM curator matching profiles to sources at scale

### Mashup Rendering (Python, in progress)
Full phase-by-phase detail: [`docs/dev/architecture.md`](docs/dev/architecture.md).
- [x] Tempo prediction + pairwise compatibility, audio-grounded
- [x] Real time-stretched, beat-aligned overlay renders + auto-alignment
- [ ] Stem separation (Demucs/`stemgen`) for selective overlay
- [ ] YouTube upload pipeline
- [ ] Bandcamp export (lowest priority — no confirmed public upload API)

### Generative Composition (planned)
- [ ] Profile-driven music generation (compose new tracks from taste models)
- [ ] Feedback loop (music influences cognition?)
- [ ] Cross-practice composition (mix epistemic state with taste profiles)
- [ ] Empirica artifact integration (log music as a form of thought)

## Project structure

Two languages, on purpose — see [`docs/dev/architecture.md`](docs/dev/architecture.md) for why:

- `src/` — existing JS/ESM MCP server: epistemic vectors → Strudel patterns (this doc, above)
- `python/` — Python MCP server: Bandcamp + YouTube integration, real audio
  analysis, calibration loop, mashup rendering. Quickstart:
  `cd python && uv sync && uv run epistemic-dj-mcp`
- `docs/human/` — product narrative and vision, for people
- `docs/dev/` — technical architecture, for engineers and future Claude sessions

## License

MIT

## Credits

Built with:
- [Strudel](https://strudel.cc) - Live coding music in the browser
- [MCP SDK](https://modelcontextprotocol.io) - Model Context Protocol
- [Empirica](https://github.com/Nubaeon/empirica) - Epistemic self-assessment
