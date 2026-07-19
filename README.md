# 🎧 Epistemic DJ

**Music taste profiling & AI-driven generation system.**

Analyze, curate, and create music grounded in learned epistemic profiles. epistemic-dj builds taste models from music analysis and listener preferences, then uses LLMs to curate, mix, and eventually compose new music matching those profiles.

**Two core workflows:**

1. **Epistemic → Sound**: Transform your cognitive state into music. High uncertainty? Dissonant, chaotic patterns. Deep focus? Clean, driving beats. Celebrating a win? Full-on euphoric build-ups.

2. **Taste Profiling & Curation**: Build epistemic profiles of musical taste (genre affinity, mood preferences, discovery vectors), then use LLMs to:
   - Analyze music sources (metadata, audio features, contextual signals)
   - Curate playlists matching taste profiles
   - Mix and match tracks for dynamic compositions
   - Generate new music grounded in learned taste models

## Features

### Epistemic State → Sound
- **MCP Tools** for Claude Code integration
- **Pattern Generator** - 13 epistemic vectors → Strudel live coding patterns
- **Mood Presets** - focus, energize, reflect, debug, celebrate
- **Web UI** - Interactive sliders + embedded Strudel REPL

### Music Taste Profiling & Curation
- **Music Analyzer** - Extract features, metadata, contextual signals from audio
- **Taste Profile Builder** - Learn genre affinity, mood preferences, discovery vectors from listener input
- **LLM Curator** - Use Claude to match music sources against taste profiles
- **Dynamic Mixing** - Create playlists, mashups, and smooth transitions from profiles
- **Audio Analyzer** - Convert existing music into epistemic vectors

### Generative Composition (upcoming)
- **Strudel Integration** - Algorithmic music composition via live-coding
- **Profile-Driven Generation** - Generate original music grounded in taste models
- **Cross-Fade Patterns** - Smooth transitions between epistemic states

## Quick Start

```bash
# Install
npm install

# Test the pattern generator
node src/generator/epistemic-to-strudel.js --mood celebrate

# Run the MCP server
node src/mcp/server.js

# Serve the web UI
npx serve src/web
```

## MCP Tools

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

## Epistemic → Musical Mappings

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

### Phase 1: Epistemic State → Sound (current)
- [x] Pattern generator (vectors → Strudel)
- [x] MCP server with tools
- [x] Web UI with sliders
- [ ] WebSocket bridge for real-time updates
- [ ] Empirica integration for automatic state tracking

### Phase 2: Taste Profiling & Curation (in progress)
- [ ] Music analyzer (audio → vectors)
- [ ] Taste profile builder (listener preferences → models)
- [ ] LLM curator (match profiles to sources)
- [ ] Playlist & mashup generation
- [ ] Music source integration (Spotify, SoundCloud, local files)

### Phase 3: Generative Composition (planned)
- [ ] Profile-driven music generation (compose new tracks from taste models)
- [ ] Feedback loop (music influences cognition?)
- [ ] Cross-practice composition (mix epistemic state with taste profiles)
- [ ] Empirica artifact integration (log music as a form of thought)

## Project structure

Two languages, on purpose — see [`docs/dev/architecture.md`](docs/dev/architecture.md) for why:

- `src/` — existing JS/ESM MCP server: epistemic vectors → Strudel patterns (this doc, above)
- `python/` — new Python MCP server: Bandcamp integration, stem separation, taste profiling (Sprints 1-3). Quickstart: `cd python && uv sync && uv run epistemic-dj-mcp`
- `docs/human/` — product narrative and vision, for people
- `docs/dev/` — technical architecture, for engineers and future Claude sessions

## License

MIT

## Credits

Built with:
- [Strudel](https://strudel.cc) - Live coding music in the browser
- [MCP SDK](https://modelcontextprotocol.io) - Model Context Protocol
- [Empirica](https://github.com/Nubaeon/empirica) - Epistemic self-assessment
