# 🎧 Epistemic DJ

**MCP server that maps epistemic state to generative music via Strudel.cc**

Transform your cognitive state into sound. High uncertainty? Dissonant, chaotic patterns. Deep focus? Clean, driving beats. Celebrating a win? Full-on euphoric build-ups.

## Features

- **MCP Tools** for Claude Code integration
- **Pattern Generator** - 13 epistemic vectors → Strudel live coding patterns
- **Mood Presets** - focus, energize, reflect, debug, celebrate
- **Web UI** - Interactive sliders + embedded Strudel REPL
- **Audio Analyzer** (coming soon) - Extract epistemic vectors from music

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

- [x] Pattern generator (vectors → Strudel)
- [x] MCP server with tools
- [x] Web UI with sliders
- [ ] Audio analyzer (music → vectors)
- [ ] WebSocket bridge for real-time updates
- [ ] Empirica integration for automatic state tracking
- [ ] Feedback loop (music influences cognition?)

## License

MIT

## Credits

Built with:
- [Strudel](https://strudel.cc) - Live coding music in the browser
- [MCP SDK](https://modelcontextprotocol.io) - Model Context Protocol
- [Empirica](https://github.com/Nubaeon/empirica) - Epistemic self-assessment
