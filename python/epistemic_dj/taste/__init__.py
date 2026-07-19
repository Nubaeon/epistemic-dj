"""The taste engine: borrows Empirica's artifact pattern (findings, distilled
patterns/anti-patterns, confidence decay) pointed at the user+music domain,
via a standalone SQLite store rather than importing Empirica's own
SessionDatabase directly (user decision, 2026-07-19 -- keeps human taste
data separate from this session's own AI-epistemic tracking). Intended
future path: graduate into a dedicated 'epistemic-dj-lab' Empirica practice
once there's an actual module/skills built around it. See
docs/dev/architecture.md.
"""

from epistemic_dj.taste.store import PatternNotFoundError, TasteStore

__all__ = ["TasteStore", "PatternNotFoundError"]
