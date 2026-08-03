"""Stem separation. First cut (2026-08-03, David's explicit choice) wraps
plain Demucs (demucs_separator.py) -- pip-installable, works today -- rather
than ZFTurbo/Music-Source-Separation-Training (higher-quality models per
the MVSEP leaderboard, but git-clone-only, not on PyPI, confirmed via
`pip index versions music-source-separation-training` returning no
match). ZFTurbo remains the real upgrade path once the selective-stem-
overlay mechanics this unblocks are proven -- see docs/dev/architecture.md.

Kept as an optional dependency group (see pyproject.toml
[project.optional-dependencies], install with `uv sync --extra separation`)
since torch/model weights are heavy and GPU-dependent. Nothing in this
package is imported at module top-level elsewhere in epistemic_dj --
callers import demucs_separator directly (and lazily) so the rest of the
MCP server works without the extra installed.
"""
