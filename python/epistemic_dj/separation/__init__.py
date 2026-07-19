"""Stem separation -- wraps ZFTurbo/Music-Source-Separation-Training rather than
hand-picking bare Demucs/HTDemucs (which is no longer SOTA; BS-RoFormer and
Mel-Band RoFormer beat it on every stem per the MVSEP leaderboard). Kept as an
optional dependency group (see pyproject.toml [project.optional-dependencies])
since torch/model weights are heavy and GPU-dependent -- not needed for
Sprint 1 (Bandcamp OAuth + MCP) work.
"""
