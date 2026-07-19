"""The taste engine: reuses Empirica's own artifact substrate (findings,
patterns, anti-patterns, calibration) pointed at the user+music domain
instead of building a bespoke black-box LLM taste-profile system.

This is deliberately thin -- it should call into Empirica's existing
finding-log/pattern/calibration primitives (via the empirica CLI or MCP
tools) rather than reimplementing artifact storage, decay math, or
calibration scoring here. See docs/dev/architecture.md.
"""
