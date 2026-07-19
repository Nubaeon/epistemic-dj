"""epistemic-dj Python MCP server.

Registered in Claude Code's MCP config ALONGSIDE the existing JS server
(src/mcp/server.js) -- this one owns Bandcamp integration, stem separation,
and taste profiling (Sprints 1-3). The JS server keeps owning
epistemic-state-to-sound generation. Two servers, not a rewrite.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("epistemic-dj")


@mcp.tool()
def ping() -> str:
    """Sanity-check tool -- confirms the Python MCP server is wired up correctly."""
    return "epistemic-dj Python MCP server is alive."


# Sprint 1 tools land here: bandcamp_oauth_start, bandcamp_get_collection, etc.
# See docs/dev/architecture.md for the full tool surface plan.


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
