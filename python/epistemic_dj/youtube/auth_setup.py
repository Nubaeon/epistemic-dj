"""One-time interactive YouTube auth setup.

Run this yourself in a real terminal -- it needs headers copied from a
logged-in browser session, a real human step that can't be automated.

Why headers, not OAuth: ytmusicapi's OAuth (setup_oauth(), device flow) is
blocked by Google's policy against the Device Authorization Grant for
non-basic scopes (anti "device code phishing" measure -- see
docs/dev/track-calibration-loop.md). This is the same trust model as
bandcamp/client.py's cookie auth: real session data, not a formal OAuth
grant, so Google's OAuth policy doesn't apply. An interim/dev-scope
solution -- real OAuth (a different flow: Desktop-app loopback, not
device flow) is the likely eventual path if this ships multi-user.

How to get the headers (Firefox or Chrome/Edge):
  1. Open https://music.youtube.com and make sure you're logged in.
  2. Open DevTools (F12) -> Network tab.
  3. Click around the site (e.g. open your Library) to trigger a request.
  4. Find a request to `/youtubei/v1/browse` (or similar), right-click it,
     and choose "Copy Request Headers" (Firefox) or "Copy as cURL" then
     extract the header lines (Chrome/Edge).
  5. Paste the raw headers when this script prompts you.

Usage:
  uv run python -m epistemic_dj.youtube.auth_setup
"""

from __future__ import annotations

from ytmusicapi import setup

from epistemic_dj.youtube.client import DEFAULT_HEADERS_PATH


def main() -> None:
    DEFAULT_HEADERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    setup(filepath=str(DEFAULT_HEADERS_PATH))
    print(f"Headers saved to {DEFAULT_HEADERS_PATH}")


if __name__ == "__main__":
    main()
