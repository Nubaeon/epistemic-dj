# Setup: connecting Bandcamp and YouTube

**Status: alpha, devs only.** This is a working MCP server you run locally
and drive from a Claude conversation — not a packaged app. Expect rough
edges; the setup below is the real, current process, not a smoothed-over
version of it.

## Run the MCP server

```bash
cd python
uv sync
uv run epistemic-dj-mcp
```

Register it in Claude Code's MCP config alongside the existing JS server
(see the root `README.md`).

## Bandcamp: get your identity token

Bandcamp has no public login/OAuth for personal collections — every
integration (including this one) authenticates the same way the unofficial
tooling ecosystem does: with your browser's session cookie.

1. Log into [bandcamp.com](https://bandcamp.com) in your browser.
2. Open developer tools → Application/Storage → Cookies → `bandcamp.com`.
3. Find the cookie named **`identity`** and copy its value.

Treat this like a password — it's your live session. Don't share it, don't
commit it, don't paste it anywhere public.

In your Claude conversation, once the MCP server is registered:

```
Use bandcamp_set_credentials with my identity token: <paste value>
```

Then:

```
Use bandcamp_get_collection to fetch my collection
```

You should get back a list of tracks/albums from your own purchased
collection. If you get an error mentioning credentials, the token wasn't
set or has expired — repeat the extraction step (these cookies rotate
periodically).

## YouTube: capture browser headers

YouTube Music has no usable OAuth path for this either — `ytmusicapi`'s
device-flow OAuth is blocked by Google's anti-phishing policy on the
Device Authorization Grant (see `docs/dev/track-calibration-loop.md`).
Same trust model as Bandcamp: real browser session data, run once,
interactively, in a real terminal (it can't be automated):

```bash
cd python
uv run python -m epistemic_dj.youtube.auth_setup
```

This walks you through: open [music.youtube.com](https://music.youtube.com)
logged in → DevTools → Network tab → trigger a request (e.g. open your
Library) → find a request to `/youtubei/v1/browse` → copy its request
headers → paste when prompted. Headers are saved locally (see
`epistemic_dj.youtube.client.DEFAULT_HEADERS_PATH`) and reused by
`youtube_get_playlist_tracks`, `youtube_get_subscribed_artists`, and
`youtube_search_tracks` — no per-session re-auth needed until the headers
expire.

Once set up, playlists are the recommended entry point over subscriptions
(subscriptions are often noisy — channels followed for unrelated reasons,
not a taste signal). In Claude:

```
Use youtube_get_playlist_tracks with my playlist id: <paste id>
```

## Rendering mashups (optional extra)

`render_mashup` (full-track overlay) works with the base install.
`render_stem_mashup` (vocals from one track over another's instrumental)
needs Demucs, which is kept out of the default install since it's heavy
and GPU-dependent:

```bash
cd python
uv sync --extra separation
```

`render_stem_mashup` defaults to `device="cuda"` — pass `device="cpu"`
explicitly if you don't have a GPU (slower, but works). Renders write
real `.wav` files to `python/renders/`; `auto_align=True` (the default,
both tools) writes both a naive and an auto-corrected version so you
can A/B them directly.

## What's not there yet

- Track tags/genre metadata on Bandcamp (collection listing is
  lightweight — see `docs/dev/architecture.md` for why)
- Lossless audio download on Bandcamp (currently an open unknown — see
  `docs/dev/architecture.md`)
- Any packaging/installer — this is source-checkout-and-run only right now
- Export/upload pipeline for rendered mashups (YouTube first, Bandcamp
  last — no confirmed public Bandcamp upload API)
