# Setup: connecting your Bandcamp collection

## Run the MCP server

```bash
cd python
uv sync
uv run epistemic-dj-mcp
```

Register it in Claude Code's MCP config alongside the existing JS server
(see the root `README.md`).

## Get your Bandcamp identity token

Bandcamp has no public login/OAuth for personal collections — every
integration (including this one) authenticates the same way the unofficial
tooling ecosystem does: with your browser's session cookie.

1. Log into [bandcamp.com](https://bandcamp.com) in your browser.
2. Open developer tools → Application/Storage → Cookies → `bandcamp.com`.
3. Find the cookie named **`identity`** and copy its value.

Treat this like a password — it's your live session. Don't share it, don't
commit it, don't paste it anywhere public.

## Connect it

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

## What's not there yet

- Track tags/genre metadata (collection listing is lightweight — see
  `docs/dev/architecture.md` for why)
- Lossless audio download (currently an open unknown — see
  `docs/dev/architecture.md`)
- Stem separation, taste profiling, curation (Sprint 2+)
