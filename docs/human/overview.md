# epistemic-dj

## What this is

A tool for people who don't want to produce a track from scratch, but want to
interact with the *architecture* of music — grab a vocal, slice a loop, build
a custom listening session — without learning a DAW.

Most music tools force a binary choice: passive consumer (Spotify) or
hardcore producer (DAW, hardware, phase alignment). There's a wide,
technically-engaged middle tier that neither serves — people who want to
engineer their listening, not just receive it. That's who this is for:
engineers, tinkerers, prosumers. Not generalists, not professional musicians.

## The core idea

Your taste isn't a black box the app owns. It's a set of inspectable,
portable epistemic artifacts — the same kind of finding/pattern/decision
tracking Empirica already uses to help AI practices calibrate, just pointed
at you and music instead of an AI and code. You can ask "why did you play me
this" and get a real answer, not a shrug.

Because it's portable, your taste profile isn't locked into one app either.
It can:
- Curate and rank tracks from Bandcamp (or any future source)
- Drive a DJ set or focus-session playlist
- Be exported as a seed for prosumer creation tools like [strudel.cc](https://strudel.cc)
- Be **shared** — a curated mixtape carries the *why* along with the tracks,
  so a friend's system can evaluate it against your own taste rather than
  just handing you a flat playlist

## Where it's going

Three tracks, roughly in order:
1. **Bandcamp integration** — connect your purchased collection
2. **Taste profiling** — an onboarding interview (including humming/whistling
   what you're after, not just picking tags) that builds your profile
3. **Curation & mixing** — an AI that discriminates what you'd want, builds
   sets, and (later) drives prosumer hardware — mixers, controllers,
   embedded gear — so you choose the material and the engineering judgment
   calls, and the AI drives execution

We deliberately don't compete with DJ software (Traktor, Serato, VirtualDJ
already do real-time stem separation better than we ever will) or with
professional production tools. The product is taste curation and
portability, not another mixer or another DAW.

See [`docs/human/setup.md`](setup.md) to connect your own Bandcamp
collection, [`docs/dev/architecture.md`](../dev/architecture.md) for the
technical design, or `AUTONOMY_BRIEFING.md` for the original MVP scope.
