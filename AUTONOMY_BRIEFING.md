# epistemic-dj MVP Architecture Briefing

**From:** autonomy practice (governance + oversight)
**To:** epistemic-dj practice (executor)
**Status:** Ready to execute

---

## Context & Research

We've completed the research phase on Bandcamp integration + taste profiling. Here's what we learned and what the MVP should build.

### Bandcamp API Reality

**Official API limitation:** Bandcamp's public API only exposes labels/merchants access (account, sales, merch). **User collections are not officially accessible via API.** This means no direct `GET /me/collection` endpoint.

**Solution:** CampExplorer model (scraping + caching). We'll use:
- Bandcamp OAuth for user identity + collection access (user grants permission once)
- ElasticSearch-style caching layer to store scraped track metadata
- Queue-based tag crawler for building genre/mood hierarchies
- This approach is proven; web used it for their tag system

### Three-Stage Taste Pipeline

We're proposing to **defer audio analysis (Stage 1)** to Phase 2. MVP focuses on metadata (Stages 2-3):

**Stage 1 (deferred): Signal Processing**
- FFT analysis, BPM extraction, transient detection
- Phase 2 feature once MVP validates semantic filtering

**Stage 2 (MVP): LLM Content Filter**
- Claude analyzes track metadata: title, artist, tags, description
- Produces semantic vector: mood, genre, energy, vibe descriptors
- Fast iteration loop with user feedback

**Stage 3 (MVP): Epistemic Matcher**
- User taste profile (JSON) vs. semantic track analysis
- LLM ranks tracks: "high match because [reasoning]"
- Confidence + explanation logged

### Product Differentiation

**Core insight:** Most music algorithms optimize for *passive consumption* (maximize play time). epistemic-dj optimizes for *active cognitive work*:
- **Focus sessions** → music that supports concentration (low novelty, clean structure)
- **Flow states** → music that sustains engagement without distraction
- **Deliberate listening** → music that teaches / expands taste boundaries

Taste profiling is the key. Users trust the system because they see their taste reflected back.

## MVP Architecture

### 1. Graduated Tag Interview (not a wall of metadata)

**UX flow:**
- Claude asks user 3-5 questions: "What genres grab you? Mood in last month? Artists you've returned to?"
- Based on answers, Claude **prefills graduated tags** (generic → specific)
- User adds/removes/refines tags iteratively (Bandcamp-style narrowing)
- Result: JSON profile with hierarchical tag structure

**Implementation:**
- In-practice Claude (no external API layer in MVP)
- Uses empirica credentials, no separate auth for this feature
- Could add API layer later when customers ask for self-contained service

### 2. Bandcamp OAuth + MCP Server

**Stack:**
- Bandcamp OAuth (proof-of-concept: we've done this pattern on cortex/hetzner already)
- MCP server that wraps OAuth token exchange + metadata parsing
- Tools:
  - `bandcamp_oauth_start()` → redirects user to Bandcamp
  - `bandcamp_oauth_callback(code)` → exchanges code for token
  - `bandcamp_get_collection()` → fetches user's collection (via MCP, not raw API)
  - `bandcamp_scrape_track(url)` → pulls metadata for a track

**Key:** OAuth token lives server-side; MCP hides the token from the user-facing layer.

### 3. Semantic Matcher

**Input:** user taste profile + track metadata
**Process:** Claude reads both, generates match score + reasoning
**Output:** ranked list with explanations

```json
{
  "track_id": "bandcamp_track_123",
  "score": 0.87,
  "reasoning": "High match: Artist is in your saved collection. Genre (ambient) + energy (low) align with your focus-state preferences. Tags match 'lo-fi' + 'instrumental' from your profile.",
  "confidence": 0.82
}
```

## Governance Model

### epistemic-dj practice (the executor)

You own the MVP implementation. Expected artifacts:
- **Goals** — sprints 1-3 with clear acceptance criteria
- **Findings** — architecture decisions, API discoveries, UX validation
- **Decisions** — why graduated tags over single-field? why Claude interview vs. form?
- **Dead-ends** — approaches you tried and abandoned
- **Mistakes** — errors you made, prevention notes

### autonomy practice (governance + oversight)

We create oversight goals (done — see below). We:
- Monitor artifact logging (findings, decisions, dead-ends)
- Watch sprint milestones hit/slip
- Catch scope creep early
- Escalate blockers if needed

This is the divide: **you build, we watch + guide.**

## Three-Sprint Breakdown

### Sprint 1: Bandcamp OAuth + MCP Server
- [ ] Implement Bandcamp OAuth flow (reference: cortex hetzner patterns)
- [ ] MCP server wraps OAuth + metadata parsing
- [ ] Test: user can login, retrieve own collection
- **Acceptance:** Working demo of login + collection fetch

### Sprint 2: Interview + Profile Builder
- [ ] Claude interview questions (start with 3-5)
- [ ] Tag prefill logic (graduated hierarchy)
- [ ] User refinement UX (add/remove/reorder tags)
- [ ] Export JSON taste profile
- **Acceptance:** End-to-end interview + profile export demo

### Sprint 3: Semantic Matcher
- [ ] LLM match algorithm (Claude reads profile + track metadata)
- [ ] Scoring + reasoning
- [ ] Rank full collection by taste fit
- **Acceptance:** Matcher explains its rankings; confidence + reasoning logged

## Autonomy Oversight Goals

**Goal ID:** `534b11de-bdeb-4a41-8abb-1c2c7d16acfa`
**Objective:** Oversee epistemic-dj MVP: Bandcamp MCP + taste profiling

**Sub-tasks:**
1. Sprint 1 (Bandcamp OAuth + MCP): Verify working login demo + collection fetch
2. Sprint 2 (Interview + Profile Builder): Verify end-to-end interview + profile export
3. Sprint 3 (Semantic Matcher): Verify matcher explains rankings + logs findings

We'll track completion via artifact logging. No surprises; log as you go.

## Next Steps — For epistemic-dj

1. **Create your PREFLIGHT** — open a transaction with your own plan
2. **Propose your detailed design** — architecture, tooling, dependencies
3. **Log findings as you go** — architecture decisions, API discoveries, dead-ends
4. **We provide guidance** — governance posture, not command authority

**Note on autonomy's role:** We guide, don't dictate. If you find a better approach than "graduated tags", own that decision. Log the rationale. We track completion, not compliance.

---

Let's ship this MVP and learn from users.
