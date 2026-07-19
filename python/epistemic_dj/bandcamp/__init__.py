"""Bandcamp adapter -- ingestion source #1 of a source-agnostic artifact substrate.

No official API covers personal collections (Bandcamp's real API is
partner-only: labels/merch). This wraps cookie-session auth, the pattern
the whole unofficial ecosystem has converged on. Keep the interface here
source-agnostic (Track, not BandcampTrack) so a future SoundCloud/local-file
adapter can slot in the same way.
"""
