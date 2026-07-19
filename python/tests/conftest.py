"""Shared test fixtures.

musdb_sample: the MUSDB18 7-second-excerpt sample set (~10MB, auto-downloaded
on first use, cached by the musdb package afterward). This is MUSDB18's own
built-in quick-eval/prototyping mode -- NOT the full research dataset, and
not something requiring a Zenodo license agreement. Ground-truth stems
(vocals/drums/bass/other) are included per track, which is what makes this
useful for validating a separation pipeline's *output correctness*, not just
"did it run."

The full MUSDB18HQ (real quality benchmarking) is a deliberate separate
download -- see docs/dev/architecture.md -- not wired into automated tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def musdb_sample():
    musdb = pytest.importorskip("musdb", reason="musdb not installed (dev dependency group)")
    return musdb.DB(download=True)
