"""Real stem separation via Demucs (pip-installable, PyTorch-based) --
the pragmatic first cut, chosen over wrapping ZFTurbo/Music-Source-
Separation-Training (not pip-installable, git-clone-only) per David's
explicit choice, 2026-08-03 ("start with demucs... upgrade the model
later without touching the overlay logic"). Higher-quality models
(BS-RoFormer etc., via ZFTurbo) are a real, deferred upgrade path once
the selective-stem-overlay mechanics this unblocks are proven -- see
docs/dev/architecture.md's build-vs-integrate research summary.
"""

from __future__ import annotations

from pathlib import Path

import demucs.api
import numpy as np

DEFAULT_MODEL = "htdemucs"


def separate_stems(
    audio_path: Path, *, model: str = DEFAULT_MODEL, device: str = "cuda"
) -> tuple[dict[str, np.ndarray], int]:
    """Real 4-stem separation (vocals/drums/bass/other, htdemucs's default
    stem set) via Demucs' pretrained model. Returns (stem_name -> mono
    numpy waveform, samplerate) -- downmixed to mono immediately at this
    boundary so nothing downstream (mixing.render's overlay/time-stretch)
    needs to know about torch or stereo channels, matching the rest of
    the render pipeline's mono convention.

    GPU by default (device='cuda') -- CPU inference is realistically
    minutes per track vs. seconds on GPU, so this is not a soft default,
    it's load-bearing for the tool being usable interactively at all.
    """
    separator = demucs.api.Separator(model=model, device=device)
    _original, stem_tensors = separator.separate_audio_file(audio_path)
    stems = {name: tensor.mean(dim=0).cpu().numpy() for name, tensor in stem_tensors.items()}
    return stems, separator.samplerate
