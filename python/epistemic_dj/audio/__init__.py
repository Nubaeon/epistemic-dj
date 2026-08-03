"""Real audio-feature extraction -- replaces metadata/title-only reasoning,
which David correctly identified as unreliable (Bandcamp titles/metadata
are often wrong about actual sound). Downloads the mp3-128 stream Bandcamp
already serves (confirmed live via get_track(), see empirica finding f21)
and extracts real signal (BPM, energy, spectral features) via librosa.
"""

from epistemic_dj.audio.analysis import (
    AudioFeatures,
    SampledAudioFeatures,
    analyze_track,
    download_stream,
    estimate_bytes_for_seconds,
    load_audio_window,
    sample_track,
    sample_track_checkpoints,
)
from epistemic_dj.audio.mapping import audio_features_to_vectors

__all__ = [
    "AudioFeatures",
    "SampledAudioFeatures",
    "analyze_track",
    "audio_features_to_vectors",
    "download_stream",
    "estimate_bytes_for_seconds",
    "load_audio_window",
    "sample_track",
    "sample_track_checkpoints",
]
