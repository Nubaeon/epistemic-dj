"""Real audio-feature extraction -- replaces metadata/title-only reasoning,
which David correctly identified as unreliable (Bandcamp titles/metadata
are often wrong about actual sound). Downloads the mp3-128 stream Bandcamp
already serves (confirmed live via get_track(), see empirica finding f21)
and extracts real signal (BPM, energy, spectral features) via librosa.
"""

from epistemic_dj.audio.analysis import AudioFeatures, analyze_track, download_stream, sample_track
from epistemic_dj.audio.mapping import audio_features_to_vectors

__all__ = [
    "AudioFeatures",
    "analyze_track",
    "audio_features_to_vectors",
    "download_stream",
    "sample_track",
]
