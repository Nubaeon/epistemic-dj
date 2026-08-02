"""Fits valence's audio-feature -> value mapping against DEAM's human-rated
valence annotations -- same dataset already downloaded for kinetic_energy
(scripts/fit_kinetic_energy_regression.py), just the unused valence_mean
column instead of arousal_mean. No new download.

Reuses the exact same 6 audio features tuned for arousal (tempo/energy/
spectral). This is a real methodological risk, not assumed to work:
valence's strongest known audio correlates are usually mode (major/minor)
and harmonic content, which this feature set doesn't capture. Fitting
anyway as the first, cheap attempt -- if R2 comes out weak, that's honest
evidence a harmonic feature is needed, not something to paper over.

Run from python/: uv run python scripts/fit_valence_regression.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from epistemic_dj.audio.analysis import analyze_file

DATA_DIR = Path(__file__).parent.parent / "data" / "deam"
ANNOTATIONS_DIR = (
    DATA_DIR / "extracted" / "annotations" / "annotations averaged per song" / "song_level"
)
AUDIO_ZIP = DATA_DIR / "DEAM_audio.zip"
OUTPUT_PATH = Path(__file__).parent.parent / "epistemic_dj" / "audio" / "valence_model.json"

SUBSAMPLE_SIZE = 300
RANDOM_SEED = 42

FEATURE_NAMES = [
    "tempo_bpm", "rms_energy", "spectral_centroid_hz",
    "onset_density_per_sec", "beat_interval_cv", "spectral_bandwidth_hz",
]


def load_static_annotations() -> dict[int, float]:
    """song_id -> valence, rescaled from DEAM's [1,9] scale to [0,1]."""
    valence_by_id: dict[int, float] = {}
    for csv_path in ANNOTATIONS_DIR.glob("static_annotations_averaged_songs_*.csv"):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                song_id = int(row["song_id"])
                valence_1_9 = float(row["valence_mean"])
                valence_by_id[song_id] = (valence_1_9 - 1.0) / 8.0
    return valence_by_id


def find_audio_names(zf: zipfile.ZipFile) -> dict[int, str]:
    """song_id -> archive member name for the audio file."""
    by_id = {}
    for name in zf.namelist():
        stem = Path(name).stem
        if stem.isdigit() and name.lower().endswith((".mp3", ".wav")):
            by_id[int(stem)] = name
    return by_id


def main() -> None:
    valence_by_id = load_static_annotations()
    print(f"Loaded {len(valence_by_id)} static valence annotations.")

    with zipfile.ZipFile(AUDIO_ZIP) as zf:
        audio_by_id = find_audio_names(zf)
        print(f"Found {len(audio_by_id)} audio files in archive.")

        available_ids = sorted(set(valence_by_id) & set(audio_by_id))
        random.seed(RANDOM_SEED)
        sample_ids = random.sample(available_ids, min(SUBSAMPLE_SIZE, len(available_ids)))
        print(f"Extracting features for {len(sample_ids)} sampled tracks...")

        rows = []
        targets = []
        failed = []
        extract_dir = DATA_DIR / "extracted_audio"
        extract_dir.mkdir(exist_ok=True)

        for i, song_id in enumerate(sample_ids):
            member = audio_by_id[song_id]
            try:
                extracted_path = Path(zf.extract(member, path=extract_dir))
                features = analyze_file(extracted_path, offset=0.0, max_duration=45.0)
                rows.append([getattr(features, name) for name in FEATURE_NAMES])
                targets.append(valence_by_id[song_id])
                extracted_path.unlink()
            except Exception as e:  # noqa: BLE001 -- one-off fitting script, log and continue
                failed.append((song_id, str(e)))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(sample_ids)} processed, {len(failed)} failed so far")

    print(f"Feature extraction done: {len(rows)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"  Sample failures: {failed[:5]}")

    X = np.array(rows)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))

    print(f"\nTrain R²={train_r2:.3f} RMSE={train_rmse:.3f} (n={len(y_train)})")
    print(f"Test  R²={test_r2:.3f} RMSE={test_rmse:.3f} (n={len(y_test)})")
    print(f"Coefficients: {dict(zip(FEATURE_NAMES, model.coef_, strict=True))}")
    print(f"Intercept: {model.intercept_}")

    output = {
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_failed": len(failed),
        "source": "DEAM (CC BY-NC), static valence annotations rescaled [1,9]->[0,1]",
        "random_seed": RANDOM_SEED,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
