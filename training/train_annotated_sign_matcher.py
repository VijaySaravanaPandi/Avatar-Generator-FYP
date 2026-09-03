"""Train the only model permitted to drive a HamNoSys avatar sequence.

BSLDict supplies video-to-English-gloss metadata, but it does not supply
HamNoSys or SiGML annotations.  This trainer intentionally accepts a separate
human-verified annotation manifest; it will not manufacture phonetic labels
from English definitions.  That distinction prevents a plausible but wrong
avatar from being described as a ground-truth result.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from bsl_gloss_features import FEATURE_VERSION, extract_video_feature

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "training" / "hamnosys_annotations.csv"
MODEL_PATH = ROOT / "Integration-20260706T062240Z-3-001" / "Integration" / "annotated_sign_matcher.joblib"


def read_manifest(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(line for line in handle if not line.lstrip().startswith("#")):
            values = {key: (value or "").strip() for key, value in row.items()}
            required = ("video_path", "sign_id", "hamnosys")
            if not all(values.get(key) for key in required):
                raise ValueError(f"Manifest row is missing one of {required}: {row}")
            video = Path(values["video_path"])
            if not video.is_absolute():
                video = ROOT / video
            if not video.is_file():
                raise FileNotFoundError(f"Annotated video does not exist: {video}")
            rows.append((video, values["sign_id"], values.get("gloss", ""), values["hamnosys"]))
    if not rows:
        raise ValueError("No annotated rows found. Populate training/hamnosys_annotations.csv first.")
    inconsistent = defaultdict(set)
    for _, sign_id, _, hamnosys in rows:
        inconsistent[sign_id].add(hamnosys)
    bad = [sign_id for sign_id, values in inconsistent.items() if len(values) != 1]
    if bad:
        raise ValueError("Each sign_id must have exactly one verified HamNoSys sequence; inconsistent: " + ", ".join(bad[:10]))
    return rows


def split_by_sign(labels, seed):
    """Keep one recording per repeatable sign for an honest signer-independent check."""
    grouped = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    rng = random.Random(seed)
    valid = [rng.choice(indices) for indices in grouped.values() if len(indices) >= 2]
    train = sorted(set(range(len(labels))) - set(valid))
    if not valid:
        raise ValueError("Need at least two annotated recordings of at least one sign for evaluation.")
    return np.asarray(train), np.asarray(valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    features, labels, glosses, notation = [], [], [], []
    for i, (video, label, gloss, hamnosys) in enumerate(rows, 1):
        feature = extract_video_feature(video)
        if feature is None:
            print(f"Skipping (MediaPipe found no hands): {video}")
            continue
        features.append(feature); labels.append(label); glosses.append(gloss); notation.append(hamnosys)
        print(f"Extracted {i}/{len(rows)}", flush=True)
    if len(features) < 2:
        raise RuntimeError("Fewer than two usable annotated videos.")
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels); glosses = np.asarray(glosses); notation = np.asarray(notation)
    train, valid = split_by_sign(labels, args.seed)
    scaler = StandardScaler().fit(features[train])
    n_components = min(args.components, len(train), features.shape[1])
    pca = PCA(n_components=n_components, whiten=True, random_state=args.seed).fit(scaler.transform(features[train]))
    train_vectors = pca.transform(scaler.transform(features[train]))
    valid_vectors = pca.transform(scaler.transform(features[valid]))
    evaluator = NearestNeighbors(n_neighbors=1).fit(train_vectors)
    _, nearest = evaluator.kneighbors(valid_vectors)
    accuracy = float(np.mean(labels[train][nearest[:, 0]] == labels[valid]))
    # A low metric is a release blocker, not a value to hide behind a UI percentage.
    print(f"Held-out top-1 sign accuracy: {accuracy:.2%} ({len(valid)} videos)")
    all_vectors = pca.transform(scaler.transform(features))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "feature_version": FEATURE_VERSION, "scaler": scaler, "pca": pca,
        "neighbours": NearestNeighbors(n_neighbors=min(5, len(features))).fit(all_vectors),
        "sign_ids": labels, "glosses": glosses, "hamnosys": notation,
        "validation_accuracy": accuracy, "validation_size": int(len(valid)),
        "release_ready": bool(accuracy >= 0.80),
    }, MODEL_PATH)
    print(f"Saved {MODEL_PATH}; release_ready={accuracy >= 0.80}")


if __name__ == "__main__":
    main()
