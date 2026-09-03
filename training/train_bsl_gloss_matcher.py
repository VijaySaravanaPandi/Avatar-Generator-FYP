"""Train a BSLDict landmark-based video-to-gloss matcher.

The model is deliberately a retrieval model: BSLDict contains thousands of glosses
with only one or two examples, so a large softmax classifier would not generalise
well. Landmark descriptors are standardised, projected with PCA, and matched to
the nearest training clips.
"""

from __future__ import annotations

import argparse
import pickle
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
METADATA_PATH = ROOT / "bsldict" / "bsldict" / "bsldict_v1.pkl"
VIDEOS_DIR = ROOT / "bsldict" / "bsldict" / "videos_original"
MODEL_PATH = ROOT / "Integration-20260706T062240Z-3-001" / "Integration" / "bsl_gloss_matcher.joblib"
FEATURE_CACHE_PATH = ROOT / "training" / "bsl_gloss_features.npz"


def available_examples():
    with open(METADATA_PATH, "rb") as handle:
        videos = pickle.load(handle)["videos"]

    examples = []
    for name, gloss in zip(videos["name"], videos["word"]):
        path = VIDEOS_DIR / str(name)
        gloss = str(gloss).strip().lower()
        if gloss and path.is_file():
            examples.append((path, gloss))
    return examples


def extract_features(examples, limit=None):
    if limit is not None:
        examples = examples[:limit]

    features, glosses, filenames = [], [], []
    total = len(examples)
    for index, (path, gloss) in enumerate(examples, start=1):
        feature = extract_video_feature(path)
        if feature is not None:
            features.append(feature)
            glosses.append(gloss)
            filenames.append(path.name)

        if index % 100 == 0 or index == total:
            print(f"Extracted {len(features)}/{index} usable clips ({index}/{total} processed)", flush=True)

    if not features:
        raise RuntimeError("MediaPipe did not detect hands in any BSLDict video.")

    np.savez_compressed(
        FEATURE_CACHE_PATH,
        features=np.asarray(features, dtype=np.float32),
        glosses=np.asarray(glosses, dtype=str),
        filenames=np.asarray(filenames, dtype=str),
        feature_version=np.asarray([FEATURE_VERSION], dtype=np.int32),
    )
    return np.asarray(features, dtype=np.float32), np.asarray(glosses, dtype=str), np.asarray(filenames, dtype=str)


def load_or_extract(examples, rebuild, limit):
    if FEATURE_CACHE_PATH.exists() and not rebuild:
        cached = np.load(FEATURE_CACHE_PATH, allow_pickle=False)
        if int(cached["feature_version"][0]) == FEATURE_VERSION:
            print(f"Using cached landmark features: {FEATURE_CACHE_PATH}")
            return cached["features"], cached["glosses"], cached["filenames"]
    return extract_features(examples, limit)


def make_split(glosses, seed):
    grouped = defaultdict(list)
    for index, gloss in enumerate(glosses):
        grouped[gloss].append(index)

    rng = random.Random(seed)
    validation = []
    for indices in grouped.values():
        if len(indices) >= 2:
            validation.append(rng.choice(indices))
    validation = np.asarray(validation, dtype=int)
    training = np.asarray(sorted(set(range(len(glosses))) - set(validation)), dtype=int)
    return training, validation


def evaluate(features, glosses, train_indices, validation_indices, scaler, pca):
    if len(validation_indices) == 0:
        return None
    train_vectors = pca.transform(scaler.transform(features[train_indices]))
    valid_vectors = pca.transform(scaler.transform(features[validation_indices]))
    neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(train_vectors)
    distances, indices = neighbours.kneighbors(valid_vectors)
    predictions = glosses[train_indices][indices[:, 0]]
    accuracy = float(np.mean(predictions == glosses[validation_indices]))
    return accuracy, distances[:, 0], distances[:, 0][predictions == glosses[validation_indices]]


def train(args):
    examples = available_examples()
    print(f"Found {len(examples)} BSLDict videos available for training.")
    features, glosses, filenames = load_or_extract(examples, args.rebuild_features, args.max_videos)
    counts = Counter(glosses)
    train_indices, validation_indices = make_split(glosses, args.seed)

    scaler = StandardScaler()
    scaler.fit(features[train_indices])
    requested_components = min(args.components, len(train_indices), features.shape[1])
    pca = PCA(n_components=requested_components, whiten=True, random_state=args.seed)
    pca.fit(scaler.transform(features[train_indices]))

    evaluation = evaluate(features, glosses, train_indices, validation_indices, scaler, pca)
    all_vectors = pca.transform(scaler.transform(features))
    neighbours = NearestNeighbors(n_neighbors=min(args.neighbours, len(features)), metric="euclidean")
    neighbours.fit(all_vectors)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_version": FEATURE_VERSION,
            "scaler": scaler,
            "pca": pca,
            "neighbours": neighbours,
            "glosses": glosses,
            "filenames": filenames,
            "feature_dim": int(features.shape[1]),
            "train_videos": int(len(features)),
            "unique_glosses": int(len(counts)),
            "repeated_glosses": int(sum(count >= 2 for count in counts.values())),
            # A nearest-neighbour index always returns *some* label.  Retain
            # held-out distance percentiles so runtime can reject videos which
            # are unlike the signs represented by the index.
            "distance_calibration": (
                {
                    "p50": float(np.percentile(evaluation[1], 50)),
                    "p90": float(np.percentile(evaluation[1], 90)),
                    "p95": float(np.percentile(evaluation[1], 95)),
                    # This is deliberately based only on correctly retrieved
                    # held-out clips.  An all-query percentile permits many
                    # wrong labels when the index itself is weak.
                    "accepted_p95": float(np.percentile(evaluation[2], 95)) if len(evaluation[2]) else 0.0,
                    "validation_accuracy": float(evaluation[0]),
                    "validation_size": int(len(validation_indices)),
                }
                if evaluation is not None else None
            ),
        },
        MODEL_PATH,
    )

    print(f"Saved matcher to: {MODEL_PATH}")
    print(f"Indexed {len(features)} clips across {len(counts)} glosses.")
    if evaluation is not None:
        accuracy, distances, correct_distances = evaluation
        print(f"Held-out repeated-gloss top-1 accuracy: {accuracy:.2%} ({len(validation_indices)} clips)")
        print(f"Held-out distance p95: {np.percentile(distances, 95):.3f}")
        if len(correct_distances):
            print(f"Correct-match distance p95 (runtime acceptance limit): {np.percentile(correct_distances, 95):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BSLDict video-to-gloss landmark matcher.")
    parser.add_argument("--max-videos", type=int, default=None, help="Use only the first N available videos.")
    parser.add_argument("--rebuild-features", action="store_true", help="Ignore the cached landmark features.")
    parser.add_argument("--components", type=int, default=128, help="PCA embedding dimensionality.")
    parser.add_argument("--neighbours", type=int, default=5, help="Stored nearest neighbours for inference.")
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
