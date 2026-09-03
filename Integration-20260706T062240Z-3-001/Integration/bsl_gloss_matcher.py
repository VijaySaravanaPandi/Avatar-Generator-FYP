"""Runtime loader for the trained BSLDict video-to-gloss matcher."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent.parent / "training"
MODEL_PATH = SCRIPT_DIR / "bsl_gloss_matcher.joblib"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from bsl_gloss_features import FEATURE_VERSION, extract_video_feature

_MODEL = None
_LOAD_ATTEMPTED = False


def _load_model():
    global _MODEL, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _MODEL
    _LOAD_ATTEMPTED = True
    if not MODEL_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    if model.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError("BSL gloss matcher feature version does not match the installed extractor. Retrain the matcher.")
    _MODEL = model
    return _MODEL


def predict_gloss_details(video_path, candidate_count=5):
    """Retrieve a BSLDict gloss and reject weak or ambiguous matches.

    The previous implementation converted distance from one nearest clip into a
    percentage.  That made every upload look confident because a nearest
    neighbour index always has an answer.  We instead aggregate evidence across
    nearby examples of each gloss and compare the distance with held-out BSLDict
    validation clips saved when the model is trained.
    """
    model = _load_model()
    if model is None:
        return {"gloss": None, "confidence": 0.0, "accepted": False, "candidates": []}

    feature = extract_video_feature(video_path)
    if feature is None:
        return {"gloss": None, "confidence": 0.0, "accepted": False, "candidates": []}
    if feature.size != model["feature_dim"]:
        raise RuntimeError("BSL gloss matcher feature dimensionality does not match the trained model.")

    vector = model["pca"].transform(model["scaler"].transform(feature.reshape(1, -1)))
    neighbour_count = min(max(candidate_count * 8, 16), len(model["glosses"]))
    distances, indices = model["neighbours"].kneighbors(vector, n_neighbors=neighbour_count)

    # More than one reference clip may represent a gloss.  Combining their
    # evidence is less sensitive to an accidental close frame sequence.
    by_gloss = {}
    for distance, index in zip(distances[0], indices[0]):
        gloss = str(model["glosses"][index])
        item = by_gloss.setdefault(gloss, {"gloss": gloss, "distance": float(distance), "evidence": 0.0})
        item["distance"] = min(item["distance"], float(distance))
        item["evidence"] += float(np.exp(-float(distance)))

    ranked = sorted(by_gloss.values(), key=lambda item: (-item["evidence"], item["distance"]))
    if not ranked:
        return {"gloss": None, "confidence": 0.0, "accepted": False, "candidates": []}
    total_evidence = sum(item["evidence"] for item in ranked) or 1.0
    for item in ranked:
        item["score"] = item["evidence"] / total_evidence
        item.pop("evidence")

    best = ranked[0]
    runner_up_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
    margin = max(0.0, best["score"] - runner_up_score)
    calibration = model.get("distance_calibration") or {}

    p50 = float(calibration.get("p50", 10.14))
    p90 = float(calibration.get("p90", 14.05))
    p95 = float(calibration.get("p95", 15.33))
    accepted_p95 = float(calibration.get("accepted_p95", 14.87))

    dist = float(best["distance"])
    exact_reference_limit = float(os.environ.get("BSL_GLOSS_EXACT_DISTANCE", "0.50"))
    exact_reference = bool(dist <= exact_reference_limit)

    # Calibrated confidence metric
    dist_factor = float(np.clip(1.0 - (dist / max(p95, 1.0)), 0.0, 1.0))
    score_factor = float(np.clip(best["score"] * 4.0, 0.0, 1.0))
    margin_factor = float(np.clip(margin * 5.0, 0.0, 1.0))

    confidence = float(np.clip(0.60 * dist_factor + 0.25 * score_factor + 0.15 * margin_factor, 0.0, 1.0))
    if exact_reference:
        confidence = max(confidence, 0.99)

    accepted = bool(exact_reference or (dist <= p90 and confidence >= 0.35 and best["score"] >= 0.05))

    return {
        "gloss": best["gloss"] if accepted else None,
        "nearest_gloss": best["gloss"],
        "confidence": confidence,
        "accepted": accepted,
        "distance": dist,
        "reason": (
            "exact_feature_reference" if exact_reference else "nearest_feature_match"
        ) if accepted else "ambiguous_or_out_of_dictionary_video",
        "candidates": ranked[:candidate_count],
    }



def predict_gloss(video_path):
    """Compatibility wrapper returning an accepted gloss and its confidence."""
    result = predict_gloss_details(video_path)
    return result["gloss"], result["confidence"]
