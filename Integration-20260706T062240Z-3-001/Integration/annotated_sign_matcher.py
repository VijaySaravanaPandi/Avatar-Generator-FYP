"""Runtime for the human-annotated video-to-HamNoSys model."""
from pathlib import Path
import sys
import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent.parent / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))
from bsl_gloss_features import FEATURE_VERSION, extract_video_feature

MODEL_PATH = HERE / "annotated_sign_matcher.joblib"

def predict_annotated_sign(video_path):
    if not MODEL_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    if model.get("feature_version") != FEATURE_VERSION or not model.get("release_ready"):
        return None
    feature = extract_video_feature(video_path)
    if feature is None:
        return None
    vector = model["pca"].transform(model["scaler"].transform(feature.reshape(1, -1)))
    distances, indices = model["neighbours"].kneighbors(vector, n_neighbors=1)
    index = int(indices[0, 0])
    return {
        "gloss": str(model["glosses"][index]), "hamnosys": str(model["hamnosys"][index]),
        "confidence": float(max(0.0, 1.0 - distances[0, 0] / 10.0)),
        "reason": "human_annotated_model",
    }
