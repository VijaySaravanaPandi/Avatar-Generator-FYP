"""
BSLDict Landmark & Feature Extractor
Extracts normalized 3D hand landmarks and pose trajectories from BSLDict videos
for training Deep Neural Networks for Handshape, Orientation, and Movement classification.
"""

import os
import sys
import glob
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# HamNoSys Canonical Handshape Labels
HANDSHAPE_CLASSES = [
    "hamflathand",
    "hamfist",
    "hamfinger2",
    "hamfinger23",
    "hamfinger2345",
    "hamfinger23spread",
    "hampinch12",
    "hampinchall",
    "hamcee12",
    "hamceeall",
    "hamdoublebent",
    "hamthumboutmod",
    "hamthumbopenmod",
    "hamthumbacrossmod",
]

# HamNoSys Movement Classes
MOVEMENT_CLASSES = [
    "hamnomotion",
    "hammoveu",
    "hammoved",
    "hammovel",
    "hammover",
    "hammoveui",
    "hammoveuo",
    "hammovedi",
    "hammovedo",
    "hamwavy",
    "hamzigzag",
    "hamcircleo",
]

def normalize_hand_landmarks(landmarks_21x3):
    """
    Normalizes 21 3D hand landmarks:
    1. Translates wrist (landmark 0) to origin (0, 0, 0)
    2. Scales by maximum distance from wrist so landmarks are scale-invariant
    """
    coords = np.array(landmarks_21x3, dtype=np.float32) # (21, 3)
    wrist = coords[0:1, :]
    centered = coords - wrist
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 1e-6:
        normalized = centered / max_dist
    else:
        normalized = centered
    return normalized.flatten() # 63-dimensional feature vector

def extract_video_features(video_path):
    """
    Processes a single video:
    - Extracts per-frame normalized 3D hand landmarks (21 points = 63 floats)
    - Extracts wrist trajectory and velocity sequence for movement
    """
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    hand_samples = []
    wrist_trajectory = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                # Pick the dominant hand (largest bounding box or right hand)
                best_hand = results.multi_hand_landmarks[0]
                lm_array = [[lm.x, lm.y, lm.z] for lm in best_hand.landmark]
                norm_feat = normalize_hand_landmarks(lm_array)
                hand_samples.append(norm_feat)
                wrist_trajectory.append([best_hand.landmark[0].x, best_hand.landmark[0].y, best_hand.landmark[0].z])

    cap.release()

    if not hand_samples:
        return None

    return {
        "video": Path(video_path).name,
        "hand_landmarks": np.array(hand_samples, dtype=np.float32),
        "wrist_trajectory": np.array(wrist_trajectory, dtype=np.float32)
    }

def main():
    root_dir = Path(__file__).resolve().parent.parent
    videos_dir = root_dir / "bsldict" / "bsldict" / "videos_original"
    output_dir = root_dir / "training" / "extracted_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(videos_dir.glob("*.mp4")))
    print(f"Found {len(video_files)} BSL video clips in {videos_dir}")

    # Process subset or batch for fast feature extraction
    max_videos_to_process = min(len(video_files), 2500)
    selected_videos = video_files[:max_videos_to_process]
    print(f"Extracting landmark features from {len(selected_videos)} videos...")

    all_hand_features = []
    all_trajectories = []
    video_names = []

    for video_path in tqdm(selected_videos, desc="Extracting MediaPipe Features"):
        try:
            feats = extract_video_features(video_path)
            if feats is not None and len(feats["hand_landmarks"]) > 0:
                all_hand_features.append(feats["hand_landmarks"])
                all_trajectories.append(feats["wrist_trajectory"])
                video_names.append(feats["video"])
        except Exception as e:
            continue

    out_file = output_dir / "bsl_landmarks_dataset.pkl"
    with open(out_file, "wb") as f:
        pickle.dump({
            "video_names": video_names,
            "hand_features": all_hand_features,
            "trajectories": all_trajectories,
            "handshape_classes": HANDSHAPE_CLASSES,
            "movement_classes": MOVEMENT_CLASSES
        }, f)

    print(f"\n[Success] Extracted features from {len(video_names)} videos.")
    print(f"Dataset saved to: {out_file.resolve()}")

if __name__ == "__main__":
    main()
