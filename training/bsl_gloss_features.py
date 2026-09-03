"""Shared landmark feature extraction for the BSLDict gloss matcher."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SAMPLES_PER_VIDEO = 32
FEATURE_VERSION = 1


def _normalise_hand(landmarks):
    points = np.asarray([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
    centred = points - points[0]
    scale = max(np.linalg.norm(centred, axis=1).max(), 1e-6)
    return centred / scale, points[0], scale


def _hand_descriptor(samples, wrists):
    """Return a fixed descriptor for one handedness stream."""
    if not samples:
        return np.zeros(267, dtype=np.float32)

    landmarks = np.asarray(samples, dtype=np.float32).reshape(len(samples), -1)
    wrist_points = np.asarray(wrists, dtype=np.float32)
    wrist_path = wrist_points - wrist_points[0]
    path_scale = max(np.linalg.norm(wrist_path, axis=1).max(), 1e-3)
    wrist_path /= path_scale

    return np.concatenate(
        [
            landmarks.mean(axis=0),
            landmarks.std(axis=0),
            landmarks[0],
            landmarks[-1],
            wrist_path.mean(axis=0),
            wrist_path.std(axis=0),
            wrist_path[-1],
            wrist_path.min(axis=0),
            wrist_path.max(axis=0),
        ]
    ).astype(np.float32)


def extract_video_feature(video_path: str | Path, samples_per_video: int = SAMPLES_PER_VIDEO):
    """Extract a 536-D, camera-scale-invariant descriptor from a sign video.

    The output order is right hand then left hand, followed by each hand's frame
    presence ratio. ``None`` means no hands were detected.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    selected = set(np.linspace(0, frame_count - 1, min(frame_count, samples_per_video), dtype=int))
    samples = {"Right": [], "Left": []}
    wrists = {"Right": [], "Left": []}
    processed = 0

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index in selected:
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                processed += 1
                if result.multi_hand_landmarks and result.multi_handedness:
                    for hand, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                        side = handedness.classification[0].label
                        normalised, wrist, _ = _normalise_hand(hand.landmark)
                        samples[side].append(normalised)
                        wrists[side].append(wrist)
            frame_index += 1
    cap.release()

    if not samples["Right"] and not samples["Left"]:
        return None

    presence = np.array(
        [len(samples["Right"]) / max(processed, 1), len(samples["Left"]) / max(processed, 1)],
        dtype=np.float32,
    )
    return np.concatenate(
        [_hand_descriptor(samples["Right"], wrists["Right"]), _hand_descriptor(samples["Left"], wrists["Left"]), presence]
    )
