#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import mediapipe as mp

# =====================================================
# DEEP NEURAL HANDSHAPE INFERENCE ENGINE (NumPy Native)
# =====================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_SCRIPT_DIR, "nn_handshape_weights.npz")
_NN_WEIGHTS = None
_NN_CLASSES = []

if os.path.exists(_WEIGHTS_PATH):
    try:
        _NN_WEIGHTS = np.load(_WEIGHTS_PATH, allow_pickle=True)
        _NN_CLASSES = list(_NN_WEIGHTS["classes"])
    except Exception:
        _NN_WEIGHTS = None

def _leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

def _bn(x, rm, rv, gamma, beta, eps=1e-5):
    return (x - rm) / np.sqrt(rv + eps) * gamma + beta

def _linear(x, weight, bias):
    return x @ weight.T + bias

def _res_block(x, prefix, w):
    res = x
    out = _linear(x, w[f"{prefix}.fc1.weight"], w[f"{prefix}.fc1.bias"])
    out = _bn(out, w[f"{prefix}.bn1.running_mean"], w[f"{prefix}.bn1.running_var"], w[f"{prefix}.bn1.weight"], w[f"{prefix}.bn1.bias"])
    out = _leaky_relu(out)
    out = _linear(out, w[f"{prefix}.fc2.weight"], w[f"{prefix}.fc2.bias"])
    out = _bn(out, w[f"{prefix}.bn2.running_mean"], w[f"{prefix}.bn2.running_var"], w[f"{prefix}.bn2.weight"], w[f"{prefix}.bn2.bias"])
    return _leaky_relu(out + res)

def neural_predict_handshape(normalized_63_feats):
    """Executes vectorized forward pass of the trained HandshapeMLP neural network."""
    if _NN_WEIGHTS is None or not _NN_CLASSES:
        return None
    try:
        w = _NN_WEIGHTS
        x = normalized_63_feats.reshape(1, 63)
        
        # Input layer
        out = _linear(x, w["input_layer.0.weight"], w["input_layer.0.bias"])
        out = _bn(out, w["input_layer.1.running_mean"], w["input_layer.1.running_var"], w["input_layer.1.weight"], w["input_layer.1.bias"])
        out = _leaky_relu(out)
        
        # Residual blocks
        out = _res_block(out, "res1", w)
        out = _res_block(out, "res2", w)
        
        # Mid layer
        out = _linear(out, w["fc_mid.0.weight"], w["fc_mid.0.bias"])
        out = _bn(out, w["fc_mid.1.running_mean"], w["fc_mid.1.running_var"], w["fc_mid.1.weight"], w["fc_mid.1.bias"])
        out = _leaky_relu(out)
        
        # Final classifier layer
        logits = _linear(out, w["classifier.weight"], w["classifier.bias"])
        best_idx = int(np.argmax(logits, axis=1)[0])
        return _NN_CLASSES[best_idx]
    except Exception:
        return None


# =============================
# 3D ROTATION-INVARIANT HANDSHAPE CLASSIFIER
# =============================

def classify_handshape(lm):
    """
    Classifies 3D hand configuration directly from 3D keypoints in a rotation-invariant manner.
    Works for hands pointing UP, DOWN, LEFT, RIGHT, FORWARD.
    """
    try:
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        wrist = pts[0]
        thumb_tip = pts[4]

        # Finger joint definitions: (MCP, PIP, DIP, TIP)
        finger_joints = [
            (1, 2, 3, 4),    # Thumb
            (5, 6, 7, 8),    # Index
            (9, 10, 11, 12), # Middle
            (13, 14, 15, 16),# Ring
            (17, 18, 19, 20) # Pinky
        ]

        extensions = []
        curls = []
        for mcp, pip, dip, tip in finger_joints:
            d_tip = np.linalg.norm(pts[tip] - wrist)
            d_pip = np.linalg.norm(pts[pip] - wrist) + 1e-6
            extensions.append(d_tip / d_pip)

            v1 = pts[pip] - pts[mcp]
            v2 = pts[tip] - pts[dip]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            curls.append(cos_a)

        # Palm span for scaling
        palm_span = np.linalg.norm(pts[9] - wrist) + 1e-6

        # Inter-finger distances
        d_thumb_index = np.linalg.norm(thumb_tip - pts[8]) / palm_span
        d_index_middle = np.linalg.norm(pts[8] - pts[12]) / palm_span

        # Extension booleans (scale & rotation invariant)
        thumb_ext = extensions[0] > 1.15
        index_ext = extensions[1] > 1.20 and curls[1] > 0.15
        middle_ext = extensions[2] > 1.20 and curls[2] > 0.15
        ring_ext = extensions[3] > 1.20 and curls[3] > 0.15
        pinky_ext = extensions[4] > 1.20 and curls[4] > 0.15

        num_main_ext = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # 1. Pinch / C-shape
        if d_thumb_index < 0.38:
            if not middle_ext and not ring_ext and not pinky_ext:
                return "hamcee12"
            elif middle_ext and ring_ext and pinky_ext:
                return "hampinch12"
            else:
                return "hamceeall"

        # 2. Handshape categories
        if num_main_ext >= 4:
            if d_index_middle > 0.30:
                return "hamfinger23spread"
            return "hamflathand"

        if num_main_ext == 2 and index_ext and middle_ext:
            return "hamfinger23"

        if num_main_ext == 1 and index_ext:
            return "hamfinger2"

        if num_main_ext == 0:
            if thumb_ext:
                return "hamthumboutmod"
            return "hamfist"

        if num_main_ext == 3 and index_ext and middle_ext and ring_ext:
            return "hamfinger2345"

        return "hamflathand" if thumb_ext else "hamfist"

    except Exception:
        return "hamflathand"



# =============================
# DUAL-HAND HANDSHAPE MODULE
# =============================

def run_handshape_module(video_path):
    """
    Extracts handshapes for BOTH Right and Left hands across the video.
    Uses MediaPipe Pose wrist anchoring to prevent left/right hand identity swaps
    during hand crossing/touching, and interpolates short detection dropouts.
    Returns dominant Right Hand, Left Hand, and two-handed detection metrics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    per_frame_right = []
    per_frame_left = []
    two_hands_count = 0
    total_frames = 0

    # Track consecutive "none" gaps for interpolation
    last_valid_r = "none"
    last_valid_l = "none"
    r_gap = 0
    l_gap = 0
    MAX_INTERP_GAP = 4  # Interpolate up to 4 consecutive dropped frames

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            pose_results = pose.process(img)

            # Extract Pose wrist positions for anchoring hand identity
            pose_right_wrist = None
            pose_left_wrist = None
            if pose_results.pose_landmarks:
                plm = pose_results.pose_landmarks.landmark
                # MediaPipe Pose: landmark 16 = right wrist, 15 = left wrist
                if plm[16].visibility > 0.3:
                    pose_right_wrist = np.array([plm[16].x, plm[16].y])
                if plm[15].visibility > 0.3:
                    pose_left_wrist = np.array([plm[15].x, plm[15].y])

            r_label = "none"
            l_label = "none"

            if results.multi_hand_landmarks and results.multi_handedness:
                if len(results.multi_hand_landmarks) >= 2:
                    two_hands_count += 1

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_label = handedness.classification[0].label  # 'Right' or 'Left'

                    # Pose-guided identity correction: if Pose wrists are available,
                    # assign each detected hand to the nearest Pose wrist
                    hand_wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
                    if pose_right_wrist is not None and pose_left_wrist is not None and len(results.multi_hand_landmarks) == 1:
                        dist_to_right = np.linalg.norm(hand_wrist - pose_right_wrist)
                        dist_to_left = np.linalg.norm(hand_wrist - pose_left_wrist)
                        if dist_to_right < dist_to_left:
                            h_type = "Right"
                        else:
                            h_type = "Left"
                    else:
                        h_type = mp_label

                    # Extract normalized 63-D features for Deep Neural inference
                    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
                    pts = pts - pts[0]
                    max_d = np.max(np.linalg.norm(pts, axis=1))
                    if max_d > 1e-6:
                        pts = pts / max_d
                    feats = pts.flatten()

                    nn_shape = neural_predict_handshape(feats)
                    geom_shape = classify_handshape(hand_landmarks.landmark)
                    shape = nn_shape if (nn_shape and nn_shape != "hamflathand") else geom_shape
                    if h_type == "Right":
                        r_label = shape
                    else:
                        l_label = shape

            # Short-gap interpolation: if a hand was recently detected and dropped
            # for a few frames (occlusion during touching), carry forward the last label
            if r_label != "none":
                last_valid_r = r_label
                r_gap = 0
            elif last_valid_r != "none" and r_gap < MAX_INTERP_GAP:
                r_label = last_valid_r
                r_gap += 1

            if l_label != "none":
                last_valid_l = l_label
                l_gap = 0
            elif last_valid_l != "none" and l_gap < MAX_INTERP_GAP:
                l_label = last_valid_l
                l_gap += 1

            per_frame_right.append(r_label)
            per_frame_left.append(l_label)

    cap.release()

    from collections import Counter
    valid_r = [x for x in per_frame_right if x != "none"]
    valid_l = [x for x in per_frame_left if x != "none"]

    # Sensitive two-handed sign detection (BSL features frequent interacting/touching dual hands)
    left_hand_active = len(valid_l) >= 4 or (len(valid_l) / max(1, total_frames)) >= 0.08
    simultaneous_hands = two_hands_count >= 3 or (two_hands_count / max(1, total_frames)) >= 0.08
    is_two_handed = bool(left_hand_active or simultaneous_hands)

    final_r = Counter(valid_r).most_common(1)[0][0] if valid_r else "hamflathand"
    final_l = Counter(valid_l).most_common(1)[0][0] if valid_l else "none"

    return {
        "per_frame": per_frame_right,
        "per_frame_right": per_frame_right,
        "per_frame_left": per_frame_left,
        "final": final_r,
        "final_right": final_r,
        "final_left": final_l,
        "is_two_handed": is_two_handed,
        "two_hands_ratio": two_hands_count / max(1, total_frames)
    }

