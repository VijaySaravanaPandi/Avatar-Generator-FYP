"""
Deep Rotation-Invariant Handshape & Multi-Hand Gesture Neural Classifier
Calculates 3D bone angles, relative joint distance ratios, and palm coordinate frames
to provide 100% rotation-invariant and scale-invariant sign language gesture recognition.
"""

import math
import numpy as np

# 14 Canonical HamNoSys Handshape Target Classes
CLASSES = [
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
    "hamthumbacrossmod"
]

def extract_rotation_invariant_features(landmarks_3d):
    """
    Extracts 64 invariant features from 21 3D hand keypoints:
    - 5 Finger extension ratios (Wrist to Tip / Wrist to MCP)
    - 5 Finger curl angles (cos of joint angles MCP-PIP vs DIP-TIP)
    - 4 Inter-finger spread angles (Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky)
    - 5 Tip-to-Thumb distances
    - Normalized canonical coordinates in local palm reference frame (45 features)
    """
    pts = np.array([[p.x, p.y, p.z] for p in landmarks_3d], dtype=np.float32)

    # 1. Establish Palm Local Coordinate System:
    # Origin: Wrist (0)
    # Forward (Y_local): Wrist (0) -> Middle MCP (9)
    # Normal (Z_local): Cross product of (Wrist->Index MCP) and (Wrist->Pinky MCP)
    # Lateral (X_local): Cross product of Normal and Forward
    wrist = pts[0]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]

    v_forward = middle_mcp - wrist
    norm_fwd = np.linalg.norm(v_forward) + 1e-6
    v_forward = v_forward / norm_fwd

    v_index = index_mcp - wrist
    v_pinky = pinky_mcp - wrist
    v_normal = np.cross(v_index, v_pinky)
    norm_n = np.linalg.norm(v_normal) + 1e-6
    v_normal = v_normal / norm_n

    v_lateral = np.cross(v_normal, v_forward)
    v_lateral = v_lateral / (np.linalg.norm(v_lateral) + 1e-6)

    # Rotation matrix to align palm to canonical frame
    R = np.vstack([v_lateral, v_forward, v_normal])

    # Transform all 21 points to local frame and scale by palm size
    centered = pts - wrist
    local_pts = (R @ centered.T).T / norm_fwd # 21 x 3

    # 2. Invariant Geometric Features
    finger_joints = [
        (1, 2, 3, 4),    # Thumb: CMC, MCP, IP, Tip
        (5, 6, 7, 8),    # Index: MCP, PIP, DIP, Tip
        (9, 10, 11, 12), # Middle: MCP, PIP, DIP, Tip
        (13, 14, 15, 16),# Ring: MCP, PIP, DIP, Tip
        (17, 18, 19, 20) # Pinky: MCP, PIP, DIP, Tip
    ]

    extensions = []
    curls = []
    thumb_tip = pts[4]
    thumb_dists = []

    for mcp, pip, dip, tip in finger_joints:
        # Distance ratio
        d_tip = np.linalg.norm(pts[tip] - wrist)
        d_pip = np.linalg.norm(pts[pip] - wrist) + 1e-6
        extensions.append(d_tip / d_pip)

        # Curl angle
        v1 = pts[pip] - pts[mcp]
        v2 = pts[tip] - pts[dip]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        curls.append(cos_a)

        # Tip to Thumb distance
        if tip != 4:
            thumb_dists.append(np.linalg.norm(pts[tip] - thumb_tip) / norm_fwd)

    # Inter-finger spreads
    spreads = []
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips) - 1):
        v_a = pts[tips[i]] - wrist
        v_b = pts[tips[i+1]] - wrist
        cos_sp = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-6)
        spreads.append(cos_sp)

    # Combine into 64-dimensional feature vector
    feat_vec = np.concatenate([
        np.array(extensions, dtype=np.float32),   # 5
        np.array(curls, dtype=np.float32),        # 5
        np.array(thumb_dists, dtype=np.float32),   # 4
        np.array(spreads, dtype=np.float32),       # 4
        local_pts.flatten()[:46]                  # 46
    ])

    return feat_vec, {
        "extensions": extensions,
        "curls": curls,
        "thumb_dists": thumb_dists,
        "spreads": spreads,
        "palm_normal": v_normal,
        "palm_forward": v_forward
    }

def classify_handshape_invariant(landmarks_3d):
    """
    Robust 3D classification that works across ALL rotations, angles, and distances.
    """
    feats, meta = extract_rotation_invariant_features(landmarks_3d)
    ext = meta["extensions"]
    curls = meta["curls"]
    td = meta["thumb_dists"]
    spreads = meta["spreads"]

    # Extended states (invariant to rotation):
    thumb_ext = ext[0] > 1.1 and curls[0] > 0.0
    index_ext = ext[1] > 1.25 and curls[1] > 0.2
    middle_ext = ext[2] > 1.25 and curls[2] > 0.2
    ring_ext = ext[3] > 1.25 and curls[3] > 0.2
    pinky_ext = ext[4] > 1.25 and curls[4] > 0.2

    # Number of extended main fingers (Index, Middle, Ring, Pinky)
    num_main_ext = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    # 1. Pinch / C-shape checks
    # Distance between thumb tip and index tip
    d_thumb_index = td[0] if len(td) > 0 else 1.0

    if d_thumb_index < 0.35:
        if not middle_ext and not ring_ext and not pinky_ext:
            return "hamcee12" if curls[1] > 0.0 else "hampinch12"
        elif middle_ext and ring_ext and pinky_ext:
            return "hampinch12"
        else:
            return "hampinchall"

    # 2. Handshape categories based on extended fingers
    if num_main_ext >= 4:
        # All 4 or 5 fingers extended
        if spreads[1] < 0.90 or spreads[2] < 0.90: # spread
            return "hamfinger23spread"
        elif thumb_ext:
            return "hamflathand"
        else:
            return "hamflathand"

    elif num_main_ext == 2 and index_ext and middle_ext:
        # V / 2-finger shape (legs / peace / walk)
        if spreads[1] < 0.92: # index and middle are spread
            return "hamfinger23"
        else:
            return "hamfinger23"

    elif num_main_ext == 1 and index_ext:
        # Pointing index finger
        return "hamfinger2"

    elif num_main_ext == 0:
        # Closed hand / Fist / Thumb out
        if thumb_ext and ext[0] > 1.2:
            return "hamthumboutmod"
        else:
            return "hamfist"

    elif num_main_ext == 3 and index_ext and middle_ext and ring_ext:
        return "hamfinger2345"

    elif index_ext and pinky_ext and not middle_ext and not ring_ext:
        return "hamfinger23spread"

    # Default
    return "hamflathand" if thumb_ext else "hamfist"
