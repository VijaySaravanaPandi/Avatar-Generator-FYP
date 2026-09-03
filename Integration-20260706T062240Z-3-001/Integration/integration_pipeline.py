#!/usr/bin/env python
# coding: utf-8
"""
HamNoSys Integration Pipeline — Multi-Modal Neural & Dual-Hand Architecture
Runs all 10 modules on a sign language video and produces accurate, grammatically valid
HamNoSys notation for single-handed and two-handed 3D avatar animations.
"""

import os
import sys
import pickle

# Suppress C++ / MediaPipe / TensorFlow Lite stderr noise completely
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import warnings
warnings.filterwarnings("ignore")

try:
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_null_fd, 2)
    os.close(_null_fd)
    sys.stderr = sys.stdout
except Exception:
    pass

import numpy as np
import cv2
import mediapipe as mp

# Ensure the Integration directory is on the path for local imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from Handshape_Model import run_handshape_module
from ori_model2 import run_orientation_module
from upper_body_locations_video import run_upper_body_location_module
from Head_and_face_location import run_head_face_location_module
from hand_location_video_P import run_hand_location_module
from fing_locations_d import run_finger_location_module
from contact_types_e import run_contact_type_module
from Arm_and_Space_positions import run_arm_space_module
from movement1_prava import run_movement1_module
from Movement_2 import run_movement2_module


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def extract_labels(module_output):
    """
    Extract useful HamNoSys labels from module outputs, ensuring
    all space-separated tags are split into individual tokens.
    """
    if isinstance(module_output, dict):
        if "per_frame" in module_output:
            labels = module_output["per_frame"]
            cleaned = []
            for item in labels:
                if item is None: continue
                if isinstance(item, tuple):
                    for x in item:
                        if x is not None:
                            cleaned.extend(str(x).split())
                else:
                    cleaned.extend(str(item).split())
            return cleaned

    if isinstance(module_output, str):
        return module_output.split()

    if isinstance(module_output, list):
        cleaned = []
        for item in module_output:
            if item is not None:
                cleaned.extend(str(item).split())
        return cleaned

    return []


def combine_hamnosys(handshape, orientation, upper_body, head_face, hand_location,
                    finger_location, contact, arm_space, movement1, movement2, video_path=None):
    """
    Combine module outputs into a strict, grammatically valid SiGML HamNoSys sequence,
    supporting both Single-Handed and Dual-Handed signs.
    """

    def pick_token(module_output, key="final"):
        if isinstance(module_output, dict):
            final = module_output.get(key)
            if final and final not in ["none", "no-contact"]:
                return str(final)
            per = module_output.get("per_frame", [])
            flat = []
            for item in per:
                if isinstance(item, tuple):
                    flat.extend([str(x) for x in item if x and x not in ["none", "no-contact"]])
                elif item and item not in ["none", "no-contact"]:
                    flat.append(str(item))
            if not flat:
                return None
            from collections import Counter
            most_common, _ = Counter(flat).most_common(1)[0]
            return most_common
        if isinstance(module_output, str) and module_output not in ["none", "no-contact"]:
            return module_output
        if isinstance(module_output, list):
            return next((str(x) for x in module_output if x and x not in ["none", "no-contact"]), None)
        return None

    def format_ori(ori_tuple):
        seq = []
        if isinstance(ori_tuple, tuple):
            for part in ori_tuple:
                part = str(part)
                if part and part not in ["none", "no-contact"] and (part.startswith("hamextfinger") or part.startswith("hampalm")):
                    seq.append(part)
        elif isinstance(ori_tuple, str) and (ori_tuple.startswith("hamextfinger") or ori_tuple.startswith("hampalm")):
            seq.append(ori_tuple)
        return seq

    sequence = []

    BASE_HANDSHAPES = {
        "hamflathand", "hamfist", "hamfinger2", "hamfinger23", "hamfinger2345",
        "hamfinger23spread", "hamceeall", "hamcee12", "hampinchall", "hampinch12"
    }

    # 1. Detect if sign is Two-Handed
    is_two_handed = False
    r_handshape = "hamflathand"
    l_handshape = None

    if isinstance(handshape, dict):
        is_two_handed = handshape.get("is_two_handed", False)
        ratio = handshape.get("two_hands_ratio", 0.0)

        r_handshape = pick_token(handshape, "final_right") or pick_token(handshape, "final") or "hamflathand"
        l_handshape = pick_token(handshape, "final_left")

        # Two-handed detection: if left hand was detected in the video or dual arms active
        if not is_two_handed and l_handshape and l_handshape != "none":
            is_two_handed = True

        # Additional fallback: if arm_space has hamdoublebent or wide arm posture
        if not is_two_handed and isinstance(arm_space, dict):
            arm_labels = arm_space.get("per_frame", [])
            doublebent_count = sum(1 for x in arm_labels if x in ["hamdoublebent", "hamneutralspace"])
            if doublebent_count / max(1, len(arm_labels)) >= 0.25:
                is_two_handed = True

        # Check dual orientation
        if not is_two_handed and isinstance(orientation, dict) and orientation.get("final_left"):
            is_two_handed = True

    # Clean Handshapes
    def clean_hs(token):
        if not token or token == "none":
            return "hamflathand"
        if token in BASE_HANDSHAPES:
            return token
        if token.endswith("mod") or "thumb" in token:
            return f"hamflathand {token}"
        if token.startswith("ham"):
            return token
        return "hamflathand"

    clean_r_hs = clean_hs(r_handshape).split()
    clean_l_hs = clean_hs(l_handshape).split() if l_handshape and l_handshape != "none" else None

    # Initial Orientation
    r_ori = orientation.get("final_right") if isinstance(orientation, dict) else None
    l_ori = orientation.get("final_left") if isinstance(orientation, dict) else None

    r_ori_seq = format_ori(r_ori) if r_ori else ["hamextfingeru", "hampalmd"]
    l_ori_seq = format_ori(l_ori) if l_ori else ["hamextfingero", "hampalmu"]

    # 2. Hand Configuration Assembly
    if is_two_handed:
        effective_l_hs = clean_l_hs if clean_l_hs else clean_r_hs
        # Symmetrical dual hand check: identical handshape and similar vertical orientation
        r_finger_dir = next((x for x in r_ori_seq if x.startswith("hamextfinger")), "hamextfingeru")
        l_finger_dir = next((x for x in l_ori_seq if x.startswith("hamextfinger")), "hamextfingeru")

        if clean_r_hs == effective_l_hs or r_finger_dir == l_finger_dir:
            sequence.append("hamsymmlr")
            sequence.extend(clean_r_hs)
            sequence.extend(r_ori_seq)
        else:
            # Dual distinct hands (Right active hand + Left base hand)
            sequence.append("hamparbegin")
            sequence.extend(clean_r_hs)
            sequence.extend(r_ori_seq)
            sequence.append("hamplus")
            sequence.extend(effective_l_hs)
            sequence.extend(l_ori_seq)
            sequence.append("hamparend")
    else:
        # Single-handed sign
        sequence.extend(clean_r_hs)
        sequence.extend(r_ori_seq)


    # 3. Location Component — prefer body location (upper_body) over face (head_face)
    FACE_ONLY_TOKENS = {"hamforehead", "hameyes", "hamnose", "hamlips", "hamchin", "hamear", "hamcheek"}
    BODY_TOKENS = {
        "hamchest", "hamshoulders", "hamshouldertop", "hamwristback",
        "hamupperarm", "hamstomach", "hamneck", "hamneutralspace",
        "hambelowstomach"
    }

    body_loc_raw = pick_token(upper_body)
    face_loc_raw = pick_token(head_face)

    if face_loc_raw and face_loc_raw in FACE_ONLY_TOKENS:
        if not body_loc_raw or body_loc_raw not in BODY_TOKENS:
            body_loc = face_loc_raw
        else:
            body_loc = body_loc_raw
    elif body_loc_raw and body_loc_raw != "none":
        body_loc = body_loc_raw
    else:
        body_loc = face_loc_raw

    if body_loc in {"hambelowstomach", "hamwristback"} and is_two_handed:
        body_loc = "hamshoulders"

    if body_loc and body_loc != "none":
        sequence.append(body_loc)
    else:
        sequence.append("hamshoulders" if is_two_handed else "hamchest")

    # For two-handed symmetric signs, specify wide open space position
    if is_two_handed and "hamsymmlr" in sequence and "hamlrbeside" not in sequence:
        sequence.append("hamlrbeside")

    # 4. Movement Component
    movement_token = pick_token(movement1) or pick_token(movement2)
    added_movement = False
    if movement_token and movement_token != "none" and movement_token != "hamnomotion":
        mov_tokens = str(movement_token).split()
        for token in mov_tokens:
            if is_two_handed and "hamsymmlr" in sequence and token in ["hammovel", "hamclose"]:
                sequence.append("hammovei")
                added_movement = True
                break
            elif token.startswith("hammove") or token.startswith("hamcircle") or token.startswith("hamarc"):
                sequence.append(token)
                added_movement = True
                break
            elif token in ["hamzigzag", "hamwavy"]:
                sequence.append("hammovei" if is_two_handed else "hammovel")
                sequence.append(token)
                added_movement = True
                break
            elif token in ["hamnodding", "hamswinging", "hamtwisting"]:
                sequence.append(token)
                added_movement = True
                break

    if not added_movement:
        sequence.append("hammovei" if is_two_handed else "hammoved")

    return " ".join(sequence)


def extract_motion_sequence(video_path, max_segments=4):
    """Return the dominant hand-path phases as HamNoSys movement tokens.

    A single final movement label loses the action in a video (for example a
    wave is reduced to just ``hammoveu``).  This lightweight tracker keeps the
    temporal path and converts its consecutive directional phases into a
    sequence CWASA can play.  It is intentionally conservative: when no hand
    track is available the caller falls back to the existing one-sign output.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    points = []
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Sampling keeps long uploads responsive while retaining movement.
            if frame_no % 3 == 0:
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    # The most visible hand is the best action stream for a
                    # single avatar; average both wrists for symmetric signs.
                    wrists = [(h.landmark[0].x, h.landmark[0].y) for h in result.multi_hand_landmarks]
                    points.append((frame_no, np.mean(wrists, axis=0)))
            frame_no += 1
    cap.release()
    if len(points) < 6:
        return []

    positions = np.asarray([p[1] for p in points], dtype=np.float32)
    # Suppress detector jitter before calculating directional phases.
    kernel = np.ones(3, dtype=np.float32) / 3.0
    smooth = np.column_stack([np.convolve(positions[:, dim], kernel, mode="same") for dim in range(2)])
    deltas = np.diff(smooth, axis=0)
    magnitudes = np.linalg.norm(deltas, axis=1)
    active = deltas[magnitudes >= max(np.percentile(magnitudes, 55), 0.006)]
    if len(active) < 3:
        return []

    # Split the active path into chronological phases and deduplicate adjacent
    # equal directions. Screen y increases downwards, hence the inverted axis.
    chunk_count = min(max_segments, max(2, len(active) // 5))
    tokens = []
    for phase in np.array_split(active, chunk_count):
        dx, dy = phase.sum(axis=0)
        if abs(dx) > abs(dy) * 1.25:
            token = "hammover" if dx > 0 else "hammovel"
        elif abs(dy) > abs(dx) * 1.25:
            token = "hammoved" if dy > 0 else "hammoveu"
        elif dx >= 0 and dy < 0:
            token = "hammoveur"
        elif dx < 0 and dy < 0:
            token = "hammoveul"
        elif dx >= 0:
            token = "hammovedr"
        else:
            token = "hammovedl"
        if not tokens or token != tokens[-1]:
            tokens.append(token)
    return tokens


def build_avatar_action_sequence(base_hamnosys, video_path):
    """Replace the final movement with tracked action phases for CWASA."""
    motion_tokens = extract_motion_sequence(video_path)
    if len(motion_tokens) < 2:
        return [base_hamnosys]

    tokens = base_hamnosys.split()
    movement_index = next((i for i, token in enumerate(tokens) if token.startswith("hammove") or token.startswith("hamcircle") or token.startswith("hamarc")), None)
    static_tokens = tokens[:movement_index] if movement_index is not None else tokens
    return [" ".join(static_tokens + [movement]) for movement in motion_tokens]



from bsl_gloss_matcher import predict_gloss_details
from bsl_lexicon import get_hamnosys_for_gloss, get_catalogue_hamnosys
from annotated_sign_matcher import predict_annotated_sign

# A nearest-neighbour model always yields a label.  Reject distant matches so
# an unknown upload does not animate as an unrelated dictionary sign.
MIN_GLOSS_CONFIDENCE = float(os.environ.get("BSL_GLOSS_MIN_CONFIDENCE", "0.45"))

# =====================================================
# MAIN PIPELINE
# =====================================================

def generate_hamnosys(video_path, original_filename=None):
    """
    Run ALL 10 modules on a video and combine results into an accurate HamNoSys string,
    with BSL Lexicon knowledge integration.
    """

    print("\nRunning modules...")

    # A trained model is used only when it has passed its held-out evaluation
    # and is based on human HamNoSys annotations.  This is the sole automatic
    # route for a user recording to select an authored avatar sequence.
    annotated = predict_annotated_sign(video_path)
    catalogue_name = original_filename or os.path.basename(video_path)
    lex_code, known_gloss = get_catalogue_hamnosys(catalogue_name)
    if annotated:
        gloss = annotated["gloss"]
        matcher_confidence = annotated["confidence"]
        lex_code = annotated["hamnosys"]
        match = {
            "nearest_gloss": gloss,
            "candidates": [{"gloss": gloss, "score": matcher_confidence, "distance": 0.0}],
            "reason": annotated["reason"],
        }
        notation_source = "human_annotated_sign_model"
        print(f"  [Annotated Sign Model] Predicted '{gloss}' ({matcher_confidence:.0%} similarity).")
    elif lex_code:
        gloss = known_gloss
        matcher_confidence = 1.0
        match = {
            "nearest_gloss": known_gloss,
            "candidates": [{"gloss": known_gloss, "score": 1.0, "distance": 0.0}],
            "reason": "exact_bsldict_catalogue_match",
        }
        notation_source = "bsldict_catalogue"
        print(f"  [BSLDict Catalogue] Exact notation loaded for '{known_gloss}'.")
    else:
        # A visual nearest-neighbour model remains a fallback for recordings
        # outside the catalogue.  Its calibrated acceptance check prevents a
        # low-quality nearest hit from animating an unrelated sign.
        match = predict_gloss_details(video_path)
        gloss = match["gloss"]
        matcher_confidence = match["confidence"]
        lex_code = get_hamnosys_for_gloss(gloss) if gloss and matcher_confidence >= MIN_GLOSS_CONFIDENCE else None
        notation_source = "bsl_gloss_matcher" if lex_code else "visual_modules"
        if lex_code:
            print(f"  [BSLDict Matcher] Predicted gloss: '{gloss}' ({matcher_confidence:.0%} similarity)")
        elif match.get("nearest_gloss"):
            print(
                f"  [BSLDict Matcher] Nearest gloss '{match['nearest_gloss']}' rejected "
                f"({matcher_confidence:.0%}; {match.get('reason', 'low confidence')})."
            )

    # An accepted dictionary notation is already an authored sign sequence.
    # Running ten heuristic feature modules afterwards cannot improve it and
    # makes the web response unnecessarily slow.  More importantly, those
    # unrelated estimates previously obscured whether the avatar was using a
    # trusted catalogue mapping.
    if lex_code:
        empty_modules = {
            "handshape": {}, "orientation": {}, "upper_body": {},
            "head_face": {}, "hand_location": {}, "finger_location": {},
            "contact": {}, "arm_space": {}, "movement1": {}, "movement2": {},
        }
        empty_modules.update({
            "predicted_gloss": gloss,
            "gloss_confidence": matcher_confidence,
            "nearest_gloss": match.get("nearest_gloss"),
            "gloss_candidates": match.get("candidates", []),
            "gloss_match_reason": match.get("reason"),
            "notation_source": notation_source,
            "avatar_sequence": [lex_code],
        })
        print("  [Avatar] Using a quality-gated annotated sign sequence.")
        return lex_code, empty_modules

    handshape       = run_handshape_module(video_path)
    print("  [1/10] Handshape (Neural Engine) - done")

    orientation     = run_orientation_module(video_path)
    print("  [2/10] Orientation (Dual Hands)  - done")

    upper_body      = run_upper_body_location_module(video_path)
    print("  [3/10] Upper Body Location       - done")

    head_face       = run_head_face_location_module(video_path)
    print("  [4/10] Head & Face Location      - done")

    hand_location   = run_hand_location_module(video_path)
    print("  [5/10] Hand Location             - done")

    finger_location = run_finger_location_module(video_path)
    print("  [6/10] Finger Location           - done")

    contact         = run_contact_type_module(video_path)
    print("  [7/10] Contact Type              - done")

    arm_space       = run_arm_space_module(video_path)
    print("  [8/10] Arm & Space               - done")

    movement1       = run_movement1_module(video_path)
    print("  [9/10] Movement 1                - done")

    movement2       = run_movement2_module(video_path)
    print("  [10/10] Movement 2               - done")

    print("\n===== MODULE OUTPUTS =====")
    print("Handshape       :", handshape)
    print("Orientation     :", orientation)
    print("Arm & Space     :", arm_space)
    print("Upper Body      :", upper_body)
    print("Head & Face     :", head_face)
    print("Hand Location   :", hand_location)
    print("Finger Location :", finger_location)
    print("Contact Type    :", contact)
    print("Movement 1      :", movement1)
    print("Movement 2      :", movement2)

    if lex_code:
        hamnosys_code = lex_code
    else:
        hamnosys_code = combine_hamnosys(
            handshape,
            orientation,
            upper_body,
            head_face,
            hand_location,
            finger_location,
            contact,
            arm_space,
            movement1,
            movement2,
            video_path=video_path
        )

    print("\n========== FINAL HAMNOSYS ==========")
    print(hamnosys_code)

    # Keep the action trajectory instead of feeding CWASA one collapsed pose.
    # Dictionary signs are already authored motion sequences, so do not replace
    # their phonetic motion with a coarse video tracker.
    avatar_sequence = [hamnosys_code] if lex_code else build_avatar_action_sequence(hamnosys_code, video_path)

    return hamnosys_code, {
        "handshape": handshape,
        "orientation": orientation,
        "upper_body": upper_body,
        "head_face": head_face,
        "hand_location": hand_location,
        "finger_location": finger_location,
        "contact": contact,
        "arm_space": arm_space,
        "movement1": movement1,
        "movement2": movement2,
        "predicted_gloss": gloss,
        "gloss_confidence": matcher_confidence,
        "nearest_gloss": match.get("nearest_gloss"),
        "gloss_candidates": match.get("candidates", []),
        "gloss_match_reason": match.get("reason"),
        "notation_source": notation_source,
        "avatar_sequence": avatar_sequence,
    }



# =====================================================
# ANNOTATED OUTPUT VIDEO
# =====================================================

def safe(x):
    if x is None:
        return "none"
    return str(x)


def get_frames(output, total_frames):
    if isinstance(output, dict):
        arr = output.get("per_frame", [])
        if len(arr) > 0:
            return arr
    if isinstance(output, str):
        return [output] * total_frames
    if isinstance(output, list):
        return output
    return ["none"] * total_frames


def create_annotated_video(video_path, output_path, modules):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    handshape_frames    = get_frames(modules["handshape"], total_frames)
    orientation_frames  = get_frames(modules["orientation"], total_frames)
    upper_frames        = get_frames(modules["upper_body"], total_frames)
    head_frames         = get_frames(modules["head_face"], total_frames)
    hand_frames         = get_frames(modules["hand_location"], total_frames)
    finger_frames       = get_frames(modules["finger_location"], total_frames)
    contact_frames      = get_frames(modules["contact"], total_frames)
    arm_frames          = get_frames(modules["arm_space"], total_frames)
    movement1_frames    = get_frames(modules["movement1"], total_frames)
    movement2_frames    = get_frames(modules["movement2"], total_frames)

    frame_id = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        idx = min(frame_id, len(handshape_frames) - 1) if handshape_frames else 0

        handshape_label = safe(handshape_frames[idx]) if handshape_frames else "none"
        orientation_label = safe(orientation_frames[min(frame_id, len(orientation_frames) - 1)]) if orientation_frames else "none"

        location_label = " ".join([
            safe(upper_frames[min(frame_id, len(upper_frames) - 1)]) if upper_frames else "none",
            safe(head_frames[min(frame_id, len(head_frames) - 1)]) if head_frames else "none",
            safe(hand_frames[min(frame_id, len(hand_frames) - 1)]) if hand_frames else "none",
            safe(finger_frames[min(frame_id, len(finger_frames) - 1)]) if finger_frames else "none",
            safe(contact_frames[min(frame_id, len(contact_frames) - 1)]) if contact_frames else "none",
            safe(arm_frames[min(frame_id, len(arm_frames) - 1)]) if arm_frames else "none",
        ])

        movement_label = " ".join([
            safe(movement1_frames[min(frame_id, len(movement1_frames) - 1)]) if movement1_frames else "none",
            safe(movement2_frames[min(frame_id, len(movement2_frames) - 1)]) if movement2_frames else "none",
        ])

        cv2.putText(frame, f"Handshape: {handshape_label}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Orientation: {orientation_label}",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Location: {location_label}",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Movement: {movement_label}",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()


def process_video(video_path, output_video=None, original_filename=None, annotate=True):
    if output_video is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_video = os.path.join(_SCRIPT_DIR, f"{base}_output.mp4")

    hamnosys_code, modules = generate_hamnosys(video_path, original_filename=original_filename)
    if annotate:
        create_annotated_video(video_path, output_video, modules)

    return {
        "hamnosys": hamnosys_code,
        "output_video": output_video if annotate else None,
        **modules,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HamNoSys Integration Pipeline")
    parser.add_argument("video", nargs="?", default="Prompt_1.mp4",
                        help="Path to input video (default: Prompt_1.mp4)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output video path (default: <input>_output.mp4)")
    args = parser.parse_args()

    result = process_video(args.video, args.output)
    print("\n✓ Done!")
    print(f"  HamNoSys: {result['hamnosys']}")
    print(f"  Video:    {result['output_video']}")
