#!/usr/bin/env python
# coding: utf-8
"""
HamNoSys Integration Pipeline — Local Version
Runs all 10 modules on a sign language video and produces a HamNoSys code string.
"""

import os
import sys
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
                    finger_location, contact, arm_space, movement1, movement2):
    """
    Combine module outputs into a strict, grammatically valid SiGML HamNoSys sequence.
    
    Formal HamNoSys CFG Rules:
    Sign ::= HandshapeStructure InitialOrientation BodyLocation [Contact] [Movement] [StateTransition]
    - InitialOrientation ::= ExtendedFinger PalmOrientation
    - BodyLocation ::= [HandPart/FingerPart] BaseLocation
    - StateTransition ::= hamreplace FinalExtendedFinger FinalPalmOrientation
    """

    def pick_token(module_output):
        if isinstance(module_output, dict):
            final = module_output.get("final")
            if final and final not in ["none", "no-contact"]:
                return final
            per = module_output.get("per_frame", [])
            flat = []
            for item in per:
                if isinstance(item, tuple):
                    flat.extend([x for x in item if x and x not in ["none", "no-contact"]])
                elif item and item not in ["none", "no-contact"]:
                    flat.append(item)
            if not flat:
                return None
            from collections import Counter
            most_common, _ = Counter(flat).most_common(1)[0]
            return most_common
        if isinstance(module_output, str) and module_output not in ["none", "no-contact"]:
            return module_output
        if isinstance(module_output, list):
            return next((x for x in module_output if x and x not in ["none", "no-contact"]), None)
        return None

    def extract_state_transitions(module_output):
        if not isinstance(module_output, dict) or "per_frame" not in module_output:
            tok = pick_token(module_output)
            return tok, tok
            
        frames = module_output["per_frame"]
        valid_frames = []
        for item in frames:
            if isinstance(item, tuple):
                valid_frames.append(tuple(x for x in item if x and x not in ["none", "no-contact"]))
            elif item and item not in ["none", "no-contact"]:
                valid_frames.append((item,))
                
        if len(valid_frames) < 5:
            tok = pick_token(module_output)
            return tok, tok
            
        chunk_size = max(1, int(len(valid_frames) * 0.3))
        start_chunk = valid_frames[:chunk_size]
        end_chunk = valid_frames[-chunk_size:]
        
        from collections import Counter
        start_tok = Counter(start_chunk).most_common(1)[0][0] if start_chunk else None
        end_tok = Counter(end_chunk).most_common(1)[0][0] if end_chunk else None
        
        if start_tok is None: start_tok = pick_token(module_output)
        if end_tok is None: end_tok = pick_token(module_output)
        
        return start_tok, end_tok

    def format_ori(ori_tuple):
        seq = []
        if isinstance(ori_tuple, tuple):
            for part in ori_tuple:
                if part and part not in ["none", "no-contact"] and (part.startswith("hamextfinger") or part.startswith("hampalm")):
                    seq.append(part)
        elif isinstance(ori_tuple, str) and (ori_tuple.startswith("hamextfinger") or ori_tuple.startswith("hampalm")):
            seq.append(ori_tuple)
        return seq

    sequence = []

    # 1. Handshape & Finger/Hand Part Component (Required Base)
    handshape_token = pick_token(handshape) or "hamflathand"
    sequence.append(handshape_token)

    hand_part = pick_token(finger_location) or pick_token(hand_location)
    if hand_part and hand_part not in ["none", "no-contact"] and hand_part != handshape_token:
        sequence.append(hand_part)

    # 2. Initial Orientation Component (Extended Finger + Palm)
    ori_start, ori_end = extract_state_transitions(orientation)
    ori_start_seq = format_ori(ori_start)
    ori_end_seq = format_ori(ori_end)

    if ori_start_seq:
        sequence.extend(ori_start_seq)
    else:
        # CFG Fallback default if model misses orientation
        sequence.extend(["hamextfingeru", "hampalmd"])

    # 3. Location Component (Body/Head/Arm Location)
    body_loc = pick_token(head_face) or pick_token(upper_body)
    if body_loc and body_loc != "none":
        sequence.append(body_loc)
    else:
        # Fallback location (chest level) if no location predicted
        sequence.append("hamshoulders")

    # 4. Contact Component (Grammatically Valid placement)
    contact_token = pick_token(contact) or pick_token(arm_space)
    if contact_token and contact_token not in ["none", "no-contact"]:
        if contact_token.startswith("hamtouch") or contact_token.startswith("hamclose") or contact_token.startswith("hambrushing"):
            sequence.append(contact_token)

    # 5. Movement Component (Base movements only to ensure SiGML parsing success)
    movement_token = pick_token(movement1) or pick_token(movement2)
    if movement_token and movement_token != "none":
        mov_tokens = str(movement_token).split()
        for token in mov_tokens:
            if token.startswith("hammove") or token.startswith("hamcircle") or token.startswith("hamarc"):
                sequence.append(token)
                break

    # 6. Dynamic State Transition (Orientation Replace)
    if ori_start_seq != ori_end_seq and ori_end_seq:
        sequence.append("hamreplace")
        sequence.extend(ori_end_seq)

    return " ".join(sequence)


# =====================================================
# MAIN PIPELINE
# =====================================================

def generate_hamnosys(video_path):
    """
    Run ALL 10 modules on a video and combine results into a HamNoSys string.
    """

    print("\nRunning modules...")

    handshape       = run_handshape_module(video_path)
    print("  [1/10] Handshape - done")

    orientation     = run_orientation_module(video_path)
    print("  [2/10] Orientation - done")

    upper_body      = run_upper_body_location_module(video_path)
    print("  [3/10] Upper Body - done")

    head_face       = run_head_face_location_module(video_path)
    print("  [4/10] Head & Face - done")

    hand_location   = run_hand_location_module(video_path)
    print("  [5/10] Hand Location - done")

    finger_location = run_finger_location_module(video_path)
    print("  [6/10] Finger Location - done")

    contact         = run_contact_type_module(video_path)
    print("  [7/10] Contact Type - done")

    arm_space       = run_arm_space_module(video_path)
    print("  [8/10] Arm & Space - done")

    movement1       = run_movement1_module(video_path)
    print("  [9/10] Movement 1 - done")

    movement2       = run_movement2_module(video_path)
    print("  [10/10] Movement 2 - done")

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
        movement2
    )

    print("\n========== FINAL HAMNOSYS ==========")
    print(hamnosys_code)

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
    """
    Create output video with per-frame labels overlaid.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Cannot open video for annotation")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        cap.release()
        return

    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract per-frame data
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

        # Safe index access
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

    print(f"Annotated video saved: {output_path}")


# =====================================================
# PROCESS VIDEO (full pipeline + annotated output)
# =====================================================

def process_video(video_path, output_video=None):
    """
    Full pipeline: generate HamNoSys + annotated video.
    """

    if output_video is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_video = os.path.join(_SCRIPT_DIR, f"{base}_output.mp4")

    hamnosys_code, modules = generate_hamnosys(video_path)

    create_annotated_video(video_path, output_video, modules)

    return {
        "hamnosys": hamnosys_code,
        "output_video": output_video
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
