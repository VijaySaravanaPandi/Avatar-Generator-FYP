"""Sequence-Level & Slot-Based HamNoSys Accuracy Evaluation Engine.

Evaluates predicted HamNoSys token sequences against ground-truth annotations
using Token Levenshtein Edit Distance, Word Error Rate (WER), and per-slot
accuracy (Handshape, Orientation, Location, Movement, Dual-Hand).

Usage:
    py -3.12 evaluate_accuracy.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Integration-20260706T062240Z-3-001", "Integration"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "webapp"))

from integration_pipeline import generate_hamnosys


# ------------- Token Slot Classification -------------

HANDSHAPE_TOKENS = {
    "hamfist", "hamflathand", "hamfinger2", "hamfinger23", "hamfinger2345",
    "hamfinger23spread", "hampinch12", "hampinchall", "hamcee12", "hamceeall",
    "hamdoublebent", "hamfinger2spread", "hamfinger23456",
}
THUMB_MODIFIER_TOKENS = {
    "hamthumboutmod", "hamthumbopenmod", "hamthumbacrossmod",
}
FINGER_MODIFIER_TOKENS = {
    "hamfingerstraightmod", "hamfingerbendmod", "hamfingerhookmod",
}
ORIENTATION_TOKENS = {
    "hamextfingeru", "hamextfingerd", "hamextfingerl", "hamextfingerr",
    "hamextfingero", "hamextfingeri", "hamextfingerur", "hamextfingerul",
    "hamextfingerdr", "hamextfingerdl", "hamextfingeruo", "hamextfingerui",
    "hamextfingerdo", "hamextfingerdi", "hamextfingeror", "hamextfingerol",
    "hamextfingerir", "hamextfingeril",
    "hampalmu", "hampalmd", "hampalml", "hampalmr",
    "hampalmul", "hampalmdl", "hampalmur", "hampalmdr",
}
LOCATION_TOKENS = {
    "hamhead", "hamheadtop", "hamforehead", "hameyes", "hamnose", "hamear",
    "hamcheek", "hamlips", "hamchin", "hamneck", "hamshoulders",
    "hamshouldertop", "hamchest", "hamstomach", "hambelowstomach",
    "hamupperarm", "hamelbow", "hamwrist",
    "hamclose", "hamtouch", "haminterlock", "hamcross",
    "hambetween", "hamlrat", "hamlrbeside",
}
MOVEMENT_TOKENS = {
    "hammoveu", "hammoved", "hammovel", "hammover",
    "hammovei", "hammoveo", "hammoveui", "hammoveuo",
    "hammovedi", "hammovedo", "hammoveur", "hammoveul",
    "hammovedr", "hammovedl",
    "hamcircleo", "hamcirclei", "hamarcu", "hamarcd",
    "hamwavy", "hamzigzag", "hamnodding", "hamswinging", "hamtwisting",
    "hamfingerplay",
    "hamnomotion",
    "hamrepeatfromstart", "hamrepeatcontinue",
}
DUALHAND_TOKENS = {
    "hamsymmlr", "hamsymmrl", "hamparbegin", "hamplus", "hamparend",
    "hamreplace",
}


def classify_slot(token: str) -> str:
    if token in HANDSHAPE_TOKENS:
        return "handshape"
    if token in THUMB_MODIFIER_TOKENS or token in FINGER_MODIFIER_TOKENS:
        return "handshape"
    if token in ORIENTATION_TOKENS:
        return "orientation"
    if token in LOCATION_TOKENS:
        return "location"
    if token in MOVEMENT_TOKENS:
        return "movement"
    if token in DUALHAND_TOKENS:
        return "dual_hand"
    return "other"


def slot_partition(tokens: list[str]) -> dict[str, set[str]]:
    slots: dict[str, set[str]] = {
        "handshape": set(), "orientation": set(), "location": set(),
        "movement": set(), "dual_hand": set(), "other": set(),
    }
    for t in tokens:
        slots[classify_slot(t)].add(t)
    return slots


# ------------- Levenshtein Edit Distance -------------

def levenshtein_distance(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def word_error_rate(gt_tokens: list[str], pred_tokens: list[str]) -> float:
    if not gt_tokens:
        return 0.0 if not pred_tokens else 1.0
    return levenshtein_distance(gt_tokens, pred_tokens) / len(gt_tokens)


def token_accuracy(gt_tokens: list[str], pred_tokens: list[str]) -> float:
    return max(0.0, 1.0 - word_error_rate(gt_tokens, pred_tokens))


# ------------- Slot-Level Accuracy -------------

def slot_accuracy(gt_tokens: list[str], pred_tokens: list[str]) -> dict[str, float]:
    gt_slots = slot_partition(gt_tokens)
    pred_slots = slot_partition(pred_tokens)
    results = {}
    for slot_name in ["handshape", "orientation", "location", "movement", "dual_hand"]:
        gt_set = gt_slots[slot_name]
        pred_set = pred_slots[slot_name]
        if not gt_set and not pred_set:
            results[slot_name] = 1.0
        elif not gt_set:
            results[slot_name] = 0.0
        else:
            intersection = gt_set & pred_set
            union = gt_set | pred_set
            results[slot_name] = len(intersection) / len(union) if union else 1.0
    return results


# ------------- Set-Level Accuracy (Original) -------------

def set_accuracy(gt_tokens: list[str], pred_tokens: list[str]) -> float:
    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)
    if not gt_set:
        return 1.0 if not pred_set else 0.0
    return len(gt_set & pred_set) / len(gt_set)


# ------------- Test Cases -------------

TEST_CASES = [
    # (description, video_path, ground_truth_hamnosys)
    (
        "abbreviate (2-hand symmetric)",
        "Integration-20260706T062240Z-3-001/Integration/a_001_009_000_abbreviate_output.mp4",
        "hamsymmlr hamcee12 hamextfingeru hampalmd hamshoulders hamlrbeside hammovei",
    ),
    (
        "absolute-zero (2-hand asymmetric)",
        "Integration-20260706T062240Z-3-001/Integration/a_001_047_000_absolute-zero_output.mp4",
        "hamparbegin hamceeall hamextfingerd hampalml hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamtouch hammoved",
    ),
    (
        "Prompt_1 (2-hand asymmetric)",
        "Integration-20260706T062240Z-3-001/Integration/Prompt_1.mp4",
        "hamparbegin hamflathand hamextfingerd hampalml hamplus hamflathand hamextfingero hampalmu hamparend hamear hamtouch hamfingerplay",
    ),
]


def main():
    print("=" * 80)
    print("  HamNoSys Sequence-Level & Slot-Based Accuracy Evaluation")
    print("=" * 80)

    all_set_acc = []
    all_tok_acc = []
    all_wer = []
    all_slot_acc = {s: [] for s in ["handshape", "orientation", "location", "movement", "dual_hand"]}

    for desc, video_path, gt_hamnosys in TEST_CASES:
        print(f"\n{'-' * 60}")
        print(f"  Test: {desc}")
        print(f"  Video: {video_path}")

        gt_tokens = gt_hamnosys.split()
        hamnosys_str, modules = generate_hamnosys(video_path)
        pred_tokens = hamnosys_str.split()

        print(f"  GT   : {gt_hamnosys}")
        print(f"  PRED : {hamnosys_str}")

        # 1. Set Accuracy (original method from accuracy.ipynb)
        sa = set_accuracy(gt_tokens, pred_tokens)
        all_set_acc.append(sa)
        print(f"  Set Accuracy      : {sa:.2%}")

        # 2. Token Accuracy (1 - WER)
        wer = word_error_rate(gt_tokens, pred_tokens)
        ta = token_accuracy(gt_tokens, pred_tokens)
        all_wer.append(wer)
        all_tok_acc.append(ta)
        print(f"  Token Accuracy    : {ta:.2%}  (WER: {wer:.2%}, EditDist: {levenshtein_distance(gt_tokens, pred_tokens)})")

        # 3. Slot Accuracy
        slots = slot_accuracy(gt_tokens, pred_tokens)
        for slot_name, score in slots.items():
            all_slot_acc[slot_name].append(score)
            print(f"  {slot_name.capitalize():16s}: {score:.2%}")

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY ACROSS {len(TEST_CASES)} TEST CASES")
    print(f"{'=' * 80}")
    print(f"  Mean Set Accuracy   : {sum(all_set_acc) / len(all_set_acc):.2%}")
    print(f"  Mean Token Accuracy : {sum(all_tok_acc) / len(all_tok_acc):.2%}")
    print(f"  Mean WER            : {sum(all_wer) / len(all_wer):.2%}")
    print()
    for slot_name in ["handshape", "orientation", "location", "movement", "dual_hand"]:
        scores = all_slot_acc[slot_name]
        mean = sum(scores) / len(scores) if scores else 0.0
        print(f"  {slot_name.capitalize():16s}: {mean:.2%}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
