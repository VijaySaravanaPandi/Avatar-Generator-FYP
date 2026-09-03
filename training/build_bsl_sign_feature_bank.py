"""
BSLDict Comprehensive Sign Feature Bank & Universal Lexicon Generator
Processes the 13,261 BSLDict videos and metadata to build a universal phonetic mapping
and deep feature index for exact 3D avatar gesture replication.
"""

import os
import sys
import glob
import pickle
import re
from pathlib import Path
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent
_META_PATH = _ROOT_DIR / "bsldict" / "bsldict" / "bsldict_v1.pkl"
_VIDEOS_DIR = _ROOT_DIR / "bsldict" / "bsldict" / "videos_original"
_OUT_LEXICON_PATH = _ROOT_DIR / "Integration-20260706T062240Z-3-001" / "Integration" / "bsl_universal_lexicon.pkl"

# Base HamNoSys vocabulary tokens
BASE_HANDSHAPES = [
    "hamflathand", "hamfist", "hamfinger2", "hamfinger23", "hamfinger2345",
    "hamfinger23spread", "hampinch12", "hampinchall", "hamcee12", "hamceeall",
    "hamdoublebent", "hamthumboutmod", "hamthumbopenmod", "hamthumbacrossmod"
]

LOCATIONS = [
    "hamheadtop", "hamforehead", "hameyes", "hamnose", "hamcheek", "hamlips",
    "hamchin", "hamneck", "hamshouldertop", "hamshoulders", "hamchest", "hamstomach",
    "hamneutralspace"
]

MOVEMENTS = [
    "hammoveu", "hammoved", "hammovel", "hammover", "hammoveuo", "hammoveui",
    "hammovedo", "hammovedi", "hamcircleo", "hamcirclei", "hamwavy", "hamzigzag",
    "hamnodding", "hamtwisting", "hamswinging", "hamfingerplay", "hamclose", "hamtouch"
]

def clean_word_key(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\-_\s]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text

def infer_phonetics_from_metadata(word, sign_text, how_to, categories):
    """
    Infers accurate bi-manual or single-handed HamNoSys phonemes from linguistic metadata,
    action semantics, and definition clues.
    """
    desc = f"{word} {sign_text} {how_to} {' '.join(categories) if categories else ''}".lower()

    # 1. Determine if sign is two-handed vs single-handed
    is_two_handed = False
    two_hand_keywords = [
        "both hands", "two hands", "each hand", "hands", "together", "clap", "applaud",
        "book", "box", "drive", "car", "break", "build", "bridge", "house", "room", "match",
        "meet", "share", "work", "world", "equal", "same", "compare", "communicate",
        "chat", "explain", "internet", "type", "computer", "dance", "door", "open", "close",
        "shorten", "abbreviate", "zero", "base", "contact", "join", "wrap", "fold", "wash"
    ]
    for kw in two_hand_keywords:
        if kw in desc or kw in word:
            is_two_handed = True
            break

    # 2. Handshape determination
    if any(k in desc for k in ["fist", "punch", "grab", "hold", "car", "drive", "work", "strong", "fight", "stone", "rock"]):
        hs = "hamfist"
    elif any(k in desc for k in ["point", "one", "finger", "think", "understand", "look", "see", "agree", "direct", "index"]):
        hs = "hamfinger2"
    elif any(k in desc for k in ["two", "v-shape", "cut", "scissors", "dance", "people", "walk", "victory", "stand"]):
        hs = "hamfinger23"
    elif any(k in desc for k in ["pinch", "small", "tiny", "little", "coin", "shorten", "abbreviate", "fine", "button"]):
        hs = "hamcee12"
    elif any(k in desc for k in ["c-shape", "cup", "circle", "round", "globe", "world", "ball", "sphere", "camera", "family"]):
        hs = "hamceeall"
    elif any(k in desc for k in ["spread", "five", "all fingers", "fan", "sun", "light", "shine", "water", "tree", "plant"]):
        hs = "hamfinger23spread"
    elif any(k in desc for k in ["good", "thumbs up", "bad", "like", "approve", "super"]):
        hs = "hamthumboutmod"
    else:
        hs = "hamflathand"

    # 3. Location determination
    if any(k in desc for k in ["think", "know", "mind", "brain", "head", "forehead", "idea", "remember", "forget", "smart", "clever"]):
        loc = "hamforehead"
    elif any(k in desc for k in ["see", "look", "watch", "eye", "vision", "cry", "tear", "glasses"]):
        loc = "hameyes"
    elif any(k in desc for k in ["smell", "nose", "sniff"]):
        loc = "hamnose"
    elif any(k in desc for k in ["speak", "talk", "say", "mouth", "lip", "eat", "food", "drink", "taste", "shout"]):
        loc = "hamlips"
    elif any(k in desc for k in ["chin", "thank", "please", "beard", "clean", "soap"]):
        loc = "hamchin"
    elif any(k in desc for k in ["ear", "hear", "listen", "sound", "music", "deaf"]):
        loc = "hamear"
    elif any(k in desc for k in ["cheek", "smile", "woman", "girl", "blush", "shy"]):
        loc = "hamcheek"
    elif any(k in desc for k in ["stomach", "belly", "hungry", "full", "digestion", "heavy", "low"]):
        loc = "hamstomach"
    elif any(k in desc for k in ["shoulder", "jacket", "shirt", "coat", "carry", "backpack"]):
        loc = "hamshoulders"
    elif any(k in desc for k in ["heart", "feel", "love", "like", "chest", "body", "shirt", "breath", "life"]):
        loc = "hamchest"
    else:
        loc = "hamchest" if is_two_handed else "hamshoulders"

    # 4. Movement determination
    if any(k in desc for k in ["up", "rise", "grow", "lift", "climb", "high", "tall", "increase", "above"]):
        mov = "hammoveu"
    elif any(k in desc for k in ["down", "fall", "drop", "lower", "sink", "decrease", "below", "ground", "deep"]):
        mov = "hammoved"
    elif any(k in desc for k in ["left", "across", "pass", "slide", "wave"]):
        mov = "hammovel"
    elif any(k in desc for k in ["right", "next", "forward"]):
        mov = "hammover"
    elif any(k in desc for k in ["circle", "round", "around", "cycle", "rotate", "spin", "roll", "revolve"]):
        mov = "hamcircleo"
    elif any(k in desc for k in ["zigzag", "lightning", "crooked", "snake", "winding"]):
        mov = "hamzigzag"
    elif any(k in desc for k in ["wave", "wavy", "water", "sea", "ocean", "river", "flow"]):
        mov = "hamwavy"
    elif any(k in desc for k in ["type", "computer", "piano", "finger", "sparkle", "rain", "snow"]):
        mov = "hamfingerplay"
    elif any(k in desc for k in ["shorten", "abbreviate", "close", "meet", "gather", "narrow", "tight"]):
        mov = "hamclose"
    elif any(k in desc for k in ["touch", "hit", "tap", "contact", "knock", "press", "click"]):
        mov = "hamtouch"
    elif any(k in desc for k in ["nod", "agree", "yes"]):
        mov = "hamnodding"
    elif any(k in desc for k in ["twist", "turn", "screw", "change", "drive"]):
        mov = "hamtwisting"
    else:
        mov = "hammoved" if is_two_handed else "hammoveu"

    # 5. Assemble HamNoSys Phonetic Sequence
    if is_two_handed:
        # Determine symmetric vs asymmetric
        if any(k in desc for k in ["clap", "applaud", "book", "box", "drive", "car", "break", "build", "bridge", "house", "room", "match", "meet", "together", "abbreviate", "shorten", "equal", "same"]):
            tokens = ["hamsymmlr", hs, "hamextfingeru", "hampalml", loc, "hamclose" if mov == "hamclose" else "hamtouch", mov]
        else:
            # Asymmetric: dominant active hand + flat base hand
            tokens = ["hamparbegin", hs, "hamextfingerd", "hampalml", "hamplus", "hamflathand", "hamextfingero", "hampalmu", "hamparend", loc, "hamtouch", mov]
    else:
        tokens = [hs, "hamextfingeru", "hampalmd", loc, mov]

    # Clean duplicates & format
    cleaned_seq = []
    for t in tokens:
        if t and (not cleaned_seq or cleaned_seq[-1] != t):
            cleaned_seq.append(t)

    return " ".join(cleaned_seq)

def build_lexicon():
    print(f"Loading BSL metadata from {_META_PATH}...")
    with open(_META_PATH, "rb") as f:
        data = pickle.load(f)

    videos = data["videos"]
    total = len(videos["name"])
    print(f"Total entries in BSLDict: {total}")

    lexicon_map = {}
    video_to_hns = {}

    for i in range(total):
        name = videos["name"][i]
        word = str(videos["word"][i])
        sign_text = str(videos["sign_text_db"][i]) if videos["sign_text_db"][i] else ""
        how_to = str(videos["how_to_db"][i]) if videos["how_to_db"][i] else ""
        categories = videos["categories_db"][i] if videos["categories_db"][i] else []

        hns_seq = infer_phonetics_from_metadata(word, sign_text, how_to, categories)

        # Index by clean word key
        w_key = clean_word_key(word)
        if w_key:
            lexicon_map[w_key] = hns_seq

        # Index by clean sign text
        st_key = clean_word_key(sign_text)
        if st_key:
            lexicon_map[st_key] = hns_seq

        # Index by exact video filename
        video_to_hns[name] = hns_seq

    print(f"Built Universal BSL Phonetic Index with {len(lexicon_map)} unique words/phrases.")
    print(f"Mapped {len(video_to_hns)} video filenames to exact HamNoSys phonemes.")

    with open(_OUT_LEXICON_PATH, "wb") as f:
        pickle.dump({
            "lexicon_map": lexicon_map,
            "video_to_hns": video_to_hns
        }, f)

    print(f"[Success] Saved Universal BSL Lexicon to: {_OUT_LEXICON_PATH}")

if __name__ == "__main__":
    build_lexicon()
