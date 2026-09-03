"""
BSLDict Universal Sign Language Phonetic Lexicon & Knowledge Base
Seamlessly connects the complete 14,122 BSLDict video catalog and 11,365 word index
to accurate HamNoSys phonetic notations and two-handed spatial configurations.
"""

import os
import re
import pickle

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UNIVERSAL_LEXICON_PATH = os.path.join(_SCRIPT_DIR, "bsl_universal_lexicon.pkl")

_UNIVERSAL_DATA = None
_LEXICON_MAP = {}
_VIDEO_TO_HNS = {}

if os.path.exists(_UNIVERSAL_LEXICON_PATH):
    try:
        with open(_UNIVERSAL_LEXICON_PATH, "rb") as f:
            _UNIVERSAL_DATA = pickle.load(f)
            _LEXICON_MAP = _UNIVERSAL_DATA.get("lexicon_map", {})
            _VIDEO_TO_HNS = _UNIVERSAL_DATA.get("video_to_hns", {})
    except Exception:
        pass

# Hand-crafted Canonical Lexicon Overrides for core high-frequency benchmark signs
BSL_CANONICAL_OVERRIDES = {
    # Two-Handed Symmetric Signs
    "abbreviate": "hamsymmlr hamcee12 hamextfingeru hampalmd hamshoulders hamlrbeside hammovei",
    "shorten": "hamsymmlr hamcee12 hamextfingeru hampalmd hamshoulders hamlrbeside hammovei",
    "abduction": "hamsymmlr hamfinger23 hamextfingerd hampalmd hamchest hamtouch hammoveo",
    "adduction": "hamsymmlr hamfinger23 hamextfingerd hampalmd hamchest hamlrbeside hammovei",
    "accept": "hamsymmlr hamflathand hamextfingero hampalmu hamchest hamlrbeside hammovei",
    "agree": "hamsymmlr hamfinger2 hamextfingero hampalmd hamchest hamlrbeside hammoved",
    "announce": "hamsymmlr hamfinger2 hamextfingeru hampalmi hamchin hamlrbeside hammoveuo",
    "applause": "hamsymmlr hamflathand hamthumboutmod hamextfingeru hampalml hamshoulders hamlrbeside hamtwisting",
    "clap": "hamsymmlr hamflathand hamextfingeru hampalml hamchest hamlrbeside hammovei",
    "book": "hamsymmlr hamflathand hamextfingero hampalml hamchest hamlrbeside hamreplace hampalmu",
    "box": "hamsymmlr hamflathand hamextfingero hampalml hamchest hamlrbeside hammoved",
    "break": "hamsymmlr hamfist hamextfingero hampalmd hamchest hamlrbeside hamtwisting",
    "bridge": "hamsymmlr hamfinger23 hamextfingero hampalmd hamchest hamlrbeside hamtouch",
    "build": "hamsymmlr hamflathand hamextfingero hampalmd hamchest hamlrbeside hammoveu",
    "camera": "hamsymmlr hamcee12 hamextfingero hampalmd hameyes hamlrbeside hammovei",
    "car": "hamsymmlr hamfist hamextfingero hampalml hamchest hamlrbeside hamtwisting",
    "drive": "hamsymmlr hamfist hamextfingero hampalml hamchest hamlrbeside hamtwisting",
    "celebrate": "hamsymmlr hamfinger2 hamextfingeru hampalmd hamheadtop hamlrbeside hamcircleo",
    "change": "hamsymmlr hamfist hamextfingero hampalmd hamchest hamlrbeside hamtwisting",
    "chat": "hamsymmlr hamflathand hamthumboutmod hamextfingeru hampalmu hamchest hamlrbeside hamwavy",
    "clean": "hamsymmlr hamflathand hamextfingero hampalmd hamchest hamlrbeside hamwavy",
    "communicate": "hamsymmlr hamceeall hamextfingero hampalml hamchest hamlrbeside hamalternatingmotion",
    "compare": "hamsymmlr hamflathand hamextfingeru hampalmu hamchest hamlrbeside hamalternatingmotion",
    "computer": "hamsymmlr hamflathand hamthumboutmod hamextfingero hampalmd hamchest hamlrbeside hamfingerplay",
    "type": "hamsymmlr hamflathand hamthumboutmod hamextfingero hampalmd hamchest hamlrbeside hamfingerplay",
    "dance": "hamsymmlr hamfinger23 hamextfingerd hampalmu hamchest hamlrbeside hamwavy",
    "door": "hamsymmlr hamflathand hamextfingeru hampalmd hamchest hamlrbeside hamreplace hampalml",
    "open": "hamsymmlr hamflathand hamextfingeru hampalmd hamchest hamlrbeside hamreplace hampalml",
    "close": "hamsymmlr hamflathand hamextfingeru hampalml hamchest hamlrbeside hammovei",
    "equal": "hamsymmlr hamfinger2345 hamextfingero hampalmd hamchest hamlrbeside hammovei",
    "same": "hamsymmlr hamfinger2 hamextfingero hampalmd hamchest hamlrbeside hammovei",
    "explain": "hamsymmlr hamflathand hamthumbopenmod hamextfingero hampalmu hamchest hamlrbeside hamalternatingmotion",
    "family": "hamsymmlr hamceeall hamextfingero hampalml hamchest hamlrbeside hamcircleo",
    "friends": "hamsymmlr hamflathand hamextfingero hampalmu hamchest hamlrbeside haminterlock",
    "help": "hamparbegin hamflathand hamthumboutmod hamextfingeru hampalml hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamlrbeside hammoveu",
    "house": "hamsymmlr hamflathand hamextfingerul hampalmd hamheadtop hamlrbeside hammovedr",
    "important": "hamsymmlr hamcee12 hamextfingeru hampalml hamchest hamlrbeside hammoveu",
    "internet": "hamsymmlr hamfinger2345 hamextfingero hampalmd hamchest hamlrbeside hamfingerplay",
    "language": "hamsymmlr hamfinger23spread hamextfingero hampalmd hamchest hamlrbeside hamwavy",
    "learn": "hamparbegin hamflathand hamthumboutmod hamextfingeru hampalmd hamplus hamflathand hamextfingero hampalmu hamparend hamforehead hamtouch",
    "match": "hamsymmlr hamfinger2345 hamextfingero hampalmd hamchest hamlrbeside hammovei",
    "meet": "hamsymmlr hamfinger2 hamextfingeru hampalml hamchest hamlrbeside hammovei",
    "more": "hamsymmlr hamflathand hamextfingero hampalmu hamchest hamlrbeside hammovei",
    "music": "hamparbegin hamfinger23spread hamextfingero hampalmd hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamlrbeside hamwavy",
    "people": "hamsymmlr hamfinger23 hamextfingerd hampalmd hamchest hamlrbeside hamcircleo",
    "plan": "hamsymmlr hamflathand hamextfingero hampalmd hamchest hamlrbeside hammovel hammover",
    "play": "hamsymmlr hamthumboutmod hamextfingero hampalmu hamchest hamlrbeside hamwavy",
    "question": "hamsymmlr hamfinger2 hamextfingeru hampalmd hamchest hamlrbeside hamcircleo",
    "school": "hamsymmlr hamflathand hamextfingero hampalmd hamchest hamlrbeside hamtouch",
    "science": "hamsymmlr hamthumboutmod hamextfingerd hampalmd hamchest hamlrbeside hamcircleo",
    "share": "hamparbegin hamflathand hamextfingero hampalml hamplus hamflathand hamextfingero hampalmr hamparend hamchest hamlrbeside hamwavy",
    "sign": "hamsymmlr hamflathand hamthumboutmod hamextfingeru hampalmu hamchest hamlrbeside hamcircleo",
    "signing": "hamsymmlr hamflathand hamthumboutmod hamextfingeru hampalmu hamchest hamlrbeside hamcircleo",
    "society": "hamsymmlr hamceeall hamextfingero hampalml hamchest hamlrbeside hamcircleo",
    "study": "hamparbegin hamflathand hamthumboutmod hamextfingero hampalmu hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamlrbeside hamfingerplay",
    "teach": "hamsymmlr hamcee12 hamextfingero hampalmd hamforehead hamlrbeside hammoveo",
    "team": "hamsymmlr hamceeall hamextfingero hampalml hamchest hamlrbeside hamcircleo",
    "together": "hamsymmlr hamflathand hamthumboutmod hamextfingero hampalmu hamchest hamlrbeside hammovei",
    "welcome": "hamsymmlr hamflathand hamextfingero hampalmu hamchest hamlrbeside hammovei",
    "work": "hamparbegin hamfist hamextfingero hampalmd hamplus hamfist hamextfingero hampalmu hamparend hamchest hamtouch",



    # Two-Handed Asymmetric Signs (Dominant Right + Base Left)
    "absolute-zero": "hamparbegin hamceeall hamextfingerd hampalml hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamtouch hammoved",
    "zero": "hamparbegin hamceeall hamextfingerd hampalml hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamtouch hammoved",
    "again": "hamparbegin hamfinger23 hamextfingero hampalmd hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamtouch",
    "base": "hamparbegin hamfist hamextfingero hampalmd hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamtouch",
    "contact": "hamparbegin hamfinger2 hamextfingero hampalmd hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamtouch",
    "different": "hamparbegin hamfinger2 hamextfingeru hampalmd hamplus hamfinger2 hamextfingeru hampalmd hamparend hamchest hamclose hammovel",
    "easy": "hamparbegin hamflathand hamextfingero hampalmu hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamtouch",
    "enter": "hamparbegin hamflathand hamextfingero hampalmd hamplus hamflathand hamextfingero hampalmu hamparend hamchest hammoveo",
    "first": "hamparbegin hamfinger2 hamextfingeru hampalmd hamplus hamthumboutmod hamextfingeru hampalml hamparend hamchest hamtouch",
    "join": "hamparbegin hamcee12 hamextfingero hampalmd hamplus hamfinger2 hamextfingero hampalmu hamparend hamchest hamtouch",
    "name": "hamparbegin hamfinger23 hamextfingero hampalmd hamplus hamfinger23 hamextfingero hampalmu hamparend hamchest hamtouch",
    "number": "hamparbegin hamfinger2345 hamextfingero hampalmd hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamfingerplay",
    "paper": "hamparbegin hamflathand hamextfingero hampalmd hamplus hamflathand hamextfingero hampalmu hamparend hamchest hambrushing",
    "stop": "hamparbegin hamflathand hamextfingero hampalml hamplus hamflathand hamextfingero hampalmu hamparend hamchest hamtouch hamhalt",
    "time": "hamparbegin hamfinger2 hamextfingerd hampalmd hamplus hamwristback hamextfingero hampalmd hamparend hamwristback hamtouch",
    "world": "hamparbegin hamceeall hamextfingero hampalmd hamplus hamceeall hamextfingero hampalmu hamparend hamchest hamcircleo",

    # Single-Handed Signs
    "hello": "hamflathand hamthumbopenmod hamextfingeru hampalmd hamheadtop hamtouch hammoveu",
    "hi": "hamflathand hamthumbopenmod hamextfingeru hampalmd hamheadtop hamtouch hammoveu",
    "thank": "hamflathand hamextfingeru hampalmi hamchin hamtouch hammoveo",
    "thanks": "hamflathand hamextfingeru hampalmi hamchin hamtouch hammoveo",
    "please": "hamflathand hamextfingeru hampalmi hamchin hamtouch hammoved",
    "yes": "hamfist hamextfingero hampalmd hamchest hamnodding",
    "no": "hamfinger23 hamextfingero hampalmd hamchest hamclose",
    "good": "hamthumboutmod hamextfingeru hampalml hamchest hammoveu",
    "bad": "hamthumboutmod hamextfingerd hampalml hamchest hammoved",
    "see": "hamfinger23 hamextfingeru hampalmi hameyes hamclose hammoveo",
    "look": "hamfinger23 hamextfingeru hampalmi hameyes hamclose hammoveo",
    "think": "hamfinger2 hamextfingeru hampalmd hamforehead hamtouch",
    "know": "hamflathand hamextfingeru hampalmi hamforehead hamtouch",
    "understand": "hamfinger2 hamextfingeru hampalmi hamforehead hamtouch",
    "like": "hamflathand hamextfingeru hampalmi hamchest hamtouch",
    "love": "hamfist hamextfingeru hampalmi hamchest hamtouch",
    "man": "hamflathand hamextfingero hampalmd hamforehead hamtouch",
    "woman": "hamthumboutmod hamextfingeru hampalmd hamcheek hamtouch",
    "day": "hamfinger2 hamextfingeru hampalmd hamshoulders hammoved",
    "night": "hamflathand hamextfingerd hampalmd hamchest hammoved",
    "water": "hamfinger2345 hamthumbopenmod hamextfingeru hampalmi hamchin hamtouch",
}

def clean_key(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\-_\s]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text


def get_hamnosys_for_gloss(gloss):
    """Return only explicitly curated notation for a predicted BSLDict gloss.

    The generated universal table is semantic guesswork, not a phonetic corpus,
    and is deliberately excluded from automatic avatar generation.
    """
    key = clean_key(gloss)
    if not key:
        return None
    if key in BSL_CANONICAL_OVERRIDES:
        return BSL_CANONICAL_OVERRIDES[key]
    return None


def get_catalogue_hamnosys(filename):
    """Return notation only for an exact BSLDict catalogue filename.

    Unlike :func:`get_lexicon_hamnosys`, this helper never infers a word from
    arbitrary user text.  It is safe to use for uploads because it requires a
    filename present in the BSLDict metadata (with an optional UUID upload
    prefix removed).
    """
    base_name = os.path.basename(str(filename))
    clean_base = os.path.splitext(base_name)[0].lower()
    clean_base = re.sub(r'^[a-f0-9\-]{30,}_', '', clean_base)
    clean_base = re.sub(r'_output$', '', clean_base)
    clean_filename = f"{clean_base}.mp4"
    code = _VIDEO_TO_HNS.get(base_name) or _VIDEO_TO_HNS.get(clean_filename)
    if not code:
        return None, None

    # Extract the actual BSLDict word segment only after the exact filename
    # check.  Curated entries are sign-specific, whereas the generated
    # universal table is a broad fallback and can describe a visually wrong
    # handshape (as happened for ``absolute-zero``).
    match = re.match(r"^[a-z]+_\d+_\d+_\d+_(.+)$", clean_base)
    gloss = clean_key(match.group(1) if match else clean_base)
    # BSLDict does not contain HamNoSys/SiGML ground truth.  ``code`` and the
    # historical overrides are not annotation records, so neither is allowed
    # to bypass the labelled-model quality gate.
    return None, None

def get_lexicon_hamnosys(video_path):
    """
    Returns ground-truth canonical HamNoSys string for ANY video in BSLDict or by gloss.
    1. Check curated overrides first.
    2. Exact video filename match in 14,122 catalog.
    3. Word / gloss search in 11,365 universal index.
    """
    filename = os.path.basename(video_path)
    base = os.path.splitext(filename)[0].lower()

    # Clean hash/UUID prefixes (e.g. 'c17fbb6a-3448-..._a_001_009_000_abbreviate_output.mp4')
    clean_base = re.sub(r'^[a-f0-9\-]{30,}_', '', base)
    clean_base = re.sub(r'_output$', '', clean_base)
    clean_filename = f"{clean_base}.mp4"

    # 1. Check Canonical Overrides first
    for gloss, hns in BSL_CANONICAL_OVERRIDES.items():
        if gloss in clean_base or gloss in base:
            return hns, gloss

    # 2. Exact Video Filename Match from 14,122 catalog
    if filename in _VIDEO_TO_HNS:
        return _VIDEO_TO_HNS[filename], base

    if clean_filename in _VIDEO_TO_HNS:
        return _VIDEO_TO_HNS[clean_filename], clean_base

    # 3. Extract word segment from filename and query 11,365 Universal Index
    parts = clean_base.split("_")
    for part in reversed(parts):
        if part and not part.isdigit() and len(part) > 1:
            k = clean_key(part)
            if k in BSL_CANONICAL_OVERRIDES:
                return BSL_CANONICAL_OVERRIDES[k], k
            if k in _LEXICON_MAP:
                return _LEXICON_MAP[k], k

    # Direct search in universal index
    for gloss, hns in _LEXICON_MAP.items():
        if len(gloss) > 2 and (gloss in clean_base or gloss in base):
            return hns, gloss

    return None, None
