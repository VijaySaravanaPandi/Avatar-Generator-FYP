#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp

# =============================
# HAMNOSYS HANDSHAPE LABELS
# =============================

NODE_A = [
    "hamfist","hamflathand","hamfinger2","hamfinger23","hamfinger23spread",
    "hamfinger2345","hampinch12","hampinch12open","hampinchall",
    "hamcee12","hamceeall","hamceeopen"
]

NODE_C = ["hamthumboutmod","hamthumbacrossmod","hamthumbopenmod"]

NODE_D = [
    "hamdoublebent","hamdoublehooked",
    "hamfingerstraightmod","hamfingerbendmod","hamfingerhookmod"
]

NODE_E = [
    "hamthumb","hamindexfinger","hammiddlefinger","hamringfinger","hampinky",
    "hambetween","hamfingernail","hamfingerpad","hamfingerside","hamfingermidjoint"
]

# =============================
# MEDIAPIPE HANDS
# =============================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =============================
# FINGER STATE
# =============================

def finger_open(tip, pip, lm):
    return lm[tip].y < lm[pip].y


def get_finger_states(lm):

    thumb = finger_open(4,3,lm)
    index = finger_open(8,6,lm)
    middle = finger_open(12,10,lm)
    ring = finger_open(16,14,lm)
    pinky = finger_open(20,18,lm)

    return thumb,index,middle,ring,pinky


# =============================
# HANDSHAPE CLASSIFICATION
# =============================

def classify_handshape(lm):

    thumb,index,middle,ring,pinky = get_finger_states(lm)

    # fist
    if not thumb and not index and not middle and not ring and not pinky:
        return "hamfist"

    # flat hand
    if index and middle and ring and pinky:
        return "hamflathand"

    # index finger
    if index and not middle and not ring and not pinky:
        return "hamindexfinger"

    # two fingers
    if index and middle and not ring and not pinky:
        return "hamfinger23"

    # pinch
    if thumb and index and not middle and not ring and not pinky:
        return "hampinch12"

    # thumb modifier
    if thumb and not index:
        return "hamthumboutmod"

    # straight fingers
    if index and middle and ring and pinky and not thumb:
        return "hamfingerstraightmod"

    # fallback finger labels
    if thumb:
        return "hamthumb"

    if index:
        return "hamindexfinger"

    if middle:
        return "hammiddlefinger"

    if ring:
        return "hamringfinger"

    if pinky:
        return "hampinky"

    return "none"


# =============================
# HANDSHAPE MODULE
# =============================

def run_handshape_module(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    per_frame = []

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(img)

        label = "none"

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                lm = hand_landmarks.landmark

                label = classify_handshape(lm)

        per_frame.append(label)

    cap.release()

    return {
        "per_frame": per_frame
    }

