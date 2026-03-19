"""
utils/analysis.py
-----------------
Facial landmark analysis helpers for the Cortisol Meter app.
Uses MediaPipe Face Mesh landmark indices to compute:
  - Eye Aspect Ratio (EAR) — eye openness
  - Eyebrow tension score
  - Mouth openness ratio
  - Composite cortisol estimate + confidence
"""

import math
import numpy as np


# ── MediaPipe Face Mesh landmark indices ─────────────────────────────────────
# Left eye (from the perspective of the person in the mirror)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Eyebrow landmarks (inner & outer, left & right)
LEFT_EYEBROW_INNER = 285
LEFT_EYEBROW_OUTER = 295
LEFT_EYEBROW_MID   = 334

RIGHT_EYEBROW_INNER = 55
RIGHT_EYEBROW_OUTER = 65
RIGHT_EYEBROW_MID   = 105

# Reference points: nose bridge and eye corners
LEFT_EYE_TOP     = 386
LEFT_EYE_BOTTOM  = 374
RIGHT_EYE_TOP    = 159
RIGHT_EYE_BOTTOM = 145

# Mouth landmarks
UPPER_LIP  = 13
LOWER_LIP  = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Nose tip (used as a vertical reference)
NOSE_TIP = 1
NOSE_BRIDGE = 168


# ── Math helpers ─────────────────────────────────────────────────────────────

def _dist(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _landmark_xy(landmarks, idx, w, h):
    """Return pixel coordinates of a landmark by index."""
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


# ── Feature extractors ────────────────────────────────────────────────────────

def compute_ear(landmarks, eye_indices, w, h):
    """
    Eye Aspect Ratio (EAR).
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    A lower EAR means the eye is more closed (drowsy / stressed).
    """
    pts = [_landmark_xy(landmarks, i, w, h) for i in eye_indices]
    A = _dist(pts[1], pts[5])
    B = _dist(pts[2], pts[4])
    C = _dist(pts[0], pts[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear


def compute_brow_tension(landmarks, w, h):
    """
    Eyebrow tension score.
    Measures how close each eyebrow is to its respective eye.
    A smaller gap → more furrowing → more tension.
    Returns a value in [0, 1] where 1 = maximum tension.
    """
    # Left brow mid → left eye top
    left_brow  = _landmark_xy(landmarks, LEFT_EYEBROW_MID,  w, h)
    left_eye_t = _landmark_xy(landmarks, LEFT_EYE_TOP,      w, h)
    left_gap   = _dist(left_brow, left_eye_t)

    # Right brow mid → right eye top
    right_brow  = _landmark_xy(landmarks, RIGHT_EYEBROW_MID, w, h)
    right_eye_t = _landmark_xy(landmarks, RIGHT_EYE_TOP,     w, h)
    right_gap   = _dist(right_brow, right_eye_t)

    # Normalise by inter-eye distance (as a face-size reference)
    left_corner  = _landmark_xy(landmarks, MOUTH_LEFT,  w, h)
    right_corner = _landmark_xy(landmarks, MOUTH_RIGHT, w, h)
    face_width   = _dist(left_corner, right_corner) + 1e-6

    avg_gap       = (left_gap + right_gap) / 2.0
    normalised    = avg_gap / face_width          # bigger gap = relaxed brows
    tension_score = max(0.0, 1.0 - normalised * 3.5)  # invert + scale
    return float(np.clip(tension_score, 0.0, 1.0))


def compute_mouth_openness(landmarks, w, h):
    """
    Mouth openness ratio.
    Ratio of vertical mouth gap to horizontal mouth width.
    A very tight (closed) mouth suggests stress; wide open may indicate surprise.
    """
    upper = _landmark_xy(landmarks, UPPER_LIP,   w, h)
    lower = _landmark_xy(landmarks, LOWER_LIP,   w, h)
    left  = _landmark_xy(landmarks, MOUTH_LEFT,  w, h)
    right = _landmark_xy(landmarks, MOUTH_RIGHT, w, h)

    vertical   = _dist(upper, lower)
    horizontal = _dist(left, right) + 1e-6
    return float(np.clip(vertical / horizontal, 0.0, 1.0))


def estimate_head_tilt(landmarks, w, h):
    """
    Estimate head tilt from nose tip y vs nose bridge y.
    Not directly cortisol-related but adds a small modifier.
    Returns a small 0-1 tension modifier from asymmetry.
    """
    nose_tip    = _landmark_xy(landmarks, NOSE_TIP,    w, h)
    nose_bridge = _landmark_xy(landmarks, NOSE_BRIDGE, w, h)
    left_eye    = _landmark_xy(landmarks, LEFT_EYE[0],  w, h)
    right_eye   = _landmark_xy(landmarks, RIGHT_EYE[0], w, h)

    # Horizontal offset of nose relative to eye midpoint
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    offset     = abs(nose_tip[0] - eye_mid_x)
    face_width = _dist(left_eye, right_eye) + 1e-6
    return float(np.clip(offset / face_width * 0.5, 0.0, 0.3))


# ── Main analysis function ────────────────────────────────────────────────────

def analyse_face(landmarks, image_w, image_h):
    """
    Analyse a set of MediaPipe Face Mesh landmarks and return a dict with:
      - ear_left, ear_right   : float  (Eye Aspect Ratio per eye)
      - ear_avg               : float
      - brow_tension          : float  [0-1]
      - mouth_openness        : float  [0-1]
      - cortisol_score        : float  [0-100]  — composite stress score
      - cortisol_level        : str    ('Low' | 'Moderate' | 'High')
      - confidence            : float  [0-100]  — confidence in estimate
    """
    w, h = image_w, image_h

    ear_left   = compute_ear(landmarks, LEFT_EYE,  w, h)
    ear_right  = compute_ear(landmarks, RIGHT_EYE, w, h)
    ear_avg    = (ear_left + ear_right) / 2.0

    brow_tension   = compute_brow_tension(landmarks, w, h)
    mouth_open     = compute_mouth_openness(landmarks, w, h)
    head_modifier  = estimate_head_tilt(landmarks, w, h)

    # ── Scoring ──────────────────────────────────────────────────────────────
    # EAR: typical relaxed EAR ≈ 0.28–0.35; stressed eyes tend to be wider or
    # squinted.  We treat deviation from ~0.30 as stress signal.
    ear_stress = float(np.clip(abs(ear_avg - 0.30) / 0.15, 0.0, 1.0))

    # Weighted composite (weights tuned empirically)
    raw_score = (
        ear_stress    * 45.0 +   # eye deviation
        brow_tension  * 65.0 +   # furrowed brows — strongest signal
        mouth_open    * 30.0 +   # mouth tension
        head_modifier * 15.0     # head asymmetry
    )

    cortisol_score = float(np.clip(raw_score, 0.0, 100.0))

    # ── Level classification ─────────────────────────────────────────────────
    if cortisol_score < 30:
        level = "Low"
    elif cortisol_score < 55:
        level = "Moderate"
    else:
        level = "High"

    # ── Confidence ───────────────────────────────────────────────────────────
    # Based on how extreme (unambiguous) the features are
    feature_strength = (
        abs(ear_avg - 0.30) / 0.15 * 0.4 +
        brow_tension               * 0.4 +
        mouth_open                 * 0.2
    )
    confidence = float(np.clip(50.0 + feature_strength * 50.0, 0.0, 100.0))

    return {
        "ear_left":      round(ear_left,  4),
        "ear_right":     round(ear_right, 4),
        "ear_avg":       round(ear_avg,   4),
        "brow_tension":  round(brow_tension, 4),
        "mouth_openness": round(mouth_open, 4),
        "cortisol_score": round(cortisol_score, 1),
        "cortisol_level": level,
        "confidence":    round(confidence, 1),
    }


# ── Stress-reduction tips ─────────────────────────────────────────────────────

TIPS = [
    ("🫁", "Box Breathing",       "Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Repeat 4×."),
    ("🚶", "Short Walk",          "Step outside for 5–10 minutes. Fresh air lowers cortisol quickly."),
    ("💧", "Hydrate",             "Drink a glass of water. Dehydration amplifies the stress response."),
    ("📵", "Screen Break",        "Look away from screens for 2 minutes; focus on something 6m away."),
    ("😌", "Progressive Relax",   "Tense each muscle group for 5s, then release, starting from your feet."),
    ("🎵", "Calming Music",       "Listen to slow-tempo music (60–80 BPM) for 5 minutes."),
    ("🌿", "Mindful Moment",      "Close your eyes, take 5 deep breaths, notice 3 things you can hear."),
    ("☀️", "Sunlight",            "Step into natural light for a few minutes to reset your circadian rhythm."),
]


def get_tips(n=4):
    """Return n random stress-reduction tips."""
    import random
    return random.sample(TIPS, min(n, len(TIPS)))
