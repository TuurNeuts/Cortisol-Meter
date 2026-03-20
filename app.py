"""
Cortisol Meter — app.py
========================
A Streamlit app that estimates stress (cortisol) levels by analysing
facial landmarks from a live webcam feed via MediaPipe Face Mesh.

Works locally AND on Streamlit Cloud via streamlit-webrtc (WebRTC).

Run with:
    streamlit run app.py
"""

import threading
import random
from io import BytesIO
from pathlib import Path

import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from utils.analysis import analyse_face, get_tips
from utils.breathing import breathing_animation_html

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cortisol Meter",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark, modern, calming design
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}
*, *::before, *::after { box-sizing: inherit; }
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0d1b2a 50%, #0d2137 100%);
    color: #e0eeff;
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main container responsive padding ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Streamlit columns: allow wrapping on small screens ── */
[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap;
    gap: 1rem;
    align-items: flex-start;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    min-width: 280px;
    flex: 1 1 300px;
}

/* ── Ensure images never overflow their column ── */
[data-testid="stImage"] img,
.stImage img {
    max-width: 100% !important;
    height: auto !important;
    border-radius: 12px;
}

/* ── WebRTC video element ── */
video {
    border-radius: 12px;
    max-width: 100% !important;
}

/* ── Card panels ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,160,255,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
    width: 100%;
}
.card-high {
    background: rgba(255,80,80,0.07);
    border: 1px solid rgba(255,100,100,0.30);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    width: 100%;
}

/* ── Level badge ── */
.badge {
    display: inline-block;
    padding: 6px 18px;
    margin: 15px auto;
    border-radius: 40px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    white-space: nowrap;
}stre
.badge-low      { background: rgba(60,200,120,0.2); color:#4ade80; border:1px solid rgba(60,200,120,0.4); }
.badge-moderate { background: rgba(250,190,50,0.2); color:#fbbf24; border:1px solid rgba(250,190,50,0.4); }
.badge-high     { background: rgba(255,80,80,0.2);  color:#f87171; border:1px solid rgba(255,80,80,0.4); }

/* ── Score gauge container — fills available width ── */
.gauge-wrap {
    text-align: center;
    padding: 8px 0;
    width: 100%;
    overflow: hidden;
}
.gauge-wrap svg {
    max-width: 100%;
    height: auto;
}

/* ── Tip card ── */
.tip-card {
    background: rgba(60,120,200,0.12);
    border: 1px solid rgba(80,150,255,0.2);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    width: 100%;
}
.tip-icon { font-size: 1.4rem; line-height:1; margin-top:2px; flex-shrink: 0; }
.tip-title { font-weight:600; font-size:0.9rem; color:#a8d8ff; margin-bottom:4px; }
.tip-desc  { font-size:0.82rem; color:#8eb8d8; line-height:1.45; }

/* ── Section headings ── */
.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a8ab0;
    margin-bottom: 10px;
}

/* ── Disclaimer ── */
.disclaimer {
    font-size: 0.72rem;
    color: #4a6a88;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 12px;
    margin-top: 8px;
    line-height: 1.6;
    width: 100%;
}

/* ── Metric boxes ── */
.metric-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(100,160,255,0.1);
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    width: 100%;
}
.metric-label { font-size:0.7rem; color:#5a8ab0; text-transform:uppercase; letter-spacing:0.8px; }
.metric-value { font-size:1.1rem; font-weight:600; color:#c0deff; margin-top:2px; }

/* ── Streamlit button override ── */
.stButton > button {
    background: linear-gradient(135deg, #1e5fa8, #2a7fd4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: all 0.2s;
    width: 100%;
    white-space: nowrap;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a7fd4, #3a9fe4);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(42,127,212,0.35);
}

/* ── Progress bar colour ── */
.stProgress > div > div > div { border-radius: 8px; }

/* ── Responsive breakpoints ── */

/* Tablet: stack columns and reduce font sizes slightly */
@media (max-width: 1050px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .section-title { font-size: 0.7rem; }
    h1 { font-size: 1.5rem !important; }
}

/* Mobile: tighten padding, reduce card padding */
@media (max-width: 600px) {
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-top: 1rem !important;
    }
    .card, .card-high {
        padding: 14px 16px;
        border-radius: 12px;
    }
    .tip-card {
        padding: 10px 12px;
        gap: 8px;
    }
    .tip-title { font-size: 0.85rem; }
    .tip-desc  { font-size: 0.78rem; }
    .metric-value { font-size: 1rem; }
    .badge { font-size: 0.78rem; padding: 5px 14px; }
    h1 { font-size: 1.3rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe initialisation (cached)
# ─────────────────────────────────────────────────────────────────────────────

# URL for the official MediaPipe face landmark model
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


@st.cache_resource
def load_face_landmarker():
    """Download (once) and return a MediaPipe FaceLandmarker (Tasks API)."""
    if not _MODEL_PATH.exists():
        with st.spinner("⬇ Downloading face landmark model (~5 MB)…"):
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

    base_opts = mp_tasks.BaseOptions(model_asset_path=str(_MODEL_PATH))
    options   = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

LANDMARK_COLOUR = (80, 180, 255)   # BGR — sky blue
CONTOUR_COLOUR  = (60, 140, 220)   # slightly darker blue
HIGHLIGHT_IDS   = [1, 33, 263, 61, 291, 159, 145, 386, 374,
                   105, 334, 65, 295, 285, 55, 13, 14]   # key points


def draw_landmarks(image: np.ndarray, landmarks, draw_all: bool = True) -> np.ndarray:
    """Overlay facial landmarks on the image."""
    h, w = image.shape[:2]

    if draw_all:
        for idx, lm in enumerate(landmarks):
            if idx % 5 == 0:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 1, LANDMARK_COLOUR, -1)

    for idx in HIGHLIGHT_IDS:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 3, (255, 220, 100), -1)
        cv2.circle(image, (cx, cy), 4, CONTOUR_COLOUR, 1)

    return image


def cortisol_colour(level: str) -> tuple:
    """Return BGR colour for the given cortisol level."""
    return {"Low": (60, 200, 100), "Moderate": (40, 190, 250), "High": (80, 80, 255)}[level]


def badge_class(level: str) -> str:
    return {"Low": "badge-low", "Moderate": "badge-moderate", "High": "badge-high"}[level]


def gauge_html(score: float, level: str, width: int = 300) -> str:
    """
    SVG speedometer gauge: five colour bands (dark-green → lime → yellow
    → orange-red → dark-red) with a pivoting needle.
    """
    import math

    scale  = width / 300
    cx     = int(150 * scale)
    cy     = int(152 * scale)
    r_out  = int(120 * scale)
    r_in   = int(66  * scale)
    h_svg  = int(215 * scale)

    bands = [
        (180, 144, "#388E3C"),
        (144, 108, "#8BC34A"),
        (108,  72, "#FDD835"),
        ( 72,  36, "#F4511E"),
        ( 36,   0, "#B71C1C"),
    ]

    def pt(deg, r):
        a = math.radians(deg)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    segs = ""
    for a1, a2, col in bands:
        ox1, oy1 = pt(a1, r_out)
        ox2, oy2 = pt(a2, r_out)
        ix2, iy2 = pt(a2, r_in)
        ix1, iy1 = pt(a1, r_in)
        d = (f"M{ox1:.2f},{oy1:.2f} "
             f"A{r_out},{r_out} 0 0,0 {ox2:.2f},{oy2:.2f} "
             f"L{ix2:.2f},{iy2:.2f} "
             f"A{r_in},{r_in} 0 0,1 {ix1:.2f},{iy1:.2f}Z")
        segs += f'  <path d="{d}" fill="{col}"/>\n'

    divs = ""
    for a in [144, 108, 72, 36]:
        x1, y1 = pt(a, r_in  - 2)
        x2, y2 = pt(a, r_out + 2)
        sw = max(2, int(3 * scale))
        divs += (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" '
                 f'x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="#0d1b2a" stroke-width="{sw}"/>\n')

    na  = math.radians(180.0 - score * 1.8)
    nl  = r_in - int(8 * scale)
    nx  = cx + nl * math.cos(na)
    ny  = cy - nl * math.sin(na)
    nw  = max(2, int(3.5 * scale))

    sc_col = {"Low": "#66BB6A", "Moderate": "#FDD835", "High": "#EF5350"}.get(level, "#fff")

    ll_x, ll_y = pt(162, r_out + int(16 * scale))
    ln_x, ln_y = pt(90,  r_out + int(16 * scale))
    lh_x, lh_y = pt(18,  r_out + int(16 * scale))

    fs_label  = max(9,  int(11 * scale))
    fs_score  = max(18, int(26 * scale))
    fs_sub    = max(7,  int(9  * scale))
    r_hub1    = max(8,  int(12 * scale))
    r_hub2    = max(3,  int(5  * scale))

    title_y = max(14, int(20 * scale))
    title_fs = max(10, int(13 * scale))

    vb_w = width
    vb_h = h_svg + title_y + 4
    return (
        '<div class="gauge-wrap">'
        f'<svg width="100%" height="auto" '
        f'viewBox="0 0 {vb_w} {vb_h}" preserveAspectRatio="xMidYMid meet" style="overflow:visible; display:block;">'
        f'  <text x="{cx}" y="{title_y}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{title_fs}" font-weight="700"'
        f' letter-spacing="2" fill="#c0deff">CORTISOL LEVEL</text>\n'
        f'  <g transform="translate(0,{title_y + 4})">'
        f'\n{segs}{divs}'
        f'  <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}"'
        f' stroke="#1a1a2e" stroke-width="{nw + 2}" stroke-linecap="round"/>'
        f'  <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}"'
        f' stroke="#e8f4ff" stroke-width="{nw}" stroke-linecap="round"/>\n'
        f'  <circle cx="{cx}" cy="{cy}" r="{r_hub1}" fill="#0d1b2a" stroke="#a0c4e0" stroke-width="1.5"/>\n'
        f'  <circle cx="{cx}" cy="{cy}" r="{r_hub2}" fill="#e8f4ff"/>\n'
        f'  <text x="{ll_x:.1f}" y="{ll_y:.1f}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_label}" font-weight="700" fill="#c0deff">LOW</text>\n'
        f'  <text x="{ln_x:.1f}" y="{ln_y:.1f}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_label}" font-weight="700" fill="#c0deff">NORMAL</text>\n'
        f'  <text x="{lh_x:.1f}" y="{lh_y:.1f}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_label}" font-weight="700" fill="#c0deff">HIGH</text>\n'
        f'  <text x="{cx}" y="{cy + int(44*scale)}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_score}" font-weight="700" fill="{sc_col}">{int(score)}</text>\n'
        f'  <text x="{cx}" y="{cy + int(62*scale)}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_sub}" fill="#5a8ab0">OUT OF 100</text>\n'
        '  </g>'
        '</svg></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC Video Processor
# ─────────────────────────────────────────────────────────────────────────────

class VideoTransformer(VideoProcessorBase):
    """
    Processes each video frame from the browser:
      - Runs MediaPipe FaceLandmarker
      - Stores the latest analysis result and annotated frame
      - Returns the annotated frame back to the browser video element
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.result: dict | None = None
        self.last_frame: np.ndarray | None = None  # BGR
        self.show_overlay: bool = True
        # Load the landmarker once per processor instance
        self._landmarker = load_face_landmarker()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert incoming frame to BGR numpy array
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror

        rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._landmarker.detect(mp_image)

        if detection.face_landmarks:
            landmarks = detection.face_landmarks[0]
            h, w = img.shape[:2]
            analysis = analyse_face(landmarks, w, h)

            if self.show_overlay:
                img = draw_landmarks(img, landmarks)

            level  = analysis["cortisol_level"]
            colour = cortisol_colour(level)
            label  = f"Cortisol: {level}  ({analysis['cortisol_score']:.0f}/100)"
            cv2.rectangle(img, (8, 8), (300, 36), (0, 0, 0), -1)
            cv2.putText(img, label, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)

            with self._lock:
                self.result     = analysis
                self.last_frame = img.copy()  # BGR
        else:
            cv2.putText(img, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 160), 2)
            with self._lock:
                self.last_frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    @property
    def latest_result(self) -> dict | None:
        with self._lock:
            return self.result

    @property
    def latest_frame_rgb(self) -> np.ndarray | None:
        with self._lock:
            f = self.last_frame
            if f is None:
                return None
            return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "last_analysis": None,
        "snapshot":      None,
        "snap_analysis": None,
        "show_overlay":  True,
        "tips":          get_tips(4),
        "history":       [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ──
st.markdown("""
<div style="text-align:center; padding: 10px 0 4px;">
  <div style="font-size:2.4rem; margin-bottom:4px;">🧠</div>
  <h1 style="font-size:1.8rem; font-weight:700; color:#c8e6ff; margin:0; letter-spacing:-0.5px;">
    Cortisol Meter
  </h1>
  <p style="color:#5a8ab0; font-size:0.88rem; margin:6px 0 0; letter-spacing:0.3px;">
    Real-time stress estimation from facial analysis · Powered by MediaPipe
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:14px 0 20px;'>",
            unsafe_allow_html=True)

# ── Two-column layout ──
col_cam, col_panel = st.columns([1, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — WebRTC camera + controls
# ─────────────────────────────────────────────────────────────────────────────

with col_cam:
    st.markdown('<div class="section-title">Live Camera Feed</div>', unsafe_allow_html=True)

    # Overlay toggle (before streamer so the processor picks it up)
    overlay_toggle = st.toggle("Show Landmarks", value=st.session_state.show_overlay, key="overlay_toggle")
    st.session_state.show_overlay = overlay_toggle

    # ── WebRTC streamer ───────────────────────────────────────────────────────
    ctx = webrtc_streamer(
        key="cortisol-cam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.stunprotocol.org:3478"]},
                {"urls": ["stun:stun.qq.com:3478"]},
                {"urls": ["stun:stun.cloudflare.com:3478"]},
                {"urls": ["stun:stun.miwifi.com:3478"]}
            ]
        },
    )

    # Keep processor overlay setting in sync
    if ctx.video_processor:
        ctx.video_processor.show_overlay = st.session_state.show_overlay

    # ── Snapshot button ───────────────────────────────────────────────────────
    snap_btn = st.button(
        "📸 Snapshot",
        use_container_width=True,
        disabled=(ctx.video_processor is None),
    )

    if snap_btn and ctx.video_processor:
        frame_rgb = ctx.video_processor.latest_frame_rgb
        analysis  = ctx.video_processor.latest_result
        if frame_rgb is not None:
            st.session_state.snapshot      = frame_rgb
            st.session_state.snap_analysis = analysis
            st.rerun()

    # ── Snapshot display ──────────────────────────────────────────────────────
    if st.session_state.snapshot is not None:
        st.markdown("---")
        st.markdown('<div class="section-title">📸 Snapshot Analysis</div>', unsafe_allow_html=True)

        sa = st.session_state.snap_analysis
        if sa:
            level = sa["cortisol_level"]
            bc    = badge_class(level)
            score = sa["cortisol_score"]
            conf  = sa["confidence"]

            snap_c1, snap_c2 = st.columns([3, 2])
            with snap_c1:
                st.image(st.session_state.snapshot, use_container_width=True)

                pil_img = Image.fromarray(st.session_state.snapshot)
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                st.download_button(
                    "⬇ Download Snapshot",
                    data=buf.getvalue(),
                    file_name="cortisol_snapshot.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with snap_c2:
                st.markdown(
                    f'<div class="card" style="padding:12px 8px;">'
                    f'{gauge_html(score, level, width=260)}'
                    f'<div style="text-align:center;margin-top:6px;">'
                    f'<span class="badge {bc}">{level} Cortisol</span>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(
                        f'<div class="metric-box"><div class="metric-label">Score</div>'
                        f'<div class="metric-value">{score:.0f}<span style="font-size:0.7rem;color:#5a8ab0;">/100</span></div></div>',
                        unsafe_allow_html=True
                    )
                with m2:
                    st.markdown(
                        f'<div class="metric-box"><div class="metric-label">Confidence</div>'
                        f'<div class="metric-value">{conf:.0f}<span style="font-size:0.7rem;color:#5a8ab0;">%</span></div></div>',
                        unsafe_allow_html=True
                    )


        else:
            st.info("No face detected in snapshot.")


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — live analysis panel (reads from WebRTC processor)
# ─────────────────────────────────────────────────────────────────────────────

with col_panel:
    # Pull the latest result from the WebRTC processor (thread-safe)
    if ctx.video_processor:
        live = ctx.video_processor.latest_result
        if live:
            st.session_state.last_analysis = live
            st.session_state.history.append(live["cortisol_score"])
            if len(st.session_state.history) > 60:
                st.session_state.history.pop(0)

    analysis = st.session_state.last_analysis

    # ── Cortisol Gauge ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Cortisol Level</div>', unsafe_allow_html=True)

    if analysis:
        level  = analysis["cortisol_level"]
        score  = analysis["cortisol_score"]
        conf   = analysis["confidence"]
        bc     = badge_class(level)

        st.markdown(
            f'<div class="card">'
            f'{gauge_html(score, level)}'
            f'<div style="text-align:center; margin-top:4px;">'
            f'<span class="badge {bc}">{level} Cortisol</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Metrics row ───────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Score</div>'
                f'<div class="metric-value">{score:.0f}<span style="font-size:0.7rem;color:#5a8ab0;">/100</span></div></div>',
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Confidence</div>'
                f'<div class="metric-value">{conf:.0f}<span style="font-size:0.7rem;color:#5a8ab0;">%</span></div></div>',
                unsafe_allow_html=True
            )
        with m3:
            ear = analysis["ear_avg"]
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Eye Ratio</div>'
                f'<div class="metric-value">{ear:.3f}</div></div>',
                unsafe_allow_html=True
            )

        # ── Feature detail bars ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Feature Detail</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.78rem;color:#5a8ab0;margin-bottom:4px;">Brow Tension</div>',
            unsafe_allow_html=True
        )
        st.progress(min(analysis["brow_tension"], 1.0))
        st.markdown(
            '<div style="font-size:0.78rem;color:#5a8ab0;margin-bottom:4px;">Mouth Openness</div>',
            unsafe_allow_html=True
        )
        st.progress(min(analysis["mouth_openness"], 1.0))

    else:
        st.markdown("""
<div class="card" style="text-align:center; padding:32px 20px;">
  <div style="font-size:2.5rem; margin-bottom:10px;">🔍</div>
  <div style="color:#4a6a88; font-size:0.88rem;">
    Start the camera and<br>position your face in the frame<br>to see your cortisol estimate.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Interventions panel ───────────────────────────────────────────────────
    if analysis and analysis["cortisol_level"] == "High":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚠️ High Stress — Let\'s Help</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="card-high">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:600;color:#f87171;margin-bottom:4px;">🫁 Box Breathing — Follow the Circle</div>',
            unsafe_allow_html=True
        )
        components.html(breathing_animation_html(phase_seconds=4), height=240)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">💡 Stress-Reduction Tips</div>', unsafe_allow_html=True)
        tips = st.session_state.tips
        tips_html = ""
        for icon, title, desc in tips:
            tips_html += (
                f'<div class="tip-card">'
                f'<div class="tip-icon">{icon}</div>'
                f'<div><div class="tip-title">{title}</div>'
                f'<div class="tip-desc">{desc}</div></div></div>'
            )
        st.markdown(tips_html, unsafe_allow_html=True)

        if st.button("🔄 Refresh Tips", key="refresh_high"):
            st.session_state.tips = get_tips(4)
            st.rerun()

    elif analysis and analysis["cortisol_level"] == "Moderate":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Elevated Stress — Time to Unwind</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:600;color:#a8d8ff;margin-bottom:4px;">🫁 Try Box Breathing</div>',
            unsafe_allow_html=True
        )
        components.html(breathing_animation_html(phase_seconds=4), height=240)
        st.markdown('</div>', unsafe_allow_html=True)

        tips = st.session_state.tips
        tips_html = ""
        for icon, title, desc in tips:
            tips_html += (
                f'<div class="tip-card">'
                f'<div class="tip-icon">{icon}</div>'
                f'<div><div class="tip-title">{title}</div>'
                f'<div class="tip-desc">{desc}</div></div></div>'
            )
        st.markdown(tips_html, unsafe_allow_html=True)

        if st.button("🔄 Refresh Tips", key="refresh_moderate"):
            st.session_state.tips = get_tips(4)
            st.rerun()

    elif analysis and analysis["cortisol_level"] == "Low":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
<div class="card">
  <div style="font-size:0.85rem;font-weight:600;color:#4ade80;margin-bottom:8px;">✅ You Look Relaxed!</div>
  <div style="font-size:0.82rem;color:#8eb8d8;line-height:1.6;">
    Your cortisol appears low — great job staying calm 🌿<br>
    Keep it up with good sleep, hydration, and regular breaks.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Disclaimer:</strong> This application is a research &amp; wellness prototype only.
  It is <strong>not a medical device</strong> and does not constitute medical advice.
  Cortisol estimates are derived from visual facial cues and are approximate.
  Do not use this tool to diagnose or treat any medical condition.
  If you are experiencing persistent stress or anxiety, please consult a healthcare professional.
</div>
""", unsafe_allow_html=True)
