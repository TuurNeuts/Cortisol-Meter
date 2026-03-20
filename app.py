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
import collections
from io import BytesIO
from pathlib import Path
import os

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
from utils.ai_tips import get_ai_tips
from utils.breathing import breathing_animation_html
from utils.theme import get_theme_css, get_gauge_html

# ─────────────────────────────────────────────────────────────────────────────
# Network & ICE Configuration
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # cache for 1 hour so we don't spam Twilio's API
def get_ice_servers():
    """Use Twilio's TURN server to fall back in case STUN is blocked by firewalls."""
    try:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", st.secrets.get("TWILIO_ACCOUNT_SID"))
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", st.secrets.get("TWILIO_AUTH_TOKEN"))
    except Exception:
        account_sid = None
        auth_token = None

    if account_sid and auth_token:
        try:
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        except Exception as e:
            print(f"Failed to get Twilio ICE servers: {e}")
            
    # Default fallback to Google STUN
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cortisol Meter",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme Toggle
if "is_dark" not in st.session_state:
    st.session_state.is_dark = True

def toggle_theme():
    st.session_state.is_dark = not st.session_state.is_dark

is_dark = st.session_state.is_dark

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dynamic light/dark modern design
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(get_theme_css(is_dark), unsafe_allow_html=True)


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
        # Smooth the rapidly changing scores over the last 15 frames (~0.5s)
        self._score_history = collections.deque(maxlen=15)

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

            # Smooth the raw score internally per-frame
            self._score_history.append(analysis["cortisol_score"])
            smoothed_score = sum(self._score_history) / len(self._score_history)
            
            # Re-apply the smoothed score and recalculate the stable level
            analysis["cortisol_score"] = float(smoothed_score)
            if smoothed_score < 30:
                level = "Low"
            elif smoothed_score < 55:
                level = "Moderate"
            else:
                level = "High"
            analysis["cortisol_level"] = level

            if self.show_overlay:
                img = draw_landmarks(img, landmarks)

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
        "getting_ai":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ──
h_col1, h_col2, h_col3 = st.columns([1, 6, 1])

with h_col3:
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.button("☀️ Light" if is_dark else "🌙 Dark", on_click=toggle_theme, use_container_width=True)

with h_col2:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 4px;">
      <div style="font-size:2.4rem; margin-bottom:4px;">🧠</div>
      <h1 style="font-size:1.8rem; font-weight:700; color:var(--text-main); margin:0; letter-spacing:-0.5px;">
        Cortisol Meter
      </h1>
      <p style="color:var(--text-muted); font-size:0.88rem; margin:6px 0 0; letter-spacing:0.3px;">
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
            "iceServers": get_ice_servers()
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
                import time
                st.markdown(
                    f'<div id="gauge-snap-{time.time()}" class="card" style="padding:12px 8px;">'
                    f'{get_gauge_html(score, level, is_dark, width=260)}'
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

        import time
        st.markdown(
            f'<div id="gauge-live-{time.time()}" class="card">'
            f'{get_gauge_html(score, level, is_dark)}'
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
        
        if st.button("✨ Get Personalised AI Tips", key="refresh_high"):
            with st.spinner("🤖 AI is reading your facial tension..."):
                st.session_state.tips = get_ai_tips(analysis, 4)

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

        if st.button("✨ Get Personalised AI Tips", key="refresh_moderate"):
            with st.spinner("🤖 AI is reading your facial tension..."):
                st.session_state.tips = get_ai_tips(analysis, 4)

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
