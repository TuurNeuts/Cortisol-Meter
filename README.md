# 🧠 Cortisol Meter

A Streamlit web app that estimates your stress (cortisol) level in real time using facial analysis via your webcam and MediaPipe Face Mesh.

> ⚠️ **Disclaimer:** This is a research & wellness prototype, not a medical device.

## Features

- 📷 Live webcam feed with facial landmark overlay
- 🧮 Real-time cortisol score (0–100) using Eye Aspect Ratio, brow tension, and mouth shape
- 📊 SVG arc gauge + confidence score
- 🫁 Guided box breathing animation (triggered on high cortisol)
- 💡 Randomised stress-reduction tips
- 📸 Snapshot + download

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## Project Structure

```
cortisol meter/
├── app.py                # Main Streamlit application
├── requirements.txt
└── utils/
    ├── analysis.py       # Facial feature extraction & cortisol scoring
    └── breathing.py      # Breathing animation HTML component
```

## How It Works

| Feature | Landmark Indices | Signal |
|---|---|---|
| Eye openness (EAR) | Both eye contours | Deviation from baseline → stress |
| Brow tension | Eyebrow midpoints vs eye tops | Smaller gap → furrowing → stress |
| Mouth shape | Lips + corners | Tight closed mouth → stress |
| Head asymmetry | Nose tip vs eye midpoint | Minor modifier |

These four signals are weighted and combined into a 0–100 **Cortisol Score**, then classified as **Low / Moderate / High**.

## Cloud Deployment

This app uses `streamlit-webrtc` so it can be deployed to remote servers (where `cv2.VideoCapture` would fail). 

**To deploy for free on Streamlit Cloud:**
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New App**, point it to your repository, branch, and `app.py`.
4. Click **Deploy**.
5. Streamlit Cloud will automatically install the system dependencies from `packages.txt` and python packages from `requirements.txt`.
6. When opening the app URL, the browser will ask for Camera Permissions.

*(Note: The `face_landmarker.task` model file is excluded from Git to save space; the app will automatically download it on the first launch).*
