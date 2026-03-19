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
# Cortisol-Meter
