import os
import streamlit as st
from google import genai
from utils.analysis import get_tips as get_fallback_tips

def get_ai_tips(analysis: dict, n: int = 4):
    """
    Generates personalized, high-quality stress-reduction tips using Google Gemini AI
    based on the exact facial analysis features.
    Falls back to normal static tips if no API key is set.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    
    if not api_key:
        return [("🔑", "API Key Missing", "Please add GEMINI_API_KEY to your Streamlit Secrets.")] + get_fallback_tips(n - 1)

    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    The user is currently exhibiting a {analysis['cortisol_level']} stress level, 
    with a cortisol score of {analysis['cortisol_score']}/100 based on facial analysis.
    Their specific facial metrics are:
    - Eye ratio (openness): {analysis['ear_avg']}
    - Eyebrow tension: {analysis['brow_tension'] * 100:.0f}%
    - Mouth openness tension: {analysis['mouth_openness'] * 100:.0f}%
    
    Based on these specific tensions (e.g. if brow tension is high, suggest relaxing the forehead; if eye ratio is strained, suggest resting the eyes), 
    generate {n} very concise, actionable, and calming stress-reduction tips. 
    
    Format EXACTLY as {n} lines. No markdown lists, no asterisks, no prefixes.
    Each line MUST strictly follow this exact format, separated by a single '|' character:
    <a single appropriate emoji>|<a short 2-3 word title>|<a 1-sentence description>
    
    Example output format:
    🫁|Box Breathing|Inhale for 4 seconds, hold for 4, and exhale for 4 to lower your heart rate.
    💆|Relax Your Brow|Gently massage your forehead to release the tension buildup.
    """

    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.7)
        )
        text = response.text.strip()
        
        tips = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                tips.append((parts[0], parts[1], parts[2]))
        
        if len(tips) < n:
            # Fallback for remaining if AI didn't format perfectly 
            tips.extend(get_fallback_tips(n - len(tips)))
            
        return tips[:n]
    except Exception as e:
        print(f"Failed to generate AI tips: {e}")
        error_msg = str(e)
        if len(error_msg) > 60:
            error_msg = error_msg[:57] + "..."
        return [("⚠️", "AI Error", f"Failed: {error_msg}")] + get_fallback_tips(n - 1)
