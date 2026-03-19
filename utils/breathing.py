"""
utils/breathing.py
------------------
Generates an HTML/JS/CSS breathing animation component
that can be injected into Streamlit via st.components.v1.html().
"""


def breathing_animation_html(phase_seconds: int = 4) -> str:
    """
    Return a self-contained HTML string with a CSS breathing circle animation.
    
    Args:
        phase_seconds: Duration of each breath phase (inhale / hold / exhale / hold)
    """
    total = phase_seconds * 4  # total cycle in seconds

    return f"""
<div style="
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 16px 8px;
    font-family: 'Segoe UI', sans-serif;
    width: 100%;
    box-sizing: border-box;
">
  <p style="color:#a0b8d8; font-size:0.9rem; margin-bottom:14px; letter-spacing:0.5px; text-align:center;">
    Follow the circle &mdash; breathe naturally
  </p>

  <div style="position:relative; width:min(140px, 55vw); height:min(140px, 55vw); margin-bottom:16px; flex-shrink:0;">
    <!-- Outer pulsing ring -->
    <div id="outerRing" style="
      position:absolute; inset:0;
      border-radius:50%;
      background: radial-gradient(circle, rgba(100,180,255,0.12), rgba(60,130,220,0.04));
      border: 2px solid rgba(100,180,255,0.25);
      animation: breatheOuter {total}s ease-in-out infinite;
    "></div>
    <!-- Inner circle -->
    <div id="innerCircle" style="
      position:absolute; inset:18px;
      border-radius:50%;
      background: radial-gradient(circle at 40% 35%, rgba(130,200,255,0.9), rgba(60,120,220,0.7));
      box-shadow: 0 0 30px rgba(80,160,255,0.5), 0 0 60px rgba(80,160,255,0.2);
      animation: breatheInner {total}s ease-in-out infinite;
    "></div>
    <!-- Phase label in the centre -->
    <div style="
      position:absolute; inset:0;
      display:flex; align-items:center; justify-content:center;
      z-index:10;
    ">
      <span id="phaseLabel" style="
        color:#e8f4ff; font-size:0.78rem; font-weight:600;
        letter-spacing:1px; text-transform:uppercase;
        text-shadow: 0 1px 4px rgba(0,0,0,0.5);
      ">Inhale</span>
    </div>
  </div>

  <!-- Timer bar -->
  <div style="width:min(160px, 70%); height:4px; background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden;">
    <div id="timerBar" style="
      height:100%; width:0%;
      background: linear-gradient(90deg, #5ab4ff, #a0d8ff);
      border-radius:2px;
      animation: timerAnim {total}s linear infinite;
    "></div>
  </div>
  <p id="countLabel" style="color:#6a9bbf; font-size:0.8rem; margin-top:10px;">0 / {phase_seconds}s</p>
</div>

<style>
@keyframes breatheInner {{
  0%                           {{ transform: scale(0.85); opacity:0.8; }}
  {round(1/total*100,1)}%      {{ transform: scale(0.85); opacity:0.8; }}   /* start inhale */
  {round((phase_seconds)/total*100,1)}%  {{ transform: scale(1.15); opacity:1.0; }}  /* end inhale */
  {round((phase_seconds*2)/total*100,1)}% {{ transform: scale(1.15); opacity:1.0; }}  /* hold */
  {round((phase_seconds*3)/total*100,1)}% {{ transform: scale(0.85); opacity:0.8; }}  /* end exhale */
  100%                         {{ transform: scale(0.85); opacity:0.8; }}   /* hold */
}}
@keyframes breatheOuter {{
  0%                            {{ transform: scale(0.9);  opacity:0.4; }}
  {round((phase_seconds)/total*100,1)}%  {{ transform: scale(1.2);  opacity:0.8; }}
  {round((phase_seconds*2)/total*100,1)}% {{ transform: scale(1.2);  opacity:0.8; }}
  {round((phase_seconds*3)/total*100,1)}% {{ transform: scale(0.9);  opacity:0.4; }}
  100%                          {{ transform: scale(0.9);  opacity:0.4; }}
}}
@keyframes timerAnim {{
  0%   {{ width: 0%; }}
  100% {{ width: 100%; }}
}}
</style>

<script>
(function() {{
  const phases = ['Inhale', 'Hold', 'Exhale', 'Hold'];
  const phaseSec = {phase_seconds};
  const total = phaseSec * 4;
  let elapsed = 0;

  function tick() {{
    const phaseIdx = Math.floor(elapsed / phaseSec) % 4;
    const withinPhase = elapsed % phaseSec;

    const label = document.getElementById('phaseLabel');
    const count = document.getElementById('countLabel');
    if (label) label.textContent = phases[phaseIdx];
    if (count) count.textContent = withinPhase + ' / ' + phaseSec + 's';

    elapsed = (elapsed + 1) % total;
    setTimeout(tick, 1000);
  }}
  tick();
}})();
</script>
"""
