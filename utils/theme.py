import math

def get_theme_css(is_dark: bool) -> str:
    if is_dark:
        vars_css = """
        :root {
            --bg: linear-gradient(135deg, #0d1117 0%, #0d1b2a 50%, #0d2137 100%);
            --text-main: #e0eeff;
            --text-muted: #5a8ab0;
            --card-bg: rgba(255,255,255,0.04);
            --card-border: rgba(100,160,255,0.15);
            --card-high-bg: rgba(255,80,80,0.07);
            --card-high-border: rgba(255,100,100,0.30);
            --badge-low-bg: rgba(60,200,120,0.2); --badge-low-color: #4ade80; --badge-low-border: rgba(60,200,120,0.4);
            --badge-mod-bg: rgba(250,190,50,0.2); --badge-mod-color: #fbbf24; --badge-mod-border: rgba(250,190,50,0.4);
            --badge-high-bg: rgba(255,80,80,0.2); --badge-high-color: #f87171; --badge-high-border: rgba(255,80,80,0.4);
            --tip-bg: rgba(60,120,200,0.12);
            --tip-border: rgba(80,150,255,0.2);
            --tip-title: #a8d8ff;
            --tip-desc: #8eb8d8;
            --metric-bg: rgba(255,255,255,0.03);
            --metric-border: rgba(100,160,255,0.1);
            --metric-val: #c0deff;
            --disc-border: rgba(255,255,255,0.06);
            --disc-color: #4a6a88;
            --gauge-div: #0d1b2a;
        }
        """
    else:
        vars_css = """
        :root {
            --bg: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --card-bg: rgba(255,255,255,0.6);
            --card-border: rgba(148,163,184,0.4);
            --card-high-bg: rgba(254,226,226,0.6);
            --card-high-border: rgba(248,113,113,0.5);
            --badge-low-bg: rgba(34,197,94,0.15); --badge-low-color: #166534; --badge-low-border: rgba(34,197,94,0.3);
            --badge-mod-bg: rgba(234,179,8,0.15); --badge-mod-color: #854d0e; --badge-mod-border: rgba(234,179,8,0.3);
            --badge-high-bg: rgba(239,68,68,0.15); --badge-high-color: #991b1b; --badge-high-border: rgba(239,68,68,0.3);
            --tip-bg: rgba(224,242,254,0.6);
            --tip-border: rgba(186,230,253,0.8);
            --tip-title: #0369a1;
            --tip-desc: #0ea5e9;
            --metric-bg: rgba(255,255,255,0.7);
            --metric-border: rgba(148,163,184,0.3);
            --metric-val: #0f172a;
            --disc-border: rgba(15,23,42,0.1);
            --disc-color: #64748b;
            --gauge-div: #94a3b8;
        }
        """

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
{vars_css}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; box-sizing: border-box; }}
*, *::before, *::after {{ box-sizing: inherit; }}
.stApp {{ background: var(--bg); color: var(--text-main); min-height: 100vh; transition: background 0.3s; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem !important; padding-bottom: 2rem !important; padding-left: 1.5rem !important; padding-right: 1.5rem !important; max-width: 100% !important; }}
[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 1rem; align-items: flex-start; }}
[data-testid="stHorizontalBlock"] > [data-testid="column"] {{ min-width: 280px; flex: 1 1 300px; }}
[data-testid="stImage"] img, .stImage img {{ max-width: 100% !important; height: auto !important; border-radius: 12px; }}
video {{ border-radius: 12px; max-width: 100% !important; }}
.card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; padding: 20px 24px; margin-bottom: 16px; backdrop-filter: blur(8px); width: 100%; transition: all 0.3s; }}
.card-high {{ background: var(--card-high-bg); border: 1px solid var(--card-high-border); border-radius: 16px; padding: 20px 24px; margin-bottom: 16px; width: 100%; transition: all 0.3s; }}
.badge {{ display: inline-block; padding: 6px 18px; margin: 15px auto; border-radius: 40px; font-size: 0.85rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; white-space: nowrap; }}
.badge-low {{ background: var(--badge-low-bg); color: var(--badge-low-color); border: 1px solid var(--badge-low-border); }}
.badge-moderate {{ background: var(--badge-mod-bg); color: var(--badge-mod-color); border: 1px solid var(--badge-mod-border); }}
.badge-high {{ background: var(--badge-high-bg); color: var(--badge-high-color); border: 1px solid var(--badge-high-border); }}
.gauge-wrap {{ text-align: center; padding: 8px 0; width: 100%; overflow: hidden; }}
.gauge-wrap svg {{ max-width: 100%; height: auto; }}
.tip-card {{ background: var(--tip-bg); border: 1px solid var(--tip-border); border-radius: 12px; padding: 12px 16px; margin-bottom: 10px; display: flex; gap: 12px; align-items: flex-start; width: 100%; transition: all 0.3s; }}
.tip-icon {{ font-size: 1.4rem; line-height:1; margin-top:2px; flex-shrink: 0; }}
.tip-title {{ font-weight:600; font-size:0.9rem; color: var(--tip-title); margin-bottom:4px; transition: color 0.3s; }}
.tip-desc  {{ font-size:0.82rem; color: var(--tip-desc); line-height:1.45; transition: color 0.3s; }}
.section-title {{ font-size: 0.75rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px; transition: color 0.3s; }}
.disclaimer {{ font-size: 0.72rem; color: var(--disc-color); border-top: 1px solid var(--disc-border); padding-top: 12px; margin-top: 8px; line-height: 1.6; width: 100%; transition: all 0.3s; }}
.metric-box {{ background: var(--metric-bg); border: 1px solid var(--metric-border); border-radius: 10px; padding: 10px 14px; text-align: center; width: 100%; transition: all 0.3s; }}
.metric-label {{ font-size:0.7rem; color: var(--text-muted); text-transform:uppercase; letter-spacing:0.8px; transition: color 0.3s; }}
.metric-value {{ font-size:1.1rem; font-weight:600; color: var(--metric-val); margin-top:2px; transition: color 0.3s; }}
.stButton > button {{ background: linear-gradient(135deg, #1e5fa8, #2a7fd4); color: white; border: none; border-radius: 10px; padding: 10px 22px; font-family: 'Inter', sans-serif; font-weight: 500; transition: all 0.2s; width: 100%; white-space: nowrap; }}
.stButton > button:hover {{ background: linear-gradient(135deg, #2a7fd4, #3a9fe4); transform: translateY(-1px); box-shadow: 0 4px 15px rgba(42,127,212,0.35); }}
.stProgress > div > div > div {{ border-radius: 8px; }}
@media (max-width: 1050px) {{
    .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{ min-width: 100% !important; flex: 1 1 100% !important; }}
    .section-title {{ font-size: 0.7rem; }}
    h1 {{ font-size: 1.5rem !important; }}
}}
@media (max-width: 600px) {{
    .block-container {{ padding-left: 0.5rem !important; padding-right: 0.5rem !important; padding-top: 1rem !important; }}
    .card, .card-high {{ padding: 14px 16px; border-radius: 12px; }}
    .tip-card {{ padding: 10px 12px; gap: 8px; }}
    .tip-title {{ font-size: 0.85rem; }}
    .tip-desc  {{ font-size: 0.78rem; }}
    .metric-value {{ font-size: 1rem; }}
    .badge {{ font-size: 0.78rem; padding: 5px 14px; }}
    h1 {{ font-size: 1.3rem !important; }}
}}
</style>
"""

def get_gauge_html(score: float, level: str, is_dark: bool, width: int = 300) -> str:
    """
    SVG speedometer gauge: five colour bands (dark-green → lime → yellow
    → orange-red → dark-red) with a pivoting needle.
    """
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
                 f'stroke="var(--gauge-div)" stroke-width="{sw}"/>\n')

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
    
    # Theme colors
    needle_outer = "#1a1a2e" if is_dark else "#334155"
    needle_inner = "#e8f4ff" if is_dark else "#f1f5f9"
    hub_outer = "#0d1b2a" if is_dark else "#e2e8f0"
    hub_stroke = "#a0c4e0" if is_dark else "#94a3b8"
    hub_inner = "#e8f4ff" if is_dark else "#ffffff"
    text_color = "#c0deff" if is_dark else "#0f172a"
    subtext_color = "#5a8ab0" if is_dark else "#64748b"

    return (
        '<div class="gauge-wrap">'
        f'<svg width="100%" height="auto" '
        f'viewBox="0 0 {vb_w} {vb_h}" preserveAspectRatio="xMidYMid meet" style="overflow:visible; display:block;">'
        f'  <text x="{cx}" y="{title_y}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{title_fs}" font-weight="700"'
        f' letter-spacing="2" fill="{text_color}">CORTISOL LEVEL</text>\n'
        f'  <g transform="translate(0,{title_y + 4})">'
        f'\n{segs}{divs}'
        f'  <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}"'
        f' stroke="{needle_outer}" stroke-width="{nw + 2}" stroke-linecap="round"/>'
        f'  <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}"'
        f' stroke="{needle_inner}" stroke-width="{nw}" stroke-linecap="round"/>\n'
        f'  <circle cx="{cx}" cy="{cy}" r="{r_hub1}" fill="{hub_outer}" stroke="{hub_stroke}" stroke-width="1.5"/>\n'
        f'  <circle cx="{cx}" cy="{cy}" r="{r_hub2}" fill="{hub_inner}"/>\n'
        f'  <text x="{ll_x:.1f}" y="{ll_y:.1f}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_label}" font-weight="700" fill="{text_color}">LOW</text>\n'
        f'  <text x="{ln_x:.1f}" y="{ln_y:.1f}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_label}" font-weight="700" fill="{text_color}">NORMAL</text>\n'
        f'  <text x="{lh_x:.1f}" y="{lh_y:.1f}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_label}" font-weight="700" fill="{text_color}">HIGH</text>\n'
        f'  <text x="{cx}" y="{cy + int(44*scale)}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_score}" font-weight="700" fill="{sc_col}">{score:.0f}</text>\n'
        f'  <text x="{cx}" y="{cy + int(62*scale)}" text-anchor="middle"'
        f' font-family="Inter,sans-serif" font-size="{fs_sub}" fill="{subtext_color}">OUT OF 100</text>\n'
        '  </g>'
        '</svg></div>'
    )
