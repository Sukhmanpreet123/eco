"""
EcoTrace v3 — Professional Dark Dashboard (v4.0)
Pure frontend redesign: dark navy theme, glassmorphism cards, pill tabs.
All backend logic unchanged.
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.graph_objects as go

RENDER_URL    = "https://eco-2-4re9.onrender.com"
GRID_FALLBACK = 475.0

GRADE_COLOURS = {"A":"#10b981","B":"#34d399","C":"#f59e0b","D":"#f97316","F":"#ef4444"}

st.set_page_config(page_title="EcoTrace v3", layout="wide", page_icon="🌱")

# ── Global dark theme CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
}
[data-testid="stHeader"] { background: #0f172a !important; border-bottom: 1px solid #1e293b; }
[data-testid="stToolbar"] { display: none !important; }

/* Hide Streamlit footer / "Learn more" branding */
footer, #MainMenu, [data-testid="stDecoration"] { display: none !important; }
a[href*="streamlit.io"] { display: none !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1526 !important;
    border-right: 1px solid #1e293b !important;
    min-width: 288px !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

/* Sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* Sidebar slider */
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #10b981 !important;
}

/* ── Tabs (Mockup Design) ── */
[data-testid="stTabs"] {
    padding-top: 5px !important;
}
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border: none !important;
    padding: 0 0 10px 0 !important;
    gap: 8px !important;
    margin-bottom: 20px !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 100px !important;
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    min-height: 38px !important;
    margin-top: 5px !important; /* Forces it down from clipping */
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #f8fafc !important; /* White pill for active tab like mockup */
    border-color: #f8fafc !important;
    color: #020617 !important; /* Very dark text */
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #f1f5f9 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]:hover {
    color: #020617 !important;
}
/* Hide tab underline bar */
[data-testid="stTabs"] [role="tabpanel"] { border: none !important; }
[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] div[data-baseweb="tab-border"]    { display: none !important; }

/* ── Main content area ── */
[data-testid="stAppViewContainer"] > section.main {
    background: #0f172a !important;
}
.block-container { padding-top: 1.5rem !important; }

/* ── Streamlit widgets in main ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #94a3b8 !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: #059669 !important; /* Darker green for perfect contrast with white text */
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    transition: background 0.2s ease !important;
}
[data-testid="stButton"] > button:hover { background: #047857 !important; }

/* ── Streamlit metrics ── */
[data-testid="stMetric"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 24px !important; }

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] { background: #1e293b !important; border-radius: 12px !important; }
[data-testid="stDataFrame"] thead th {
    background: #0f172a !important;
    color: #10b981 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    border-bottom: 1px solid #334155 !important;
}
[data-testid="stDataFrame"] tbody tr { background: #1e293b !important; }
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: #162032 !important; }
[data-testid="stDataFrame"] tbody td { color: #cbd5e1 !important; font-size: 12px !important; }

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    background: #1e293b !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}
.stSuccess { border-left-color: #10b981 !important; }
.stInfo    { border-left-color: #3b82f6 !important; }
.stWarning { border-left-color: #f59e0b !important; }
.stError   { border-left-color: #ef4444 !important; }

/* ── Form submit button ── */
[data-testid="stFormSubmitButton"] > button {
    background: #059669 !important; /* Darker green */
    color: #fff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
[data-testid="stFormSubmitButton"] > button:hover { background: #047857 !important; }

/* ── Headings ── */
h1,h2,h3 { color: #f1f5f9 !important; font-family: 'Inter', sans-serif !important; }
h1 { font-size: 26px !important; font-weight: 700 !important; }
h2 { font-size: 20px !important; font-weight: 600 !important; }
h3 { font-size: 16px !important; font-weight: 600 !important; }
p, caption, small { color: #94a3b8 !important; }

/* ── Divider ── */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)


# ── API helper ────────────────────────────────────────────────
def api(path, params=None, method="GET", json_body=None, timeout=6):
    try:
        url = f"{RENDER_URL}{path}"
        if method == "POST":
            r = requests.post(url, params=params, json=json_body, timeout=timeout)
        else:
            r = requests.get(url, params=params, timeout=timeout)
        return r.json()
    except Exception:
        return {}


def fmt_val(val, unit="", decimals=2):
    if val is None: return "—"
    try:
        v = float(val)
        if v == 0: return f"0 {unit}".strip()
        if abs(v) < 0.01: return f"<0.01 {unit}".strip()
        return f"{round(v, decimals)} {unit}".strip()
    except Exception:
        return str(val)


# ── Session state ─────────────────────────────────────────────
if "last_device" not in st.session_state:
    st.session_state.last_device = None


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    # Logo header
    st.markdown("""
    <div style="padding:16px 0 8px; border-bottom:1px solid #1e293b; margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:28px;">🌱</span>
        <div>
          <div style="font-size:18px;font-weight:700;color:#f1f5f9;">EcoTrace</div>
          <div style="font-size:11px;color:#10b981;font-weight:600;letter-spacing:1px;">v3 · AI CARBON GOVERNANCE</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    device_res  = api("/active_devices")
    active_list = device_res.get("devices", []) or ["no-active-sessions"]
    st.markdown("<div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>MONITORING TARGET</div>", unsafe_allow_html=True)
    target = st.selectbox("", active_list, label_visibility="collapsed")

    if st.session_state.last_device != target:
        st.session_state.last_device = target

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Budget settings
    st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:8px;'>⚙️ BUDGET SETTINGS</div>", unsafe_allow_html=True)
    BUDGET_G = st.slider("Carbon budget (g CO₂)", 10, 500, 100, step=10)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Pre-run estimator
    st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:8px;'>🔬 PRE-RUN ESTIMATOR</div>", unsafe_allow_html=True)
    est_task   = st.text_input("Task type",   "image_classification")
    est_model  = st.text_input("Model name",  "ResNet-50")
    est_epochs = st.number_input("Epochs",    1, 500, 25)
    est_batch  = st.number_input("Batch size",1, 512, 32)

    if st.button("Estimate CO₂ ↗", use_container_width=True):
        est = api("/estimate", params={
            "task_type": est_task, "model_name": est_model,
            "epochs": est_epochs, "batch_size": est_batch})
        if est.get("similar_count", 0) > 0:
            st.success(
                f"Based on {est['similar_count']} similar runs:\n"
                f"CO₂: {est['co2_min_g']}g – {est['co2_max_g']}g "
                f"(avg {est['co2_avg_g']}g)\n"
                f"Duration: {est['duration_min_mins']}–{est['duration_max_mins']} min")
            bc = est.get("best_config", {})
            if bc:
                st.info(f"Best config: batch={bc['batch_size']}, epochs={bc['epochs']} "
                        f"→ {bc['total_co2_g']}g CO₂, "
                        f"{round(bc['final_accuracy']*100,1)}% acc [Grade {bc['grade']}]")
        else:
            st.info("No past runs to estimate from yet.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Project stats — custom HTML to avoid duplicate st.metric bug
    @st.cache_data(ttl=30)
    def _project_totals():
        fp_all  = api("/fingerprint/all")
        fp_list = fp_all.get("runs", []) or []
        return (
            fp_list,
            sum(r.get("total_co2_g")  or 0 for r in fp_list),
            sum(r.get("wasted_co2_g") or 0 for r in fp_list),
        )

    fp_list, proj_co2, proj_wasted = _project_totals()
    waste_pct = round(proj_wasted / proj_co2 * 100, 1) if proj_co2 > 0 else 0
    bar_color = "#10b981" if waste_pct < 20 else ("#f59e0b" if waste_pct < 40 else "#ef4444")

    st.markdown(f"""
    <div style='font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:10px;'>📊 PROJECT STATS</div>
    <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px;margin-bottom:8px;'>
      <div style='font-size:11px;color:#94a3b8;margin-bottom:2px;'>Total CO₂ emitted</div>
      <div style='font-size:22px;font-weight:700;color:#f1f5f9;'>{fmt_val(proj_co2,"g",4)}</div>
    </div>
    <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px;margin-bottom:8px;'>
      <div style='font-size:11px;color:#94a3b8;margin-bottom:2px;'>Wasted CO₂</div>
      <div style='font-size:22px;font-weight:700;color:#ef4444;'>{fmt_val(proj_wasted,"g",4)}</div>
    </div>
    <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:12px;'>
      <div style='font-size:11px;color:#94a3b8;margin-bottom:6px;'>Waste efficiency</div>
      <div style='background:#0f172a;border-radius:6px;height:8px;overflow:hidden;'>
        <div style='width:{min(waste_pct,100)}%;height:8px;background:{bar_color};border-radius:6px;'></div>
      </div>
      <div style='font-size:12px;color:{bar_color};font-weight:600;margin-top:4px;'>{waste_pct}% wasted</div>
    </div>
    <div style='font-size:11px;color:#64748b;margin-top:8px;'>{len(fp_list)} completed run(s)</div>
    """, unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────
tab_live, tab_history, tab_sla, tab_lb, tab_audit, tab_beh = st.tabs([
    "📡 Live Monitor", "🔬 Run History",
    "📋 Carbon SLA",   "🏆 Leaderboard",
    "🔒 Audit Trail",  "🧠 Behavior Report"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR (pure JS — zero Streamlit rerun)
# ══════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("""
    <div style='margin-bottom:4px;'>
      <span style='font-size:24px;font-weight:700;color:#f1f5f9;'>🌱 EcoTrace — Real‑Time AI Carbon Governance</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Monitoring: **{target}** · Updates every 5s in place · No page reload")

    live_html = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --bg:     #0f172a;
  --card:   #1e293b;
  --border: #334155;
  --green:  #10b981;
  --red:    #ef4444;
  --amber:  #f59e0b;
  --blue:   #3b82f6;
  --text:   #f1f5f9;
  --muted:  #94a3b8;
}}
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); }}

/* LIVE badge */
#status-badge {{
  display:inline-flex; align-items:center; gap:10px;
  padding:8px 18px; border-radius:100px;
  background:#0d2b1e; border:1px solid #10b981;
  color:#10b981; font-size:13px; font-weight:600;
  margin-bottom:18px;
}}
#status-badge.ended {{
  background:#2d1f00; border-color:#f59e0b; color:#f59e0b;
}}
.status-dot {{
  width:9px; height:9px; border-radius:50%;
  background:#10b981; flex-shrink:0;
  box-shadow: 0 0 0 0 rgba(16,185,129,0.7);
  animation: pulse 2s infinite;
}}
#status-badge.ended .status-dot {{
  background:#f59e0b; animation:none;
  box-shadow: none;
}}
@keyframes pulse {{
  0%   {{ box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }}
  70%  {{ box-shadow: 0 0 0 8px rgba(16,185,129,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0); }}
}}

/* Metric cards */
.metric-row {{
  display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:16px;
}}
.metric-card {{
  background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:16px 18px;
  transition: border-color 0.2s;
}}
.metric-card:hover {{ border-color:#10b981; }}
.metric-label {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }}
.metric-value {{ font-size:26px; font-weight:700; color:var(--text); line-height:1.1; }}
.metric-delta {{ font-size:11px; margin-top:5px; }}
.metric-delta.good  {{ color:#10b981; }}
.metric-delta.bad   {{ color:#ef4444; }}
.metric-delta.muted {{ color:var(--muted); }}

/* Banners */
.banner {{
  padding:10px 16px; border-radius:10px; font-size:13px;
  margin-bottom:10px; display:none;
  border-left:4px solid transparent;
  background:var(--card);
}}
.banner.show {{ display:block; }}
.banner.red    {{ border-left-color:#ef4444; color:#fca5a5; }}
.banner.yellow {{ border-left-color:#f59e0b; color:#fcd34d; }}
.banner.green  {{ border-left-color:#10b981; color:#6ee7b7; }}
.banner.blue   {{ border-left-color:#3b82f6; color:#93c5fd; }}

hr {{ border:none; border-top:1px solid var(--border); margin:16px 0; }}

/* Chart grid */
.chart-grid {{ display:grid; grid-template-columns:3fr 2fr; gap:16px; margin-top:4px; }}
.panel-title {{ font-size:13px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px; }}
canvas {{ width:100% !important; height:200px !important; }}

/* Carbon debt */
.debt-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
.debt-card {{
  background:var(--card); border:1px solid var(--border);
  border-radius:12px; padding:14px; text-align:center;
}}
.debt-icon {{ font-size:22px; margin-bottom:4px; }}
.debt-val  {{ font-size:18px; font-weight:700; color:var(--text); margin:4px 0 2px; }}
.debt-desc {{ font-size:11px; color:var(--muted); }}

/* Bottom grid */
.bottom-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }}
.panel {{
  background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:16px 18px;
}}
.panel h4  {{ margin:0 0 12px; font-size:13px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }}

.grid-val {{ font-size:28px; font-weight:700; color:var(--text); }}
.grid-badge {{
  display:inline-block; padding:3px 12px; border-radius:100px;
  font-size:11px; font-weight:600; margin-left:8px; vertical-align:middle;
}}
.grid-badge.red    {{ background:#450a0a; color:#fca5a5; }}
.grid-badge.yellow {{ background:#422006; color:#fcd34d; }}
.grid-badge.green  {{ background:#052e16; color:#6ee7b7; }}

.timestamp {{ font-size:11px; color:var(--muted); margin-top:12px; }}

/* FP table */
.fp-table {{ width:100%; font-size:11px; border-collapse:collapse; }}
.fp-table th {{ padding:6px 8px; color:var(--green); font-weight:600; text-align:left; border-bottom:1px solid var(--border); font-size:10px; text-transform:uppercase; }}
.fp-table td {{ padding:6px 8px; color:#cbd5e1; border-top:1px solid #1e3a4a; }}
.grade-pill {{ padding:2px 8px; border-radius:100px; font-weight:700; font-size:11px; }}
</style>

<!-- Status badge -->
<div id="status-badge"><div class="status-dot"></div><span id="status-text">Connecting...</span></div>
<hr/>

<!-- Metric cards -->
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Current Power</div>
    <div class="metric-value" id="val-power">—</div>
    <div class="metric-delta muted" id="delta-power"></div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Predicted (1 hr)</div>
    <div class="metric-value" id="val-pred">—</div>
    <div class="metric-delta muted">Linear regression</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Carbon Rate</div>
    <div class="metric-value" id="val-carbon">—</div>
    <div class="metric-delta" id="delta-carbon"></div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Efficiency vs Best</div>
    <div class="metric-value" id="val-eff">—</div>
    <div class="metric-delta" id="delta-eff"></div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Session CO₂</div>
    <div class="metric-value" id="val-co2">—</div>
    <div class="metric-delta muted" id="delta-co2">of {BUDGET_G}g budget</div>
  </div>
</div>

<!-- Alert banners -->
<div id="banner-budget"  class="banner"></div>
<div id="banner-fp"      class="banner"></div>
<div id="banner-dna"     class="banner"></div>
<div id="banner-shapley" class="banner"></div>

<hr/>

<!-- Chart + Carbon debt -->
<div class="chart-grid">
  <div>
    <div class="panel-title">⚡ Live Heartbeat</div>
    <canvas id="powerChart"></canvas>
    <div id="anomaly-caption" class="timestamp"></div>
  </div>
  <div>
    <div class="panel-title">🌍 Carbon Debt Equivalent</div>
    <div id="debt-section">
      <div class="debt-grid">
        <div class="debt-card"><div class="debt-icon">🚗</div><div class="debt-val" id="debt-car">—</div><div class="debt-desc">km petrol car</div></div>
        <div class="debt-card"><div class="debt-icon">📱</div><div class="debt-val" id="debt-phone">—</div><div class="debt-desc">phone charges</div></div>
        <div class="debt-card"><div class="debt-icon">🌳</div><div class="debt-val" id="debt-tree">—</div><div class="debt-desc">tree absorption</div></div>
      </div>
    </div>
  </div>
</div>

<!-- Bottom panels -->
<div class="bottom-grid">
  <div class="panel"><h4>🔬 Recent Run Fingerprints</h4><div id="fp-table">Loading...</div></div>
  <div class="panel">
    <h4>⚡ Grid Intensity</h4>
    <span class="grid-val" id="grid-val">{GRID_FALLBACK:.0f} g/kWh</span>
    <span class="grid-badge red" id="grid-badge">Coal-heavy</span>
    <p style="font-size:12px;color:#94a3b8;margin:8px 0" id="grid-advice">Grid is dirty. Waiting saves ~30–40% CO₂.</p>
    <div class="banner blue show" id="grid-saving">Scheduling for 280 g/kWh saves ~41.1% CO₂.</div>
  </div>
</div>

<div class="timestamp" id="last-updated">Fetching data...</div>

<script>
const SERVER   = "{RENDER_URL}";
const SESSION  = "{target}";
const BUDGET   = {BUDGET_G};
const GRID     = {GRID_FALLBACK};
const INTERVAL = 5000;

let powerHistory = [];
let prevWatts    = null;
let anomalyData  = [];
let dnaFetched   = false;
let fpFetched    = false;
let wasLive      = false;

function fmt(val, unit="", dec=2) {{
  if (val === null || val === undefined) return "—";
  const v = parseFloat(val);
  if (isNaN(v)) return "—";
  if (v === 0)  return "0" + (unit ? " "+unit : "");
  if (Math.abs(v) < 0.01) return "<0.01" + (unit ? " "+unit : "");
  return v.toFixed(dec) + (unit ? " "+unit : "");
}}
function setBanner(id, text, cls) {{
  const el = document.getElementById(id);
  el.className = "banner show " + cls;
  el.textContent = text;
}}
function hideBanner(id) {{ document.getElementById(id).className = "banner"; }}
function setText(id, val) {{ const el=document.getElementById(id); if(el) el.textContent=val; }}

function drawChart() {{
  const canvas = document.getElementById("powerChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width  = canvas.offsetWidth || 500;
  const H = canvas.height = 200;
  ctx.clearRect(0, 0, W, H);
  if (powerHistory.length < 2) return;

  const watts  = powerHistory.map(p => p.watts);
  const maxW   = Math.max(...watts, 1);
  const minW   = Math.min(...watts, 0);
  const rangeW = maxW - minW || 1;
  const padL=40, padR=10, padT=10, padB=30;
  const chartW = W-padL-padR, chartH = H-padT-padB;

  // Background
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth   = 1;
  for (let i=0; i<=4; i++) {{
    const y = padT + (chartH/4)*i;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W-padR, y); ctx.stroke();
    const label = (maxW-(rangeW/4)*i).toFixed(1);
    ctx.fillStyle="#475569"; ctx.font="10px Inter,sans-serif";
    ctx.textAlign="right"; ctx.fillText(label, padL-4, y+4);
  }}

  // Gradient fill under line
  const grad = ctx.createLinearGradient(0, padT, 0, padT+chartH);
  grad.addColorStop(0,   "rgba(16,185,129,0.3)");
  grad.addColorStop(1,   "rgba(16,185,129,0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  powerHistory.forEach((p,i) => {{
    const x = padL+(i/(powerHistory.length-1))*chartW;
    const y = padT+chartH-((p.watts-minW)/rangeW)*chartH;
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  }});
  ctx.lineTo(padL+chartW, padT+chartH);
  ctx.lineTo(padL, padT+chartH);
  ctx.closePath(); ctx.fill();

  // Line
  ctx.strokeStyle="#10b981"; ctx.lineWidth=2.5; ctx.lineJoin="round";
  ctx.beginPath();
  powerHistory.forEach((p,i) => {{
    const x = padL+(i/(powerHistory.length-1))*chartW;
    const y = padT+chartH-((p.watts-minW)/rangeW)*chartH;
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  }});
  ctx.stroke();

  // Dots
  ctx.fillStyle="#10b981";
  powerHistory.forEach((p,i) => {{
    const x = padL+(i/(powerHistory.length-1))*chartW;
    const y = padT+chartH-((p.watts-minW)/rangeW)*chartH;
    ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
  }});

  // Anomaly X markers
  if (anomalyData.length > 0 && powerHistory.length > 1) {{
    ctx.strokeStyle="#ef4444"; ctx.lineWidth=2;
    anomalyData.forEach(ad => {{
      const anomHHMM = ad.timeStr ? ad.timeStr.slice(0,5) : null;
      if (!anomHHMM) return;
      let matchIdx = -1;
      powerHistory.forEach((p,i) => {{ if(p.time && p.time.slice(0,5)===anomHHMM) matchIdx=i; }});
      if (matchIdx < 0) return;
      const x = padL+(matchIdx/(powerHistory.length-1))*chartW;
      const y = padT+chartH-((ad.watts-minW)/rangeW)*chartH;
      const r=6;
      ctx.beginPath();
      ctx.moveTo(x-r,y-r); ctx.lineTo(x+r,y+r);
      ctx.moveTo(x+r,y-r); ctx.lineTo(x-r,y+r);
      ctx.stroke();
    }});
  }}

  // X-axis labels
  ctx.fillStyle="#475569"; ctx.font="10px Inter,sans-serif"; ctx.textAlign="center";
  powerHistory.forEach((p,i) => {{
    if (i%5===0 || i===powerHistory.length-1) {{
      const x = padL+(i/(powerHistory.length-1))*chartW;
      ctx.fillText(p.time, x, H-5);
    }}
  }});
}}

function renderFPTable(runs) {{
  if (!runs || runs.length===0) {{
    document.getElementById("fp-table").innerHTML="<small style='color:#64748b'>No completed runs yet.</small>";
    return;
  }}
  const gc = {{A:"#10b981",B:"#34d399",C:"#f59e0b",D:"#f97316",F:"#ef4444"}};
  const gb = {{A:"#052e16",B:"#052e16",C:"#431407",D:"#431407",F:"#450a0a"}};
  let html=`<table class="fp-table"><tr><th>Model</th><th>Epochs</th><th>CO₂</th><th>Wasted</th><th>Acc</th><th>Grade</th></tr>`;
  runs.slice(0,5).forEach(r => {{
    const g=r.efficiency_grade||"?";
    html+=`<tr>
      <td>${{r.model_name||"—"}}</td>
      <td>${{r.epochs||"—"}}</td>
      <td>${{fmt(r.total_co2_g,"g",4)}}</td>
      <td>${{fmt(r.wasted_co2_g,"g",4)}}</td>
      <td>${{r.final_accuracy?(r.final_accuracy*100).toFixed(1)+"%":"—"}}</td>
      <td><span class="grade-pill" style="color:${{gc[g]||"#888"}};background:${{gb[g]||"#1e293b"}}">${{g}}</span></td>
    </tr>`;
  }});
  html+="</table>";
  document.getElementById("fp-table").innerHTML=html;
}}

async function fetchDNAMatch(live) {{
  if (!live) {{ hideBanner("banner-dna"); return; }}
  if (dnaFetched || powerHistory.length < 10) return;
  const watts=powerHistory.map(p=>p.watts);
  if (Math.max(...watts)-Math.min(...watts) < 2.0) return;
  try {{
    const r = await fetch(SERVER+"/dna/match", {{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{powers:watts}})}});
    const d = await r.json();
    if (d.prediction) {{ setBanner("banner-dna","🧬 "+d.prediction,"blue"); dnaFetched=true; }}
  }} catch(e) {{}}
}}

async function fetchFPCompare() {{
  if (fpFetched) return;
  try {{
    const r = await fetch(`${{SERVER}}/fingerprint/compare?session_id=${{SESSION}}`);
    const d = await r.json();
    const similar = d.similar_runs||[];
    if (similar.length>0) {{
      const best=similar[0];
      const acc=best.final_accuracy?(best.final_accuracy*100).toFixed(1)+"%":"?";
      setBanner("banner-fp",`🔬 Most similar past run: ${{best.model_name||"?"}} — CO₂: ${{fmt(best.total_co2_g,"g",4)}} | Acc: ${{acc}} | Grade: ${{best.efficiency_grade||"?"}} | Similarity: ${{best.similarity_score||"?"}}`, "blue");
      fpFetched=true;
    }}
  }} catch(e) {{}}
}}

async function fetchAndUpdate() {{
  try {{
    const [predResp,budgetResp,shapResp,fpResp,anomResp,activeResp] = await Promise.all([
      fetch(`${{SERVER}}/predict?session_id=${{SESSION}}`),
      fetch(`${{SERVER}}/budget_check?session_id=${{SESSION}}&budget_g=${{BUDGET}}`),
      fetch(`${{SERVER}}/shapley?session_id=${{SESSION}}`),
      fetch(`${{SERVER}}/fingerprint/all`),
      fetch(`${{SERVER}}/anomalies?session_id=${{SESSION}}`),
      fetch(`${{SERVER}}/active_devices`),
    ]);
    const pred=await predResp.json(), budget=await budgetResp.json(), shap=await shapResp.json();
    const fp=await fpResp.json(), anom=await anomResp.json(), active=await activeResp.json();

    const activeDevices=active.devices||[];
    const isLive=activeDevices.includes(SESSION)&&!pred.error;

    if (wasLive && !isLive) {{
      hideBanner("banner-dna"); hideBanner("banner-fp");
      dnaFetched=false; fpFetched=false; anomalyData=[];
    }}
    wasLive=isLive;

    const badge=document.getElementById("status-badge");
    if (isLive) {{
      badge.className="";
      document.getElementById("status-text").textContent="LIVE — agent is active";
    }} else {{
      badge.className="ended";
      document.getElementById("status-text").textContent="SESSION ENDED — showing last known values";
    }}

    const currW=parseFloat(pred.current_avg_w)||0;
    const predW=parseFloat(pred.predicted_w)||0;
    setText("val-power", fmt(currW,"W",2));
    setText("val-pred",  fmt(predW,"W",2));

    const deltaPowerEl=document.getElementById("delta-power");
    if (prevWatts!==null && currW>0) {{
      const delta=currW-prevWatts, sign=delta>=0?"+":"";
      deltaPowerEl.textContent=sign+delta.toFixed(2)+" W since last tick";
      deltaPowerEl.className="metric-delta "+(delta>0?"bad":"good");
    }} else {{ deltaPowerEl.textContent="Tracking power draw…"; deltaPowerEl.className="metric-delta muted"; }}
    if (currW>0) prevWatts=currW;

    const carbonGhr=parseFloat(pred.carbon_g_hr)||0;
    setText("val-carbon", fmt(carbonGhr,"g/hr",2));
    const carbDelta=document.getElementById("delta-carbon");
    carbDelta.textContent=carbonGhr>5?"↑ above average":"↑ normal range";
    carbDelta.className="metric-delta "+(carbonGhr>5?"bad":"good");

    const serverSamples=parseInt(pred.samples)||0;
    const sessionCO2=(currW/1000)*GRID*(serverSamples*5/3600);
    setText("val-co2", fmt(sessionCO2,"g",4));

    const bestCO2=parseFloat(budget.best_past_co2), projCO2=parseFloat(budget.projected_co2)||0;
    const effEl=document.getElementById("val-eff"), effDelta=document.getElementById("delta-eff");
    if (!isNaN(bestCO2)&&bestCO2>=1.0&&projCO2>=1.0) {{
      let dp=Math.max(-999,Math.min(999,(projCO2-bestCO2)/bestCO2*100));
      effEl.textContent=(dp>0?"+":"")+dp.toFixed(1)+"% CO₂";
      effDelta.textContent=dp>0?"↑ worse than best":"↓ better than best";
      effDelta.className="metric-delta "+(dp>0?"bad":"good");
    }} else {{
      effEl.textContent="—";
      effDelta.textContent="Complete a run to set baseline";
      effDelta.className="metric-delta muted";
    }}

    const bStatus=budget.status||"green", bRec=budget.recommendation||"";
    if(bStatus==="red") setBanner("banner-budget","🛑 Budget alert — "+bRec,"red");
    else if(bStatus==="yellow") setBanner("banner-budget","⚠️ Budget warning — "+bRec,"yellow");
    else hideBanner("banner-budget");

    const fairCO2=parseFloat(shap.co2_fair_g), savedCO2=parseFloat(shap.co2_saved_g);
    if(!isNaN(fairCO2)&&!isNaN(savedCO2)&&savedCO2>0)
      setBanner("banner-shapley","⚖️ Shapley attribution: your fair CO₂ share = "+fairCO2.toFixed(6)+"g (saved "+savedCO2.toFixed(6)+"g vs naive attribution)","green");
    else hideBanner("banner-shapley");

    const debtSection=document.getElementById("debt-section");
    if(sessionCO2>0.001) {{
      debtSection.style.opacity="1";
      setText("debt-car",   fmt(sessionCO2*0.00417,"km",3));
      setText("debt-phone", fmt(sessionCO2/5.5,"×",2));
      setText("debt-tree",  fmt(sessionCO2/0.0095,"min",1));
    }} else {{
      debtSection.style.opacity="0.3";
      setText("debt-car","—"); setText("debt-phone","—"); setText("debt-tree","—");
    }}

    const anomList=anom.anomalies||[];
    if(isLive&&anomList.length>0) {{
      anomalyData=anomList.map(a=>({{timeStr:a.timestamp?a.timestamp.slice(-8):null,watts:parseFloat(a.power_w)||0}})).filter(a=>a.timeStr);
    }} else if(!isLive) anomalyData=[];
    const anomCap=document.getElementById("anomaly-caption");
    anomCap.textContent=(isLive&&anomalyData.length>0)?"⚠️ "+anomalyData.length+" anomaly event(s) — red × markers on chart.":"";

    if(isLive&&currW>0) {{
      const now=new Date().toLocaleTimeString("en-GB",{{hour:"2-digit",minute:"2-digit",second:"2-digit"}});
      powerHistory.push({{time:now,watts:currW}});
      if(powerHistory.length>40) powerHistory.shift();
    }}
    drawChart();
    renderFPTable(fp.runs||[]);
    fetchDNAMatch(isLive);
    if(isLive) fetchFPCompare();

    document.getElementById("last-updated").textContent="🔗 Connected · "+SESSION+" · Last updated "+new Date().toLocaleTimeString();
  }} catch(err) {{
    document.getElementById("last-updated").textContent="⚠️ Fetch error: "+err.message;
  }}
}}
fetchAndUpdate();
setInterval(fetchAndUpdate, 5000);
</script>
"""
    components.html(live_html, height=1150, scrolling=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — RUN HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("""
    <div style='margin-bottom:4px;'>
      <span style='font-size:22px;font-weight:700;color:#f1f5f9;'>🔬 Run Fingerprint History</span>
    </div>
    <p style='color:#94a3b8;font-size:13px;margin-bottom:16px;'>
      Every completed training run is permanently logged with its carbon footprint, efficiency grade, and wasted CO₂.
    </p>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh History", use_container_width=False):
        st.cache_data.clear()
        st.rerun()

    hist_list = api("/fingerprint/all").get("runs", []) or []

    if hist_list:
        fp_df = pd.DataFrame(hist_list)
        st.dataframe(fp_df, use_container_width=True, height=260)

        col1, col2 = st.columns(2)
        if "efficiency_grade" in fp_df.columns:
            grade_counts = fp_df["efficiency_grade"].value_counts()
            bar_colours  = [GRADE_COLOURS.get(g, "#888") for g in grade_counts.index.tolist()]
            fig_g = go.Figure(go.Bar(
                x=grade_counts.index.tolist(),
                y=grade_counts.values.tolist(),
                marker_color=bar_colours,
                text=grade_counts.values.tolist(),
                textposition="auto"))
            fig_g.update_layout(
                title="Efficiency Grade Distribution",
                xaxis_title="Grade", yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                height=300, margin=dict(l=20,r=20,t=40,b=20))
            col1.plotly_chart(fig_g, use_container_width=True)

        if "timestamp" in fp_df.columns and "total_co2_g" in fp_df.columns:
            fp_sorted = fp_df.sort_values("timestamp")
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=fp_sorted["timestamp"], y=fp_sorted["total_co2_g"],
                mode="lines+markers",
                line=dict(color="#10b981", width=2),
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.15)",
                name="Total CO₂",
                hovertemplate="<b>%{x}</b><br>CO₂: %{y:.4f} g<extra></extra>"))
            if "wasted_co2_g" in fp_sorted.columns:
                fig_t.add_trace(go.Bar(
                    x=fp_sorted["timestamp"], y=fp_sorted["wasted_co2_g"],
                    name="Wasted CO₂",
                    marker_color="rgba(239,68,68,0.45)",
                    hovertemplate="<b>%{x}</b><br>Wasted: %{y:.4f} g<extra></extra>"))
            fig_t.update_layout(
                title="CO₂ Per Run Over Time",
                yaxis_title="CO₂ (g)",
                template="plotly_dark",
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                height=300, barmode="overlay",
                margin=dict(l=20,r=20,t=40,b=20))
            col2.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No completed runs yet. Run Cell 4 in Colab to log your first session.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — CARBON SLA
# ══════════════════════════════════════════════════════════════
with tab_sla:
    st.markdown("""
    <div style='margin-bottom:4px;'>
      <span style='font-size:22px;font-weight:700;color:#f1f5f9;'>📋 Carbon SLA Manager</span>
    </div>
    <p style='color:#94a3b8;font-size:13px;margin-bottom:16px;'>
      Set maximum CO₂ and minimum accuracy compliance targets per model. SLA breaches are permanently logged on the Audit Trail.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1e293b;border:1px solid #334155;border-radius:14px;padding:20px;margin-bottom:20px;'>
      <div style='font-size:13px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:14px;'>
        ⚙️ Set New SLA Rule
      </div>
    """, unsafe_allow_html=True)

    with st.form("sla_form"):
        sla_model   = st.text_input("Model name", "ResNet-50")
        c1, c2 = st.columns(2)
        sla_max_co2 = c1.number_input("Max CO₂ (g)", 0.0, 10000.0, 50.0, step=1.0)
        sla_min_acc = c2.number_input("Min accuracy (0–1)", 0.0, 1.0, 0.90, step=0.01)
        if st.form_submit_button("💾 Save SLA", use_container_width=True):
            r = api("/sla/set", method="POST",
                    json_body={"model_name": sla_model,
                               "max_co2_g":  sla_max_co2,
                               "min_accuracy": sla_min_acc})
            if r.get("status") == "SLA saved":
                st.success(f"✅ SLA saved for {sla_model}")
            else:
                st.error("Failed to save SLA. Check server connection.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:13px;font-weight:600;color:#94a3b8;text-transform:uppercase;
                letter-spacing:0.5px;margin-bottom:10px;'>📊 Active SLA Rules</div>
    """, unsafe_allow_html=True)
    slas = api("/sla/all").get("slas", []) or []
    if slas:
        st.dataframe(pd.DataFrame(slas), use_container_width=True)
    else:
        st.info("No SLA rules configured yet.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — LEADERBOARD
# ══════════════════════════════════════════════════════════════
with tab_lb:
    st.markdown("""
    <div style='margin-bottom:4px;'>
      <span style='font-size:22px;font-weight:700;color:#f1f5f9;'>🏆 Team Carbon Efficiency Leaderboard</span>
    </div>
    <p style='color:#94a3b8;font-size:13px;margin-bottom:16px;'>
      Privacy-safe ranking (k-anonymity ≥ 3). Only shown when at least 3 researchers have submitted runs.
    </p>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Leaderboard", use_container_width=False):
        st.rerun()

    lb    = api("/leaderboard", params={"k": 3})
    board = lb.get("leaderboard", []) or []
    if board:
        medals = ["🥇","🥈","🥉"]
        for i, row in enumerate(board[:3]):
            m = medals[i] if i < 3 else ""
            st.markdown(f"""
            <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;
                        padding:14px 20px;margin-bottom:10px;display:flex;
                        align-items:center;justify-content:space-between;'>
              <div style='font-size:20px;'>{m}</div>
              <div style='flex:1;margin-left:14px;'>
                <div style='font-weight:600;color:#f1f5f9;font-size:14px;'>{row.get("researcher_id","?")}</div>
                <div style='font-size:11px;color:#94a3b8;'>Efficiency: {row.get("efficiency_pct","?")}%</div>
              </div>
              <div style='font-size:22px;font-weight:700;color:#10b981;'>{row.get("efficiency_pct","?")}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.dataframe(pd.DataFrame(board), use_container_width=True)
        fig_lb = go.Figure(go.Bar(
            x=[r["researcher_id"] for r in board],
            y=[r["efficiency_pct"] for r in board],
            marker=dict(
                color=[r["efficiency_pct"] for r in board],
                colorscale=[[0,"#059669"],[1,"#34d399"]],
                showscale=False),
            text=[f"{r['efficiency_pct']}%" for r in board],
            textposition="auto"))
        fig_lb.update_layout(
            title="Efficiency % by Researcher",
            yaxis_title="Efficiency %",
            template="plotly_dark",
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            height=300, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_lb, use_container_width=True)
    else:
        st.info(lb.get("message", "No leaderboard data yet. Need ≥3 researchers."))


# ══════════════════════════════════════════════════════════════
# TAB 5 — AUDIT TRAIL
# ══════════════════════════════════════════════════════════════
with tab_audit:
    st.markdown("""
    <div style='margin-bottom:4px;'>
      <span style='font-size:22px;font-weight:700;color:#f1f5f9;'>🔒 Cryptographic Audit Trail</span>
    </div>
    <p style='color:#94a3b8;font-size:13px;margin-bottom:16px;'>
      Every session event is SHA-256 hashed and chained. Any tampering breaks the chain — making EcoTrace tamper-proof.
    </p>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Audit", use_container_width=False):
        st.rerun()

    verify = api("/audit/verify")
    if verify.get("valid") is True:
        st.success(f"✅ Audit chain intact — {verify.get('entries', 0)} entries verified.")
    elif verify.get("valid") is False:
        st.error(f"🚨 Chain broken at id={verify.get('broken_at')}! Possible tampering detected.")
    else:
        st.info("Audit chain status unknown.")

    audit_sid = st.text_input("🔍 Filter by session ID (leave blank for all entries)")
    if audit_sid:
        st.caption(f"Showing audit entries for: {audit_sid}")
    else:
        st.caption("Showing last 100 audit entries across all sessions.")

    audit_data = api("/audit", params={"session_id": audit_sid} if audit_sid else {})
    audit_rows = audit_data.get("audit", []) or []

    if audit_rows:
        # Render color-coded event type badges
        EVENT_COLORS = {
            "run_completed":   ("#052e16", "#10b981"),
            "sla_breach":      ("#450a0a", "#ef4444"),
            "anomaly_detected":("#422006", "#f59e0b"),
        }
        rows_html = ""
        for row in audit_rows:
            et   = row.get("event_type", "unknown")
            bg, fg = EVENT_COLORS.get(et, ("#1e293b", "#94a3b8"))
            pill = f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:100px;font-size:11px;font-weight:600;'>{et}</span>"
            sid  = str(row.get("session_id",""))[:20]+"…" if len(str(row.get("session_id","")))>20 else str(row.get("session_id",""))
            h    = str(row.get("entry_hash",""))[:16]+"…"
            rows_html += f"""<tr>
              <td style='padding:8px 10px;color:#64748b;font-size:11px;'>{row.get("id","")}</td>
              <td style='padding:8px 10px;color:#94a3b8;font-size:11px;font-family:monospace;'>{sid}</td>
              <td style='padding:8px 10px;'>{pill}</td>
              <td style='padding:8px 10px;color:#64748b;font-size:11px;'>{row.get("timestamp","")}</td>
              <td style='padding:8px 10px;color:#475569;font-size:11px;font-family:monospace;'>{h}</td>
            </tr>"""

        st.markdown(f"""
        <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;overflow:hidden;'>
          <table style='width:100%;border-collapse:collapse;'>
            <thead>
              <tr style='border-bottom:1px solid #334155;'>
                <th style='padding:10px;text-align:left;color:#10b981;font-size:11px;font-weight:600;text-transform:uppercase;'>ID</th>
                <th style='padding:10px;text-align:left;color:#10b981;font-size:11px;font-weight:600;text-transform:uppercase;'>Session</th>
                <th style='padding:10px;text-align:left;color:#10b981;font-size:11px;font-weight:600;text-transform:uppercase;'>Event Type</th>
                <th style='padding:10px;text-align:left;color:#10b981;font-size:11px;font-weight:600;text-transform:uppercase;'>Timestamp</th>
                <th style='padding:10px;text-align:left;color:#10b981;font-size:11px;font-weight:600;text-transform:uppercase;'>Entry Hash</th>
              </tr>
            </thead>
            <tbody style='divide-y:#334155;'>{rows_html}</tbody>
          </table>
        </div>
        <p style='color:#475569;font-size:11px;margin-top:8px;'>entry_hash = SHA-256(event + prev_hash). Any modification breaks the chain.</p>
        """, unsafe_allow_html=True)
    else:
        st.info("No audit entries yet.")


# ══════════════════════════════════════════════════════════════
# TAB 6 — BEHAVIOR REPORT
# ══════════════════════════════════════════════════════════════
with tab_beh:
    st.markdown("""
    <div style='margin-bottom:4px;'>
      <span style='font-size:22px;font-weight:700;color:#f1f5f9;'>🧠 Carbon Behavior Analytics</span>
    </div>
    <p style='color:#94a3b8;font-size:13px;margin-bottom:16px;'>
      AI-powered analysis of your team's carbon habits — identifying duplicate runs, off-peak waste, and providing personalized insights.
    </p>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Report", use_container_width=False):
        st.rerun()

    beh_rid = st.text_input("Researcher ID (leave blank for all)")
    beh     = api("/behavior", params={"researcher_id": beh_rid} if beh_rid else {})

    if beh.get("total_runs"):
        # Top stat cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs",    beh["total_runs"])
        c2.metric("Total CO₂",     fmt_val(beh["total_co2_g"],    "g"))
        c3.metric("Total Wasted",  fmt_val(beh["total_wasted_g"], "g"))
        c4.metric("Waste %",       fmt_val(beh["waste_pct"],      "%"))

        st.markdown("<hr/>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        # Duplicate runs panel
        dup_count = beh["duplicate_runs"]
        dup_waste = beh["duplicate_waste_g"]
        dup_color = "#ef4444" if dup_count > 0 else "#10b981"
        dup_badge_bg = "#450a0a" if dup_count > 0 else "#052e16"
        col_a.markdown(f"""
        <div style='background:#1e293b;border:1px solid #334155;border-radius:14px;padding:20px;'>
          <div style='font-size:12px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;'>
            🔁 Duplicate Runs
          </div>
          <div style='display:flex;align-items:center;gap:12px;'>
            <span style='font-size:36px;font-weight:700;color:{dup_color};'>{dup_count}</span>
            <span style='background:{dup_badge_bg};color:{dup_color};padding:4px 12px;border-radius:100px;font-size:12px;font-weight:600;'>
              ↑ {dup_waste}g CO₂ wasted
            </span>
          </div>
          <div style='font-size:11px;color:#64748b;margin-top:8px;'>Same model + epochs + batch within 10 min.</div>
        </div>
        """, unsafe_allow_html=True)

        # Late night runs panel
        night_count = beh["night_runs"]
        night_waste = beh["night_waste_g"]
        night_color = "#f59e0b" if night_count > 0 else "#10b981"
        night_bg    = "#422006" if night_count > 0 else "#052e16"
        col_b.markdown(f"""
        <div style='background:#1e293b;border:1px solid #334155;border-radius:14px;padding:20px;'>
          <div style='font-size:12px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;'>
            🌙 Late Night Runs (10 PM–5 AM)
          </div>
          <div style='display:flex;align-items:center;gap:12px;'>
            <span style='font-size:36px;font-weight:700;color:{night_color};'>{night_count}</span>
            <span style='background:{night_bg};color:{night_color};padding:4px 12px;border-radius:100px;font-size:12px;font-weight:600;'>
              ↑ {night_waste}g wasted
            </span>
          </div>
          <div style='font-size:11px;color:#64748b;margin-top:8px;'>Grid is dirtier at night — higher CO₂ per kWh.</div>
        </div>
        """, unsafe_allow_html=True)

        # AI Insight callout
        st.markdown(f"""
        <div style='margin-top:16px;background:#1c1a0a;border-left:4px solid #f59e0b;
                    border-radius:10px;padding:14px 18px;'>
          <span style='font-size:15px;'>💡</span>
          <span style='color:#fcd34d;font-size:13px;margin-left:8px;'>{beh.get("insight","")}</span>
        </div>
        """, unsafe_allow_html=True)

    elif beh.get("message"):
        st.info(beh["message"])
    else:
        st.info("No behavior data available yet. Complete at least one run.")
