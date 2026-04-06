"""
EcoTrace v3 — Zero-Flicker Dashboard

The live monitor tab uses pure JavaScript fetch() to update numbers
in place every 5 seconds. Streamlit never reruns for live data.
The page stays completely solid — no fade, no flash, no transparency.
Only the non-live tabs (History, SLA, Leaderboard, Audit, Behavior)
use normal Streamlit rendering with manual Refresh buttons.
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────
RENDER_URL    = "https://eco-2-4re9.onrender.com"
GRID_FALLBACK = 475.0
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EcoTrace v3",
    layout="wide",
    page_icon="🌱",
)

# ── Global CSS — no opacity transitions, no fade ──────────────
st.markdown("""
<style>
/* Remove ALL Streamlit fade animations */
*, *::before, *::after {
    transition: none !important;
    animation-duration: 0s !important;
}
section[data-testid="stSidebar"] { min-width: 280px !important; }
</style>
""", unsafe_allow_html=True)


# ── SAFE API FETCH ─────────────────────────────────────────────
def api(path, params=None, method="GET", json_body=None, timeout=6):
    try:
        url = f"{RENDER_URL}{path}"
        if method == "POST":
            r = requests.post(url, params=params,
                              json=json_body, timeout=timeout)
        else:
            r = requests.get(url, params=params, timeout=timeout)
        return r.json()
    except Exception:
        return {}


def fmt_val(val, unit="", decimals=2):
    if val is None:
        return "—"
    try:
        v = float(val)
        if v == 0:
            return f"0 {unit}".strip()
        if abs(v) < 0.01:
            return f"<0.01 {unit}".strip()
        return f"{round(v, decimals)} {unit}".strip()
    except Exception:
        return str(val)


# ── SESSION STATE ─────────────────────────────────────────────
if "last_device" not in st.session_state:
    st.session_state.last_device = None


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🌱 EcoTrace v3")

    device_res  = api("/active_devices")
    active_list = device_res.get("devices", []) or ["no-active-sessions"]
    target      = st.selectbox("Select device to monitor", active_list)

    if st.session_state.last_device != target:
        st.session_state.last_device = target

    st.markdown("---")
    st.subheader("⚙️ Budget settings")
    BUDGET_G = st.slider("Carbon budget (g CO₂)", 10, 500, 100, step=10)

    st.markdown("---")
    st.subheader("🔬 Pre-run estimator")
    est_task   = st.text_input("Task type",    "image_classification")
    est_model  = st.text_input("Model name",   "ResNet-50")
    est_epochs = st.number_input("Epochs",     1, 500, 25)
    est_batch  = st.number_input("Batch size", 1, 512, 32)

    if st.button("Estimate CO₂ before training ↗"):
        est = api("/estimate", params={
            "task_type":  est_task,   "model_name": est_model,
            "epochs":     est_epochs, "batch_size": est_batch})
        if est.get("similar_count", 0) > 0:
            st.success(
                f"Based on {est['similar_count']} similar runs:\n"
                f"CO₂: {est['co2_min_g']}g – {est['co2_max_g']}g "
                f"(avg {est['co2_avg_g']}g)\n"
                f"Duration: {est['duration_min_mins']}–"
                f"{est['duration_max_mins']} min")
            bc = est.get("best_config", {})
            if bc:
                st.info(
                    f"Best config: batch={bc['batch_size']}, "
                    f"epochs={bc['epochs']} → "
                    f"{bc['total_co2_g']}g CO₂, "
                    f"{round(bc['final_accuracy']*100,1)}% acc "
                    f"[Grade {bc['grade']}]")
        else:
            st.info("No past runs to estimate from yet.")

    st.markdown("---")
    fp_all      = api("/fingerprint/all")
    fp_list     = fp_all.get("runs", []) or []
    proj_co2    = sum(r.get("total_co2_g")  or 0 for r in fp_list)
    proj_wasted = sum(r.get("wasted_co2_g") or 0 for r in fp_list)
    st.metric("Project total CO₂",  fmt_val(proj_co2,    "g"))
    st.metric("Project wasted CO₂", fmt_val(proj_wasted, "g"))
    st.caption(f"{len(fp_list)} completed run(s)")


# ── TABS ──────────────────────────────────────────────────────
tab_live, tab_history, tab_sla, tab_lb, tab_audit, tab_beh = st.tabs([
    "📡 Live Monitor", "🔬 Run History",
    "📋 Carbon SLA",   "🏆 Leaderboard",
    "🔒 Audit Trail",  "🧠 Behavior Report"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR (pure JS — zero Streamlit rerun)
# ══════════════════════════════════════════════════════════════
with tab_live:
    st.title("🌱 EcoTrace — Real-Time AI Carbon Governance")
    st.caption(f"Monitoring: **{target}** · Updates every 5s in place · No page reload")

    # The entire live section is one HTML component.
    # JavaScript fetches data from the Render API every 5 seconds
    # and updates only the specific DOM elements that changed.
    # Streamlit is completely uninvolved in the refresh cycle.
    live_html = f"""
<style>
  :root {{
    --green:  #2ecc71;
    --red:    #e74c3c;
    --amber:  #f39c12;
    --blue:   #3498db;
    --text:   #262730;
    --muted:  #6b6b76;
    --card:   #f8f9fa;
    --border: #e0e0e0;
    --live:   #d4edda;
    --ended:  #fff3cd;
  }}

  body {{ margin: 0; font-family: sans-serif; color: var(--text); }}

  /* Status badge */
  #status-badge {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 16px; border-radius: 8px;
    font-size: 14px; font-weight: 500;
    margin-bottom: 16px;
    background: var(--live); color: #155724;
    border: 1px solid #c3e6cb;
  }}
  #status-badge.ended {{
    background: var(--ended); color: #856404;
    border-color: #ffeeba;
  }}
  .status-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: #28a745; flex-shrink: 0;
  }}
  #status-badge.ended .status-dot {{ background: #ffc107; }}

  /* Metric cards row */
  .metric-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }}
  .metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }}
  .metric-label {{
    font-size: 12px; color: var(--muted); margin-bottom: 4px;
  }}
  .metric-value {{
    font-size: 24px; font-weight: 600; color: var(--text);
    line-height: 1.1;
  }}
  .metric-delta {{
    font-size: 12px; margin-top: 4px;
  }}
  .metric-delta.good  {{ color: #28a745; }}
  .metric-delta.bad   {{ color: #dc3545; }}
  .metric-delta.muted {{ color: var(--muted); }}

  /* Alert banners */
  .banner {{
    padding: 10px 16px; border-radius: 8px;
    font-size: 13px; margin-bottom: 10px;
    display: none;
  }}
  .banner.show {{ display: block; }}
  .banner.red    {{ background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; }}
  .banner.yellow {{ background:#fff3cd; color:#856404; border:1px solid #ffeeba; }}
  .banner.green  {{ background:#d4edda; color:#155724; border:1px solid #c3e6cb; }}
  .banner.blue   {{ background:#d1ecf1; color:#0c5460; border:1px solid #bee5eb; }}

  /* Chart area */
  .chart-grid {{ display: grid; grid-template-columns: 3fr 2fr; gap: 16px; margin-top: 16px; }}

  /* Carbon debt cards */
  .debt-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
  }}
  .debt-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; text-align: center;
  }}
  .debt-icon {{ font-size: 20px; }}
  .debt-val  {{ font-size: 18px; font-weight: 600; margin: 4px 0 2px; }}
  .debt-desc {{ font-size: 11px; color: var(--muted); }}

  /* Bottom grid */
  .bottom-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
  .panel {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }}
  .panel h4 {{ margin: 0 0 10px; font-size: 14px; color: var(--text); }}

  .grid-badge {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 500; margin-left: 8px;
  }}
  .grid-badge.red    {{ background:#f8d7da; color:#721c24; }}
  .grid-badge.yellow {{ background:#fff3cd; color:#856404; }}
  .grid-badge.green  {{ background:#d4edda; color:#155724; }}

  /* Simple line chart via canvas */
  canvas {{ width: 100% !important; height: 200px !important; }}

  .section-title {{
    font-size: 15px; font-weight: 600; margin: 0 0 12px;
    color: var(--text);
  }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 16px 0; }}
  .timestamp {{ font-size: 11px; color: var(--muted); margin-top: 8px; }}
</style>

<!-- Status badge -->
<div id="status-badge">
  <div class="status-dot"></div>
  <span id="status-text">Connecting...</span>
</div>

<hr/>

<!-- 5 metric cards -->
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Current power</div>
    <div class="metric-value" id="val-power">—</div>
    <div class="metric-delta muted" id="delta-power"></div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Predicted (1 hr)</div>
    <div class="metric-value" id="val-pred">—</div>
    <div class="metric-delta muted">Linear regression</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Carbon rate</div>
    <div class="metric-value" id="val-carbon">—</div>
    <div class="metric-delta" id="delta-carbon"></div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Efficiency vs best</div>
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
<div id="banner-budget"   class="banner"></div>
<div id="banner-fp"       class="banner"></div>
<div id="banner-dna"      class="banner"></div>
<div id="banner-shapley"  class="banner"></div>

<hr/>

<!-- Chart + Carbon debt -->
<div class="chart-grid">
  <div>
    <div class="section-title">⚡ Live heartbeat</div>
    <canvas id="powerChart"></canvas>
    <div id="anomaly-caption" class="timestamp"></div>
  </div>
  <div>
    <div class="section-title">🌍 Carbon debt</div>
    <div class="debt-grid">
      <div class="debt-card">
        <div class="debt-icon">🚗</div>
        <div class="debt-val" id="debt-car">—</div>
        <div class="debt-desc">km petrol car</div>
      </div>
      <div class="debt-card">
        <div class="debt-icon">📱</div>
        <div class="debt-val" id="debt-phone">—</div>
        <div class="debt-desc">phone charges</div>
      </div>
      <div class="debt-card">
        <div class="debt-icon">🌳</div>
        <div class="debt-val" id="debt-tree">—</div>
        <div class="debt-desc">tree absorption</div>
      </div>
    </div>
  </div>
</div>

<!-- Fingerprint table + Grid -->
<div class="bottom-grid">
  <div class="panel">
    <h4>🔬 Recent run fingerprints</h4>
    <div id="fp-table">Loading...</div>
  </div>
  <div class="panel">
    <h4>⚡ Grid intensity</h4>
    <div style="font-size:28px;font-weight:600" id="grid-val">{GRID_FALLBACK:.0f} g/kWh</div>
    <span class="grid-badge red" id="grid-badge">Coal-heavy</span>
    <p style="font-size:12px;color:#6b6b76;margin:8px 0" id="grid-advice">
      Grid is dirty. Waiting saves ~30–40% CO₂.
    </p>
    <div class="banner blue show" id="grid-saving">
      Scheduling for 280 g/kWh saves ~41.1% CO₂.
    </div>
  </div>
</div>

<div class="timestamp" id="last-updated">Fetching data...</div>

<script>
// ── Config ──────────────────────────────────────────────────
const SERVER   = "{RENDER_URL}";
const SESSION  = "{target}";
const BUDGET   = {BUDGET_G};
const GRID     = {GRID_FALLBACK};
const INTERVAL = 5000;  // 5 seconds

// ── State ───────────────────────────────────────────────────
let powerHistory   = [];   // {{time, watts}}
let totalReadings  = 0;
let sessionCO2     = 0;
let chartCtx       = null;
let sampleCount    = 0;

// ── Helpers ─────────────────────────────────────────────────
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

function hideBanner(id) {{
  document.getElementById(id).className = "banner";
}}

function setText(id, val) {{
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}}

// ── Draw chart (pure canvas, no library needed) ─────────────
function drawChart() {{
  const canvas = document.getElementById("powerChart");
  if (!canvas) return;
  const ctx    = canvas.getContext("2d");
  const W = canvas.width  = canvas.offsetWidth  || 500;
  const H = canvas.height = 200;
  ctx.clearRect(0, 0, W, H);

  if (powerHistory.length < 2) return;

  const watts  = powerHistory.map(p => p.watts);
  const maxW   = Math.max(...watts, 1);
  const minW   = Math.min(...watts, 0);
  const rangeW = maxW - minW || 1;

  const padL = 40, padR = 10, padT = 10, padB = 30;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  // Grid lines
  ctx.strokeStyle = "#e0e0e0";
  ctx.lineWidth   = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = padT + (chartH / 4) * i;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    const label = (maxW - (rangeW / 4) * i).toFixed(1);
    ctx.fillStyle = "#888"; ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(label, padL - 4, y + 4);
  }}

  // Power line
  ctx.strokeStyle = "#2ecc71";
  ctx.lineWidth   = 2;
  ctx.lineJoin    = "round";
  ctx.beginPath();
  powerHistory.forEach((p, i) => {{
    const x = padL + (i / (powerHistory.length - 1)) * chartW;
    const y = padT + chartH - ((p.watts - minW) / rangeW) * chartH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Dots
  ctx.fillStyle = "#2ecc71";
  powerHistory.forEach((p, i) => {{
    const x = padL + (i / (powerHistory.length - 1)) * chartW;
    const y = padT + chartH - ((p.watts - minW) / rangeW) * chartH;
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  }});

  // X-axis labels (show every 5th)
  ctx.fillStyle   = "#888";
  ctx.font        = "10px sans-serif";
  ctx.textAlign   = "center";
  powerHistory.forEach((p, i) => {{
    if (i % 5 === 0 || i === powerHistory.length - 1) {{
      const x = padL + (i / (powerHistory.length - 1)) * chartW;
      ctx.fillText(p.time, x, H - 5);
    }}
  }});
}}

// ── Fingerprint table ────────────────────────────────────────
function renderFPTable(runs) {{
  if (!runs || runs.length === 0) {{
    document.getElementById("fp-table").innerHTML =
      "<small style='color:#888'>No completed runs yet.</small>";
    return;
  }}
  let html = `<table style="width:100%;font-size:11px;border-collapse:collapse">
    <tr style="background:#f0f0f0;text-align:left">
      <th style="padding:4px 6px">Model</th>
      <th style="padding:4px 6px">Epochs</th>
      <th style="padding:4px 6px">CO₂ (g)</th>
      <th style="padding:4px 6px">Wasted</th>
      <th style="padding:4px 6px">Accuracy</th>
      <th style="padding:4px 6px">Grade</th>
    </tr>`;
  runs.slice(0, 5).forEach(r => {{
    const grade = r.efficiency_grade || "?";
    const gradeColor = {{A:"#28a745",B:"#5cb85c",C:"#f0ad4e",
                         D:"#e67e22",F:"#e74c3c"}}[grade] || "#888";
    html += `<tr style="border-top:1px solid #eee">
      <td style="padding:4px 6px">${{r.model_name||"—"}}</td>
      <td style="padding:4px 6px">${{r.epochs||"—"}}</td>
      <td style="padding:4px 6px">${{fmt(r.total_co2_g,"g",4)}}</td>
      <td style="padding:4px 6px">${{fmt(r.wasted_co2_g,"g",4)}}</td>
      <td style="padding:4px 6px">${{r.final_accuracy ?
        (r.final_accuracy*100).toFixed(1)+"%" : "—"}}</td>
      <td style="padding:4px 6px;font-weight:600;color:${{gradeColor}}">${{grade}}</td>
    </tr>`;
  }});
  html += "</table>";
  document.getElementById("fp-table").innerHTML = html;
}}

// ── Main fetch loop ──────────────────────────────────────────
async function fetchAndUpdate() {{
  try {{
    // Parallel fetch all endpoints
    const [predResp, budgetResp, shapResp, fpResp, anomResp, activeResp] =
      await Promise.all([
        fetch(`${{SERVER}}/predict?session_id=${{SESSION}}`),
        fetch(`${{SERVER}}/budget_check?session_id=${{SESSION}}&budget_g=${{BUDGET}}`),
        fetch(`${{SERVER}}/shapley?session_id=${{SESSION}}`),
        fetch(`${{SERVER}}/fingerprint/all`),
        fetch(`${{SERVER}}/anomalies?session_id=${{SESSION}}`),
        fetch(`${{SERVER}}/active_devices`),
      ]);

    const pred   = await predResp.json();
    const budget = await budgetResp.json();
    const shap   = await shapResp.json();
    const fp     = await fpResp.json();
    const anom   = await anomResp.json();
    const active = await activeResp.json();

    // ── Session live/ended detection ─────────────────────────
    const activeDevices = active.devices || [];
    const isLive = activeDevices.includes(SESSION) && !pred.error;

    const badge = document.getElementById("status-badge");
    if (isLive) {{
      badge.className = "";
      document.getElementById("status-text").textContent =
        "LIVE — agent is active";
    }} else {{
      badge.className = "ended";
      document.getElementById("status-text").textContent =
        "SESSION ENDED — showing last known values";
    }}

    // ── Power & prediction ────────────────────────────────────
    const currW = parseFloat(pred.current_avg_w) || 0;
    const predW = parseFloat(pred.predicted_w)   || 0;

    setText("val-power",  fmt(currW, "W", 2));
    setText("val-pred",   fmt(predW, "W", 2));

    // ── Carbon rate ───────────────────────────────────────────
    const carbonGhr = parseFloat(pred.carbon_g_hr) || 0;
    setText("val-carbon", fmt(carbonGhr, "g/hr", 2));
    const carbDelta = document.getElementById("delta-carbon");
    if (carbonGhr > 5) {{
      carbDelta.textContent = "↑ above avg";
      carbDelta.className   = "metric-delta bad";
    }} else {{
      carbDelta.textContent = "↑ normal";
      carbDelta.className   = "metric-delta good";
    }}

    // ── Session CO₂ accumulation ──────────────────────────────
    sampleCount++;
    const elapsedHrs = sampleCount * 5 / 3600;
    sessionCO2 = (currW / 1000) * {GRID_FALLBACK} * elapsedHrs;
    setText("val-co2", fmt(sessionCO2, "g", 4));

    // ── Efficiency vs best ────────────────────────────────────
    const bestCO2   = parseFloat(budget.best_past_co2);
    const projCO2   = parseFloat(budget.projected_co2) || 0;
    const effEl     = document.getElementById("val-eff");
    const effDelta  = document.getElementById("delta-eff");
    if (!isNaN(bestCO2) && bestCO2 >= 1.0 && projCO2 >= 1.0) {{
      let dp = ((projCO2 - bestCO2) / bestCO2 * 100);
      dp = Math.max(-999, Math.min(999, dp));
      effEl.textContent    = (dp > 0 ? "+" : "") + dp.toFixed(1) + "% CO₂";
      effDelta.textContent = dp > 0 ? "↑ worse than best" : "↓ better than best";
      effDelta.className   = dp > 0 ? "metric-delta bad" : "metric-delta good";
    }} else {{
      effEl.textContent    = "—";
      effDelta.textContent = "Complete a run to set baseline";
      effDelta.className   = "metric-delta muted";
    }}

    // ── Budget banner ─────────────────────────────────────────
    const bStatus = budget.status || "green";
    const bRec    = budget.recommendation || "";
    if (bStatus === "red") {{
      setBanner("banner-budget", "🛑 Budget alert — " + bRec, "red");
    }} else if (bStatus === "yellow") {{
      setBanner("banner-budget", "⚠️ Budget warning — " + bRec, "yellow");
    }} else {{
      hideBanner("banner-budget");
    }}

    // ── Shapley banner ────────────────────────────────────────
    const fairCO2  = parseFloat(shap.co2_fair_g);
    const savedCO2 = parseFloat(shap.co2_saved_g);
    if (!isNaN(fairCO2) && !isNaN(savedCO2) && savedCO2 > 0) {{
      setBanner("banner-shapley",
        "⚖️ Shapley attribution: your fair CO₂ share = " +
        fairCO2.toFixed(4) + "g (saved " + savedCO2.toFixed(4) +
        "g vs naive attribution)", "green");
    }} else {{
      hideBanner("banner-shapley");
    }}

    // ── Carbon debt ───────────────────────────────────────────
    const totalG = Math.max(sessionCO2, 0.001);
    setText("debt-car",   fmt(totalG * 0.00417, "km", 3));
    setText("debt-phone", fmt(totalG / 5.5,     "×",  2));
    setText("debt-tree",  fmt(totalG / 0.0095,  "min",1));

    // ── Anomaly caption ───────────────────────────────────────
    const anomList = anom.anomalies || [];
    const anomCap  = document.getElementById("anomaly-caption");
    anomCap.textContent = anomList.length > 0
      ? "⚠️ " + anomList.length + " anomaly event(s) — wasted carbon logged."
      : "";

    // ── Power history & chart ─────────────────────────────────
    if (isLive && currW > 0) {{
      const now = new Date().toLocaleTimeString("en-GB",
        {{hour:"2-digit", minute:"2-digit", second:"2-digit"}});
      powerHistory.push({{time: now, watts: currW}});
      if (powerHistory.length > 40) powerHistory.shift();
    }}
    drawChart();

    // ── Fingerprint table ─────────────────────────────────────
    renderFPTable(fp.runs || []);

    // ── Timestamp ─────────────────────────────────────────────
    const ts = new Date().toLocaleTimeString();
    document.getElementById("last-updated").textContent =
      "🔗 Connected · " + SESSION + " · Last updated " + ts;

  }} catch (err) {{
    document.getElementById("last-updated").textContent =
      "⚠️ Fetch error: " + err.message;
  }}
}}

// ── Start immediately, then repeat every 5s ──────────────────
// This is a plain setInterval — no page reload, no rerun,
// no Streamlit involvement. Only the text inside the DOM
// elements above changes.
fetchAndUpdate();
setInterval(fetchAndUpdate, {REFRESH_MS});
</script>
"""

    # REFRESH_MS is a Python variable — substitute it
    live_html = live_html.replace("{REFRESH_MS}", "5000")

    components.html(live_html, height=900, scrolling=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — RUN HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    st.header("🔬 Run fingerprint history")
    if st.button("🔄 Refresh"):
        st.rerun()
    if fp_list:
        fp_df = pd.DataFrame(fp_list)
        st.dataframe(fp_df, use_container_width=True)

        if "efficiency_grade" in fp_df.columns:
            grade_counts = fp_df["efficiency_grade"].value_counts()
            fig_g = go.Figure(go.Bar(
                x=grade_counts.index.tolist(),
                y=grade_counts.values.tolist(),
                marker_color=["#2ecc71","#27ae60",
                              "#f39c12","#e67e22","#e74c3c"]))
            fig_g.update_layout(title="Efficiency grade distribution",
                                template="plotly_dark", height=250)
            st.plotly_chart(fig_g, use_container_width=True)

        if "timestamp" in fp_df.columns and "total_co2_g" in fp_df.columns:
            fig_t = go.Figure(go.Scatter(
                x=fp_df["timestamp"], y=fp_df["total_co2_g"],
                mode="lines+markers", line=dict(color="#3498db"),
                name="CO₂ per run"))
            if "wasted_co2_g" in fp_df.columns:
                fig_t.add_trace(go.Bar(
                    x=fp_df["timestamp"], y=fp_df["wasted_co2_g"],
                    name="Wasted CO₂",
                    marker_color="rgba(231,76,60,0.5)"))
            fig_t.update_layout(title="CO₂ per run over time",
                                template="plotly_dark", height=300)
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No completed runs yet.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — CARBON SLA
# ══════════════════════════════════════════════════════════════
with tab_sla:
    st.header("📋 Carbon SLA manager")
    st.caption("Set maximum CO₂ and minimum accuracy targets per model.")

    with st.form("sla_form"):
        sla_model   = st.text_input("Model name", "ResNet-50")
        sla_max_co2 = st.number_input("Max CO₂ (g)", 1.0, 10000.0, 50.0, step=1.0)
        sla_min_acc = st.number_input("Min accuracy (0–1)", 0.0, 1.0, 0.90, step=0.01)
        if st.form_submit_button("Save SLA"):
            r = api("/sla/set", method="POST",
                    json_body={"model_name":   sla_model,
                               "max_co2_g":    sla_max_co2,
                               "min_accuracy": sla_min_acc})
            st.success(f"SLA saved for {sla_model}") \
                if r.get("status") == "SLA saved" \
                else st.error("Failed to save SLA.")

    st.subheader("Active SLAs")
    slas = api("/sla/all").get("slas", []) or []
    if slas:
        st.dataframe(pd.DataFrame(slas), use_container_width=True)
    else:
        st.info("No SLAs set yet.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — LEADERBOARD
# ══════════════════════════════════════════════════════════════
with tab_lb:
    st.header("🏆 Team carbon efficiency leaderboard")
    st.caption("Privacy-safe (k-anonymity): shown only when ≥3 researchers.")
    if st.button("🔄 Refresh leaderboard"):
        st.rerun()
    lb    = api("/leaderboard", params={"k": 3})
    board = lb.get("leaderboard", []) or []
    if board:
        st.dataframe(pd.DataFrame(board), use_container_width=True)
        fig_lb = go.Figure(go.Bar(
            x=[r["researcher_id"] for r in board],
            y=[r["efficiency_pct"] for r in board],
            marker_color="#2ecc71"))
        fig_lb.update_layout(
            title="Efficiency % by researcher",
            yaxis_title="Efficiency %",
            template="plotly_dark", height=300)
        st.plotly_chart(fig_lb, use_container_width=True)
    else:
        st.info(lb.get("message", "No leaderboard data yet."))


# ══════════════════════════════════════════════════════════════
# TAB 5 — AUDIT TRAIL
# ══════════════════════════════════════════════════════════════
with tab_audit:
    st.header("🔒 Cryptographic audit trail")
    if st.button("🔄 Refresh audit"):
        st.rerun()

    verify = api("/audit/verify")
    if verify.get("valid") is True:
        st.success(
            f"✅ Audit chain intact — "
            f"{verify.get('entries', 0)} entries verified.")
    elif verify.get("valid") is False:
        st.error(
            f"🚨 Chain broken at id={verify.get('broken_at')}! "
            f"Possible tampering.")
    else:
        st.info("Audit chain status unknown.")

    audit_sid  = st.text_input(
        "Filter by session ID (leave blank for last 100)")
    audit_data = api("/audit",
                     params={"session_id": audit_sid}
                     if audit_sid else {})
    audit_rows = audit_data.get("audit", []) or []
    if audit_rows:
        au_df     = pd.DataFrame(audit_rows)
        disp_cols = [c for c in
                     ["id", "session_id", "event_type",
                      "timestamp", "entry_hash", "prev_hash"]
                     if c in au_df.columns]
        st.dataframe(au_df[disp_cols],
                     use_container_width=True, height=350)
        st.caption("entry_hash = SHA-256(event + prev_hash). "
                   "Any modification breaks the chain.")
    else:
        st.info("No audit entries yet.")


# ══════════════════════════════════════════════════════════════
# TAB 6 — BEHAVIORAL REPORT
# ══════════════════════════════════════════════════════════════
with tab_beh:
    st.header("🧠 Carbon behavior analytics")
    if st.button("🔄 Refresh report"):
        st.rerun()

    beh_rid = st.text_input("Researcher ID (leave blank for all)")
    beh     = api("/behavior",
                  params={"researcher_id": beh_rid} if beh_rid else {})

    if beh.get("total_runs"):
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total runs",   beh["total_runs"])
        b2.metric("Total CO₂",    fmt_val(beh["total_co2_g"],    "g"))
        b3.metric("Total wasted", fmt_val(beh["total_wasted_g"], "g"))
        b4.metric("Waste %",      fmt_val(beh["waste_pct"],      "%"))

        st.markdown("---")
        ca, cb = st.columns(2)
        with ca:
            st.metric("Duplicate runs",
                      beh["duplicate_runs"],
                      delta=f"{beh['duplicate_waste_g']}g CO₂ wasted",
                      delta_color="inverse")
            st.caption("Same model+epochs+batch within 10 min.")
        with cb:
            st.metric("Late-night runs (10 PM–5 AM)",
                      beh["night_runs"],
                      delta=f"{beh['night_waste_g']}g wasted",
                      delta_color="inverse")
        st.info(f"💡 {beh['insight']}")
    elif beh.get("message"):
        st.info(beh["message"])
    else:
        st.info("No completed runs to analyse yet.")
