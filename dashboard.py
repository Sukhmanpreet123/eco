"""
EcoTrace v3 — Streamlit Dashboard
All features: live heartbeat, 5 metric cards, budget guard,
fingerprint comparison, Carbon DNA match, anomaly markers,
carbon debt equivalents, Shapley attribution, pre-run estimator,
SLA manager, leaderboard, behavioral report, audit trail.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────
RENDER_URL   = "https://eco-2-4re9.onrender.com"
GRID_FALLBACK = 475.0
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EcoTrace v3",
    layout="wide",
    page_icon="🌱",
)


# ── SAFE API FETCH ────────────────────────────────────────────
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
    """Format a number cleanly — avoids truncation."""
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


# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.title("🌱 EcoTrace v3")

device_res   = api("/active_devices")
active_list  = device_res.get("devices", []) or ["google-colab"]
target       = st.sidebar.selectbox("Select device to monitor", active_list)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Budget settings")
BUDGET_G = st.sidebar.slider("Carbon budget (g CO₂)", 10, 500, 100, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Pre-run estimator")
est_task    = st.sidebar.text_input("Task type",    "image_classification")
est_model   = st.sidebar.text_input("Model name",   "ResNet-50")
est_epochs  = st.sidebar.number_input("Epochs",  1, 500, 25)
est_batch   = st.sidebar.number_input("Batch size", 1, 512, 32)
if st.sidebar.button("Estimate CO₂ before training ↗"):
    est = api("/estimate", params={
        "task_type": est_task, "model_name": est_model,
        "epochs": est_epochs, "batch_size": est_batch})
    if est.get("similar_count", 0) > 0:
        st.sidebar.success(
            f"Based on {est['similar_count']} similar runs:\n"
            f"CO₂: {est['co2_min_g']}g – {est['co2_max_g']}g "
            f"(avg {est['co2_avg_g']}g)\n"
            f"Duration: {est['duration_min_mins']}–{est['duration_max_mins']} min")
        bc = est.get("best_config", {})
        if bc:
            st.sidebar.info(
                f"Most efficient config seen:\n"
                f"batch={bc['batch_size']}, epochs={bc['epochs']} → "
                f"{bc['total_co2_g']}g CO₂, "
                f"{round(bc['final_accuracy']*100,1)}% acc [Grade {bc['grade']}]")
    else:
        st.sidebar.info("No past runs to estimate from yet.")

st.sidebar.markdown("---")
# Project totals
fp_all      = api("/fingerprint/all")
fp_list     = fp_all.get("runs", []) or []
proj_co2    = sum(r.get("total_co2_g") or 0 for r in fp_list)
proj_wasted = sum(r.get("wasted_co2_g") or 0 for r in fp_list)
st.sidebar.metric("Project total CO₂", fmt_val(proj_co2, "g"),
                  help="All completed runs")
st.sidebar.metric("Project wasted CO₂", fmt_val(proj_wasted, "g"),
                  help="Carbon burned after overfitting")
st.sidebar.caption(f"{len(fp_list)} completed run(s)")

# ── SESSION STATE ─────────────────────────────────────────────
if (st.session_state.get("last_device") != target):
    st.session_state.history     = pd.DataFrame(columns=["Time", "Power"])
    st.session_state.last_device = target
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Power"])

# ── PAGE TABS ─────────────────────────────────────────────────
tab_live, tab_history, tab_sla, tab_leaderboard, tab_audit, tab_behavior = \
    st.tabs(["📡 Live Monitor", "🔬 Run History",
             "📋 Carbon SLA", "🏆 Leaderboard",
             "🔒 Audit Trail", "🧠 Behavior Report"])


# ═════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ═════════════════════════════════════════════════════════════
with tab_live:
    st.title("🌱 EcoTrace — Real-Time AI Carbon Governance")
    placeholder = st.empty()

    while True:
        res    = api(f"/predict", params={"session_id": target})
        budget = api("/budget_check",
                     params={"session_id": target, "budget_g": BUDGET_G})
        anom   = api("/anomalies", params={"session_id": target})
        cmp    = api("/fingerprint/compare",
                     params={"session_id": target})
        shap   = api("/shapley", params={"session_id": target})
        dna_r  = {}

        curr_w = res.get("current_avg_w", 0) or 0
        new_row = {"Time": datetime.now().strftime("%H:%M:%S"), "Power": curr_w}
        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([new_row])],
            ignore_index=True).tail(40)

        # Live DNA match from current readings
        live_powers = st.session_state.history["Power"].tolist()
        if len(live_powers) >= 5:
            dna_r = api("/dna/match", method="POST",
                        json_body={"powers": live_powers})

        with placeholder.container():

            # ── Row 1: 5 metric cards ─────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Current power",
                      fmt_val(curr_w, "W"),
                      help="Avg wattage in last window")
            c2.metric("Predicted (1 hr)",
                      fmt_val(res.get("predicted_w"), "W"),
                      help="Linear regression forecast")

            carbon_val = res.get("carbon_g_hr") or 0
            c3.metric("Carbon rate",
                      fmt_val(carbon_val, "g/hr"),
                      delta="above avg" if carbon_val > 5 else "normal",
                      delta_color="inverse" if carbon_val > 5 else "normal")

            best_co2 = budget.get("best_past_co2")
            proj_co2_live = budget.get("projected_co2", 0) or 0
            if best_co2 and float(best_co2) >= 1.0 and proj_co2_live >= 1.0:
                dp    = round(((proj_co2_live - best_co2) / best_co2) * 100, 1)
                dp    = max(min(dp, 999.0), -999.0)
                sign  = "+" if dp > 0 else ""
                c4.metric("Efficiency vs best",
                          f"{sign}{dp}% CO₂",
                          delta="worse" if dp > 0 else "better",
                          delta_color="inverse" if dp > 0 else "normal")
            else:
                c4.metric("Efficiency vs best", "—",
                          help="Complete a full run (≥1g) to set baseline")

            elapsed_hrs  = len(st.session_state.history) * 5 / 3600
            sess_co2     = round((curr_w / 1000) * GRID_FALLBACK * elapsed_hrs, 4)
            c5.metric("Session CO₂",
                      fmt_val(sess_co2, "g"),
                      delta=f"of {BUDGET_G}g budget")

            st.write("---")

            # ── Row 2: Alert banners ───────────────────────
            b_status = budget.get("status", "green")
            b_rec    = budget.get("recommendation", "")
            if b_status == "red":
                st.error(f"🛑 Budget alert — {b_rec}")
            elif b_status == "yellow":
                st.warning(f"⚠️ Budget warning — {b_rec}")

            # Fingerprint banner
            similar = cmp.get("similar_runs", []) or []
            if similar:
                best_run = similar[0]
                bco2     = best_run.get("total_co2_g") or 0
                bacc     = best_run.get("final_accuracy") or 0
                bgrade   = best_run.get("efficiency_grade", "?")
                short_id = str(best_run.get("run_id", ""))[-20:]
                acc_str  = f"{round(bacc*100,1)}%" if bacc else "N/A"
                if proj_co2_live >= 1.0 and bco2 >= 1.0:
                    pct = round(((proj_co2_live - bco2) / bco2) * 100, 1)
                    pct = max(min(pct, 999.0), -999.0)
                    if pct > 20:
                        st.warning(
                            f"📊 Fingerprint alert — projected "
                            f"**{pct}% more CO₂** than best run "
                            f"(...{short_id}: {bco2}g → {acc_str}, Grade {bgrade}). "
                            f"Consider reducing epochs or batch size.")
                    else:
                        st.success(
                            f"📊 On track — similar to ...{short_id} "
                            f"({bco2}g CO₂, {acc_str}, Grade {bgrade}).")
                else:
                    st.info("📊 Past runs found but none ≥1g yet. "
                            "Call end_session() to save your first fingerprint.")

            # DNA match banner
            if dna_r.get("match"):
                m = dna_r["match"]
                st.info(
                    f"🧬 Carbon DNA match ({round(m['similarity']*100,1)}% similar) — "
                    f"{dna_r.get('prediction','')}")

            # Shapley banner
            if shap.get("co2_fair_g") is not None:
                saved = shap.get("co2_saved_g", 0) or 0
                if saved > 0:
                    st.success(
                        f"⚖️ Shapley attribution: your fair CO₂ share = "
                        f"{shap['co2_fair_g']}g "
                        f"(saved {shap['co2_saved_g']}g vs naive full attribution)")

            # ── Row 3: Chart + Carbon debt ─────────────────
            col_chart, col_debt = st.columns([3, 2])

            with col_chart:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.history["Time"],
                    y=st.session_state.history["Power"],
                    mode="lines+markers",
                    line=dict(color="#2ecc71", width=2),
                    marker=dict(size=4),
                    name="Power (W)"))

                anomalies = anom.get("anomalies", []) or []
                if anomalies:
                    hist_times = set(st.session_state.history["Time"].tolist())
                    at = [a["timestamp"] for a in anomalies
                          if a.get("timestamp") in hist_times]
                    ap = [a["power_w"] for a in anomalies
                          if a.get("timestamp") in hist_times]
                    if at:
                        fig.add_trace(go.Scatter(
                            x=at, y=ap, mode="markers",
                            marker=dict(color="red", size=10, symbol="x"),
                            name=f"Anomaly ({len(anomalies)})"))

                fig.update_layout(
                    title=f"Live heartbeat: {target}",
                    template="plotly_dark", height=280,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig, use_container_width=True)

                if anomalies:
                    st.caption(
                        f"⚠️ {len(anomalies)} anomaly event(s) detected — "
                        f"wasted carbon logged to audit trail.")

            with col_debt:
                st.subheader("🌍 Carbon debt")
                total_g  = max(sess_co2, 0.001)
                car_km   = total_g * 0.00417
                phone_x  = total_g / 5.5
                tree_min = total_g / 0.0095
                d1, d2, d3 = st.columns(3)
                d1.metric("🚗 Car km",    fmt_val(car_km, "km", 3))
                d2.metric("📱 Charges",   fmt_val(phone_x, "×", 2))
                d3.metric("🌳 Tree abs.", fmt_val(tree_min, "min", 1))

                if proj_co2 > 0:
                    st.markdown("---")
                    st.caption("Project total")
                    pc = round(proj_co2 * 0.00417, 2)
                    pt = round(proj_co2 / 0.0095 / 60, 1)
                    st.metric("All runs", fmt_val(proj_co2, "g CO₂"),
                              delta=f"≈ {pc} km driving")
                    st.caption(f"Tree needs {pt} hrs to absorb.")

            # ── Row 4: Fingerprint table + Grid ───────────
            col_fp, col_grid = st.columns(2)

            with col_fp:
                st.subheader("🔬 Recent run fingerprints")
                if fp_list:
                    fp_df = pd.DataFrame(fp_list)
                    cols  = [c for c in
                             ["run_id", "model_name", "epochs",
                              "total_co2_g", "wasted_co2_g",
                              "final_accuracy", "efficiency_grade", "timestamp"]
                             if c in fp_df.columns]
                    disp = fp_df[cols].copy()
                    disp.columns = [c.replace("_", " ").title()
                                    for c in cols]
                    st.dataframe(disp, use_container_width=True, height=220)
                else:
                    st.info("No completed runs yet. "
                            "Call end_session() in your notebook.")

            with col_grid:
                st.subheader("⚡ Grid intensity")
                gi = GRID_FALLBACK
                if gi > 400:
                    label   = "🔴 Coal-heavy"
                    advice  = f"Grid is dirty ({gi:.0f} g/kWh). Waiting saves ~30–40% CO₂."
                elif gi > 200:
                    label   = "🟡 Mixed grid"
                    advice  = "Grid is average. Training now is acceptable."
                else:
                    label   = "🟢 Clean grid"
                    advice  = "Great time to train!"
                st.metric("Current intensity", f"{gi:.0f} g/kWh", delta=label)
                st.caption(advice)
                sav = round((1 - 280 / gi) * 100, 1)
                if sav > 0:
                    st.info(f"Scheduling for cleaner grid (280 g/kWh) "
                            f"saves ~**{sav}%** CO₂.")
                st.caption("💡 Connect Electricity Maps API for real-time data.")

            st.write("---")
            st.caption(
                f"🔗 Cloud Tower connected | {target} | "
                f"Updated {datetime.now().strftime('%H:%M:%S')}")

        time.sleep(5)
        st.rerun()


# ═════════════════════════════════════════════════════════════
# TAB 2 — RUN HISTORY
# ═════════════════════════════════════════════════════════════
with tab_history:
    st.header("🔬 Run fingerprint history")
    if fp_list:
        fp_df = pd.DataFrame(fp_list)
        st.dataframe(fp_df, use_container_width=True)

        # Grade distribution
        if "efficiency_grade" in fp_df.columns:
            grade_counts = fp_df["efficiency_grade"].value_counts()
            fig_g = go.Figure(go.Bar(
                x=grade_counts.index.tolist(),
                y=grade_counts.values.tolist(),
                marker_color=["#2ecc71","#27ae60","#f39c12","#e67e22","#e74c3c"]))
            fig_g.update_layout(title="Efficiency grade distribution",
                                template="plotly_dark", height=250)
            st.plotly_chart(fig_g, use_container_width=True)

        # CO2 over time
        if "timestamp" in fp_df.columns and "total_co2_g" in fp_df.columns:
            fig_t = go.Figure(go.Scatter(
                x=fp_df["timestamp"], y=fp_df["total_co2_g"],
                mode="lines+markers", line=dict(color="#3498db"),
                name="CO₂ per run"))
            if "wasted_co2_g" in fp_df.columns:
                fig_t.add_trace(go.Bar(
                    x=fp_df["timestamp"], y=fp_df["wasted_co2_g"],
                    name="Wasted CO₂", marker_color="rgba(231,76,60,0.5)"))
            fig_t.update_layout(title="CO₂ per run over time",
                                template="plotly_dark", height=300)
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No completed runs yet.")


# ═════════════════════════════════════════════════════════════
# TAB 3 — CARBON SLA
# ═════════════════════════════════════════════════════════════
with tab_sla:
    st.header("📋 Carbon SLA manager")
    st.caption("Set maximum CO₂ and minimum accuracy targets per model. "
               "EcoTrace will flag any run that breaches these in the audit trail.")

    with st.form("sla_form"):
        sla_model   = st.text_input("Model name", "ResNet-50")
        sla_max_co2 = st.number_input("Max CO₂ (g)", 1.0, 10000.0, 50.0, step=1.0)
        sla_min_acc = st.number_input("Min accuracy (0–1)", 0.0, 1.0, 0.90, step=0.01)
        if st.form_submit_button("Save SLA"):
            r = api("/sla/set", method="POST",
                    json_body={"model_name": sla_model,
                               "max_co2_g": sla_max_co2,
                               "min_accuracy": sla_min_acc})
            if r.get("status") == "SLA saved":
                st.success(f"SLA saved for {sla_model}")
            else:
                st.error("Failed to save SLA.")

    st.subheader("Active SLAs")
    slas = api("/sla/all").get("slas", []) or []
    if slas:
        st.dataframe(pd.DataFrame(slas), use_container_width=True)
    else:
        st.info("No SLAs set yet.")


# ═════════════════════════════════════════════════════════════
# TAB 4 — LEADERBOARD
# ═════════════════════════════════════════════════════════════
with tab_leaderboard:
    st.header("🏆 Team carbon efficiency leaderboard")
    st.caption("Privacy-safe (k-anonymity): shown only when ≥3 researchers have runs.")
    lb = api("/leaderboard", params={"k": 3})
    board = lb.get("leaderboard", []) or []
    msg   = lb.get("message", "")
    if board:
        lb_df = pd.DataFrame(board)
        st.dataframe(lb_df, use_container_width=True)

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
        st.info(msg or "No leaderboard data yet.")


# ═════════════════════════════════════════════════════════════
# TAB 5 — AUDIT TRAIL
# ═════════════════════════════════════════════════════════════
with tab_audit:
    st.header("🔒 Cryptographic audit trail")

    verify = api("/audit/verify")
    if verify.get("valid") is True:
        st.success(
            f"✅ Audit chain intact — {verify.get('entries', 0)} entries verified.")
    elif verify.get("valid") is False:
        st.error(
            f"🚨 Chain broken at entry id={verify.get('broken_at')}! "
            f"Possible tampering detected.")
    else:
        st.info("Audit chain status unknown.")

    audit_sid = st.text_input("Filter by session ID (leave blank for last 100)")
    audit_data = api("/audit",
                     params={"session_id": audit_sid} if audit_sid else {})
    audit_rows = audit_data.get("audit", []) or []
    if audit_rows:
        au_df = pd.DataFrame(audit_rows)
        disp_cols = [c for c in
                     ["id", "session_id", "event_type", "timestamp",
                      "entry_hash", "prev_hash"]
                     if c in au_df.columns]
        st.dataframe(au_df[disp_cols], use_container_width=True, height=350)
        st.caption("entry_hash is SHA-256 of (event + prev_hash). "
                   "Any modification breaks the chain.")
    else:
        st.info("No audit entries yet.")


# ═════════════════════════════════════════════════════════════
# TAB 6 — BEHAVIORAL REPORT
# ═════════════════════════════════════════════════════════════
with tab_behavior:
    st.header("🧠 Carbon behavior analytics")
    st.caption("Analyses your training habits to find carbon waste patterns.")

    beh_rid = st.text_input("Researcher ID (leave blank for all)")
    beh     = api("/behavior",
                  params={"researcher_id": beh_rid} if beh_rid else {})

    if beh.get("total_runs"):
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total runs",    beh["total_runs"])
        b2.metric("Total CO₂",     fmt_val(beh["total_co2_g"], "g"))
        b3.metric("Total wasted",  fmt_val(beh["total_wasted_g"], "g"))
        b4.metric("Waste %",       fmt_val(beh["waste_pct"], "%"))

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Duplicate runs detected",
                      beh["duplicate_runs"],
                      delta=f"{beh['duplicate_waste_g']}g CO₂ wasted",
                      delta_color="inverse")
            st.caption("Duplicate = same model+epochs+batch within 10 min.")
        with col_b:
            st.metric("Late-night runs (10 PM–5 AM)",
                      beh["night_runs"],
                      delta=f"{beh['night_waste_g']}g wasted at night",
                      delta_color="inverse")
            st.caption("Night runs often have higher waste % (tired researcher effect).")

        st.info(f"💡 Insight: {beh['insight']}")
    elif beh.get("message"):
        st.info(beh["message"])
    else:
        st.info("No completed runs to analyse yet.")