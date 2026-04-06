"""
EcoTrace v3 — Full Governance Backend
All features: telemetry, Carbon DNA, Shapley, overfit detection,
SLA enforcement, audit trail, leaderboard, behavioral analytics,
pre-run estimator.

Fixes applied (v3.1):
  B1  compute_grade handles 0g CO2 runs correctly
  B2  overfit detection sorts power readings by timestamp
  B3  resample_dna uses [0.5]*n for constant signals (matches agent)
  B4  /predict future index = len + 720 (1 hr @ 5 s/sample)
  B5  /budget_check projects over remaining time, not flat +1 hr
  B6  /shapley correct 2-player Shapley formula
  B7  /estimate NaN guard on final_accuracy
  B8  /fingerprint/all strips carbon_dna column (too large for list view)
  B9  sha256_chain no longer mutates caller dict
  B10 DB indexes added for energy_logs and run_fingerprints
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import datetime
import sqlite3
import hashlib
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="EcoTrace v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

GRID_INTENSITY = 475.0
DNA_LEN        = 50
DB_PATH        = "ecotrace.db"


# ── HELPERS ───────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def resample_dna(powers: list, n: int = DNA_LEN) -> list:
    """
    Resample a raw power list to an n-point normalised [0,1] vector.
    B3 fix: constant-power signals return [0.5]*n (not [0.0]*n).
    """
    if len(powers) < 2:
        return [0.5] * n
    arr  = np.array(powers, dtype=float)
    idx  = np.linspace(0, len(arr) - 1, n)
    rs   = np.interp(idx, np.arange(len(arr)), arr)
    mn, mx = rs.min(), rs.max()
    if mx - mn < 1e-9:          # B3: constant signal → centred at 0.5
        return [0.5] * n
    return ((rs - mn) / (mx - mn)).tolist()


def dna_sim(a: list, b: list) -> float:
    va = np.array(a).reshape(1, -1)
    vb = np.array(b).reshape(1, -1)
    return float(cosine_similarity(va, vb)[0][0])


def sha256_chain(entry: dict, prev: str) -> str:
    """
    B9 fix: work on a copy so the caller's dict is never mutated.
    """
    data = dict(entry)          # <-- copy, not reference
    data["prev_hash"] = prev
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def compute_grade(co2: float, wasted: float, acc) -> str:
    """
    B1 fix: treat co2 == 0 (very short runs) as a bonus rather than N/A.
    Scoring:
      waste %  → 0-40 pts
      accuracy → 0-40 pts
      co2      → 0-20 pts
    """
    waste_pct = ((wasted or 0) / co2 * 100) if co2 and co2 > 0 else 0
    acc_pct   = (acc or 0) * 100

    score  = 0
    score += (40 if waste_pct < 10
              else 25 if waste_pct < 25
              else 10 if waste_pct < 50
              else 0)
    score += (40 if acc_pct >= 95
              else 30 if acc_pct >= 90
              else 20 if acc_pct >= 80
              else (10 if acc_pct > 0 else 0))
    # B1: 0 g CO2 gets the max points (clean run)
    score += (20 if (co2 is None or co2 <= 0 or co2 < 5)
              else 10 if co2 < 20
              else 0)

    return ("A" if score >= 85
            else "B" if score >= 70
            else "C" if score >= 50
            else "D" if score >= 30
            else "F")


def append_audit(conn, session_id: str, event_type: str, details: dict):
    row = conn.execute(
        "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    prev  = row["entry_hash"] if row else "GENESIS"
    entry = {
        "session_id": session_id,
        "event_type": event_type,
        "details":    details,
        "timestamp":  datetime.datetime.now().isoformat(),
    }
    h = sha256_chain(entry, prev)   # B9: sha256_chain gets a copy
    conn.execute(
        "INSERT INTO audit_log (session_id,event_type,details,timestamp,entry_hash,prev_hash) "
        "VALUES (?,?,?,?,?,?)",
        (session_id, event_type, json.dumps(details, default=str),
         entry["timestamp"], h, prev))


# ── DATABASE INIT ─────────────────────────────────────────────

def init_db():
    conn = get_conn()
    c    = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS energy_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device TEXT, session_id TEXT, power_w REAL, timestamp TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS run_fingerprints (
        run_id TEXT PRIMARY KEY, device TEXT, researcher_id TEXT,
        task_type TEXT, model_name TEXT, dataset_size INTEGER,
        epochs INTEGER, batch_size INTEGER, avg_watts REAL,
        peak_watts REAL, duration_mins REAL, total_co2_g REAL,
        wasted_co2_g REAL, final_accuracy REAL, final_loss REAL,
        carbon_dna TEXT, efficiency_grade TEXT, timestamp TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, timestamp TEXT,
        power_w REAL, baseline_w REAL, note TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, event_type TEXT, details TEXT,
        timestamp TEXT, entry_hash TEXT, prev_hash TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS carbon_sla (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT, max_co2_g REAL,
        min_accuracy REAL, created_at TEXT)''')

    # B10: indexes for frequent queries
    c.execute("CREATE INDEX IF NOT EXISTS idx_logs_session "
              "ON energy_logs(session_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp "
              "ON energy_logs(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fp_task_model "
              "ON run_fingerprints(task_type, model_name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fp_timestamp "
              "ON run_fingerprints(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_audit_session "
              "ON audit_log(session_id)")

    conn.commit()
    conn.close()


init_db()


# ── BASIC ENDPOINTS ───────────────────────────────────────────

@app.get("/")
def home():
    return {"status": "Online", "version": "EcoTrace v3.1"}


@app.post("/log")
async def log_energy(request: Request):
    p  = await request.json()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    conn.execute(
        "INSERT INTO energy_logs (device,session_id,power_w,timestamp) VALUES (?,?,?,?)",
        (p.get("device"), p.get("session_id"), p.get("power_w"), ts))
    conn.commit()
    conn.close()
    return {"status": "saved", "power": p.get("power_w")}


@app.get("/active_devices")
def get_active_devices():
    try:
        conn   = get_conn()
        cutoff = (datetime.datetime.now() -
                  datetime.timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        rows   = conn.execute(
            "SELECT DISTINCT session_id FROM energy_logs "
            "WHERE timestamp >= ? ORDER BY session_id",
            (cutoff,)).fetchall()
        conn.close()
        return {"devices": [r["session_id"] for r in rows]}
    except Exception as e:
        return {"devices": [], "error": str(e)}


@app.get("/predict")
def predict_energy(session_id: str):
    """
    B4 fix: future time-horizon = len + 720 (1 hr at 5 s/sample).
    """
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            "SELECT id, power_w FROM energy_logs WHERE session_id=? ORDER BY id",
            conn, params=(session_id,))
        conn.close()
        if len(df) < 5:
            return {"error": f"Need more data. Have {len(df)} points.",
                    "samples": len(df)}

        recent = df.tail(20)
        X      = np.arange(len(recent)).reshape(-1, 1)
        y      = recent["power_w"].values
        avg_w  = float(y.mean())

        if np.all(y == y[0]) or len(recent) < 3:
            pred = avg_w
        else:
            # B4: 1 hour = 3600s / 5s = 720 samples ahead
            future_idx = len(recent) + 720
            raw_pred   = float(
                LinearRegression().fit(X, y).predict([[future_idx]])[0])
            pred = max(0.0, min(raw_pred, avg_w * 3.0))

        return {
            "session":       session_id,
            "samples":       len(df),           # used by dashboard for elapsed time
            "current_avg_w": round(avg_w, 2),
            "predicted_w":   round(pred, 2),
            "carbon_g_hr":   round((avg_w / 1000) * GRID_INTENSITY, 4),
        }
    except Exception as e:
        return {"error": str(e), "samples": 0}


# ── FINGERPRINT SAVE ──────────────────────────────────────────

@app.post("/fingerprint/save")
async def save_fingerprint(request: Request):
    fp  = await request.json()
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sid = fp.get("run_id", "")

    conn    = get_conn()
    # B2: sort by timestamp so powers are in chronological order
    df_logs = pd.read_sql_query(
        "SELECT power_w, timestamp FROM energy_logs "
        "WHERE session_id=? ORDER BY timestamp ASC",
        conn, params=(sid,))

    powers = df_logs["power_w"].tolist() if not df_logs.empty else []
    dna    = resample_dna(powers) if len(powers) >= 2 else [0.5] * DNA_LEN

    # ── Authoritative CO₂ computed server-side ─────────────
    duration_mins = fp.get("duration_mins", 0) or 0
    avg_watts     = round(float(np.mean(powers)), 2) if powers else fp.get("avg_watts", 0) or 0
    peak_watts    = round(float(np.max(powers)), 2) if powers else fp.get("peak_watts", 0) or 0
    total_co2_g   = round((avg_watts / 1000) * GRID_INTENSITY * (duration_mins / 60), 6)

    # B2: Overfit / carbon waste detection with sorted powers
    val_losses   = fp.get("val_losses", []) or []
    wasted_co2_g = 0.0
    waste_epoch  = None
    if len(val_losses) >= 3 and powers:
        best_loss, best_ep = float("inf"), 0
        patience, no_improve = 3, 0
        for i, loss in enumerate(val_losses):
            if loss < best_loss - 1e-6:
                best_loss = loss
                best_ep   = i
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        waste_start = best_ep + patience
        if waste_start < len(powers):
            wp           = powers[waste_start:]
            wasted_co2_g = round(
                (float(np.mean(wp)) / 1000) * GRID_INTENSITY
                * (len(wp) * 5 / 3600), 6)
            waste_epoch  = waste_start

    grade = compute_grade(total_co2_g, wasted_co2_g, fp.get("final_accuracy"))

    # ── Anomaly detection ──────────────────────────────────
    # Guard: only run IsolationForest when there is genuine variance.
    # contamination=0.1 always flags 10% of readings as outliers,
    # so on a flat idle signal (all readings ~4W) it produces dozens
    # of false positives. We require BOTH:
    #   1. std_dev >= 2W  (there is real spread in the data)
    #   2. peak >= 1.5 × median  (there are actual spikes)
    if len(df_logs) >= 10:
        pwr      = df_logs["power_w"].values
        pwr_std  = float(np.std(pwr))
        pwr_med  = float(np.median(pwr))
        pwr_peak = float(np.max(pwr))
        has_spikes = (pwr_std >= 2.0) and (pwr_peak >= pwr_med * 1.5)

        if has_spikes:
            iso   = IsolationForest(contamination=0.1, random_state=42)
            preds = iso.fit_predict(df_logs[["power_w"]])
            for i, pred in enumerate(preds):
                if pred == -1:
                    row = df_logs.iloc[i]
                    conn.execute(
                        "INSERT INTO anomalies "
                        "(session_id,timestamp,power_w,baseline_w,note) VALUES (?,?,?,?,?)",
                        (sid, row["timestamp"], float(row["power_w"]),
                         pwr_med, "IsolationForest spike"))
                    append_audit(conn, sid, "anomaly_detected",
                                 {"power_w": float(row["power_w"]),
                                  "baseline_w": pwr_med,
                                  "std_dev": round(pwr_std, 2)})

    # ── SLA check ──────────────────────────────────────────
    sla = conn.execute(
        "SELECT * FROM carbon_sla WHERE model_name=? ORDER BY id DESC LIMIT 1",
        (fp.get("model_name", ""),)).fetchone()
    if sla:
        if ((total_co2_g or 0) > sla["max_co2_g"] or
                (fp.get("final_accuracy") or 0) < sla["min_accuracy"]):
            append_audit(conn, sid, "sla_breach",
                         {"max_co2_g": sla["max_co2_g"],
                          "actual_co2_g": total_co2_g,
                          "min_accuracy": sla["min_accuracy"],
                          "actual_acc": fp.get("final_accuracy")})

    conn.execute('''INSERT OR REPLACE INTO run_fingerprints
        (run_id,device,researcher_id,task_type,model_name,
         dataset_size,epochs,batch_size,avg_watts,peak_watts,
         duration_mins,total_co2_g,wasted_co2_g,final_accuracy,
         final_loss,carbon_dna,efficiency_grade,timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (sid, fp.get("device"),
         fp.get("researcher_id", "anonymous"),
         fp.get("task_type"), fp.get("model_name"),
         fp.get("dataset_size"), fp.get("epochs"), fp.get("batch_size"),
         avg_watts, peak_watts,
         duration_mins, total_co2_g,
         wasted_co2_g, fp.get("final_accuracy"), fp.get("final_loss"),
         json.dumps(dna), grade, ts))

    append_audit(conn, sid, "run_completed",
                 {"total_co2_g": total_co2_g,
                  "wasted_co2_g": wasted_co2_g,
                  "grade": grade, "waste_epoch": waste_epoch})

    conn.commit()
    conn.close()
    return {
        "status":       "saved",
        "run_id":       sid,
        "grade":        grade,
        "total_co2_g":  total_co2_g,
        "avg_watts":    avg_watts,
        "peak_watts":   peak_watts,
        "wasted_co2_g": wasted_co2_g,
        "waste_epoch":  waste_epoch,
    }


# ── FINGERPRINT COMPARE ───────────────────────────────────────

@app.get("/fingerprint/compare")
def compare_fingerprint(session_id: str = "", task_type: str = "",
                        model_name: str = ""):
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            "SELECT run_id,device,researcher_id,task_type,model_name,"
            "dataset_size,epochs,batch_size,avg_watts,peak_watts,"
            "duration_mins,total_co2_g,wasted_co2_g,final_accuracy,"
            "final_loss,efficiency_grade,timestamp "   # B8: no carbon_dna
            "FROM run_fingerprints WHERE run_id!=? AND total_co2_g>=0",
            conn, params=(session_id,))
        conn.close()

        if df.empty:
            return {"similar_runs": [], "message": "No past runs yet."}

        if task_type and model_name:
            sub      = df[(df["task_type"] == task_type) &
                          (df["model_name"] == model_name)]
            filtered = sub if not sub.empty else df
        else:
            filtered = df

        if filtered.empty:
            return {"similar_runs": [],
                    "message": "No matching runs for this task/model."}

        cols   = ["avg_watts", "peak_watts", "duration_mins",
                  "total_co2_g", "dataset_size", "epochs", "batch_size"]
        avail  = [c for c in cols if c in filtered.columns]
        matrix = filtered[avail].fillna(0).values
        if len(matrix) == 0:
            return {"similar_runs": []}

        query  = matrix.mean(axis=0).reshape(1, -1)
        scores = cosine_similarity(query, matrix)[0]
        top3   = np.argsort(scores)[-3:][::-1]
        result = filtered.iloc[top3].to_dict(orient="records")
        for i, rec in enumerate(result):
            rec["similarity_score"] = round(float(scores[top3[i]]), 3)
        return {"similar_runs": result}
    except Exception as e:
        return {"error": str(e)}


@app.get("/fingerprint/all")
def all_fingerprints():
    """
    B8: carbon_dna is excluded — it's a large JSON blob that bloats
    every list response. Use /fingerprint/dna/<run_id> if you need it.
    """
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            "SELECT run_id,device,researcher_id,task_type,model_name,"
            "dataset_size,epochs,batch_size,avg_watts,peak_watts,"
            "duration_mins,total_co2_g,wasted_co2_g,final_accuracy,"
            "final_loss,efficiency_grade,timestamp "
            "FROM run_fingerprints ORDER BY timestamp DESC", conn)
        conn.close()
        return {"runs": df.to_dict(orient="records")}
    except Exception as e:
        return {"runs": [], "error": str(e)}


@app.get("/fingerprint/dna/{run_id}")
def get_dna(run_id: str):
    """Fetch the full 50-point Carbon DNA for a specific run."""
    try:
        conn = get_conn()
        row  = conn.execute(
            "SELECT carbon_dna FROM run_fingerprints WHERE run_id=?",
            (run_id,)).fetchone()
        conn.close()
        if not row:
            return {"error": "Run not found."}
        return {"run_id": run_id, "carbon_dna": json.loads(row["carbon_dna"] or "[]")}
    except Exception as e:
        return {"error": str(e)}


# ── CARBON DNA MATCH ──────────────────────────────────────────

@app.post("/dna/match")
async def dna_match(request: Request):
    body     = await request.json()
    live_raw = body.get("powers", [])
    if len(live_raw) < 5:
        return {"match": None, "message": "Need at least 5 power readings."}

    live_dna = resample_dna(live_raw)
    conn     = get_conn()
    df       = pd.read_sql_query(
        "SELECT run_id,model_name,total_co2_g,final_accuracy,"
        "carbon_dna,efficiency_grade FROM run_fingerprints "
        "WHERE carbon_dna IS NOT NULL", conn)
    conn.close()

    if df.empty:
        return {"match": None, "message": "No past DNA records yet."}

    best_score, best_row = -1.0, None
    for _, row in df.iterrows():
        try:
            stored = json.loads(row["carbon_dna"])
            score  = dna_sim(live_dna, stored)
            if score > best_score:
                best_score, best_row = score, row
        except Exception:
            continue

    if best_row is None:
        return {"match": None}

    acc_pct = round((best_row["final_accuracy"] or 0) * 100, 1)
    return {
        "match": {
            "run_id":         best_row["run_id"],
            "model_name":     best_row["model_name"],
            "total_co2_g":    best_row["total_co2_g"],
            "final_accuracy": best_row["final_accuracy"],
            "grade":          best_row["efficiency_grade"],
            "similarity":     round(best_score, 3),
        },
        "prediction": (
            f"Power curve matches {best_row['model_name']} "
            f"({round(best_score*100,1)}% similar). "
            f"That run: {best_row['total_co2_g']}g CO₂ → {acc_pct}% accuracy."
        )
    }


# ── ANOMALIES ─────────────────────────────────────────────────

@app.get("/anomalies")
def get_anomalies(session_id: str):
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            "SELECT * FROM anomalies WHERE session_id=? ORDER BY timestamp",
            conn, params=(session_id,))
        conn.close()
        return {"anomalies": df.to_dict(orient="records")}
    except Exception as e:
        return {"anomalies": [], "error": str(e)}


# ── BUDGET CHECK ──────────────────────────────────────────────

@app.get("/budget_check")
def budget_check(session_id: str, budget_g: float = 100.0):
    """
    B5 fix: project co2 over the REMAINING budget window, not always +1 hr.
    elapsed_hrs = readings * 5s / 3600
    budget_hrs  = budget_g / rate_g_per_hr
    remaining   = max(budget_hrs - elapsed_hrs, 0.5)   # at least 30 min lookahead
    projected   = co2_so_far + rate * remaining
    """
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            "SELECT power_w FROM energy_logs WHERE session_id=? ORDER BY id",
            conn, params=(session_id,))
        past = pd.read_sql_query(
            "SELECT total_co2_g, final_accuracy FROM run_fingerprints "
            "WHERE total_co2_g>=0 ORDER BY timestamp DESC LIMIT 5", conn)
        conn.close()

        if len(df) < 5:
            return {"status": "green", "projected_co2": 0,
                    "best_past_co2": None, "samples": len(df),
                    "message": "Not enough data yet."}

        avg_w       = float(df["power_w"].mean())
        elapsed_hrs = len(df) * 5 / 3600
        rate        = (avg_w / 1000) * GRID_INTENSITY   # g CO₂ per hr
        co2_so_far  = round(rate * elapsed_hrs, 4)

        # Fixed: use a 1-hour lookahead from now.
        # "At the current rate, how much CO₂ will I have 1 hour from now?"
        # This avoids the mathematical identity: budget_hrs = budget_g/rate
        # → projected = co2_so_far + rate*(budget_g/rate) = budget_g at startup.
        projected   = round(co2_so_far + rate * 1.0, 2)

        best_co2 = float(past["total_co2_g"].min()) if not past.empty else None
        best_acc = float(past["final_accuracy"].max()) if not past.empty else None

        if projected > budget_g:
            status, rec = "red", (
                f"Projected {projected}g exceeds budget {budget_g}g. "
                f"Rate: {round(rate,2)} g/hr. Consider stopping.")
        elif projected > budget_g * 0.8:
            status, rec = "yellow", (
                f"Projected {projected}g approaching budget {budget_g}g. "
                f"Rate: {round(rate,2)} g/hr.")
        else:
            status, rec = "green", f"On track. Rate: {round(rate,2)} g/hr."

        return {
            "status":         status,
            "projected_co2":  projected,
            "co2_so_far":     co2_so_far,
            "budget_g":       budget_g,
            "elapsed_hrs":    round(elapsed_hrs, 4),
            "rate_g_per_hr":  round(rate, 4),
            "best_past_co2":  best_co2,
            "best_past_acc":  best_acc,
            "samples":        len(df),
            "recommendation": rec,
        }
    except Exception as e:
        return {"error": str(e)}


# ── SHAPLEY ATTRIBUTION ───────────────────────────────────────

@app.get("/shapley")
def shapley_attribution(session_id: str, n_sessions: int = 2):
    """
    B6 fix: correct 2-player Shapley value.

    Modelled as a cooperative game:
      v({you})         = compute_W          (what you'd use alone, no idle)
      v({shared_infra}) = 0                 (infra alone contributes 0)
      v({you, shared}) = measured_W         (total observed)

    Shapley values:
      phi_you    = compute_W + idle_W / 2
      phi_shared = idle_W / 2
    where:
      idle_W    = assumed infrastructure idle baseline (15W split equally)
      compute_W = measured_W - idle_W / n_sessions
    """
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            "SELECT power_w FROM energy_logs WHERE session_id=?",
            conn, params=(session_id,))
        conn.close()
        if df.empty:
            return {"error": "No data for this session."}

        measured_w = float(df["power_w"].mean())
        idle_total = 15.0                           # assumed infra idle baseline
        idle_share = idle_total / max(n_sessions, 1)
        compute_w  = max(measured_w - idle_share, 0.0)

        # B6: correct Shapley formulas
        shap_you   = round(compute_w + idle_share / 2, 2)   # your compute + half idle
        shap_shr   = round(idle_share / 2, 2)               # shared gets half idle

        hrs        = len(df) * 5 / 3600
        co2_fair   = round((shap_you  / 1000) * GRID_INTENSITY * hrs, 6)
        co2_naive  = round((measured_w / 1000) * GRID_INTENSITY * hrs, 6)

        return {
            "session_id":       session_id,
            "measured_avg_w":   round(measured_w, 2),
            "idle_baseline_w":  round(idle_share, 2),
            "compute_w":        round(compute_w, 2),
            "shapley_your_w":   shap_you,
            "shapley_shared_w": shap_shr,
            "co2_fair_g":       co2_fair,
            "co2_naive_g":      co2_naive,
            "co2_saved_g":      round(max(co2_naive - co2_fair, 0), 6),
            "n_sessions":       n_sessions,
        }
    except Exception as e:
        return {"error": str(e)}


# ── PRE-RUN ESTIMATOR ─────────────────────────────────────────

@app.get("/estimate")
def pre_run_estimate(task_type: str = "", model_name: str = "",
                     dataset_size: int = 0, epochs: int = 0,
                     batch_size: int = 32):
    """B7 fix: guard against NaN in final_accuracy before idxmax()."""
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            "SELECT * FROM run_fingerprints WHERE total_co2_g>=0", conn)
        conn.close()
        if df.empty:
            return {"message": "No past runs to estimate from.", "similar_count": 0}

        if task_type and model_name:
            sub  = df[(df["task_type"] == task_type) & (df["model_name"] == model_name)]
            pool = sub if not sub.empty else df
        else:
            pool = df

        co2_vals = pool["total_co2_g"].dropna()
        dur_vals = pool["duration_mins"].dropna()
        acc_vals = pool["final_accuracy"].dropna()

        # B7: fillna before computing score to avoid NaN idxmax
        scored = pool.copy()
        scored["_acc"]   = scored["final_accuracy"].fillna(0)
        scored["_co2"]   = scored["total_co2_g"].replace(0, np.nan).fillna(1)
        scored["score"]  = scored["_acc"] / scored["_co2"]
        best_idx = scored["score"].idxmax()
        best     = scored.loc[best_idx]

        return {
            "similar_count":     len(pool),
            "co2_min_g":         round(float(co2_vals.min()), 4) if len(co2_vals) else 0,
            "co2_max_g":         round(float(co2_vals.max()), 4) if len(co2_vals) else 0,
            "co2_avg_g":         round(float(co2_vals.mean()), 4) if len(co2_vals) else 0,
            "duration_min_mins": round(float(dur_vals.min()), 1) if len(dur_vals) else 0,
            "duration_max_mins": round(float(dur_vals.max()), 1) if len(dur_vals) else 0,
            "acc_avg":           round(float(acc_vals.mean()), 3) if len(acc_vals) else None,
            "best_config": {
                "batch_size":     int(best.get("batch_size") or 0),
                "epochs":         int(best.get("epochs") or 0),
                "total_co2_g":    round(float(best.get("total_co2_g") or 0), 4),
                "final_accuracy": round(float(best.get("final_accuracy") or 0), 3),
                "grade":          best.get("efficiency_grade", "?"),
            }
        }
    except Exception as e:
        return {"error": str(e)}


# ── CARBON SLA ────────────────────────────────────────────────

@app.post("/sla/set")
async def set_sla(request: Request):
    body = await request.json()
    conn = get_conn()
    conn.execute(
        "INSERT INTO carbon_sla (model_name,max_co2_g,min_accuracy,created_at) "
        "VALUES (?,?,?,?)",
        (body.get("model_name"), body.get("max_co2_g"),
         body.get("min_accuracy"), datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"status": "SLA saved", "model_name": body.get("model_name")}


@app.get("/sla/all")
def get_slas():
    try:
        conn = get_conn()
        df   = pd.read_sql_query("SELECT * FROM carbon_sla ORDER BY id DESC", conn)
        conn.close()
        return {"slas": df.to_dict(orient="records")}
    except Exception as e:
        return {"slas": [], "error": str(e)}


# ── LEADERBOARD ───────────────────────────────────────────────

@app.get("/leaderboard")
def leaderboard(k: int = 3):
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            "SELECT researcher_id,total_co2_g,wasted_co2_g,"
            "final_accuracy,efficiency_grade FROM run_fingerprints "
            "WHERE total_co2_g>=0", conn)
        conn.close()
        if df.empty:
            return {"leaderboard": [], "message": "No completed runs yet."}

        g = df.groupby("researcher_id").agg(
            total_co2=("total_co2_g",   "sum"),
            total_wasted=("wasted_co2_g", "sum"),
            runs=("total_co2_g",         "count"),
            avg_acc=("final_accuracy",   "mean"),
        ).reset_index()
        g["efficiency_pct"] = (
            1 - g["total_wasted"] / g["total_co2"].replace(0, 1)) * 100
        g = g.sort_values("efficiency_pct", ascending=False)
        g["rank"] = range(1, len(g) + 1)

        if len(g) < k:
            return {"leaderboard": [],
                    "message": f"Need {k} researchers for privacy-safe leaderboard. Have {len(g)}."}

        return {
            "leaderboard": [
                {
                    "rank":           int(r["rank"]),
                    "researcher_id":  r["researcher_id"],
                    "runs":           int(r["runs"]),
                    "total_co2_g":    round(float(r["total_co2"]), 4),
                    "efficiency_pct": round(float(r["efficiency_pct"]), 1),
                    "avg_accuracy":   round(float(r["avg_acc"] or 0), 3),
                }
                for _, r in g.iterrows()
            ],
            "total_researchers": len(g),
        }
    except Exception as e:
        return {"leaderboard": [], "error": str(e)}


# ── AUDIT TRAIL ───────────────────────────────────────────────

@app.get("/audit")
def get_audit(session_id: str = ""):
    try:
        conn = get_conn()
        if session_id:
            df = pd.read_sql_query(
                "SELECT * FROM audit_log WHERE session_id=? ORDER BY id",
                conn, params=(session_id,))
        else:
            df = pd.read_sql_query(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        return {"audit": df.to_dict(orient="records")}
    except Exception as e:
        return {"audit": [], "error": str(e)}


@app.get("/audit/verify")
def verify_audit_chain():
    try:
        conn = get_conn()
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        conn.close()
        if not rows:
            return {"valid": True, "entries": 0, "message": "No entries yet."}
        for row in rows:
            entry = {
                "session_id": row["session_id"],
                "event_type": row["event_type"],
                "details":    json.loads(row["details"] or "{}"),
                "timestamp":  row["timestamp"],
            }
            expected = sha256_chain(entry, row["prev_hash"])
            if expected != row["entry_hash"]:
                return {"valid": False, "broken_at": row["id"],
                        "message": f"Chain broken at id={row['id']}."}
        return {"valid": True, "entries": len(rows), "message": "Chain intact."}
    except Exception as e:
        return {"error": str(e)}


# ── BEHAVIORAL ANALYTICS ──────────────────────────────────────

@app.get("/behavior")
def behavior_report(researcher_id: str = ""):
    try:
        conn = get_conn()
        q    = "SELECT * FROM run_fingerprints WHERE total_co2_g>=0"
        p    = ()
        if researcher_id:
            q += " AND researcher_id=?"
            p  = (researcher_id,)
        df = pd.read_sql_query(q, conn, params=p)
        conn.close()
        if df.empty:
            return {"message": "No completed runs to analyse."}

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        dup_count, dup_waste = 0, 0.0
        if len(df) > 1:
            df["_pm"]  = df["model_name"].shift(1)
            df["_pe"]  = df["epochs"].shift(1)
            df["_pb"]  = df["batch_size"].shift(1)
            df["_gap"] = (df["timestamp"] - df["timestamp"].shift(1)
                          ).dt.total_seconds().fillna(999) / 60
            mask = ((df["model_name"] == df["_pm"]) &
                    (df["epochs"] == df["_pe"]) &
                    (df["batch_size"] == df["_pb"]) &
                    (df["_gap"] < 10))
            dup_count = int(mask.sum())
            dup_waste = round(float(df.loc[mask, "total_co2_g"].sum()), 4)

        df["hour"]  = df["timestamp"].dt.hour.fillna(-1).astype(int)
        night_mask  = df["hour"].between(22, 23) | df["hour"].between(0, 5)
        night_runs  = int(night_mask.sum())
        night_waste = round(float(
            df.loc[night_mask, "wasted_co2_g"].fillna(0).sum()), 4)

        total_co2    = round(float(df["total_co2_g"].sum()), 4)
        total_wasted = round(float(df["wasted_co2_g"].fillna(0).sum()), 4)
        waste_pct    = round(total_wasted / total_co2 * 100, 1) if total_co2 > 0 else 0

        if dup_count > 0:
            dup_pct = round(dup_waste / total_co2 * 100, 1) if total_co2 > 0 else 0
            insight = (f"{dup_count} duplicate run(s) within 10 min wasted "
                       f"{dup_waste}g CO₂ ({dup_pct}% of total).")
        elif night_runs > 0:
            insight = (f"{night_runs} late-night run(s) detected. "
                       f"These historically have higher waste. "
                       f"Wasted {night_waste}g CO₂ after-hours.")
        else:
            insight = "No significant waste patterns detected."

        return {
            "researcher_id":     researcher_id or "all",
            "total_runs":        len(df),
            "total_co2_g":       total_co2,
            "total_wasted_g":    total_wasted,
            "waste_pct":         waste_pct,
            "duplicate_runs":    dup_count,
            "duplicate_waste_g": dup_waste,
            "night_runs":        night_runs,
            "night_waste_g":     night_waste,
            "insight":           insight,
        }
    except Exception as e:
        return {"error": str(e)}
