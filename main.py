from fastapi import FastAPI, Request
import datetime
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

GRID_INTENSITY = 475.0  # g CO2 per kWh (global average fallback)

# ─────────────────────────────────────────
# 1. DATABASE INIT  (adds 2 new tables)
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("ecotrace.db")
    c = conn.cursor()

    # Original table — unchanged
    c.execute('''CREATE TABLE IF NOT EXISTS energy_logs (
                   id         INTEGER PRIMARY KEY AUTOINCREMENT,
                   device     TEXT,
                   session_id TEXT,
                   power_w    REAL,
                   timestamp  TEXT)''')

    # NEW: one row per completed training run
    c.execute('''CREATE TABLE IF NOT EXISTS run_fingerprints (
                   run_id          TEXT PRIMARY KEY,
                   device          TEXT,
                   task_type       TEXT,
                   model_name      TEXT,
                   dataset_size    INTEGER,
                   epochs          INTEGER,
                   batch_size      INTEGER,
                   avg_watts       REAL,
                   peak_watts      REAL,
                   duration_mins   REAL,
                   total_co2_g     REAL,
                   final_accuracy  REAL,
                   final_loss      REAL,
                   timestamp       TEXT)''')

    # NEW: anomaly events detected per session
    c.execute('''CREATE TABLE IF NOT EXISTS anomalies (
                   id         INTEGER PRIMARY KEY AUTOINCREMENT,
                   session_id TEXT,
                   timestamp  TEXT,
                   power_w    REAL,
                   baseline_w REAL,
                   note       TEXT)''')

    conn.commit()
    conn.close()

init_db()


# ─────────────────────────────────────────
# 2. EXISTING ENDPOINTS (unchanged)
# ─────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "Online", "mode": "EcoTrace v2 — Full Governance"}


@app.post("/log")
async def log_energy(request: Request):
    payload = await request.json()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("ecotrace.db")
    c = conn.cursor()
    c.execute("INSERT INTO energy_logs (device, session_id, power_w, timestamp) VALUES (?, ?, ?, ?)",
              (payload['device'], payload['session_id'], payload['power_w'], ts))
    conn.commit()
    conn.close()
    return {"status": "saved", "power": payload['power_w']}


@app.get("/active_devices")
def get_active_devices():
    try:
        conn = sqlite3.connect("ecotrace.db")
        df = pd.read_sql_query("SELECT DISTINCT session_id FROM energy_logs", conn)
        conn.close()
        return {"devices": df['session_id'].tolist()}
    except Exception as e:
        return {"devices": [], "error": str(e)}


@app.get("/predict")
def predict_energy(session_id: str):
    try:
        conn = sqlite3.connect("ecotrace.db")
        df = pd.read_sql_query(
            "SELECT id, power_w FROM energy_logs WHERE session_id = ?",
            conn, params=(session_id,))
        conn.close()

        if len(df) < 5:
            return {"error": f"Need more data. Currently have {len(df)} points."}

        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['power_w'].values

        if np.all(y == y[0]):
            predicted_power = float(y[0])
        else:
            model = LinearRegression()
            model.fit(X, y)
            predicted_power = float(model.predict([[len(df) + 60]])[0])

        avg_power = float(df['power_w'].mean())
        carbon    = round((avg_power / 1000) * GRID_INTENSITY, 2)

        return {
            "session":        session_id,
            "samples":        len(df),
            "current_avg_w":  round(avg_power, 2),
            "predicted_w":    round(predicted_power, 2),
            "carbon_g_hr":    carbon,
        }
    except Exception as e:
        return {"error": "Math Processing Error", "details": str(e)}


# ─────────────────────────────────────────
# 3. NEW: FINGERPRINT ENDPOINTS
# ─────────────────────────────────────────

@app.post("/fingerprint/save")
async def save_fingerprint(request: Request):
    """
    Called by eco_agent.end_session().
    Saves a completed run summary and detects anomalies via Isolation Forest.
    """
    fp = await request.json()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("ecotrace.db")
    c = conn.cursor()

    # Save fingerprint row
    c.execute('''INSERT OR REPLACE INTO run_fingerprints
                 (run_id, device, task_type, model_name, dataset_size, epochs,
                  batch_size, avg_watts, peak_watts, duration_mins,
                  total_co2_g, final_accuracy, final_loss, timestamp)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
              (fp.get('run_id'), fp.get('device'), fp.get('task_type'),
               fp.get('model_name'), fp.get('dataset_size'), fp.get('epochs'),
               fp.get('batch_size'), fp.get('avg_watts'), fp.get('peak_watts'),
               fp.get('duration_mins'), fp.get('total_co2_g'),
               fp.get('final_accuracy'), fp.get('final_loss'), ts))

    # ── Anomaly detection on raw power readings for this session ──
    session_id = fp.get('run_id', '')
    df_logs = pd.read_sql_query(
        "SELECT power_w, timestamp FROM energy_logs WHERE session_id = ?",
        conn, params=(session_id,))

    if len(df_logs) >= 10:
        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(df_logs[['power_w']])
        baseline = df_logs['power_w'].median()
        anomaly_rows = df_logs[preds == -1]
        for _, row in anomaly_rows.iterrows():
            c.execute(
                "INSERT INTO anomalies (session_id, timestamp, power_w, baseline_w, note) VALUES (?,?,?,?,?)",
                (session_id, row['timestamp'], row['power_w'], baseline,
                 "IsolationForest spike"))

    conn.commit()
    conn.close()
    return {"status": "fingerprint saved", "run_id": fp.get('run_id')}


@app.get("/fingerprint/compare")
def compare_fingerprint(session_id: str, task_type: str = "", model_name: str = ""):
    """
    Returns the 3 most similar past runs using cosine similarity.
    Called at the start of a new session so dashboard can show comparison.
    """
    try:
        conn = sqlite3.connect("ecotrace.db")
        df = pd.read_sql_query(
            "SELECT * FROM run_fingerprints WHERE run_id != ?",
            conn, params=(session_id,))
        conn.close()

        if df.empty:
            return {"similar_runs": [], "message": "No past runs yet. This is your baseline."}

        # Filter by same task/model if possible
        filtered = df[(df['task_type'] == task_type) & (df['model_name'] == model_name)]
        if filtered.empty:
            filtered = df  # fall back to all runs

        feature_cols = ['avg_watts', 'peak_watts', 'duration_mins',
                        'total_co2_g', 'dataset_size', 'epochs', 'batch_size']
        available = [c for c in feature_cols if c in filtered.columns]
        matrix = filtered[available].fillna(0).values

        if len(matrix) == 0:
            return {"similar_runs": []}

        # Use the new session's metadata as query vector
        # (we use column averages of filtered set as placeholder if run not saved yet)
        query = matrix.mean(axis=0).reshape(1, -1)
        scores = cosine_similarity(query, matrix)[0]
        top_idx = np.argsort(scores)[-3:][::-1]
        top_runs = filtered.iloc[top_idx].to_dict(orient='records')

        return {"similar_runs": top_runs}
    except Exception as e:
        return {"error": str(e)}


@app.get("/fingerprint/all")
def all_fingerprints():
    """Returns all saved run fingerprints for the history table in dashboard."""
    try:
        conn = sqlite3.connect("ecotrace.db")
        df = pd.read_sql_query(
            "SELECT * FROM run_fingerprints ORDER BY timestamp DESC", conn)
        conn.close()
        return {"runs": df.to_dict(orient='records')}
    except Exception as e:
        return {"runs": [], "error": str(e)}


# ─────────────────────────────────────────
# 4. NEW: ANOMALY ENDPOINT
# ─────────────────────────────────────────

@app.get("/anomalies")
def get_anomalies(session_id: str):
    """Returns all anomaly events for a given session."""
    try:
        conn = sqlite3.connect("ecotrace.db")
        df = pd.read_sql_query(
            "SELECT * FROM anomalies WHERE session_id = ? ORDER BY timestamp",
            conn, params=(session_id,))
        conn.close()
        return {"anomalies": df.to_dict(orient='records')}
    except Exception as e:
        return {"anomalies": [], "error": str(e)}


# ─────────────────────────────────────────
# 5. NEW: BUDGET GUARD ENDPOINT
# ─────────────────────────────────────────

@app.get("/budget_check")
def budget_check(session_id: str, budget_g: float = 100.0):
    """
    Compares projected total CO2 against budget.
    Returns: status (green/yellow/red), projected_co2, best_past_co2, recommendation.
    """
    try:
        conn = sqlite3.connect("ecotrace.db")
        df = pd.read_sql_query(
            "SELECT power_w FROM energy_logs WHERE session_id = ?",
            conn, params=(session_id,))
        past = pd.read_sql_query(
            "SELECT total_co2_g, final_accuracy FROM run_fingerprints ORDER BY timestamp DESC LIMIT 5",
            conn)
        conn.close()

        if len(df) < 5:
            return {"status": "green", "message": "Not enough data yet."}

        avg_w       = df['power_w'].mean()
        elapsed_hrs = len(df) * 5 / 3600          # 5s intervals → hours
        rate        = (avg_w / 1000) * GRID_INTENSITY  # g/hr
        # Extrapolate: assume we're ~10% done (simple heuristic)
        projected   = round(rate / 0.10, 2) if elapsed_hrs > 0 else 0

        best_co2   = float(past['total_co2_g'].min()) if not past.empty else None
        best_acc   = float(past['final_accuracy'].max()) if not past.empty else None

        if projected > budget_g * 1.0:
            status = "red"
            rec    = f"Projected {projected}g exceeds budget {budget_g}g. Consider stopping now."
        elif projected > budget_g * 0.8:
            status = "yellow"
            rec    = f"Projected {projected}g is 80%+ of budget. Monitor closely."
        else:
            status = "green"
            rec    = "On track within budget."

        return {
            "status":        status,
            "projected_co2": projected,
            "budget_g":      budget_g,
            "best_past_co2": best_co2,
            "best_past_acc": best_acc,
            "recommendation": rec,
        }
    except Exception as e:
        return {"error": str(e)}
