from fastapi import FastAPI, Request
import datetime
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# --- 1. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect("ecotrace.db")
    cursor = conn.cursor()
    # We store device, power, and a session_id to group training runs
    cursor.execute('''CREATE TABLE IF NOT EXISTS energy_logs 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       device TEXT, session_id TEXT, power_w REAL, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.get("/")
def home():
    return {"status": "Online", "mode": "Analytics & Prediction"}

# --- 2. THE COLLECTOR (Updated for SQL) ---
@app.post("/log")
async def log_energy(request: Request):
    payload = await request.json()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = sqlite3.connect("ecotrace.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO energy_logs (device, session_id, power_w, timestamp) VALUES (?, ?, ?, ?)",
                   (payload['device'], payload['session_id'], payload['power_w'], ts))
    conn.commit()
    conn.close()
    return {"status": "saved", "power": payload['power_w']}

# --- 3. THE BRAIN (Linear Regression Prediction) ---
@app.get("/predict")
def predict_energy(session_id: str):
    try:
        conn = sqlite3.connect("ecotrace.db")
        # FIX: params must be a tuple (session_id,)
        df = pd.read_sql_query("SELECT id, power_w FROM energy_logs WHERE session_id = ?", conn, params=(session_id,))
        conn.close()
        
        if len(df) < 5:
            return {"error": f"Need more data. Currently have {len(df)} points."}

        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['power_w'].values
        
        if np.all(y == y[0]):
            predicted_power = y[0]
        else:
            model = LinearRegression()
            model.fit(X, y)
            future_step = len(df) + 60 
            predicted_power = model.predict([[future_step]])[0]
        
        avg_power = df['power_w'].mean()
        carbon = (avg_power / 1000) * 475

        return {
            "session": session_id,
            "samples": len(df),
            "current_avg_w": round(float(avg_power), 2),
            "predicted_w": round(float(predicted_power), 2),
            "carbon_g_hr": round(float(carbon), 2)
        }
    except Exception as e:
        return {"error": "Math Processing Error", "details": str(e)}
