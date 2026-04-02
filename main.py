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
    conn = sqlite3.connect("ecotrace.db")
    # Load history for this specific session
    df = pd.read_sql_query("SELECT id, power_w FROM energy_logs WHERE session_id = ?", conn)
    conn.close()
    
    if len(df) < 10:
        return {"error": "Need at least 10 data points to predict. Keep training!"}

    # Prepare ML Model
    X = np.array(range(len(df))).reshape(-1, 1) # Time steps
    y = df['power_w'].values                   # Power recorded
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict power 1 hour (60 mins) into the future
    future_step = len(df) + 60 
    predicted_power = model.predict([[future_step]])[0]
    
    # Carbon Calculation (Avg 475g CO2 per kWh)
    avg_power = df['power_w'].mean()
    est_kwh_per_hour = avg_power / 1000
    carbon_per_hour = est_kwh_per_hour * 475

    return {
        "session": session_id,
        "samples_collected": len(df),
        "current_avg_watts": round(avg_power, 2),
        "predicted_watts_next_hour": round(predicted_power, 2),
        "est_carbon_grams_per_hour": round(carbon_per_hour, 2)
    }
