from fastapi import FastAPI, Request
import datetime

app = FastAPI()

# A simple list to act as a temporary database for Phase 2
data_store = []

@app.get("/")
def home():
    return {"status": "online", "message": "EcoTrace Control Tower is Live"}

@app.post("/log")
async def log_energy(request: Request):
    payload = await request.json()
    
    # Add a timestamp so we know when the data arrived
    payload["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Store and Print the incoming data
    data_store.append(payload)
    
    print(f"--- [Inbound Data] ---")
    print(f"Device: {payload.get('device')}")
    print(f"Power: {payload.get('power_w')}W")
    print(f"Session: {payload.get('session_id')}")
    
    return {"status": "success", "received": payload["power_w"]}

@app.get("/view_logs")
def view_logs():
    return {"logs": data_store[-10:]} # Show the last 10 readings