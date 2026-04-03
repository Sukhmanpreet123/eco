import threading
import time
import requests
import datetime
import socket

# --- 1. SETTINGS ---
# Change this to your actual Render URL!
SERVER_URL = "https://eco-2-4re9.onrender.com/log"

def start(session_id=None):
    """Starts the Carbon Sniffer in a background thread."""
    
    # Auto-detect device name if no session_id is provided
    if session_id is None:
        session_id = f"Agent-{socket.gethostname()}"

    def sniffer_loop():
        print(f"🌱 Eco-Agent Started: Monitoring {session_id}...")
        
        # Try to initialize GPU (for Colab/Gaming Laptops)
        has_gpu = False
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            has_gpu = True
        except:
            print("⚠️ No GPU detected. Monitoring system average power.")

        while True:
            try:
                # Get Power logic
                if has_gpu:
                    import pynvml
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                else:
                    # Mock or CPU-based power estimation (simplified for now)
                    power = 12.5 

                payload = {
                    "device": socket.gethostname(),
                    "session_id": session_id,
                    "power_w": round(power, 2)
                }

                # Send to Cloud
                requests.post(SERVER_URL, json=payload, timeout=5)
                
            except Exception as e:
                # Fail silently in background to not disturb main code
                pass
            
            time.sleep(5) # Record every 5 seconds

    # Start the thread as a 'daemon' (dies when main program stops)
    thread = threading.Thread(target=sniffer_loop, daemon=True)
    thread.start()