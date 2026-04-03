import threading
import time
import requests
import datetime
import socket
import uuid

# ── CONFIG ──────────────────────────────────────────────
SERVER_URL  = "https://eco-2-4re9.onrender.com/log"
SERVER_BASE = "https://eco-2-4re9.onrender.com"
# ────────────────────────────────────────────────────────


def start(session_id=None):
    """
    ORIGINAL function — unchanged.
    Starts a background power sniffer with no metadata.
    Used by run_local.py and the Windows startup service.
    """
    if session_id is None:
        session_id = f"Agent-{socket.gethostname()}"

    def sniffer_loop():
        print(f"🌱 Eco-Agent Started: Monitoring {session_id}...")

        has_gpu = False
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            has_gpu = True
        except:
            print("⚠️  No GPU detected. Using CPU estimation.")

        while True:
            try:
                if has_gpu:
                    import pynvml
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                else:
                    try:
                        import psutil
                        cpu_pct = psutil.cpu_percent(interval=1)
                        power   = (cpu_pct / 100) * 28.0   # i5-1135G7 TDP
                    except:
                        power = 12.5

                requests.post(SERVER_URL,
                              json={"device":     socket.gethostname(),
                                    "session_id": session_id,
                                    "power_w":    round(power, 2)},
                              timeout=5)
            except:
                pass
            time.sleep(5)

    thread = threading.Thread(target=sniffer_loop, daemon=True)
    thread.start()


# ════════════════════════════════════════════════════════
# NEW FEATURE 1 — SESSION-AWARE MONITORING
# ════════════════════════════════════════════════════════

class Session:
    """
    Holds state for one training run.
    Created by start_session(), consumed by end_session().
    """
    def __init__(self, run_id, task_type, model_name,
                 dataset_size, epochs, batch_size):
        self.run_id       = run_id
        self.task_type    = task_type
        self.model_name   = model_name
        self.dataset_size = dataset_size
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.start_time   = time.time()
        self.device       = socket.gethostname()
        self._thread      = None
        self._stop        = threading.Event()
        self.readings     = []   # local buffer of (timestamp, power_w)


def start_session(task_type="unknown", model_name="unknown",
                  dataset_size=0, epochs=0, batch_size=0):
    """
    ── HOW TO USE IN COLAB ──────────────────────────────
    from eco_agent import start_session, end_session

    session = start_session(
        task_type    = "image_classification",
        model_name   = "ResNet-50",
        dataset_size = 3000,
        epochs       = 25,
        batch_size   = 32
    )

    # ... your training code here ...

    end_session(session, final_accuracy=0.913, final_loss=0.243)
    ────────────────────────────────────────────────────
    """
    run_id  = f"{socket.gethostname()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = Session(run_id, task_type, model_name,
                      dataset_size, epochs, batch_size)

    # ── Check similar past runs immediately ──
    try:
        res = requests.get(f"{SERVER_BASE}/fingerprint/compare",
                           params={"session_id": run_id,
                                   "task_type":  task_type,
                                   "model_name": model_name},
                           timeout=5).json()
        similar = res.get("similar_runs", [])
        if similar:
            best = similar[0]
            print(f"\n📊 Similar past run found: {best.get('run_id', 'unknown')}")
            print(f"   CO₂: {best.get('total_co2_g', '?')}g  |  "
                  f"Accuracy: {best.get('final_accuracy', '?')}")
            print(f"   Use this as your efficiency baseline.\n")
        else:
            print("📊 No similar past runs. This run will become your baseline.\n")
    except:
        print("📊 Could not fetch past runs (server offline?).\n")

    # ── Start background sniffer ──
    def sniffer_loop():
        has_gpu = False
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            has_gpu = True
        except:
            pass

        while not session._stop.is_set():
            try:
                if has_gpu:
                    import pynvml
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                else:
                    try:
                        import psutil
                        cpu_pct = psutil.cpu_percent(interval=1)
                        power   = (cpu_pct / 100) * 28.0
                    except:
                        power = 12.5

                ts = datetime.datetime.now().strftime("%H:%M:%S")
                session.readings.append((ts, power))

                requests.post(SERVER_URL,
                              json={"device":     session.device,
                                    "session_id": session.run_id,
                                    "power_w":    round(power, 2)},
                              timeout=5)
            except:
                pass
            time.sleep(5)

    session._thread = threading.Thread(target=sniffer_loop, daemon=True)
    session._thread.start()

    print(f"🌱 Session started → run_id: {run_id}")
    print(f"   Task: {task_type} | Model: {model_name} | "
          f"Epochs: {epochs} | Batch: {batch_size}\n")
    return session


def end_session(session, final_accuracy=None, final_loss=None):
    """
    Call this in the last cell of your notebook after training finishes.
    Stops the sniffer, computes the fingerprint, and sends it to the server.
    """
    # Stop background thread
    session._stop.set()

    duration_mins = (time.time() - session.start_time) / 60.0
    powers        = [r[1] for r in session.readings]

    if not powers:
        print("⚠️  No power readings recorded.")
        return

    avg_watts  = round(sum(powers) / len(powers), 2)
    peak_watts = round(max(powers), 2)
    total_co2  = round((avg_watts / 1000) * 475 * (duration_mins / 60), 4)

    fingerprint = {
        "run_id":         session.run_id,
        "device":         session.device,
        "task_type":      session.task_type,
        "model_name":     session.model_name,
        "dataset_size":   session.dataset_size,
        "epochs":         session.epochs,
        "batch_size":     session.batch_size,
        "avg_watts":      avg_watts,
        "peak_watts":     peak_watts,
        "duration_mins":  round(duration_mins, 2),
        "total_co2_g":    total_co2,
        "final_accuracy": final_accuracy,
        "final_loss":     final_loss,
    }

    try:
        requests.post(f"{SERVER_BASE}/fingerprint/save",
                      json=fingerprint, timeout=10)
        print("\n✅ Session complete — fingerprint saved.")
    except:
        print("\n⚠️  Could not save fingerprint (server offline).")

    # ── Print run summary ──
    print(f"\n{'─'*45}")
    print(f"  Run summary: {session.run_id}")
    print(f"  Duration  : {duration_mins:.1f} min")
    print(f"  Avg power : {avg_watts} W  |  Peak: {peak_watts} W")
    print(f"  Total CO₂ : {total_co2} g")
    if final_accuracy:
        print(f"  Accuracy  : {final_accuracy*100:.1f}%")
    if final_loss:
        print(f"  Loss      : {final_loss:.4f}")
    print(f"{'─'*45}\n")

    return fingerprint
