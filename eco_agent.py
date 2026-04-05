"""
EcoTrace v3 — Smart Agent
Drop-in for Colab / Kaggle / Local.
Inject with: !curl -O https://raw.githubusercontent.com/YOUR_REPO/eco_agent.py

Usage in notebook:
    from eco_agent import start_session, end_session
    session = start_session(task_type="image_classification",
                            model_name="ResNet-50",
                            dataset_size=3000, epochs=25, batch_size=32,
                            researcher_id="your_name")
    # ... training ...
    end_session(session, final_accuracy=0.913, final_loss=0.243,
                val_losses=[0.8, 0.6, 0.4, 0.35, 0.34, 0.34, 0.34])

For persistent local monitoring (run_local.py):
    from eco_agent import start
    start(session_id="Intel-Laptop-Ldh")
"""

import threading
import time
import datetime
import socket
import json

try:
    import requests
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "requests", "-q"])
    import requests

# ── CONFIG ────────────────────────────────────────────────────
SERVER_URL  = "https://eco-2-4re9.onrender.com/log"
SERVER_BASE = "https://eco-2-4re9.onrender.com"
GRID_G_KWH  = 475.0
# ─────────────────────────────────────────────────────────────


def _get_power(handle=None, has_gpu=False):
    """Read power from GPU (pynvml) or estimate from CPU (psutil)."""
    if has_gpu and handle is not None:
        try:
            import pynvml
            return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.5)
        return max((cpu_pct / 100) * 28.0, 2.0)   # i5-1135G7 TDP, 2W floor
    except Exception:
        return 12.5   # absolute last resort


def _init_gpu():
    """Try to initialise pynvml. Returns (has_gpu, handle)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name   = pynvml.nvmlDeviceGetName(handle)
        print(f"   GPU detected: {name}")
        return True, handle
    except Exception:
        print("   No NVIDIA GPU — using CPU estimation.")
        return False, None


# ─────────────────────────────────────────────────────────────
# ORIGINAL start() — unchanged for run_local.py compatibility
# ─────────────────────────────────────────────────────────────

def start(session_id=None):
    """
    Persistent background sniffer — no metadata, no fingerprint.
    Used by run_local.py and the Windows startup folder service.
    """
    if session_id is None:
        session_id = f"Agent-{socket.gethostname()}"

    has_gpu, handle = _init_gpu()

    def _loop():
        print(f"🌱 Eco-Agent running: {session_id}")
        while True:
            try:
                power = _get_power(handle, has_gpu)
                requests.post(SERVER_URL,
                              json={"device":     socket.gethostname(),
                                    "session_id": session_id,
                                    "power_w":    round(power, 2)},
                              timeout=5)
            except Exception:
                pass
            time.sleep(5)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ─────────────────────────────────────────────────────────────
# SESSION CLASS
# ─────────────────────────────────────────────────────────────

class Session:
    def __init__(self, run_id, task_type, model_name,
                 dataset_size, epochs, batch_size, researcher_id):
        self.run_id        = run_id
        self.task_type     = task_type
        self.model_name    = model_name
        self.dataset_size  = dataset_size
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.researcher_id = researcher_id
        self.device        = socket.gethostname()
        self.start_time    = time.time()
        self.readings      = []          # list of (timestamp_str, power_w)
        self._stop         = threading.Event()
        self._thread       = None


# ─────────────────────────────────────────────────────────────
# start_session()
# ─────────────────────────────────────────────────────────────

def start_session(task_type="unknown", model_name="unknown",
                  dataset_size=0, epochs=0, batch_size=32,
                  researcher_id="anonymous"):
    """
    Call at the TOP of your notebook before training.
    Returns a session object — pass it to end_session() at the end.
    """
    run_id  = (f"{socket.gethostname()}_"
               f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    session = Session(run_id, task_type, model_name,
                      dataset_size, epochs, batch_size, researcher_id)

    print(f"\n{'='*50}")
    print(f"  EcoTrace v3 — Session started")
    print(f"  run_id  : {run_id}")
    print(f"  task    : {task_type}  |  model: {model_name}")
    print(f"  epochs  : {epochs}  |  batch: {batch_size}")
    print(f"{'='*50}")

    # ── Pre-run estimate from past fingerprints ───────────
    try:
        est = requests.get(
            f"{SERVER_BASE}/estimate",
            params={"task_type": task_type, "model_name": model_name,
                    "epochs": epochs, "batch_size": batch_size},
            timeout=5).json()
        if est.get("similar_count", 0) > 0:
            print(f"\n📊 Pre-run estimate (from {est['similar_count']} similar runs):")
            print(f"   CO₂ range : {est['co2_min_g']}g – {est['co2_max_g']}g "
                  f"(avg {est['co2_avg_g']}g)")
            print(f"   Duration  : {est['duration_min_mins']}–"
                  f"{est['duration_max_mins']} min")
            best = est.get("best_config", {})
            if best:
                print(f"   Best config seen: batch={best['batch_size']}, "
                      f"epochs={best['epochs']} → "
                      f"{best['total_co2_g']}g CO₂, "
                      f"{round(best['final_accuracy']*100,1)}% acc "
                      f"[Grade {best['grade']}]")
        else:
            print("\n📊 No past runs — this will become your baseline.")
    except Exception:
        print("\n📊 Could not fetch pre-run estimate (server offline?).")

    # ── Similar run comparison ────────────────────────────
    try:
        cmp = requests.get(
            f"{SERVER_BASE}/fingerprint/compare",
            params={"session_id": run_id, "task_type": task_type,
                    "model_name": model_name},
            timeout=5).json()
        similar = cmp.get("similar_runs", [])
        if similar:
            best = similar[0]
            print(f"\n🔬 Most similar past run: "
                  f"...{str(best.get('run_id',''))[-16:]}")
            print(f"   CO₂: {best.get('total_co2_g','?')}g  |  "
                  f"Accuracy: {round((best.get('final_accuracy') or 0)*100,1)}%  |  "
                  f"Grade: {best.get('efficiency_grade','?')}")
    except Exception:
        pass

    # ── Background sniffer ────────────────────────────────
    has_gpu, handle = _init_gpu()

    def _loop():
        while not session._stop.is_set():
            try:
                power = _get_power(handle, has_gpu)
                ts    = datetime.datetime.now().strftime("%H:%M:%S")
                session.readings.append((ts, power))
                requests.post(SERVER_URL,
                              json={"device":     session.device,
                                    "session_id": session.run_id,
                                    "power_w":    round(power, 2)},
                              timeout=5)
            except Exception:
                pass
            time.sleep(5)

    session._thread = threading.Thread(target=_loop, daemon=True)
    session._thread.start()

    print(f"\n✅ Monitoring active — call end_session(session, ...) when done.\n")
    return session


# ─────────────────────────────────────────────────────────────
# end_session()
# ─────────────────────────────────────────────────────────────

def end_session(session, final_accuracy=None, final_loss=None,
                val_losses=None):
    """
    Call in the LAST CELL of your notebook.

    val_losses : optional list of validation loss per epoch
                 e.g. [0.9, 0.7, 0.5, 0.4, 0.39, 0.39]
                 Used to detect the carbon waste zone (overfitting).
    """
    session._stop.set()
    time.sleep(1)   # let final reading land

    powers        = [r[1] for r in session.readings]
    duration_mins = (time.time() - session.start_time) / 60.0

    if not powers:
        print("⚠️  No power readings recorded — is the agent running?")
        return None

    avg_watts  = round(sum(powers) / len(powers), 2)
    peak_watts = round(max(powers), 2)
    total_co2  = round((avg_watts / 1000) * GRID_G_KWH * (duration_mins / 60), 4)

    # Carbon DNA — 50-point normalised power curve
    try:
        import numpy as np
        arr  = np.array(powers, dtype=float)
        idx  = np.linspace(0, len(arr) - 1, 50)
        rs   = np.interp(idx, np.arange(len(arr)), arr)
        mn, mx = rs.min(), rs.max()
        dna  = ((rs - mn) / (mx - mn + 1e-9)).tolist()
    except Exception:
        dna  = []

    fingerprint = {
        "run_id":         session.run_id,
        "device":         session.device,
        "researcher_id":  session.researcher_id,
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
        "val_losses":     val_losses or [],
        "carbon_dna":     json.dumps(dna),
    }

    result = {}
    try:
        r      = requests.post(f"{SERVER_BASE}/fingerprint/save",
                               json=fingerprint, timeout=15)
        result = r.json()
        print("\n✅ Fingerprint saved to EcoTrace server.")
    except Exception as e:
        print(f"\n⚠️  Could not save fingerprint: {e}")

    # ── DNA match — find most similar past run ────────────
    if len(powers) >= 5:
        try:
            dm = requests.post(
                f"{SERVER_BASE}/dna/match",
                json={"powers": powers},
                timeout=8).json()
            if dm.get("match"):
                print(f"\n🧬 Carbon DNA match: {dm['prediction']}")
        except Exception:
            pass

    # ── Shapley attribution ───────────────────────────────
    try:
        sh = requests.get(
            f"{SERVER_BASE}/shapley",
            params={"session_id": session.run_id},
            timeout=5).json()
        if "co2_fair_g" in sh:
            print(f"\n⚖️  Shapley attribution: your fair share = "
                  f"{sh['co2_fair_g']}g CO₂ "
                  f"(naive would be {sh['co2_naive_g']}g, "
                  f"saved {sh['co2_saved_g']}g by fair attribution)")
    except Exception:
        pass

    # ── Print run summary ─────────────────────────────────
    grade      = result.get("grade", "?")
    wasted     = result.get("wasted_co2_g", 0)
    waste_ep   = result.get("waste_epoch")

    print(f"\n{'─'*50}")
    print(f"  EcoTrace Run Summary")
    print(f"  run_id     : {session.run_id}")
    print(f"  Duration   : {round(duration_mins, 1)} min")
    print(f"  Avg power  : {avg_watts} W  |  Peak: {peak_watts} W")
    print(f"  Total CO₂  : {total_co2} g")
    if wasted and wasted > 0:
        waste_pct = round(wasted / total_co2 * 100, 1) if total_co2 > 0 else 0
        print(f"  Wasted CO₂ : {wasted} g ({waste_pct}%) "
              f"— overfitting after epoch {waste_ep}")
    if final_accuracy:
        print(f"  Accuracy   : {round(final_accuracy * 100, 2)}%")
    if final_loss:
        print(f"  Loss       : {final_loss:.4f}")
    print(f"  Grade      : {grade}")
    print(f"  Carbon DNA : {len(dna)}-point vector saved")

    # Human equivalents
    car_km    = round(total_co2 * 0.00417, 4)
    phone_x   = round(total_co2 / 5.5, 3)
    tree_min  = round(total_co2 / 0.0095, 1)
    print(f"\n  Carbon debt equivalents:")
    print(f"    🚗  {car_km} km driving")
    print(f"    📱  {phone_x}x phone charges")
    print(f"    🌳  {tree_min} min of tree absorption")
    print(f"{'─'*50}\n")

    return fingerprint
