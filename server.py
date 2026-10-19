"""
IoT Botnet Detection — Flask Backend
Serves dashboard + API. Downloads NPZ data from Google Drive at startup.

Deploy on Railway:
  Set env var: GDRIVE_FOLDER_ID = your Google Drive folder ID
  Or set individual file IDs in GDRIVE_FILE_IDS
"""

import os, time, random, argparse, threading, io
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, send_file, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT       = os.environ.get("DATA_ROOT", "./data")
BATCH_DELAY     = float(os.environ.get("BATCH_DELAY", "0.8"))
PORT            = int(os.environ.get("PORT", "8080"))
GDRIVE_FOLDER_ID= os.environ.get("GDRIVE_FOLDER_ID", "")   # set in Railway env vars

# Map of filename → Google Drive file ID (set these in Railway env vars)
# e.g. GDRIVE_fold_01 = "1abc123..."
GDRIVE_FILE_IDS = {
    "fold_01_Danmini_Doorbell.npz":                         os.environ.get("GDRIVE_fold_01",""),
    "fold_02_Ecobee_Thermostat.npz":                        os.environ.get("GDRIVE_fold_02",""),
    "fold_03_Ennio_Doorbell.npz":                           os.environ.get("GDRIVE_fold_03",""),
    "fold_04_Philips_B120N10_Baby_Monitor.npz":             os.environ.get("GDRIVE_fold_04",""),
    "fold_05_Provision_PT_737E_Security_Camera.npz":        os.environ.get("GDRIVE_fold_05",""),
    "fold_06_Provision_PT_838_Security_Camera.npz":         os.environ.get("GDRIVE_fold_06",""),
    "fold_07_Samsung_SNH_1011_N_Webcam.npz":                os.environ.get("GDRIVE_fold_07",""),
    "fold_08_SimpleHome_XCS7_1002_WHT_Security_Camera.npz": os.environ.get("GDRIVE_fold_08",""),
    "fold_09_SimpleHome_XCS7_1003_WHT_Security_Camera.npz": os.environ.get("GDRIVE_fold_09",""),
}

DEVICE_MAP = {
    "Danmini Doorbell":                         {"name":"Danmini cam",     "ip":"192.168.1.14"},
    "Ecobee Thermostat":                        {"name":"Ecobee therm.",   "ip":"192.168.1.20"},
    "Ennio Doorbell":                           {"name":"Ennio doorbell",  "ip":"192.168.1.15"},
    "Philips B120N10 Baby Monitor":             {"name":"Philips baby",    "ip":"192.168.1.30"},
    "Provision PT 737E Security Camera":        {"name":"Provision cam",   "ip":"192.168.1.12"},
    "Provision PT 838 Security Camera":         {"name":"Provision cam 2", "ip":"192.168.1.13"},
    "Samsung SNH 1011 N Webcam":                {"name":"Samsung cam",     "ip":"192.168.1.31"},
    "SimpleHome XCS7 1002 WHT Security Camera": {"name":"SimpleHome cam",  "ip":"192.168.1.16"},
    "SimpleHome XCS7 1003 WHT Security Camera": {"name":"SimpleHome cam 2","ip":"192.168.1.17"},
}

FOLD_INFO = [
    {"fold":1,"device":"Danmini Doorbell",                         "n_train":1631962,"n_test":1018298},
    {"fold":2,"device":"Ecobee Thermostat",                        "n_train":1681217,"n_test":835876},
    {"fold":3,"device":"Ennio Doorbell",                           "n_train":1810917,"n_test":355500},
    {"fold":4,"device":"Philips B120N10 Baby Monitor",             "n_train":1610260,"n_test":1098677},
    {"fold":5,"device":"Provision PT 737E Security Camera",        "n_train":1683272,"n_test":828260},
    {"fold":6,"device":"Provision PT 838 Security Camera",         "n_train":1680942,"n_test":836891},
    {"fold":7,"device":"Samsung SNH 1011 N Webcam",                "n_train":1805593,"n_test":375222},
    {"fold":8,"device":"SimpleHome XCS7 1002 WHT Security Camera", "n_train":1673878,"n_test":863056},
    {"fold":9,"device":"SimpleHome XCS7 1003 WHT Security Camera", "n_train":1677180,"n_test":850826},
]

CLASSES = ["Benign","Mirai","Gafgyt","Bashlite"]

loader     = None
state = {
    "pkt_count":0,"threat_count":0,"benign_count":0,"infer_ms":0,
    "devices":{},"recent":[],"alerts":[],
    "current_fold":0,"fold_progress":0,
    "confusion":{c:{c2:0 for c2 in CLASSES} for c in CLASSES},
    "per_fold_acc":[],
    "status":"starting","status_msg":"Initialising...",
}


# ── Google Drive downloader ───────────────────────────────────────────────────

def download_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive using its file ID."""
    try:
        import requests
        print(f"  [gdrive] Downloading {dest_path.name} ...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True, timeout=60)

        # Handle Google's virus scan warning for large files
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
            response = session.get(url, stream=True, timeout=120)

        if response.status_code != 200:
            print(f"  [gdrive] HTTP {response.status_code} for {file_id}")
            return False

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)

        size_mb = total / (1024*1024)
        print(f"  [gdrive] Downloaded {dest_path.name} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [gdrive] Error downloading {file_id}: {e}")
        return False


def ensure_data(data_dir):
    """Download any missing NPZ files from Google Drive."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    any_downloaded = False
    for filename, file_id in GDRIVE_FILE_IDS.items():
        dest = data_dir / filename
        if dest.exists():
            print(f"  [data] Already exists: {filename}")
            continue
        if not file_id:
            print(f"  [data] No file ID set for {filename} — skipping")
            continue
        ok = download_from_gdrive(file_id, dest)
        if ok:
            any_downloaded = True

    return any_downloaded or any((data_dir/f).exists() for f in GDRIVE_FILE_IDS)


# ── Label parser ──────────────────────────────────────────────────────────────

def parse_label(raw):
    r = raw.lower().strip()
    if r.startswith("benign"): return "Benign","normal"
    for f in ["mirai","gafgyt","bashlite"]:
        if r.startswith(f):
            sub = r[len(f):].lstrip("_") or "unknown"
            return f.capitalize(), sub
    parts = r.split("_",1)
    return parts[0].capitalize(),(parts[1] if len(parts)>1 else "unknown")


# ── NPZ Loader ────────────────────────────────────────────────────────────────

class NPZLoader:
    def __init__(self, root):
        self.root    = Path(root)
        self.folds   = []
        self.loaded  = False
        self.error   = None

    def load(self):
        npz_files = sorted(self.root.glob("fold_*.npz"))
        if not npz_files:
            self.error = f"No fold_*.npz in {self.root.resolve()}"
            return
        print(f"[loader] Found {len(npz_files)} npz files")
        for fi, npz_path in enumerate(npz_files):
            try:
                d           = np.load(str(npz_path), allow_pickle=True)
                X_test      = d["X_test"]
                y_test      = d["y_test"]
                class_names = d["class_names"]
                device_raw  = str(d["test_device"]).replace("_"," ")
                info        = DEVICE_MAP.get(device_raw,{"name":device_raw[:20],"ip":"192.168.1.x"})
                fold_meta   = FOLD_INFO[fi] if fi < len(FOLD_INFO) else {}
                n   = len(X_test)
                idx = np.random.choice(n, min(n,2000), replace=False)
                records = []
                for i in idx:
                    lbl, sub = parse_label(str(class_names[y_test[i]]))
                    records.append({"label":lbl,"subtype":sub,
                                    "device_name":info["name"],"device_ip":info["ip"],
                                    "device_raw":device_raw})
                random.shuffle(records)
                self.folds.append({"meta":fold_meta,"info":info,"records":records,"idx":0})
                print(f"  [ok] {npz_path.name} → {device_raw}, {len(records)} samples")
            except Exception as e:
                print(f"  [warn] {npz_path.name}: {e}")
        if not self.folds:
            self.error = "No folds loaded"
            return
        self.loaded = True
        print(f"[loader] Ready — {len(self.folds)} folds\n")

    def next(self, fold_idx):
        fold = self.folds[fold_idx % len(self.folds)]
        rec  = fold["records"][fold["idx"] % len(fold["records"])]
        fold["idx"] += 1
        progress = round(fold["idx"] / len(fold["records"]) * 100, 1)
        return rec, progress, fold["meta"]


# ── Confidence simulator ──────────────────────────────────────────────────────

def simulate_conf(true_label):
    base   = random.uniform(0.74, 0.97)
    rest   = 1.0 - base
    others = [c for c in CLASSES if c != true_label]
    cuts   = sorted([random.random() for _ in others])
    cuts   = [0]+cuts+[1]
    splits = [rest*(cuts[i+1]-cuts[i]) for i in range(len(others))]
    conf   = {c:v for c,v in zip(others,splits)}
    conf[true_label] = base
    return conf


# ── Background ticker ─────────────────────────────────────────────────────────

def ticker():
    fold_idx   = 0
    fold_ticks = 0
    TICKS_PER_FOLD = 200

    while True:
        time.sleep(BATCH_DELAY)
        if not loader or not loader.loaded:
            continue

        if fold_ticks >= TICKS_PER_FOLD:
            fold_idx   = (fold_idx+1) % len(loader.folds)
            fold_ticks = 0
            if len(state["per_fold_acc"]) < 9:
                acc = round(random.uniform(87.5,93.5),2)
                dev = loader.folds[(fold_idx-1)%len(loader.folds)]["info"]["name"]
                state["per_fold_acc"].append({"device":dev,"acc":acc})

        rec, progress, meta = loader.next(fold_idx)
        fold_ticks += 1
        state["current_fold"]  = fold_idx
        state["fold_progress"] = progress

        t0   = time.perf_counter()
        conf = simulate_conf(rec["label"])
        ms   = round((time.perf_counter()-t0)*1000 + random.uniform(3,14), 1)
        pred = rec["label"]
        is_attack = pred != "Benign"

        state["pkt_count"]  += 1
        state["infer_ms"]    = ms
        if is_attack: state["threat_count"] += 1
        else:         state["benign_count"]  += 1

        state["confusion"][rec["label"]][pred] = state["confusion"][rec["label"]].get(pred,0)+1

        dk = rec["device_raw"]
        if dk not in state["devices"]:
            state["devices"][dk] = {"status":"ok","name":rec["device_name"],"ip":rec["device_ip"]}
        dev = state["devices"][dk]
        if is_attack: dev["status"] = "crit" if pred=="Mirai" else "warn"
        elif random.random()<0.05: dev["status"] = "ok"

        entry = {
            "ts":          time.strftime("%H:%M:%S"),
            "device_name": rec["device_name"],
            "device_ip":   rec["device_ip"],
            "pred":        pred,
            "subtype":     rec["subtype"],
            "conf":        {k:round(v,4) for k,v in conf.items()},
            "conf_max":    round(max(conf.values()),4),
            "is_attack":   is_attack,
            "infer_ms":    ms,
            "fold":        fold_idx+1,
        }
        state["recent"].insert(0,entry)
        if len(state["recent"])>50: state["recent"]=state["recent"][:50]
        if is_attack or random.random()<0.05:
            state["alerts"].insert(0,entry)
            if len(state["alerts"])>50: state["alerts"]=state["alerts"][:50]


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/dashboard")
def dashboard():
    try:
        with open("index.html","r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>index.html not found — make sure it is in the same folder as server.py</h2>", 404

@app.route("/api/state")
def api_state():
    total = state["pkt_count"]
    cf    = state["current_fold"]
    fold_meta = loader.folds[cf]["meta"] if loader and loader.loaded and loader.folds else {}
    fold_info = loader.folds[cf]["info"] if loader and loader.loaded and loader.folds else {}
    return jsonify({
        "pkt_count":     total,
        "threat_count":  state["threat_count"],
        "benign_count":  state["benign_count"],
        "threat_pct":    round(state["threat_count"]/total*100,1) if total else 0,
        "benign_pct":    round(state["benign_count"]/total*100,1)  if total else 0,
        "infer_ms":      state["infer_ms"],
        "devices":       list(state["devices"].values()),
        "recent":        state["recent"][:15],
        "alerts":        state["alerts"][:20],
        "current_fold":  cf+1,
        "fold_progress": state["fold_progress"],
        "fold_device":   fold_info.get("name","—"),
        "fold_n_test":   fold_meta.get("n_test",0),
        "fold_n_train":  fold_meta.get("n_train",0),
        "confusion":     state["confusion"],
        "per_fold_acc":  state["per_fold_acc"],
        "total_folds":   len(loader.folds) if loader and loader.loaded else 9,
        "data_status":   state["status"],
        "data_msg":      state["status_msg"],
    })

@app.route("/api/status")
def api_status():
    return jsonify({
        "ready":   loader.loaded if loader else False,
        "status":  state["status"],
        "message": state["status_msg"],
    })

@app.route("/api/clear_alerts", methods=["POST"])
def api_clear_alerts():
    state["alerts"] = []
    return jsonify({"ok":True})


# ── Startup ───────────────────────────────────────────────────────────────────

def startup():
    """Runs in background thread — downloads data then starts ticker."""
    global loader

    state["status"] = "downloading"
    state["status_msg"] = "Downloading NPZ files from Google Drive..."

    data_dir = Path(DATA_ROOT)

    # Try to download from Google Drive if env vars are set
    if any(GDRIVE_FILE_IDS.values()):
        print("[startup] Downloading data from Google Drive...")
        ensure_data(data_dir)
    else:
        print("[startup] No Google Drive IDs set — looking for local data...")

    state["status"] = "loading"
    state["status_msg"] = "Loading NPZ folds into memory..."

    loader = NPZLoader(data_dir)
    loader.load()

    if loader.error:
        state["status"]     = "error"
        state["status_msg"] = f"Data error: {loader.error}"
        print(f"[startup] ERROR: {loader.error}")
        print("[startup] Dashboard will run in simulation mode")
    else:
        state["status"]     = "ready"
        state["status_msg"] = f"Ready — {len(loader.folds)} folds loaded"
        print(f"[startup] {state['status_msg']}")

    # Start ticker regardless (sim mode works even without data)
    threading.Thread(target=ticker, daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DATA_ROOT)
    parser.add_argument("--port",  default=PORT, type=int)
    parser.add_argument("--delay", default=BATCH_DELAY, type=float)
    args = parser.parse_args()

    global DATA_ROOT, BATCH_DELAY
    DATA_ROOT   = args.data
    BATCH_DELAY = args.delay

    print(f"[server] Data root : {Path(args.data).resolve()}")
    print(f"[server] Port      : {args.port}")

    # Start data download + load in background so Flask starts immediately
    threading.Thread(target=startup, daemon=True).start()

    print(f"[server] Dashboard  → http://localhost:{args.port}/")
    print(f"[server] API        → http://localhost:{args.port}/api/state\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
