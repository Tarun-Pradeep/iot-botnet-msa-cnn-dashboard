# IoT Botnet Detection — Live Demo Dashboard
### MSA-CNN · N-BaIoT · Leave-One-Device-Out Cross-Validation

An interactive dashboard that demonstrates our attention-based CNN approach to IoT botnet detection. It replays real traffic samples from the **N-BaIoT** dataset device-by-device and visualises classification, alerts and device status in real time.

The underlying research — *“IoT Botnet Detection Using a Multi-Scale Attention CNN: A Scalable Framework for Multi-Class Botnet Traffic Classification”* — was selected for **oral presentation at an IEEE conference (Jul 2026)**.

> **Note:** this repository is a *presentation dashboard*, not the model. Traffic rows are real N-BaIoT samples, but the per-inference confidence scores and the live metrics panel are **randomly simulated** for display purposes and do not reflect measured model performance.
>
> The real experiments, code and results are in **[iot-botnet-detection-nbaiot](https://github.com/Tarun-Pradeep/iot-botnet-detection-nbaiot)** — under leave-one-device-out cross-validation the attention-residual CNN reaches **84.97 % accuracy** (best baseline: MLP 87.38 %).

## What it shows

- **9 real N-BaIoT devices** — Danmini doorbell, Ecobee thermostat, Ennio doorbell, Philips & Samsung baby monitors, Provision & SimpleHome cameras, TP-Link router and more
- **Traffic classes** — Mirai (5 variants), Bashlite/Gafgyt (5 variants) and Benign
- **Per-class confidence bars**, live traffic stream table, timestamped alert log
- **Metrics panel** (simulated values) and packet-rate sparkline
- Device status that flips to *compromised* when attacks are detected

## Run locally

```bash
pip install -r requirements.txt
python server.py --data ./data --port 8080
```

Open http://localhost:8080. Place the leave-one-device-out fold files (`fold_XX_<device>.npz`) in `./data`, or set the `GDRIVE_fold_XX` environment variables and the server downloads them at startup.

### Deploy (Railway / Docker)

A `Dockerfile` and `railway.toml` are included — set the `GDRIVE_fold_XX` variables in your project settings and deploy.

## Project summary

| | |
|---|---|
| Dataset | N-BaIoT (UCI) — https://archive.ics.uci.edu/dataset/442 |
| Architecture | Attention-residual 1-D CNN (feature attention + residual conv blocks) |
| Validation | Leave-One-Device-Out cross-validation |
| Baselines | Logistic Regression, Random Forest, XGBoost, LightGBM, MLP |
| Measured accuracy (LODO CV) | 84.97 % — [full results](https://github.com/Tarun-Pradeep/iot-botnet-detection-nbaiot#results-9-fold-leave-one-device-out) |
| Authors | Tarun Pradeep, Varun Suresh Kumar |
| Supervisor | Dr. Ceronmani Sharmila V |

## Tech stack

Python · Flask · NumPy · HTML/CSS/JavaScript
