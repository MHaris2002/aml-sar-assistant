# Delivery Damage Claims Automation

Computer vision + LLM pipeline to automate first-pass triage of damaged parcel claims.
Detects package damage from photos, reasons about severity using an LLM, and logs claim decisions.

## Status
🚧 In progress — Day 1: dataset acquisition

## Datasets
- [Damaged Package Detection](https://universe.roboflow.com/iot-project/damaged-package-detection) (classification)
- [Parcel Damage Detection](https://universe.roboflow.com/university-of-moratuwa-ztkqd/parcel-damage-detection) (object detection)

## Setup
1. `python -m venv venv` and activate it
2. `pip install -r requirements.txt`
3. Add your Roboflow API key to `.env` as `ROBOFLOW_API_KEY=...`
4. `python scripts/download_data.py`