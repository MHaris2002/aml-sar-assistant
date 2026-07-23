"""
Downloads the two source datasets from Roboflow Universe and saves them
into data/raw/. Requires ROBOFLOW_API_KEY to be set in a .env file at
the project root.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# Load API key from .env
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError(
        "ROBOFLOW_API_KEY not found. Make sure you have a .env file "
        "with ROBOFLOW_API_KEY=your_key_here"
    )

rf = Roboflow(api_key=api_key)

# Ensure output directory exists
raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# --- Dataset 1: Damaged Package Detection (IoT Project) — CLASSIFICATION dataset ---
print("Downloading Dataset 1: Damaged Package Detection...")
project1 = rf.workspace("iot-project").project("damaged-package-detection")
dataset1 = project1.version(2).download("folder", location=str(raw_dir / "damaged_package_detection"))
print(f"Dataset 1 saved to: {dataset1.location}")

# --- Dataset 2: Parcel Damage Detection (University of Moratuwa) — OBJECT DETECTION dataset ---
print("Downloading Dataset 2: Parcel Damage Detection...")
project2 = rf.workspace("university-of-moratuwa-ztkqd").project("parcel-damage-detection")
dataset2 = project2.version(1).download("yolov8", location=str(raw_dir / "parcel_damage_detection"))
print(f"Dataset 2 saved to: {dataset2.location}")

print("\nBoth datasets downloaded successfully.")