import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)

ROADS = [
    "Ring Road", "NH-24", "MG Road", "Outer Ring Road",
    "GT Road", "DND Flyway", "Noida Expressway", "Yamuna Expressway",
]
INCIDENT_TYPES = ["Accident", "Congestion", "Road Work", "Signal Failure", "Vehicle Breakdown"]
WEATHER        = ["Clear", "Rainy", "Foggy", "Cloudy"]
COMPLAINT_TEXTS = [
    "Heavy traffic near {} junction causing long delays.",
    "Pothole on {} making driving dangerous.",
    "Signal not working at {} crossing since morning.",
    "Truck blocking {} lane for over an hour.",
    "Accident on {} has caused a massive jam.",
]

def random_timestamp(days_back: int = 30) -> str:
    base = datetime.now() - timedelta(days=random.randint(0, days_back))
    return base.strftime("%Y-%m-%d %H:%M")

def congestion_label(speed: int, vehicle_count: int) -> str:
    if speed < 15 or vehicle_count > 400:
        return "High"
    elif speed < 35 or vehicle_count > 200:
        return "Medium"
    return "Low"

def build_sensor_data(n: int = 1000) -> pd.DataFrame:
    speeds = np.random.randint(5, 80, n)
    counts = np.random.randint(50, 600, n)
    return pd.DataFrame({
        "timestamp":       [random_timestamp() for _ in range(n)],
        "road":            np.random.choice(ROADS, n),
        "vehicle_speed":   speeds,
        "vehicle_count":   counts,
        "weather":         np.random.choice(WEATHER, n),
        "congestion_level": [congestion_label(s, c) for s, c in zip(speeds, counts)],
    })

def build_accident_data(n: int = 300) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp":         [random_timestamp() for _ in range(n)],
        "location":          np.random.choice(ROADS, n),
        "incident_type":     np.random.choice(INCIDENT_TYPES, n),
        "severity":          np.random.choice(["Minor", "Moderate", "Severe"], n),
        "vehicles_involved": np.random.randint(1, 8, n),
        "report": [
            (f"{random.choice(INCIDENT_TYPES)} reported on {random.choice(ROADS)}. "
             f"Weather was {random.choice(WEATHER)}. "
             f"Severity: {random.choice(['Minor', 'Moderate', 'Severe'])}.")
            for _ in range(n)
        ],
    })

def build_police_data(n: int = 200) -> pd.DataFrame:
    actions = ["Diverted traffic", "Issued challan", "Cleared road", "Called ambulance"]
    return pd.DataFrame({
        "timestamp":    [random_timestamp() for _ in range(n)],
        "location":     np.random.choice(ROADS, n),
        "action_taken": np.random.choice(actions, n),
        "police_report": [
            (f"Officer responded to {random.choice(INCIDENT_TYPES)} on {random.choice(ROADS)}. "
             f"Action: {random.choice(['Diverted traffic', 'Cleared road'])}.")
            for _ in range(n)
        ],
    })

def build_complaint_data(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [random_timestamp() for _ in range(n)],
        "location":  np.random.choice(ROADS, n),
        "complaint": [
            random.choice(COMPLAINT_TEXTS).format(random.choice(ROADS))
            for _ in range(n)
        ],
        "status": np.random.choice(["Resolved", "Pending", "Under Review"], n),
    })

def main():
    os.makedirs("data", exist_ok=True)
    datasets = {
        "data/sensor_data.csv":    build_sensor_data(1000),
        "data/accident_data.csv":  build_accident_data(300),
        "data/police_data.csv":    build_police_data(200),
        "data/complaints.csv":     build_complaint_data(200),
    }
    for path, df in datasets.items():
        df.to_csv(path, index=False)
        print(f"Saved {path}  ({len(df)} rows)")
    print("\nPhase 1 complete — all datasets saved to /data/")

if __name__ == "__main__":
    main()
