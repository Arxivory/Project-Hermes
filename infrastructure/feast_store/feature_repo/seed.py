# Build mock BPO agent data for now

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parent
DATA_DIR = REPO_PATH / "data"

def build_local_feature_lake():
    print("Creating baseline operational feature data frames...")
    os.makedirs("data", exist_ok=True)

    agents_pool = ["agent-senior-expert", "agent-stressed-rookie"]

    base_data = {
        "agent_id": agents_pool,
        "event_timestamp": [datetime.utcnow() - timedelta(hours=1) for _ in agents_pool],
        "created_timestamp": [datetime.utcnow() for _ in agents_pool],
        "historical_mean_aht_fraud": [90.0, 300.0],
        "historical_mean_aht_billing": [60.0, 120.0],
        "historical_fcr_fraud": [0.95, 0.45],
        "historical_fcr_billing": [0.99, 0.55],
        "current_cognitive_stress_index": [1.0, 2.2]
    }

    df = pd.DataFrame(base_data)
    parquet_path = DATA_DIR / "agent_historical_telemetry.parquet"
    df.to_parquet(parquet_path)
    print(f"Mock profile records saved to offline store path: {parquet_path}")

if __name__ == "__main__":
    build_local_feature_lake()