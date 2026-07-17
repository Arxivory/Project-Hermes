import time
import random
import pandas as pd
from datetime import datetime, timezone
from prefect import flow, task
from feast import FeatureStore
import os

PARQUET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../feast_store/feature_repo/data/agent_historical_telemetry.parquet"))
FEAST_REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../feast_store/feature_repo"))

@task(name="Ingest Live Agent Metrics")
def ingest_agent_metrics():
    """
    Grabs fresh, real operational telemetry.
    In Production, this queries a database or API.
    And here we simulate pulling the live active states.
    """
    print("Ingesting real-time queue states and agent telemetry...")

    # Simulate stress updates
    telemetry_updates = {
        "agent-senior-expert": {
            "current_cognitive_stress_index": 1.15
        },
        "agent-stressed-rookie": {
            "current_cognitive_stress_index": 3.85
        }
    }
    return telemetry_updates

@task(name="Update Feast Feature Store")
def update_feast_features(telemetry_updates: dict):
    """
    Pushes the fresh ingested features into the Offline Parquet Store,
    and then programmatically materializes them to Feast's Online store.
    """

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"Parquet Offline Database not found at {PARQUET_PATH}. "
            "Please run your seed.py script first to generate the data."
        )
    
    df = pd.read_parquet(PARQUET_PATH)

    now_utc = datetime.now(timezone.utc)
    for agent_id, updates in telemetry_updates.items():
        mask = df["agent_id"] == agent_id
        if mask.any():
            print(f"Updating metrics in offline parquet store for agent: {agent_id}")
            df.loc[mask, "current_cognitive_stress_index"] = updates["current_cognitive_stress_index"]
            df.loc[mask, "event_timestamp"] = now_utc
            df.loc[mask, "created_timestamp"] = now_utc

    df.to_parquet(PARQUET_PATH)
    print("Offline parquet feature lake successfully updated.")

    print(f"Initializing Feast Feature Store from: {FEAST_REPO_PATH}")
    store = FeatureStore(repo_path=FEAST_REPO_PATH)

    print("Materializing updated offline metrics to the online store...")

    start_date = datetime.now(timezone.utc) - pd.Timedelta(days=1)
    end_date = datetime.now(timezone.utc) + pd.Timedelta(minutes=5)

    store.materialize(start_date=start_date, end_date=end_date)
    print("Feast online store materialized and fully synced!")

@task(name="Continous Model Drift Evaluation")
def evaluate_model_performance():
    """
    Simulates matching predictions against ground-truth labels to detect accuracy decay.
    """
    simulated_accuracy = random.uniform(0.91, 0.95)
    print(f"Evaluating Model Drift... Current Accuracy: {simulated_accuracy:.3f}")
    if simulated_accuracy < 0.92:
        print("ALERT: Model accuracy dropped below threshold! Flagging for retraining.")
    else:
        print("Model health is stable.")

@flow(name="Hermes Core Operations Pipeline")
def hermes_orchestration_flow():
    """
    The master DAG coordinating our feature backfills and evaluation loops.
    """
    metrics = ingest_agent_metrics()
    update_feast_features(metrics)
    evaluate_model_performance()

if __name__ == "__main__":
    hermes_orchestration_flow()