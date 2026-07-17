import time
import random
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

