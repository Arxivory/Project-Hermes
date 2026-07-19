import torch
import pandas as pd
from datetime import datetime, timezone
from prefect import flow, task
from feast import FeatureStore
from datasets import load_dataset
from transformers import pipeline
import os

PARQUET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../feast_store/feature_repo/data/agent_historical_telemetry.parquet"))
FEAST_REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../feast_store/feature_repo"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../triton/intent_model/1"))

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
    Evaluates our classifier using a live slice from the Bitext dataset via load_dataset
    to compute absolute accuracy and identify performance decay.
    """
    print("Initializing dataset evaluation task...")

    print("Fetching Bitext Financial Dataset via Hugging Face datasets...")
    full_dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )

    SAMPLE_SIZE = 250
    eval_slice = full_dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))
    print(f"Sampled {SAMPLE_SIZE} production instances for performance checking.")

    print(f"Loading registered Hugging Face model from: {MODEL_DIR}")
    classifier = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        device=0 if torch.cuda.is_available() else -1
    )

    texts = eval_slice["instruction"]
    true_labels = eval_slice["intent"]
    
    print("Evaluating batch accuracy...")
    predictions = classifier(texts, batch_size=32)

    correct = 0
    for pred, true_label in zip(predictions, true_labels):
        if pred["label"] == true_label:
            correct += 1

    accuracy = correct / len(texts)
    print(f"Live Evaluation Complete. Calculated Accuracy: {accuracy:.4f}")

    DRIFT_THRESHOLD = 0.90

    if accuracy < DRIFT_THRESHOLD:
        print(f"ALERT: performance drop detected! Current: {accuracy:.2f}")
    else:
        print("Model performance metrics are stable")

    return accuracy

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