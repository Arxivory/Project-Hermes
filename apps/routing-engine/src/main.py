import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from feast import FeatureStore

from src.domain.entities import CustomerTicket
from src.domain.optimizer import CognitiveRoutingOptimizer
from src.domain.classifier import IntentClassifier

app = FastAPI(title="Project Hermes Core Routing Microservice")
optimizer = CognitiveRoutingOptimizer()
classifier = IntentClassifier()

FEAST_REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../infrastructure/feast_store/feature_repo"))
try:
    feature_store = FeatureStore(repo_path=FEAST_REPO_PATH)
    print("Secure native connection mapped to SQLite Feast Online Store.")
except Exception as e:
    print(f"Failed to connect to Feast Registry Path at {FEAST_REPO_PATH}: {e}")
    feature_store = None

class InboundTicketSchema(BaseModel):
    ticket_id: str
    session_id: str
    timestamp: str
    customer_tier: str
    channel: str
    raw_utterance: str

def process_ticket_logic(ticket: CustomerTicket):
    """Encapsulates the core routing decision framework"""
    print(f"Ingested ticket event: [{ticket.ticket_id}] ({ticket.customer_tier})")
    print(f"TODO for NLP Prediction and Feature Store")

@app.post("/api/v1/tickets")
async def receive_ticket_http(payload: InboundTicketSchema, background_tasks: BackgroundTasks):
    """HTTP Endpoint allowing native system execution without running Kafka container for now."""
    domain_ticket = CustomerTicket(
        ticket_id=payload.ticket_id,
        customer_tier=payload.customer_tier,
        raw_utterance=payload.raw_utterance
    )
    background_tasks.add_task(process_ticket_logic, domain_ticket)
    return {"status": "queued", "ticket_id": payload.ticket_id}

@app.get("/health")
def health_check():
    return {"status": "operational", "mode": "native-local"}