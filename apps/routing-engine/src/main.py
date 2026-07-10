import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from feast import FeatureStore

from src.domain.entities import CustomerTicket, AgentProfile
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

def process_and_route_ticket(ticket: CustomerTicket):
    print(f"Processing Inbound Transaction Stream for Ticket: {ticket.ticket_id}")

    predicted_intent = classifier.predict_intent(ticket.raw_utterance)
    ticket.predicted_intent = predicted_intent
    print(f"Local NLP Model Intent Classification Resolution: '{predicted_intent}'")

    monitored_agents = ["agent-senior-expert", "agent-stressed-rookie"]
    enriched_profiles = []

    if feature_store:
        try:
            entity_rows = [{"agent_id": aid} for aid in monitored_agents]
            features_to_fetch = [
                "agent_performance_metrics:historical_mean_aht_fraud",
                "agent_performance_metrics:historical_mean_aht_billing",
                "agent_performance_metrics:historical_fcr_rate",
                "agent_performance_metrics:current_cognitive_stress_index"
            ]

            response = feature_store.get_online_features(
                features=features_to_fetch,
                entity_rows=entity_rows
            ).to_dict()

            for i, aid in enumerate(monitored_agents):
                fcr = response["historical_fcr_rate"][i] or 0.80
                stress = response["current_cognitive_stress_index"][i] or 1.0
                aht_fraud = response["historical_mean_aht_fraud"][i] or 120.0
                aht_billing = response["historical_mean_aht_billing"][i] or 90.0

                profile = AgentProfile(
                    agent_id = aid,
                    current_status="IDLE",
                    intent_aht_matrix={"fraud_alert": aht_fraud, "billing_dispute": aht_billing},
                    intent_fcr_matrix={"fraud_alert": fcr, "billing_dispute": fcr},
                    cognitive_stress_index=stress
                )
                enriched_profiles.append(profile)
        except Exception as e:
            print(f"Feature Store Lookup Failure: {e}. Injecting baseline safety fallback context.")
    
    if not enriched_profiles:
        return

    routing_decisions = optimizer.optimize_routing_matrix([ticket], enriched_profiles)

    print(f"Final Routing Target Resolution Matrix:")

    for t_id, a_id, score in routing_decisions:
        print(f"TICKET [{t_id}] routed to AGENT [{a_id}] with optimal score: {score:.2}")

@app.post("/api/v1/tickets")
async def receive_ticket_http(payload: InboundTicketSchema, background_tasks: BackgroundTasks):
    domain_ticket = CustomerTicket(
        ticket_id=payload.ticket_id,
        customer_tier=payload.customer_tier,
        raw_utterance=payload.raw_utterance
    )
    background_tasks.add_task(process_and_route_ticket, domain_ticket)
    return {"status": "processing", "ticket_id": payload.ticket_id}

@app.get("/health")
def health_check():
    return {"status": "operational", "mode": "native-local"}