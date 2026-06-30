from src.domain.entities import CustomerTicket, AgentProfile
from src.domain.optimizer import CognitiveRoutingOptimizer

def test_routing_optimizer():

    # Mock Customer Tickets

    critical_fraud_ticket = CustomerTicket(
        ticket_id="t-101",
        customer_tier="PLATINUM",
        raw_utterance="My credit card has unauthorized charges! Please freeze it immediately!",
        predicted_intent="fraud_alert",
        urgency_score=2.5
    )

    billing_ticket = CustomerTicket(
        ticket_id="t-102",
        customer_tier="REGULAR",
        raw_utterance="Why was I charged 5 dollars extra on my monthly invoice statements?",
        predicted_intent="billing_dispute",
        urgency_score=1.0
    )

    # Mock BPO Agents

    expert_agent = AgentProfile(
        agent_id="agent-senior-expert",
        current_status="IDLE",
        intent_aht_matrix={"fraud_alert": 90.0, "billing_dispute": 60.0},
        intent_fcr_matrix={"fraud_alert": 0.98, "billing_dispute": 0.95},
        cognitive_stress_index=1.0
    )

    stressed_rookie_agent = AgentProfile(
        agent_id="agent-stressed-rookie",
        current_status="IDLE",
        intent_aht_matrix={"fraud_alert": 300.0, "billing_dispute": 120.0},
        intent_fcr_matrix={"fraud_alert": 0.50, "billing_dispute": 0.80},
        cognitive_stress_index=2.2
    )

    optimizer = CognitiveRoutingOptimizer()
    batch_tickets = [critical_fraud_ticket, billing_ticket]
    floor_agents = [expert_agent, stressed_rookie_agent]

    results = optimizer.optimize_routing_matrix(batch_tickets, floor_agents)

    print("\nOptimal Optimization Resolution Matrix Assignments Output:")
    for t_id, a_id, score in results:
        print(f"Ticket [{t_id}] Assigned to [{a_id}] (Calculated Match Weight: {score:.2f})")

if __name__ == "__main__":
    test_routing_optimizer()