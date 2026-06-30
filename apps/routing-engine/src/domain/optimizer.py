import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from .entities import CustomerTicket, AgentProfile

class CognitiveRoutingOptimizer:
    def __init__(self, weight_fcr: float = 100.0, weight_time: float = 0.1, weight_tier: float = 5.0):
        """
        Initializes the operations research routing engine.
        Weights allow operational managers to bias for equality (FCR) vs speed (AHT/SLA)
        """
        self.w_fcr = weight_fcr
        self.w_time = weight_time
        self.w_tier = weight_tier

    def _get_tier_multiplier(self, tier: str) -> float:
        """
        Translates customer SLA classification into an operational penalty scalar.
        """
        mapping = {
            "PLATINUM": 3.0,
            "VIP": 2.0,
            "REGULAR": 1.0
        }

        return mapping.get(tier.upper(), 1.0)

    def calculate_match_score(self, ticket: CustomerTicket, agent: AgentProfile) -> float:
        """
        Computes the stochastic match score between an explicit ticket intent and an agent.
        Higher Score = More optimal allocation alignment.
        """

        intent = ticket.predicted_intent or "default"

        p_fcr = agent.intent_fcr_matrix.get(intent, 0.70)
        base_aht = agent.intent_aht_matrix.get(intent, 180)

        effective_aht = base_aht * agent.cognitive_stress_index

        tier_penalty = self._get_tier_multiplier(ticket.customer_tier) * ticket.urgency_score
        queue_friction = agent.current_queue_wait_time_sec * tier_penalty

        quality_component = self.w_fcr * p_fcr
        latency_component = self.w_time * effective_aht
        queue_component = self.w_tier * queue_friction

        score = quality_component - latency_component - queue_component
        return float(score)
    
    def optimize_routing_matrix(self, tickets: List[CustomerTicket], agents: List[AgentProfile]) -> List[Tuple[str, str, float]]:
        """
        Uses the Hungarian Algorithm (linear sum assignment) via SciPy to solve optimal pairings
        simultaneously across a matrix batch size of pending tickets and idle agents.
        """

        if not tickets or not agents:
            return []

        viable_agents = [a for a in agents if a.current_status in ["IDLE", "AFTER_CALL_WORK"]]
        if not viable_agents:
            return []
        
        num_tickets = len(tickets)
        num_agents = len(viable_agents)

        cost_matrix = np.zeros((num_tickets, num_agents))

        for t_idx, ticket in enumerate(tickets):
            for a_idx, agent in enumerate(viable_agents):
                cost_matrix[t_idx, a_idx] = self.calculate_match_score(ticket, agent)

        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        assignments = []
        for r, c in zip(row_ind, col_ind):
            ticket_id = tickets[r].ticket_id
            agent_id = viable_agents[c].agent_id
            match_score = cost_matrix[r, c]
            assignments.append((ticket_id, agent_id, match_score))

        return assignments