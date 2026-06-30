import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from .entities import CustomerTicket, AgentProfile

class CognitiveRoutingOptimizer:
    def __init__(self, weight_fcr: float = 100.0, weight_time: float = 0.1, weight_tier = 5.0):
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

        intent = ticket.predict_intent or "default"

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