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