from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class CustomerTicket(BaseModel):
    """
    Represents an active customer interaction ticket entering the routing mesh.
    """
    ticket_id: str
    customer_tier: str
    raw_utterance: str
    predicted_intent: Optional[str] = None
    urgency_score: float = 1.0

class AgentProfile(BaseModel):
    """
    Represents an active BPO agent currently tracked on the production floor.
    """
    agent_id: str
    current_status: str

    intent_aht_matrix: Dict[str, float]

    intent_fcr_matrix: Dict[str, float]

    cognitive_stress_index: float
    current_queue_wait_time_sec: float = 0.0