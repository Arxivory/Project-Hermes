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
    predict_intent: Optional[str] = None
    urgency_score: float = 1.0