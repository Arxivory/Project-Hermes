import re

class LocalIntentClassifier:
    def __init__(self):
        print("Loading local-native memory NLP Intent Classifier core...")

        self.intent_rules = {
            r"(charge|unauthorized|freeze|fraud|stolen|card)": "fraud_alert",
            r"(invoice|billing|charged|fee|payment|extra)": "billing_dispute",
            r"(cancel|close|terminate|leave)": "account_cancellation",
            r"(limit|increase|raise|max)": "limit_upgrade"
        }

    def predict_intent(self, utterance: str) -> str:
        """
        Runs deterministic keyword-boundary embedding simulation over incoming strings.
        Returns the mapped intent class or a default routing bucket.
        """
        normalized_text = utterance.lower()
        for pattern, intent in self.intent_rules.items():
            if re.search(pattern, normalized_text):
                return intent
            
        return "general_inquiry"