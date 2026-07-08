import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class LocalIntentClassifier:
    def __init__(self, model_dir: str = "infrastructure/triton/intent_model/1"):
        print(f"Initializing ML Pipeline. Loading weights from: {model_dir}...")

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", model_dir))

        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(base_path)

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label if hasattr(self.model.config, 'id2label') else {
            0: "fraud_alert",
            1: "billing_dispute",
            2: "account_cancellation",
            3: "limit_upgrade",
            4: "general_inquiry"
        }
        
        print("Trained NLP Weights loaded successfully into local system memory!")

    def predict_intent(self, utterance: str) -> str:
        """
        Runs inference over the inbound customer string.
        """
        inputs = self.tokenizer(
            utterance,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()

        return self.id2label.get(predicted_class_id, "general_inquiry")