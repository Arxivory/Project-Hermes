import os
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Hermes_Model_Registry")

def register_local_model():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../triton/intent_model/1"))

    print(f"Loading trained weights from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    explicit_pip_requirements = [
        "torch",
        "transformers",
        "accelerate"
    ]

    with mlflow.start_run(run_name="intent_classifier_registration") as run:
        mlflow.log_param("model_type", "bert-sequence-classification")
        mlflow.log_param("vocab_size", tokenizer.vocab_size)
        mlflow.log_metric("eval_accuracy", 0.942) # last project evaluation

        print("Logging and registering model to MLflow (this may take a moment)...")

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="intent_classifier",
            task="text-classification",
            pip_requirements=explicit_pip_requirements,
            registered_model_name="Hermes_Intent_Classifier"
        )

        print("Success! Model is now registered.")

if __name__ == "__main__":
    register_local_model()