import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

def train_intent_model():
    print("Initializing Project Hermes NLP Training Pipeline...")

    print("Sourcing Bitext Financial Support Dataset from Hugging Face...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    unique_intents = sorted(list(set(dataset["train"]["intent"])))
    intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
    id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}

    print(f"Successfully extracted {len(unique_intents)} unique financial intent classes.")

    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretained(model_ckpt)

    def preprocess_function(examples):
        inputs = tokenizer(examples["utterance"], truncation=True, padding="max_length", max_length=64)
        inputs["labels"] = [intent_to_id[intent] for intent in examples["intent"]]
        return inputs
    
    print("Tokenizing textual utterances across splits...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=len(unique_intents),
        id2label=id_to_intent,
        label2id=intent_to_id
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return metric.compute(predictions=preds, references=labels)
    
    output_dir = "./results"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Running fine-tuning sequence on dataset...")
    trainer.train()

    model_save_path = "../../../infrastructure/triton/intent_model/1/model.safetensors"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_pretrained(os.path.dirname(model_save_path))
    tokenizer.save_pretrained(os.path.dirname(model_save_path))
    print(f"Model artifact successfully saved to local directory framework: {model_save_path}")

if __name__ == "main":
    train_intent_model()