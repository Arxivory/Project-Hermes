import time
import json
import uuid
import random
from datetime import datetime
from datasets import load_dataset
import requests

class BPOTrafficSimulator:
    def __init__(self, use_kafka: bool = False, bootstrap_servers: str = "localhost:9092", topic: str = "customer.events"):
        self.topic = topic
        self.use_kafka = use_kafka
        self.customer_tiers = ["REGULAR", "VIP", "PLATINUM"]

        print("Sourcing benchmark utterances from Bitext Financial Hub...")
        self.dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

        if self.use_kafka:
            try:
                from confluent_kafka import Producer
                conf = {'bootstrap.servers': bootstrap_servers, 'client.id': 'bpo-simulator-producer', 'acks': 1}
                self.producer = Producer(conf)
                print("Kafka producer initialized successfully.")
            except Exception as e:
                print("Kafka initialization failed ({e}). Falling back to Native HTTP mode.")
                self.use_kafka = False

    def start_streaming(self, delay_sec: float = 2.0):
        print(f"Project Hermes traffic pipeline active. Mode: {'Kafka' if self.use_kafka else 'Direct HTTP Request'}")

        try:
            while True:
                sample = random.choice(self.dataset)
                
                event_payload = {
                    "ticket_id": str(uuid.uuid4()),
                    "session_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "customer_tier": random.choice(self.customer_tiers),
                    "channel": random.choice(["CHAT", "MOBILE_APP"]),
                    "raw_utterance": sample["instruction"]
                }

                if self.use_kafka:
                    self.producer.produce(self.topic, key=event_payload["ticket_id"], value=json.dumps(event_payload))
                    self.producer.poll(0)
                    print(f"Dispatched event to Kafka topic: {self.topic}")
                else:
                    try:
                        response = requests.post("http://localhost:8000/api/v1/tickets", json=event_payload, timeout=1.0)
                        print(f"Pushed event directly to API Router: Code {response.status_code}")
                    except requests.exceptions.ConnectionError:
                        print("Router app is offline. Retrying next tick...")

                time.sleep(delay_sec)
        except KeyboardInterrupt:
            print("Traffic generator Stopped by user.")