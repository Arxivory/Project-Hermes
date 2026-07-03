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