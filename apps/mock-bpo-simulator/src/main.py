from traffic_generator import BPOTrafficSimulator

def boot_traffic():
    simulator = BPOTrafficSimulator(bootstrap_servers="localhost:9092")
    simulator.start_streaming(delay_sec=2.0)

if __name__ == "__main__":
    boot_traffic()