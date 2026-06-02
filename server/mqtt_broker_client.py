class MQTTBrokerClient:
    def publish(self, topic: str, payload: dict):
        print(f"[MQTT] {topic}: {payload}")
