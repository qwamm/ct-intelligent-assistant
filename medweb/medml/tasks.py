import dramatiq
from dramatiq.brokers.rabbitmq import RabbitmqBroker

broker = RabbitmqBroker(host="localhost", port=5672)
dramatiq.set_broker(broker)


def send_prediction_task(file_path: str, projection_type: str, image_id: int):
    message = dramatiq.Message(
        queue_name="predict_all",
        actor_name="predict_all",
        args=(file_path, projection_type, image_id),
        kwargs={},
        options={},
    )

    broker.enqueue(message)
    return message