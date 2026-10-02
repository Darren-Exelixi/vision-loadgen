import threading
import time

from vision_loadgen.config import KafkaConfig
from vision_loadgen.kafka_io import TopicWaker
from vision_loadgen.registry import Registry


class FakeProducer:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[str, str, dict]] = []
        self.lock = threading.Lock()
        self.flushed = 0
        self.closed = False

    def send(self, topic, key, value):
        if self.fail:
            raise RuntimeError("broker down")
        with self.lock:
            self.sent.append((topic, key, value))

    def flush(self, timeout=None):
        self.flushed += 1

    def close(self, timeout=None):
        self.closed = True

    def count(self) -> int:
        with self.lock:
            return len(self.sent)


def _message(now: float) -> dict:
    return {"camera_id": "loadgen-wake-r", "timestamp": now, "image_path": "/frames/a.jpg"}


def _waker(producer: FakeProducer, make_message=_message, **kafka) -> TopicWaker:
    config = KafkaConfig(**{"topic": "frames", "wake_interval_s": 0.02, **kafka})
    return TopicWaker(config, make_message, producer_factory=lambda cfg: producer)


def test_sends_only_while_the_block_runs():
    producer = FakeProducer()
    waker = _waker(producer)
    with waker.during("emotion sync"):
        time.sleep(0.2)
    during = producer.count()
    time.sleep(0.1)

    assert during >= 3
    assert producer.count() == during  # stopped when the sync ended
    assert producer.flushed == 1
    topic, key, value = producer.sent[0]
    assert (topic, key) == ("frames", "loadgen-wake-r")
    waker.close()
    assert producer.closed


def test_first_message_goes_out_immediately():
    producer = FakeProducer()
    with _waker(producer, wake_interval_s=60).during("sync"):
        deadline = time.monotonic() + 1
        while producer.count() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert producer.count() == 1


def test_disabled_or_without_a_frame_sends_nothing():
    producer = FakeProducer()
    with _waker(producer, wake_during_sync=False).during("sync"):
        time.sleep(0.05)
    with _waker(producer, make_message=lambda now: None).during("sync"):
        time.sleep(0.05)
    assert producer.count() == 0


def test_failures_never_reach_the_sync():
    producer = FakeProducer(fail=True)
    ran = []
    with _waker(producer).during("sync"):
        time.sleep(0.05)
        ran.append(True)
    assert ran

    def broken_factory(cfg):
        raise RuntimeError("no brokers")

    waker = TopicWaker(KafkaConfig(), _message, producer_factory=broken_factory)
    with waker.during("sync"):
        ran.append(True)
    assert len(ran) == 2


def test_registrar_wakes_the_topic_while_a_worker_syncs(tmp_path):
    from vision_loadgen.environment import build_app_config
    from vision_loadgen.registrar import Registrar

    environ = {"LOADGEN_ENVIRONMENT": "staging", "POSTGRES_URL": "postgresql://x@127.0.0.1:1"}
    app = build_app_config(worker_settings_spec="", environ=environ, package_env={})
    app.kafka.wake_interval_s = 0.02
    registry = Registry(run_id="r1", environment="staging", path=str(tmp_path / "r.json"))
    registrar = Registrar(app, ["emotion"], registry)
    registrar.set_wake_frame({"image_path": "/frames/source.jpg", "camera_id": "real"})
    assert Registry.load(tmp_path / "r.json").wake_image_path == "/frames/source.jpg"

    producer = FakeProducer()
    seen_during_sync = []

    class Client:
        def sync(self):
            time.sleep(0.1)
            seen_during_sync.append(producer.count())
            return "status"

    import vision_loadgen.registrar as registrar_module
    original = registrar_module.TopicWaker
    registrar_module.TopicWaker = lambda kafka, make: TopicWaker(kafka, make, producer_factory=lambda cfg: producer)
    try:
        assert registrar._sync("emotion", Client()) == "status"
    finally:
        registrar_module.TopicWaker = original
    registrar.close_waker()

    assert seen_during_sync[0] >= 2
    _, key, message = producer.sent[0]
    assert key == "loadgen-wake-r1"
    assert message["image_path"] == "/frames/source.jpg"
    assert message["loadgen"] == "r1" and message["wake"] is True
    assert message["frame_date"] and message["frame_hour"]
    assert producer.closed
