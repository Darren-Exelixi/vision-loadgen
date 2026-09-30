from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer, TopicPartition

from vision_loadgen.config import KafkaConfig

log = logging.getLogger(__name__)


@dataclass
class ProducerStats:
    sent: int = 0
    acked: int = 0
    errors: int = 0
    last_error: str = ""


class FrameProducer:
    def __init__(self, kafka: KafkaConfig) -> None:
        self._topic = kafka.topic
        self._producer = KafkaProducer(
            bootstrap_servers=kafka.bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            key_serializer=lambda key: key.encode("utf-8"),
            acks=1,
            linger_ms=5,
            retries=3,
        )
        self._lock = threading.Lock()
        self._stats = ProducerStats()

    def send(self, message: dict) -> None:
        with self._lock:
            self._stats.sent += 1
        future = self._producer.send(self._topic, key=message["camera_id"], value=message)
        future.add_callback(self._on_ack)
        future.add_errback(self._on_error)

    def stats(self) -> ProducerStats:
        with self._lock:
            return ProducerStats(**vars(self._stats))

    def close(self, timeout_s: float = 10.0) -> None:
        try:
            self._producer.flush(timeout=timeout_s)
        finally:
            self._producer.close(timeout=timeout_s)

    def _on_ack(self, _metadata) -> None:
        with self._lock:
            self._stats.acked += 1

    def _on_error(self, exc: Exception) -> None:
        with self._lock:
            self._stats.errors += 1
            self._stats.last_error = str(exc)


class TopicTap:
    """Reads the frame topic without a consumer group, from the latest offset."""

    def __init__(self, kafka: KafkaConfig, on_message: Callable[[dict], None]) -> None:
        self._kafka = kafka
        self._on_message = on_message
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="loadgen-tap", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        consumer = None
        try:
            consumer = KafkaConsumer(
                self._kafka.topic,
                bootstrap_servers=self._kafka.bootstrap_servers,
                group_id=None,
                auto_offset_reset="latest",
                enable_auto_commit=False,
                value_deserializer=_decode_json,
            )
            while not self._stop.is_set():
                batches = consumer.poll(timeout_ms=500)
                for records in batches.values():
                    for record in records:
                        if isinstance(record.value, dict):
                            self._on_message(record.value)
        except Exception as exc:
            self.error = str(exc)
            log.exception("Topic tap stopped")
        finally:
            if consumer is not None:
                consumer.close()


def _decode_json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


class LagReader:
    """Consumer lag of each worker's newest consumer group on the frame topic."""

    def __init__(self, kafka: KafkaConfig) -> None:
        if not hasattr(KafkaAdminClient, "list_consumer_groups"):
            raise RuntimeError("consumer lag needs kafka-python 2.x (3.x renamed the admin group APIs)")
        self._topic = kafka.topic
        self._admin = KafkaAdminClient(bootstrap_servers=kafka.bootstrap_servers, client_id="loadgen-lag")
        self._offsets = KafkaConsumer(bootstrap_servers=kafka.bootstrap_servers, group_id=None)
        partitions = self._offsets.partitions_for_topic(self._topic) or set()
        self._partitions = [TopicPartition(self._topic, partition) for partition in sorted(partitions)]

    def newest_group(self, prefix: str) -> Optional[str]:
        groups = [entry[0] for entry in self._admin.list_consumer_groups()]
        candidates = [group for group in groups if group.startswith(prefix)]
        if not candidates:
            return None
        return max(candidates, key=lambda group: _numeric_suffix(group[len(prefix):]))

    def lag(self, prefix: str) -> Optional[int]:
        group = self.newest_group(prefix)
        if group is None or not self._partitions:
            return None
        committed = self._admin.list_consumer_group_offsets(group)
        end_offsets = self._offsets.end_offsets(self._partitions)
        total = 0
        found = False
        for partition in self._partitions:
            offset = committed.get(partition)
            if offset is None or offset.offset < 0:
                continue
            found = True
            total += max(0, end_offsets[partition] - offset.offset)
        return total if found else None

    def close(self) -> None:
        self._admin.close()
        self._offsets.close()


def _numeric_suffix(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return -1
