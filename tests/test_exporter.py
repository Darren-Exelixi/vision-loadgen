import socket
import time
import urllib.request

import pytest

from vision_loadgen.analysis import WorkerSample
from vision_loadgen.config import OutputConfig
from vision_loadgen.environment import build_app_config
from vision_loadgen.exporter import CONTENT_TYPE, METRICS, MetricsExporter
from vision_loadgen.kafka_io import ProducerStats
from vision_loadgen.metrics import Recorder


def _lines(exporter: MetricsExporter) -> list[str]:
    return [line for line in exporter.render().splitlines() if not line.startswith("#")]


def test_render_text_format_with_run_id_and_escaping():
    exporter = MetricsExporter("20260930-120000-ab12")
    exporter.set("loadgen_active_cameras", 5)
    exporter.set("loadgen_stage_info", 1, stage='level "5"\\a\nb', stage_index="01")
    exporter.set("loadgen_frames_sent_total", 12)
    text = exporter.render()
    assert "# TYPE loadgen_active_cameras gauge" in text
    assert "# TYPE loadgen_frames_sent_total counter" in text
    assert "# HELP loadgen_active_cameras " in text
    assert 'loadgen_active_cameras{run_id="20260930-120000-ab12"} 5.0' in text
    assert 'loadgen_stage_info{run_id="20260930-120000-ab12",stage="level \\"5\\"\\\\a\\nb",stage_index="01"} 1.0' in text
    assert text.endswith("\n")


def test_empty_metrics_are_left_out():
    exporter = MetricsExporter("r")
    assert exporter.render() == ""
    exporter.set("loadgen_consumer_lag", 3, worker="crowd")
    assert "loadgen_active_cameras" not in exporter.render()


def test_none_removes_and_clear_matches_labels():
    exporter = MetricsExporter("r")
    for copy in (0, 1):
        for worker in ("crowd", "emotion"):
            exporter.set("loadgen_gpu_copy_staleness_p95_seconds", 1.5, worker=worker, gpu_copy=copy)
    exporter.clear("loadgen_gpu_copy_staleness_p95_seconds", worker="crowd")
    assert len(_lines(exporter)) == 2 and all('worker="emotion"' in line for line in _lines(exporter))
    exporter.set("loadgen_gpu_copy_staleness_p95_seconds", None, worker="emotion", gpu_copy=0)
    assert len(_lines(exporter)) == 1
    exporter.clear("loadgen_gpu_copy_staleness_p95_seconds")
    assert _lines(exporter) == []


def test_inc_and_special_values():
    exporter = MetricsExporter("r")
    exporter.inc("loadgen_publish_errors_total")
    exporter.inc("loadgen_publish_errors_total", 2)
    assert exporter.value("loadgen_publish_errors_total") == 3.0
    exporter.set("loadgen_clock_offset_seconds", float("nan"), worker="crowd")
    assert 'loadgen_clock_offset_seconds{run_id="r",worker="crowd"} NaN' in exporter.render()


def test_undeclared_metric_or_wrong_labels_raise():
    exporter = MetricsExporter("r")
    with pytest.raises(KeyError):
        exporter.set("loadgen_nope", 1)
    with pytest.raises(ValueError):
        exporter.set("loadgen_consumer_lag", 1)
    with pytest.raises(ValueError):
        exporter.set("loadgen_consumer_lag", 1, worker="crowd", extra="x")


def test_metric_names_follow_conventions():
    for name, (kind, help_text, labels) in METRICS.items():
        assert name.startswith("loadgen_") and help_text
        assert kind in ("gauge", "counter")
        assert name.endswith("_total") == (kind == "counter")
        assert "run_id" not in labels


def test_http_scrape_and_linger():
    exporter = MetricsExporter("r", "127.0.0.1", 0)
    assert exporter.start()
    exporter.set("loadgen_active_cameras", 7)
    with urllib.request.urlopen(f"http://127.0.0.1:{exporter.port}/metrics", timeout=5) as response:
        assert response.headers["Content-Type"] == CONTENT_TYPE
        assert 'loadgen_active_cameras{run_id="r"} 7.0' in response.read().decode()
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"http://127.0.0.1:{exporter.port}/other", timeout=5)
    assert error.value.code == 404
    started = time.monotonic()
    exporter.stop(linger_s=0.3)
    assert time.monotonic() - started >= 0.3
    with pytest.raises(OSError):
        urllib.request.urlopen(f"http://127.0.0.1:{exporter.port}/metrics", timeout=2)
    exporter.stop()  # stopping twice is harmless


def test_busy_port_warns_and_does_not_fail(caplog):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        exporter = MetricsExporter("r", "127.0.0.1", busy.getsockname()[1])
        assert exporter.start() is False
    assert "Metrics endpoint not started" in caplog.text
    exporter.stop(linger_s=5)  # never started: returns at once


def test_recorder_exports_each_batch(tmp_path):
    exporter = MetricsExporter("r")
    recorder = Recorder(tmp_path, ["crowd", "emotion"], exporter)
    batch = [
        WorkerSample(worker="crowd", t=time.time(), ok=True, running=True, lag=4,
                     synthetic_staleness=[1.0, 2.0, 3.0], gpu_staleness={0: [1.0, 2.0], 1: [3.0]}),
        WorkerSample(worker="emotion", t=time.time(), ok=False, lag=9, error="timeout"),
    ]
    recorder.record("level 5", 5, ProducerStats(sent=10, acked=9, errors=1), batch, {"crowd": 0.5})
    assert exporter.value("loadgen_active_cameras") == 5
    assert exporter.value("loadgen_frames_acked_total") == 9
    assert exporter.value("loadgen_publish_errors_total") == 1
    assert exporter.value("loadgen_worker_up", worker="crowd") == 1
    assert exporter.value("loadgen_worker_up", worker="emotion") == 0
    assert exporter.value("loadgen_worker_running", worker="crowd") == 1
    assert exporter.value("loadgen_consumer_lag", worker="emotion") == 9
    assert exporter.value("loadgen_synthetic_staleness_seconds", worker="crowd", quantile="max") == 3.0
    assert exporter.value("loadgen_synthetic_staleness_seconds", worker="emotion", quantile="max") is None
    assert exporter.value("loadgen_real_staleness_excess_seconds", worker="crowd") == 0.5
    assert exporter.value("loadgen_gpu_copy_staleness_p95_seconds", worker="crowd", gpu_copy=1) == 3.0

    # The next batch has one GPU copy fewer: its series goes away.
    batch[0].gpu_staleness = {0: [1.0]}
    recorder.record("level 5", 5, ProducerStats(sent=20, acked=19), batch, {"crowd": 0.5})
    assert exporter.value("loadgen_gpu_copy_staleness_p95_seconds", worker="crowd", gpu_copy=1) is None
    assert exporter.value("loadgen_frames_sent_total") == 20
    recorder.close()


def test_recorder_without_exporter_is_unchanged(tmp_path):
    recorder = Recorder(tmp_path, ["crowd"])
    recorder.record("level 5", 5, ProducerStats(), [WorkerSample(worker="crowd", t=time.time(), ok=True)], {})
    recorder.close()
    assert (tmp_path / "timeseries.csv").read_text().count("\n") == 2


def test_metrics_port_from_env_and_overrides():
    assert OutputConfig().metrics_port is None
    environ = {"LOADGEN_METRICS_PORT": "9464", "LOADGEN_METRICS_ADDR": "127.0.0.1"}
    app = build_app_config(worker_settings_spec="", environ=environ, package_env={})
    assert app.output.metrics_port == 9464 and app.output.metrics_addr == "127.0.0.1"
    app = build_app_config(worker_settings_spec="", overrides={"output": {"metrics_port": 9999}}, environ=environ, package_env={})
    assert app.output.metrics_port == 9999
    assert build_app_config(worker_settings_spec="", environ={"LOADGEN_METRICS_PORT": ""}, package_env={}).output.metrics_port is None
