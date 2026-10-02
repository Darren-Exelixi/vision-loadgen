import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from vision_loadgen.environment import build_app_config
from vision_loadgen.ui.server import (
    BadRequest, JobManager, UiServer, WorkerLogs, build_command, free_port, list_corpora, list_runs, load_run, parse_prometheus,
)


def test_free_port_skips_a_port_held_on_loopback():
    # Runs listen on 127.0.0.1; on Windows a wildcard-only probe would call this port free.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        assert free_port(port, attempts=5) != port

TEMPLATE = "7797ee15-32d5-46b7-b736-ac078e9b9b8c"


@pytest.fixture
def app(tmp_path):
    environ = {"LOADGEN_ENVIRONMENT": "staging", "LOADGEN_RESULTS_DIR": str(tmp_path / "results"),
               "LOADGEN_CORPUS_DIR": str(tmp_path / "corpora"), "POSTGRES_URL": "postgresql://x@127.0.0.1:1"}
    return build_app_config(worker_settings_spec="", environ=environ, package_env={})


def test_check_and_run_arguments(app, tmp_path):
    params = {"workers": ["emotion", "ai-attendance"], "server_ip": "10.10.10.22", "scenario": "latency",
              "template_camera": TEMPLATE, "source": "corpus", "corpus": "lobby", "set": ["levels=[5,10]"]}
    kind, argv = build_command(app, "run", {**params, "no_guard": True}, False, tmp_path)
    assert kind == "run"
    assert argv == ["run", "--scenario", "latency", "--worker", "emotion", "--worker", "attendance",
                    "--server-ip", "10.10.10.22", "--template-camera", TEMPLATE, "--set", "levels=[5,10]",
                    "--set", "source.mode=corpus", "--set", "source.corpus_name=lobby", "--no-guard"]
    assert build_command(app, "run", params, True, tmp_path)[1][-1] == "--allow-production"
    assert build_command(app, "check", params, False, tmp_path)[1][0] == "check"
    # Sizing comes first, so a typed `sizing=...` override still wins.
    argv = build_command(app, "run", {**params, "sizing": "fixed"}, False, tmp_path)[1]
    assert argv[argv.index("--set"):argv.index("--set") + 4] == ["--set", "sizing=fixed", "--set", "levels=[5,10]"]


@pytest.mark.parametrize("change", [
    {"workers": []}, {"workers": ["nope"]}, {"server_ip": "10.0.0.1; rm -rf /"}, {"scenario": "stress"},
    {"template_camera": "not-a-uuid"}, {"set": ["bad key=1"]}, {"set": ["novalue"]}, {"source": "corpus", "corpus": "../x"},
    {"sizing": "huge"},
])
def test_invalid_test_parameters_are_rejected(app, tmp_path, change):
    params = {"workers": ["emotion"], "scenario": "latency", **change}
    with pytest.raises(BadRequest):
        build_command(app, "run", params, False, tmp_path)


def test_corpus_and_camera_actions(app, tmp_path):
    with pytest.raises(BadRequest):
        build_command(app, "corpus", {"video": "clip.mp4", "name": "lobby"}, False, tmp_path)  # not uploaded
    (tmp_path / "clip.mp4").write_bytes(b"x")
    _, argv = build_command(app, "corpus", {"video": "clip.mp4", "name": "lobby", "fps": "5",
                                            "image_root": "/app/events/loadgen_corpus/lobby",
                                            "copy_target": "admin1@10.10.10.22:/srv/events/loadgen_corpus/"},
                            False, tmp_path)
    assert argv[:4] == ["corpus", "from-video", "--video", str(tmp_path / "clip.mp4")]
    assert "--image-root" in argv and "--copy-target" in argv
    with pytest.raises(BadRequest):
        build_command(app, "corpus", {"video": "../clip.mp4", "name": "lobby"}, False, tmp_path)
    with pytest.raises(BadRequest):
        build_command(app, "corpus", {"video": "clip.mp4", "name": "lobby", "fps": "500"}, False, tmp_path)
    assert build_command(app, "video-camera-add", {"name": "Lobby video", "like": TEMPLATE}, False, tmp_path)[1] == \
        ["video-camera", "add", "--name", "Lobby video", "--like", TEMPLATE]
    assert build_command(app, "video-camera-remove", {"camera_id": TEMPLATE}, False, tmp_path)[1][-1] == TEMPLATE
    assert build_command(app, "cleanup-orphans", {"yes": True}, False, tmp_path)[1] == ["cleanup", "--orphans", "--yes"]
    with pytest.raises(BadRequest):
        build_command(app, "shell", {}, False, tmp_path)


def test_parse_prometheus_keeps_loadgen_metrics():
    text = ('# HELP loadgen_active_cameras x\n# TYPE loadgen_active_cameras gauge\n'
            'loadgen_active_cameras{run_id="r"} 5.0\n'
            'loadgen_stage_info{run_id="r",stage="level \\"5\\"",stage_index="01"} 1.0\n'
            'other_metric 3\n')
    metrics = parse_prometheus(text)
    assert set(metrics) == {"loadgen_active_cameras", "loadgen_stage_info"}
    assert metrics["loadgen_active_cameras"][0] == {"labels": {"run_id": "r"}, "value": 5.0}
    assert metrics["loadgen_stage_info"][0]["labels"]["stage"] == 'level "5"'


def test_job_lifecycle_output_and_stop(tmp_path):
    # Stands in for a run: watches LOADGEN_STOP_FILE as Runner does.
    script = ("import os, sys, time\n"
              "print('Run 20261001-120000-abcd: test', flush=True)\n"
              "for _ in range(300):\n"
              "    if os.path.exists(os.environ['LOADGEN_STOP_FILE']):\n"
              "        print('stopping', flush=True); sys.exit(3)\n"
              "    time.sleep(0.1)\n")
    manager = JobManager(tmp_path / "jobs", [sys.executable, "-c", script])
    job = manager.start("run", [])
    assert job.metrics_port and job.argv[-2:] == ["--metrics-port", str(job.metrics_port)]
    deadline = time.time() + 15
    text = ""
    while "Run 2026" not in text and time.time() < deadline:
        time.sleep(0.2)
        text, _ = manager.output(job, 0)
    assert job.run_id == "20261001-120000-abcd"
    manager.stop(job)
    job.process.wait(timeout=15)
    time.sleep(0.3)
    assert job.ended_at is not None and job.returncode == 3
    assert "stopping" in manager.output(job, 0)[0]


def test_exclusive_jobs(tmp_path):
    manager = JobManager(tmp_path / "jobs", [sys.executable, "-c", "import time; time.sleep(30)"])
    first = manager.start("cleanup", [])
    try:
        with pytest.raises(BadRequest):
            manager.start("cleanup", [])
        other = manager.start("check", [])  # not exclusive
        manager.stop(other)  # jobs other than runs are ended
        assert other.process.wait(timeout=15) != 0
    finally:
        first.process.kill()


def _write_run(results, run_id="20261001-120000-abcd"):
    directory = results / run_id
    directory.mkdir(parents=True)
    summary = {"run_id": run_id, "scenario": {"name": "s", "type": "latency", "workers": ["emotion"],
                                              "source": {"mode": "corpus"}},
               "stages": [{"name": "level 5", "workers": {"emotion": {"verdict": {"kept_up": True}}}}]}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "timeseries.csv").write_text("time,elapsed_s,active_cameras,emotion_staleness_p95\nt,5.0,5,0.40\n")
    return run_id


def test_history(tmp_path):
    run_id = _write_run(tmp_path)
    runs = list_runs(tmp_path)
    assert runs[0]["run_id"] == run_id and runs[0]["source"] == "corpus" and runs[0]["stages"] == 1
    loaded = load_run(tmp_path, run_id)
    assert loaded["timeseries"][0]["emotion_staleness_p95"] == "0.40"
    with pytest.raises(KeyError):
        load_run(tmp_path, "../etc")
    corpus = tmp_path / "corpora" / "lobby"
    corpus.mkdir(parents=True)
    (corpus / "manifest.jsonl").write_text('{"image_path": "000000.jpg"}\n')
    (corpus / "corpus.json").write_text('{"frames": 1, "fps": 5.0, "image_root": "/app/x"}')
    assert list_corpora(tmp_path / "corpora") == [{"name": "lobby", "frames": 1, "fps": 5.0, "image_root": "/app/x",
                                                   "source_video": None, "has_header": True}]


def test_http_page_api_upload_and_origin_check(app, tmp_path):
    _write_run(tmp_path / "results")
    ui = UiServer(app, base_argv=[sys.executable, "-c", "print('hello')"])
    httpd = ui.http_server("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<title>vision-loadgen</title>" in page and "chart.umd" in page
        config = json.loads(urllib.request.urlopen(base + "/api/config").read())
        assert config["environment"] == "staging" and {w["name"] for w in config["workers"]} >= {"emotion", "attendance"}
        assert json.loads(urllib.request.urlopen(base + "/api/runs").read())[0]["scenario"] == "s"

        upload = urllib.request.Request(base + "/api/videos?name=clip.mp4", data=b"0123456789", method="POST")
        assert json.loads(urllib.request.urlopen(upload).read()) == {"video": "clip.mp4", "bytes": 10}
        assert json.loads(urllib.request.urlopen(base + "/api/uploads").read()) == ["clip.mp4"]

        body = json.dumps({"action": "check", "params": {"workers": ["emotion"]}}).encode()
        request = urllib.request.Request(base + "/api/jobs", data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        job = json.loads(urllib.request.urlopen(request).read())
        deadline = time.time() + 15
        while time.time() < deadline:
            state = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{job['id']}?since=0").read())
            if not state["running"]:
                break
            time.sleep(0.2)
        assert state["returncode"] == 0 and "hello" in state["output"]

        bad = urllib.request.Request(base + "/api/jobs", data=json.dumps({"action": "run", "params": {}}).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 400
        cross = urllib.request.Request(base + "/api/jobs", data=body, method="POST",
                                       headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(cross)
        assert error.value.code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_worker_log_commands(app, tmp_path):
    logs = WorkerLogs(app, tmp_path / "logs")
    assert logs.command("emotion") == ["docker", "logs", "--follow", "--timestamps", "--tail", "300",
                                       "sentiment_analysis_backend"]
    assert {t["worker"]: t["container"] for t in logs.targets()} == {
        "crowd": "crowd_monitoring_backend", "emotion": "sentiment_analysis_backend", "attendance": "frs_backend",
        "ppe": "ppe_backend", "intrusion": "intrusion_backend", "fire_smoke": "fire_smoke_backend",
        "obstacle": "obstacle_detection_backend", "productivity": "productivity_monitoring_backend",
        "fall": "fall_detection_backend"}
    app.worker_logs.ssh_target, app.worker_logs.docker_command = "admin1@10.10.10.22", "sudo -n docker"
    assert logs.command("attendance") == [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "admin1@10.10.10.22",
        "sudo -n docker logs --follow --timestamps --tail 300 frs_backend"]
    app.worker_logs.password = "s3cret"
    argv = logs.command("attendance")
    assert argv == [sys.executable, "-m", "vision_loadgen.ssh_follow", "admin1@10.10.10.22",
                    "sudo", "-S", "-p", "", "docker", "logs", "--follow", "--timestamps", "--tail", "300", "frs_backend"]
    assert "s3cret" not in " ".join(argv) and "s3cret" not in repr(app.worker_logs)
    app.worker_logs.password = ""
    for change in ({"ssh_target": "admin1@host; rm -rf /"}, {"docker_command": "docker; reboot"}):
        for key, value in change.items():
            setattr(app.worker_logs, key, value)
        with pytest.raises(BadRequest):
            logs.command("emotion")
        app.worker_logs.ssh_target, app.worker_logs.docker_command = "admin1@10.10.10.22", "docker"
    app.workers["emotion"].container = "x y"
    with pytest.raises(BadRequest):
        logs.command("emotion")
    with pytest.raises(BadRequest):
        logs.command("nope")


def test_worker_log_follow_read_and_stop(app, tmp_path, monkeypatch):
    script = "import time\nprint('2026-10-01T10:00:00Z INFO ready', flush=True)\nprint('WARNING slow', flush=True)\ntime.sleep(60)\n"
    monkeypatch.setattr(WorkerLogs, "command", lambda self, worker: [sys.executable, "-c", script])
    logs = WorkerLogs(app, tmp_path / "logs")
    follow = logs.start("emotion")
    assert logs.start("emotion") is follow  # already following
    deadline = time.time() + 15
    data = logs.read("emotion", 0)
    while "WARNING slow" not in data["output"] and time.time() < deadline:
        time.sleep(0.2)
        data = logs.read("emotion", 0)
    assert data["running"] and "INFO ready" in data["output"]
    assert logs.read("emotion", data["offset"])["output"] == ""
    logs.stop("emotion")
    follow.process.wait(timeout=10)
    time.sleep(0.3)
    assert not logs.read("emotion", 0)["running"]
    with pytest.raises(KeyError):
        logs.read("crowd", 0)
