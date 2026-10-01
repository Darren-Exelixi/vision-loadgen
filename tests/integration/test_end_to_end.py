"""End-to-end run against real Postgres + Kafka with fake workers. Needs Docker; run with LOADGEN_IT=1."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import psycopg2
import pytest
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError
from psycopg2.extras import Json, RealDictCursor

from vision_loadgen import cli
from vision_loadgen.config import ConfigError, KafkaConfig, load_scenario
from vision_loadgen.environment import build_app_config
from vision_loadgen.kafka_io import LagReader
from vision_loadgen.registrar import Registrar, find_orphans, registry_for_orphan
from vision_loadgen.registry import Registry
from vision_loadgen.runner import RunOptions, Runner
from vision_loadgen.sources import CorpusSource, capture_corpus

# tests/ has no __init__.py; pytest puts this directory on sys.path, which makes the
# helpers importable by name.
import it_schema as schema
from fake_worker import FakeWorker

pytestmark = pytest.mark.skipif(os.environ.get("LOADGEN_IT") != "1", reason="set LOADGEN_IT=1 (needs Docker)")

HERE = Path(__file__).parent
COMPOSE = ["docker", "compose", "-f", str(HERE / "docker-compose.it.yml"), "-p", "loadgen-it"]
# The compat service runs inside the compose network against an already running stack.
MANAGE_STACK = os.environ.get("LOADGEN_IT_STACK") != "external"
PG = os.environ.get("LOADGEN_IT_PG", "postgresql://it:it@localhost:55432/{}")
BOOTSTRAP = os.environ.get("LOADGEN_IT_BOOTSTRAP", "localhost:59092")
TOPIC = "exelixi.frames.raw"
SECRET = "it-secret"


def _sql(db: str, statement: str, params=None, fetch: bool = False):
    conn = psycopg2.connect(PG.format(db), cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    finally:
        conn.close()


def _one(db: str, statement: str, params=None):
    rows = _sql(db, statement, params, fetch=True)
    return rows[0] if rows else None


def _wait_for_kafka() -> None:
    deadline = time.monotonic() + 90
    while True:
        try:
            admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP)
            if TOPIC not in admin.list_topics():
                admin.create_topics([NewTopic(TOPIC, num_partitions=1, replication_factor=1)])
            admin.close()
            return
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def _wait_for_consumer_groups(*prefixes: str) -> None:
    """A fresh broker takes a while to elect the group coordinator; wait until the fake workers commit."""
    reader = LagReader(KafkaConfig(bootstrap_servers=BOOTSTRAP, topic=TOPIC))
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                if all(reader.lag(prefix) is not None for prefix in prefixes):
                    return
            except KafkaError:  # e.g. CoordinatorLoadInProgressError right after the broker starts
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("fake workers never committed consumer offsets")
            time.sleep(1)
    finally:
        reader.close()


def _create_databases() -> None:
    conn = psycopg2.connect(PG.format("it"))
    conn.autocommit = True
    with conn.cursor() as cur:
        for name in ("vision_main", "crowd", "frs"):
            cur.execute(f"DROP DATABASE IF EXISTS {name}")
            cur.execute(f"CREATE DATABASE {name}")
    conn.close()
    _sql("vision_main", schema.MAIN)
    _sql("crowd", schema.CROWD)
    _sql("frs", schema.FRS)


def _seed(crowd_port: int, frs_port: int) -> dict:
    server = _one("vision_main", "INSERT INTO servers (server_ip, server_protocol, is_active) "
                                 "VALUES ('127.0.0.1', 'http', true) RETURNING id::text")["id"]
    functions = {}
    for key, port in (("crowd-monitoring", crowd_port), ("ai-attendance", frs_port)):
        function = _one("vision_main", "INSERT INTO functions (key, container_port, is_exelixi_activated) "
                                       "VALUES (%s, %s, true) RETURNING id::text", (key, port))["id"]
        functions[key] = _one("vision_main", "INSERT INTO server_functions (server_id, function_id, is_active) "
                                             "VALUES (%s, %s, true) RETURNING id::text", (server, function))["id"]
    region = _one("vision_main", "INSERT INTO camera_regions (name) VALUES ('Lobby') RETURNING id::text")["id"]
    template = _one("vision_main", """
        INSERT INTO cameras (name, region_id, type, ip, port, "user", password, timezone, rtsp_url, is_active, created_at)
        VALUES ('Lobby cam', %s, 'ip', '10.0.0.5', 554, 'admin', 'secret', 'Asia/Dubai', 'rtsp://10.0.0.5/live',
                true, now()) RETURNING id::text""", (region,))["id"]
    for server_function in functions.values():
        _sql("vision_main", "INSERT INTO function_camera_regions (server_function_id, camera_region_id, status) "
                            "VALUES (%s, %s, 'assigned')", (server_function, region))

    setting = _one("crowd", "INSERT INTO crowd_gathering_settings (name, selected_cameras) VALUES ('default', %s) "
                            "RETURNING id", (Json([template]),))["id"]
    _sql("crowd", "INSERT INTO crowd_gathering_camera_lines (camera_id, setting_id, line_start) VALUES (%s, %s, %s)",
         (template, setting, Json({"x": 0.5, "y": 0.0})))
    _sql("crowd", "INSERT INTO crowd_gathering_events (camera_id) VALUES (%s)", (template,))

    _sql("frs", "INSERT INTO frs_settings (name, check_in_cameras, is_enabled) VALUES ('default', %s, true)",
         (Json([template]),))
    employees = {name: str(uuid.uuid4()) for name in ("a", "b", "c")}
    attendance = {}
    for name in ("a", "b"):
        attendance[name] = _one("frs", """
            INSERT INTO frs_attendance (employee_id, date, check_in_at, is_late, shift_working_days)
            VALUES (%s, CURRENT_DATE, '2026-09-29 08:00:00+04', false, %s) RETURNING id::text""",
            (employees[name], Json(["Mon", "Tue"])))["id"]
    return {"template": template, "region": region, "employees": employees, "attendance": attendance}


class AttendanceSideEffects:
    """What a real attendance worker would do when a synthetic camera recognises employees."""

    def __init__(self, seed: dict) -> None:
        self.seed = seed
        self.done = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, camera: str, _payload: dict) -> None:
        if camera == self.seed["template"]:
            return
        with self._lock:
            if self.done.is_set():
                return
            employees, attendance, template = self.seed["employees"], self.seed["attendance"], self.seed["template"]
            _sql("frs", "UPDATE frs_attendance SET check_in_at = now(), is_late = true WHERE id = %s",
                 (attendance["a"],))
            _sql("frs", "INSERT INTO frs_recognition_events (employee_id, attendance_id, camera_id) VALUES (%s, %s, %s)",
                 (employees["a"], attendance["a"], camera))
            new_row = _one("frs", "INSERT INTO frs_attendance (employee_id, date, check_in_at) "
                                  "VALUES (%s, CURRENT_DATE, now()) RETURNING id::text", (employees["c"],))["id"]
            _sql("frs", "INSERT INTO frs_recognition_events (employee_id, attendance_id, camera_id) VALUES (%s, %s, %s)",
                 (employees["c"], new_row, camera))
            _sql("frs", "UPDATE frs_attendance SET is_late = true WHERE id = %s", (attendance["b"],))
            for source in (camera, template):
                _sql("frs", "INSERT INTO frs_recognition_events (employee_id, attendance_id, camera_id) "
                            "VALUES (%s, %s, %s)", (employees["b"], attendance["b"], source))
            self.done.set()


def _event_image(events: Path, camera: str) -> str:
    """Write an event image the way the crowd worker lays them out; returns the path relative to EVENTS_DIR."""
    relative = f"crowd-monitoring/crowd_monitoring_images/{camera}/2026-09-29/hc_event.jpg"
    (events / relative).parent.mkdir(parents=True, exist_ok=True)
    (events / relative).write_bytes(b"\xff\xd8\xff\xd9")
    return relative


def _crowd_event_writer(template: str, events: Path):
    seen: set[str] = set()

    def on_frame(camera: str, _payload: dict) -> None:
        if camera != template and camera not in seen:
            seen.add(camera)
            _sql("crowd", "INSERT INTO crowd_gathering_events (camera_id, image_path) VALUES (%s, %s)",
                 (camera, _event_image(events, camera)))

    return on_frame


class FrameRouter:
    """Publishes the template camera's frames like the real frame router."""

    def __init__(self, camera: str, image: Path) -> None:
        self._camera = camera
        self._image = image
        self._stop = threading.Event()
        self._producer = KafkaProducer(bootstrap_servers=BOOTSTRAP,
                                       value_serializer=lambda value: json.dumps(value).encode("utf-8"))

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._producer.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            self._producer.send(TOPIC, {"camera_id": self._camera, "timestamp": now, "image_path": str(self._image),
                                        "frame_date": time.strftime("%Y-%m-%d"), "frame_hour": time.strftime("%H")})
            self._stop.wait(0.25)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    if MANAGE_STACK:
        subprocess.run(COMPOSE + ["up", "-d", "--wait"], check=True)
    workers: list[FakeWorker] = []
    router = None
    try:
        _wait_for_kafka()
        _create_databases()
        tmp = tmp_path_factory.mktemp("loadgen")
        events = tmp / "events"
        # GPU: 2 cameras per model copy, 100 MB each, 200 MB free -> 4 cameras (1 real + 3 synthetic).
        crowd = FakeWorker("crowd-monitoring", PG.format("vision_main"), PG.format("crowd"),
                           "crowd_gathering_settings", "selected_cameras", BOOTSTRAP, TOPIC, SECRET, max_cameras=4)
        frs = FakeWorker("ai-attendance", PG.format("vision_main"), PG.format("frs"),
                         "frs_settings", "check_in_cameras", BOOTSTRAP, TOPIC, SECRET)
        seed = _seed(crowd.port, frs.port)
        _event_image(events, seed["template"])
        crowd.on_frame = _crowd_event_writer(seed["template"], events)
        side_effects = AttendanceSideEffects(seed)
        frs.on_frame = side_effects
        workers = [crowd, frs]
        for worker in workers:
            worker.start()

        image = tmp / "frame.jpg"
        image.write_bytes(b"\xff\xd8\xff\xd9")
        router = FrameRouter(seed["template"], image)
        router.start()

        (tmp / "loadgen.yaml").write_text(f"""
environment: staging
kafka: {{bootstrap_servers: "{BOOTSTRAP}", topic: {TOPIC}}}
database: {{main_url: "{PG.format('vision_main')}"}}
auth: {{jwt_secret_key: {SECRET}, jwt_algorithm: HS256}}
frames: {{shared_mount_path: "{tmp.as_posix()}", corpus_dir: "{(tmp / 'corpus').as_posix()}", timezone: Asia/Dubai}}
registration: {{template_camera_id: "{seed['template']}", sync_timeout_s: 30}}
output: {{results_dir: "{(tmp / 'results').as_posix()}", sample_interval_s: 1}}
events: {{dir: "{events.as_posix()}"}}
workers:
  crowd:
    db_url: "{PG.format('crowd')}"
    cameras_per_worker: 2
    vram_per_worker_mb: 100
    gpu_free_vram_mb: 200
    metrics: {{path: /metrics, stage_histogram: fake_stage_seconds, frames_counter: fake_frames_processed_total,
              stages: [detect], gpu_stage_limits: {{detect: 0.15}}}}
  attendance: {{db_url: "{PG.format('frs')}"}}
""", encoding="utf-8")
        (tmp / "scenario.yaml").write_text("""
name: it-latency
type: latency
workers: [crowd, attendance]
fps_per_camera: 4
levels: [2, 3, 5]
level_duration_s: 14
settle_s: 2
saturation: {window_s: 8, max_staleness_s: 5, max_lag_growth_per_s: 5}
guard: {baseline_samples: 2, grace_s: 5}
""", encoding="utf-8")
        _wait_for_consumer_groups("crowd-monitoring_consumer_group_", "ai-attendance_consumer_group_")
        yield {"tmp": tmp, "events": events, "seed": seed, "crowd": crowd, "frs": frs, "side_effects": side_effects}
    finally:
        if router:
            router.stop()
        for worker in workers:
            worker.stop()
        if MANAGE_STACK:
            subprocess.run(COMPOSE + ["down", "-v"], check=False)


def _app(stack):
    return build_app_config(str(stack["tmp"] / "loadgen.yaml"), worker_settings_spec="")


def _cli(stack, *args: str) -> int:
    return cli.main(["--config", str(stack["tmp"] / "loadgen.yaml"), "--worker-settings", "", *args])


def test_check_preflight_passes(stack, capsys):
    assert _cli(stack, "check", "--scenario", str(stack["tmp"] / "scenario.yaml")) == 0
    output = capsys.readouterr().out
    assert "Preflight passed" in output and "consumer group crowd-monitoring_consumer_group_" in output
    assert "room for 3 synthetic cameras next to 1 real" in output
    assert "caps the run at 3 synthetic cameras" in output and "events dir" in output


def test_capture_then_replay_corpus(stack):
    app = _app(stack)
    result = capture_corpus(app.kafka, app.frames, "it-corpus", duration_s=3, source_camera_ids=[stack["seed"]["template"]])
    assert result.frames >= 5 and result.missing_images == 0
    source = CorpusSource(app.frames, "it-corpus")
    first, second = source.frame_for(0), source.frame_for(0)
    assert first["image_path"] != second["image_path"] and Path(first["image_path"]).is_file()
    with pytest.raises(FileExistsError):
        capture_corpus(app.kafka, app.frames, "it-corpus", duration_s=1, source_camera_ids=[])


def test_run_publishes_measures_and_cleans_up(stack):
    app = _app(stack)
    scenario = load_scenario(str(stack["tmp"] / "scenario.yaml"), app)
    seed = stack["seed"]
    template = seed["template"]
    syncs_before = stack["crowd"].syncs
    app.output.metrics_port, app.output.metrics_addr, app.output.metrics_linger_s = 0, "127.0.0.1", 0.5

    runner = Runner(app, scenario, RunOptions())
    scraped: list[str] = []

    def scrape_mid_run():
        deadline = time.time() + 120
        while time.time() < deadline and not scraped:
            exporter = runner.exporter
            if (exporter is not None and exporter.value("loadgen_consumer_lag", worker="crowd") is not None
                    and exporter.value("loadgen_synthetic_staleness_seconds", worker="crowd", quantile="0.95")
                    is not None):
                with urllib.request.urlopen(f"http://127.0.0.1:{exporter.port}/metrics", timeout=5) as response:
                    scraped.append(response.read().decode())
            time.sleep(0.5)

    scraper = threading.Thread(target=scrape_mid_run, daemon=True)
    scraper.start()
    summary = runner.run()
    scraper.join(timeout=10)

    assert summary.get("error") is None, summary.get("error")
    assert scraped, "no mid-run scrape of /metrics"
    run_label = f'run_id="{summary["run_id"]}"'
    for series in ("loadgen_run_info{", "loadgen_stage_info{", "loadgen_active_cameras{", 'phase="running"',
                   'loadgen_synthetic_staleness_seconds{' + run_label + ',worker="crowd",quantile="0.95"}',
                   'loadgen_consumer_lag{' + run_label + ',worker="crowd"}'):
        assert series in scraped[0], series
    exporter = runner.exporter
    assert exporter.value("loadgen_phase", phase="done") == 1
    assert exporter.value("loadgen_stage_kept_up", worker="crowd", stage="level 3", stage_index="02") == 1
    assert abs(exporter.value("loadgen_clock_offset_seconds", worker="crowd")) <= 1.0
    assert summary["aborted"] is None
    assert summary["producer"]["acked"] > 50 and summary["producer"]["errors"] == 0
    assert summary["guard_baseline_real_cameras"] == {"crowd": 1, "attendance": 1}
    # Fake workers share this machine's clock, so the estimated offset is about zero.
    for clock in summary["clock_offset"].values():
        assert clock["offset_s"] is not None and abs(clock["offset_s"]) <= 1.0
    # The GPU cap (3 synthetic next to 1 real) drops level 5, so the worker never refuses a sync.
    assert "3 synthetic cameras" in summary["capacity_note"]
    assert [stage["cameras"] for stage in summary["stages"]] == [2, 3]
    assert stack["crowd"].peak_cameras == 4
    for stage in summary["stages"]:
        for name in ("crowd", "attendance"):
            entry = stage["workers"][name]
            assert entry["verdict"]["kept_up"], entry["verdict"]
            assert entry["staleness_p95"] is not None and entry["staleness_p95"] < 5
        assert stage["workers"]["crowd"]["gpu_workers_expected"] == 2
        assert set(stage["workers"]["crowd"]["gpu_workers"]) <= {"0", "1"}
        # The worker's own metrics: 4 ms detect (under the 5 ms bucket) and frames it processed.
        crowd_stats = stage["workers"]["crowd"]
        assert 0.002 < crowd_stats["worker_stage_p95_s"]["detect"] <= 0.005
        assert crowd_stats["processed_fps"] > 0
        assert crowd_stats["verdict"]["hint"] == ""
        assert "worker_stage_p95_s" not in stage["workers"]["attendance"]
    assert exporter.value("loadgen_worker_stage_seconds", worker="crowd", stage="detect", quantile="0.95") <= 0.005
    assert exporter.value("loadgen_publish_rate") is not None
    # One re-sync per stage plus the teardown sync.
    assert stack["crowd"].syncs - syncs_before == 3 and stack["frs"].syncs >= 3
    assert stack["side_effects"].done.is_set()

    run_dir = Path(app.output.results_dir) / summary["run_id"]
    with (run_dir / "timeseries.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and any(row["crowd_lag"] not in ("", "None") for row in rows)
    assert any(row["crowd_stage_detect_p95"] not in ("", "-") for row in rows)
    assert any(row["crowd_processed_fps"] not in ("", "-", "0.0") for row in rows)

    teardown = summary["teardown"]
    assert teardown["errors"] == []
    assert Registry.load(run_dir / "registry.json").status == "done"

    assert _one("vision_main", "SELECT count(*) AS n FROM cameras WHERE name LIKE 'loadtest-%%'")["n"] == 0
    assert _one("vision_main", "SELECT count(*) AS n FROM camera_regions WHERE name LIKE 'loadtest-%%'")["n"] == 0
    assert _one("vision_main", "SELECT count(*) AS n FROM function_camera_regions")["n"] == 2
    template_row = _one("vision_main", "SELECT rtsp_url FROM cameras WHERE id = %s", (template,))
    assert template_row["rtsp_url"] == "rtsp://10.0.0.5/live"

    assert _one("crowd", "SELECT selected_cameras::text AS c FROM crowd_gathering_settings")["c"] == json.dumps([template])
    assert _one("crowd", "SELECT count(*) AS n FROM crowd_gathering_camera_lines")["n"] == 1
    assert [row["camera_id"] for row in _sql("crowd", "SELECT camera_id FROM crowd_gathering_events", fetch=True)] == [template]
    assert teardown["steps"]["purge:crowd_gathering_events"]["files_deleted"] == 3
    images = stack["events"] / "crowd-monitoring" / "crowd_monitoring_images"
    assert sorted(path.name for path in images.iterdir()) == [template]

    assert _one("frs", "SELECT check_in_cameras AS c FROM frs_settings")["c"] == [template]
    a = _one("frs", "SELECT check_in_at, is_late, shift_working_days FROM frs_attendance WHERE id = %s",
             (seed["attendance"]["a"],))
    assert a["is_late"] is False and a["check_in_at"].isoformat() == "2026-09-29T04:00:00+00:00"
    assert a["shift_working_days"] == ["Mon", "Tue"]
    assert _one("frs", "SELECT count(*) AS n FROM frs_attendance WHERE employee_id = %s",
                (seed["employees"]["c"],))["n"] == 0
    assert _one("frs", "SELECT is_late FROM frs_attendance WHERE id = %s", (seed["attendance"]["b"],))["is_late"] is True
    assert teardown["manual_review"]["attendance"] == [seed["attendance"]["b"]]
    assert teardown["restored"]["attendance"] == {"restored": 1, "deleted": 1}
    events = _sql("frs", "SELECT camera_id::text AS camera FROM frs_recognition_events", fetch=True)
    assert [row["camera"] for row in events] == [template]


def test_unresponsive_worker_aborts_and_tears_down(stack):
    """The worker-health guard stops a run whose worker stops answering, even with --no-guard."""
    app = _app(stack)
    scenario_path = stack["tmp"] / "health-scenario.yaml"
    scenario_path.write_text("""
name: it-health
type: throughput
workers: [crowd]
fps_per_camera: 4
start_cameras: 1
step_cameras: 1
max_cameras: 2
step_duration_s: 60
settle_s: 2
saturation: {window_s: 8}
guard: {baseline_samples: 2, unreachable_s: 4}
""", encoding="utf-8")
    scenario = load_scenario(str(scenario_path), app)
    crowd = stack["crowd"]
    runner = Runner(app, scenario, RunOptions(no_guard=True))

    def hang_after_a_few_samples():
        deadline = time.time() + 90
        while time.time() < deadline and len(runner.samples["crowd"]) < 4:
            time.sleep(0.2)
        crowd.unresponsive = True

    hang = threading.Thread(target=hang_after_a_few_samples, daemon=True)
    hang.start()
    started = time.time()
    try:
        summary = runner.run()
    finally:
        crowd.unresponsive = False
        hang.join(timeout=5)

    assert summary.get("error") is None, summary.get("error")
    # 1 s samples: the trip lands 4-5 s after the first failed one.
    assert re.match(r"worker health: crowd: worker unreachable for [45]s \(limit 4s\)", summary["aborted"] or ""), \
        summary["aborted"]
    assert time.time() - started < 50  # well before the 60 s stage would have ended
    assert summary["teardown"]["errors"] == []
    assert _one("vision_main", "SELECT count(*) AS n FROM cameras WHERE name LIKE 'loadtest-%%'")["n"] == 0
    assert _one("crowd", "SELECT selected_cameras::text AS c FROM crowd_gathering_settings")["c"] == json.dumps(
        [stack["seed"]["template"]])


def test_video_corpus_with_video_camera_template(stack, capsys):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from vision_loadgen import video_camera

    app = _app(stack)
    tmp = stack["tmp"]
    clip = tmp / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for index in range(40):
        writer.write(np.full((48, 64, 3), index * 6, dtype=np.uint8))
    writer.release()
    image_root = "/app/events/loadgen_corpus/it-video"
    assert _cli(stack, "corpus", "from-video", "--video", str(clip), "--name", "it-video", "--fps", "4",
                "--image-root", image_root) == 0
    assert "Wrote 8 frames" in capsys.readouterr().out

    template = stack["seed"]["template"]
    camera = video_camera.add_video_camera(app, "IT video", template, ["crowd"], zones_from=template, corpus="it-video")
    assert camera["name"] == "loadgen-video-it-video"
    assert camera["cloned_rows"] == {"crowd:crowd_gathering_camera_lines": 1}
    assert [row["id"] for row in video_camera.list_video_cameras(app)] == [camera["id"]]
    row = _one("vision_main", "SELECT is_active, rtsp_url FROM cameras WHERE id = %s", (camera["id"],))
    assert row["is_active"] is False and row["rtsp_url"] == ""

    scenario_path = tmp / "video-scenario.yaml"
    scenario_path.write_text("""
name: it-video
type: latency
workers: [crowd]
fps_per_camera: 4
levels: [2]
level_duration_s: 10
settle_s: 2
saturation: {window_s: 6, max_staleness_s: 5, max_lag_growth_per_s: 5}
guard: {baseline_samples: 2, grace_s: 5}
source: {mode: corpus, corpus_name: it-video}
""", encoding="utf-8")
    assert _cli(stack, "check", "--scenario", str(scenario_path), "--template-camera", camera["id"]) == 0
    output = capsys.readouterr().out
    assert "corpus it-video: 8 frames sampled at 4.0 fps, read from /app/events/loadgen_corpus/it-video" in output
    assert "cannot be checked from here" in output

    seen: list[dict] = []
    crowd = stack["crowd"]
    original = crowd.on_frame

    def record(camera_id, payload):
        if payload.get("loadgen"):
            seen.append(payload)
        if original:
            original(camera_id, payload)

    crowd.on_frame = record
    try:
        app.registration.template_camera_id = camera["id"]
        summary = Runner(app, load_scenario(str(scenario_path), app), RunOptions()).run()
    finally:
        crowd.on_frame = original
    assert summary.get("error") is None, summary.get("error")
    assert summary["template_camera_id"] == camera["id"]
    assert summary["stages"][0]["workers"]["crowd"]["verdict"]["kept_up"], summary["stages"][0]
    assert seen and all(payload["image_path"].startswith(image_root + "/") for payload in seen)
    assert len({payload["camera_id"] for payload in seen}) == 2
    assert summary["teardown"]["errors"] == []

    # Teardown removed the run's cameras but kept the video camera; `remove` deletes it.
    assert _one("vision_main", "SELECT count(*) AS n FROM cameras WHERE name LIKE 'loadtest-%%'")["n"] == 0
    assert [row["id"] for row in video_camera.list_video_cameras(app)] == [camera["id"]]
    with pytest.raises(ConfigError):
        video_camera.remove_video_camera(app, template)
    report = video_camera.remove_video_camera(app, camera["id"])
    assert report["errors"] == [] and report["camera"] == "deleted"
    assert report["rows"]["crowd:crowd_gathering_camera_lines"] == 1
    assert video_camera.list_video_cameras(app) == []


def test_orphan_cleanup_after_crash(stack):
    app = _app(stack)
    registry = Registry.create(app.output.results_dir, "crashed-run", app.environment)
    Registrar(app, ["crowd"], registry).setup(2, stack["seed"]["template"])
    Path(registry.path).unlink()

    orphans = find_orphans(app)
    assert set(orphans) == {"crashed-run"} and len(orphans["crashed-run"]["camera_ids"]) == 2

    rebuilt = registry_for_orphan(app, app.output.results_dir, "crashed-run", orphans["crashed-run"])
    report = Registrar(app, app.configured_workers(), rebuilt).teardown(keep_events=False)
    assert report["errors"] == [], report["errors"]
    assert find_orphans(app) == {}
    assert _cli(stack, "cleanup", "--orphans") == 0
    assert _one("crowd", "SELECT selected_cameras::text AS c FROM crowd_gathering_settings")["c"] == \
        json.dumps([stack["seed"]["template"]])
    assert _one("crowd", "SELECT count(*) AS n FROM crowd_gathering_camera_lines")["n"] == 1
