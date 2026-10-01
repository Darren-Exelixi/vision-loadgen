from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from vision_loadgen import db
from vision_loadgen.config import AppConfig, ConfigError, load_scenario
from vision_loadgen.environment import DEFAULT_WORKER_SETTINGS, build_app_config, load_env_file
from vision_loadgen.kafka_io import LagReader
from vision_loadgen.registrar import Registrar, find_orphans, registry_for_orphan
from vision_loadgen.registry import REGISTRY_NAME, Registry
from vision_loadgen.runner import RunOptions, Runner, check_gates, plan_capacities, template_camera_id
from vision_loadgen.scenarios import build_stages, cap_stages
from vision_loadgen.sources import CorpusSource, LiveTapSource, capture_corpus
from vision_loadgen.workers import WorkerClient, resolve_deployment

log = logging.getLogger("vision_loadgen")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vision_loadgen",
        description="Kafka frame load generator. Run it beside a worker (a one-off container of the "
                    "worker's service) to test that worker with its own settings, or standalone with env vars.",
    )
    parser.add_argument("--config", help="Optional YAML/JSON file merged over the built-in presets")
    parser.add_argument("--env-file",
                        help="KEY=value file loaded before anything else; variables already set win "
                             "(default: LOADGEN_ENV_FILE, else ./.env if present; '' to skip)")
    parser.add_argument("--environment", choices=["staging", "production"],
                        help="Overrides LOADGEN_ENVIRONMENT (unset means production)")
    parser.add_argument("--worker-settings", default=DEFAULT_WORKER_SETTINGS,
                        help="module:attribute of the worker's Settings; '' to ignore it")
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    def scenario_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--scenario", default="latency",
                         help="throughput | latency | soak, or a scenario file (default: latency)")
        sub.add_argument("--worker", action="append", default=[],
                         help="Worker to test, by name or function key (repeatable); default: the worker "
                              "this runs beside")
        sub.add_argument("--template-camera", help="Real camera to copy; default: the worker's freshest live camera")
        sub.add_argument("--server-ip", help="Pick the deployment on this server when a module runs on several")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="Scenario override, e.g. --set max_cameras=40 --set guard.action=backoff")

    check = commands.add_parser("check", help="Read-only preflight: config, DBs, workers, GPU capacity, Kafka, frames")
    scenario_args(check)

    capture = commands.add_parser("capture", help="Record real frames into a replayable corpus")
    capture.add_argument("--name", required=True)
    capture.add_argument("--duration", type=float, default=300.0, help="Seconds to record")
    capture.add_argument("--camera", action="append", default=[], help="Source camera id (repeatable)")
    capture.add_argument("--max-frames", type=int, default=0)

    corpus = commands.add_parser("corpus", help="Build a replayable corpus from a video file")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    from_video = corpus_commands.add_parser("from-video", help="Sample a video into JPEGs + manifest (needs OpenCV)")
    from_video.add_argument("--video", required=True, help="Video file (mp4, avi, mkv, ...)")
    from_video.add_argument("--name", required=True, help="Corpus name (folder name)")
    from_video.add_argument("--fps", type=float, default=5.0,
                            help="Frames per second to keep; match the scenario's fps_per_camera (default 5)")
    from_video.add_argument("--max-seconds", type=float, default=0.0, help="Only the first N seconds of video")
    from_video.add_argument("--out", help="Where to write it (default: LOADGEN_CORPUS_DIR)")
    from_video.add_argument("--image-root",
                            help="Folder the workers will read it from (default /app/events/loadgen_corpus/<name>)")
    from_video.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90)")
    from_video.add_argument("--copy-target", default="",
                            help="scp destination to print, e.g. admin1@10.10.10.22:/srv/vision/events/loadgen_corpus/ "
                                 "(default LOADGEN_CORPUS_COPY_TARGET)")

    video_camera = commands.add_parser("video-camera", help="Persistent template camera for a video's perspective")
    video_commands = video_camera.add_subparsers(dest="video_command", required=True)
    add = video_commands.add_parser("add", help="Clone an existing camera into an inactive 'loadgen-video-' camera")
    add.add_argument("--name", required=True)
    add.add_argument("--like", required=True, help="Existing camera to copy region, timezone, ... from")
    add.add_argument("--zones-from", help="Camera whose per-module rows (e.g. emotion zones) to copy")
    add.add_argument("--worker", action="append", default=[],
                     help="Workers whose per-camera rows --zones-from copies (repeatable; default: all)")
    add.add_argument("--corpus", default="", help="Corpus it stands for (recorded in its description)")
    video_commands.add_parser("list", help="List video cameras")
    remove = video_commands.add_parser("remove", help="Delete a video camera and its per-module rows")
    remove.add_argument("camera_id")

    ui = commands.add_parser("ui", help="Local web page to run these commands and watch runs")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--allow-production", action="store_true", help="Allow runs against production from the page")

    run = commands.add_parser("run", help="Run a load test scenario")
    scenario_args(run)
    run.add_argument("--no-guard", action="store_true", help="Disable the real-camera guard (staging throughput only)")
    run.add_argument("--keep-events", action="store_true", help="Keep events created by synthetic cameras")
    run.add_argument("--allow-production", action="store_true")
    run.add_argument("--metrics-port", type=int,
                     help="Serve Prometheus metrics on this port during the run (default LOADGEN_METRICS_PORT; off)")

    cleanup = commands.add_parser("cleanup", help="Undo a run, or remove leftovers from crashed runs")
    target = cleanup.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--orphans", action="store_true")
    cleanup.add_argument("--yes", action="store_true", help="Actually delete orphans (default lists them)")
    cleanup.add_argument("--keep-events", action="store_true")
    return parser


def _overrides(args) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.environment:
        overrides["environment"] = args.environment
    if getattr(args, "template_camera", None):
        overrides["registration"] = {"template_camera_id": args.template_camera}
    if getattr(args, "metrics_port", None) is not None:
        overrides["output"] = {"metrics_port": args.metrics_port}
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("kafka").setLevel(logging.WARNING)
    try:
        env_file = load_env_file(args.env_file)
        if env_file:
            log.info("Loaded variables from %s (already-set variables kept)", env_file.resolve())
        app = build_app_config(args.config, args.worker_settings, _overrides(args))
        if app.local_worker:
            log.info("Running beside the %s worker (%s)", app.local_worker, app.workers[app.local_worker].function_key)
        if args.command == "check":
            return _check(app, args)
        if args.command == "capture":
            return _capture(app, args)
        if args.command == "corpus":
            return _corpus(app, args)
        if args.command == "video-camera":
            return _video_camera(app, args)
        if args.command == "ui":
            from vision_loadgen.ui.server import serve
            return serve(app, args.host, args.port, args.allow_production, env_file=args.env_file)
        if args.command == "run":
            return _run(app, args)
        return _cleanup(app, args)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2


def _scenario(app: AppConfig, args):
    scenario = load_scenario(args.scenario, app, args.set, args.worker)
    if args.server_ip:
        for name in scenario.workers:
            app.workers[name].server_ip = args.server_ip
    return scenario


def _report_clock(name: str, client: WorkerClient, ok, warn) -> None:
    for _ in range(4):  # a few more responses narrow the 1 s Date resolution
        client.status()
    if not client.clock.known:
        warn(f"{name}: no Date header, so clock offset is unknown (real-camera staleness uncorrected)")
        return
    message = (f"{name}: worker clock is {client.clock.seconds:+.1f}s vs this machine "
               f"(±{client.clock.error_s:.1f}s); real-camera staleness is corrected for it, "
               "synthetic staleness does not depend on it")
    (warn if abs(client.clock.seconds) > 2 else ok)(message)


def _check(app: AppConfig, args) -> int:
    scenario = _scenario(app, args)
    problems: list[str] = []

    def ok(message: str) -> None:
        print(f"  ok    {message}")

    def warn(message: str) -> None:
        print(f"  warn  {message}")

    def fail(message: str) -> None:
        problems.append(message)
        print(f"  FAIL  {message}")

    print(f"Environment: {app.environment}; scenario {scenario.name} ({scenario.type}, sizing {scenario.sizing}) "
          f"on {', '.join(scenario.workers)}, up to {scenario.registered_cameras} cameras at "
          f"{scenario.fps_per_camera} fps")
    print(f"Mode: {'beside the ' + app.local_worker + ' worker' if app.local_worker else 'standalone'}; "
          f"results in {app.output.results_dir}")
    try:
        check_gates(app, scenario, RunOptions(allow_production=True))
    except ConfigError as exc:
        fail(str(exc))
        return 1

    registrar = Registrar(app, scenario.workers, Registry(run_id="check", environment=app.environment, path=""))
    clients: dict[str, WorkerClient] = {}
    real_cameras: dict[str, int] = {}
    try:
        conn = db.connect(app.main_db_url())
        try:
            with conn.cursor() as cur:
                for name in scenario.workers:
                    worker = app.workers[name]
                    try:
                        deployment = resolve_deployment(cur, name, worker)
                        ok(f"{name}: deployed on {deployment.server_ip} -> {deployment.base_url}")
                    except ConfigError as exc:
                        fail(str(exc))
                        continue
                    client = WorkerClient(name, deployment.base_url, worker.api_prefix, app.auth)
                    try:
                        status = client.status()
                        clients[name] = client
                        real_cameras[name] = len(status.active_cameras)
                        ok(f"{name}: status reachable, running={status.running}, "
                           f"{len(status.active_cameras)} active cameras")
                        _report_clock(name, client, ok, warn)
                    except Exception as exc:
                        fail(f"{name}: /worker/status failed: {exc}")
                    if worker.note:
                        print(f"  note  {name}: {worker.note}")
        finally:
            conn.close()
    except Exception as exc:
        fail(f"vision-main database: {exc}")

    registrar.clients = clients
    template_id = template_camera_id(app, scenario)
    try:
        if not template_id and clients:
            template_id = registrar.pick_template()
        camera = registrar.template_camera(template_id)
        ok(f"template camera {camera['id']} '{camera['name']}' in region {camera['region_id']}")
    except Exception as exc:
        fail(f"template camera: {exc}")

    try:
        for name, key in registrar.check_settings_rows().items():
            ok(f"{name}: enabled settings row {key}")
    except Exception as exc:
        fail(f"settings: {exc}")

    capacities = plan_capacities(app, scenario.workers, real_cameras)
    for name, capacity in capacities.items():
        if capacity.max_synthetic_cameras is None:
            warn(f"{name}: GPU capacity unknown ({capacity.budget_source}); set gpu_free_vram_mb to cap the run")
        else:
            ok(f"{name}: {capacity.cameras_per_worker} cameras per model copy, {capacity.vram_per_worker_mb} MB each, "
               f"budget {capacity.budget_mb} MB ({capacity.budget_source}) -> room for "
               f"{capacity.max_synthetic_cameras} synthetic cameras next to {capacity.real_cameras} real")
    try:
        stages, note = cap_stages(build_stages(scenario), capacities, scenario)
        if note:
            warn(note)
        for stage in stages:
            copies = ", ".join(f"{name} {capacity.gpu_workers(stage.cameras) or '?'}"
                               for name, capacity in capacities.items())
            print(f"  plan  {stage.name:<22} {stage.duration_s:>6.0f}s   model copies: {copies}")
    except ConfigError as exc:
        fail(str(exc))

    if app.events.dir and Path(app.events.dir).is_dir():
        ok(f"events dir {app.events.dir} is mounted; event files will be cleaned")
    else:
        warn(f"events dir '{app.events.dir or 'unset'}' not available here; event files will not be cleaned")

    try:
        reader = LagReader(app.kafka)
        try:
            for name in scenario.workers:
                group = reader.newest_group(app.workers[name].group_prefix)
                if group:
                    ok(f"{name}: consumer group {group}, lag {reader.lag(app.workers[name].group_prefix)}")
                else:
                    warn(f"{name}: no consumer group starting with {app.workers[name].group_prefix} yet "
                         "(a worker with no active cameras has not started its Kafka pipeline)")
        finally:
            reader.close()
    except Exception as exc:
        fail(f"kafka: {exc}")

    if scenario.source.mode == "live" and template_id:
        source = LiveTapSource(app.kafka, scenario.source.source_camera_ids or [template_id], set())
        source.start()
        try:
            if source.wait_for_first_frame(scenario.source.first_frame_timeout_s):
                frame = source.frame_for(0)
                if Path(frame["image_path"]).is_file():
                    ok(f"live frame readable: {frame['image_path']}")
                else:
                    fail(f"live frame not readable here: {frame['image_path']} (check the frames mount)")
            else:
                fail("no live frames from the source camera(s)")
        finally:
            source.stop()

    if scenario.source.mode == "corpus":
        try:
            corpus = CorpusSource(app.frames, scenario.source.corpus_name,
                                  scenario.source.image_root or app.frames.corpus_image_root)
            ok(f"corpus {scenario.source.corpus_name}: {corpus.frames} frames"
               + (f" sampled at {corpus.fps} fps" if corpus.fps else "") + f", read from {corpus.image_root}")
            if corpus.fps and abs(corpus.fps - scenario.fps_per_camera) > 0.01:
                warn(f"corpus fps {corpus.fps} differs from fps_per_camera {scenario.fps_per_camera}: "
                     f"playback runs {scenario.fps_per_camera / corpus.fps:.1f}x real time")
            if not corpus.verified:
                warn(f"{corpus.image_root} is on the workers' side and cannot be checked from here; make sure the "
                     "corpus was copied there (a worker that cannot read it shows no synthetic staleness)")
        except Exception as exc:
            fail(f"corpus: {exc}")

    print("Preflight passed" if not problems else f"Preflight found {len(problems)} problem(s)")
    return 0 if not problems else 1


def _capture(app: AppConfig, args) -> int:
    cameras = args.camera or ([app.registration.template_camera_id] if app.registration.template_camera_id else [])
    if not cameras:
        raise ConfigError("Pass --camera or set LOADGEN_TEMPLATE_CAMERA_ID")
    result = capture_corpus(app.kafka, app.frames, args.name, args.duration, cameras, args.max_frames)
    print(f"Captured {result.frames} frames into {result.directory} ({result.missing_images} images already gone)")
    return 0 if result.frames else 1


def _corpus(app: AppConfig, args) -> int:
    from vision_loadgen.video import copy_hint, corpus_from_video

    def progress(done: int, expected: int) -> None:
        print(f"progress {done}/{expected or '?'} frames", flush=True)

    corpus = corpus_from_video(args.video, args.name, args.out or app.frames.corpus_dir, args.fps,
                               args.max_seconds, args.image_root or "", args.quality, progress)
    print(f"Wrote {corpus.frames} frames ({corpus.width}x{corpus.height}, {corpus.fps} fps from "
          f"{corpus.source_fps} fps) to {corpus.directory}")
    print(f"The workers must read it at {corpus.image_root}. Copy it there, e.g.:")
    print(f"  {copy_hint(corpus, args.copy_target or os.environ.get('LOADGEN_CORPUS_COPY_TARGET', ''))}")
    print(f"Then run with --set source.mode=corpus --set source.corpus_name={corpus.name}")
    return 0


def _video_camera(app: AppConfig, args) -> int:
    from vision_loadgen import video_camera

    if args.video_command == "add":
        workers = [app.worker_name(name) for name in args.worker] or app.configured_workers()
        result = video_camera.add_video_camera(app, args.name, args.like, workers, args.zones_from, args.corpus)
        print(json.dumps(result, indent=2))
        print(f"Use it with --template-camera {result['id']}")
        return 0
    if args.video_command == "list":
        print(json.dumps(video_camera.list_video_cameras(app), indent=2))
        return 0
    report = video_camera.remove_video_camera(app, args.camera_id)
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


def _run(app: AppConfig, args) -> int:
    scenario = _scenario(app, args)
    options = RunOptions(no_guard=args.no_guard, keep_events=args.keep_events, allow_production=args.allow_production)
    summary = Runner(app, scenario, options).run()
    _print_summary(summary)
    failed = summary.get("error") or summary.get("aborted") or summary["teardown"]["errors"]
    return 1 if failed else 0


def _print_summary(summary: dict) -> None:
    print(f"\nRun {summary['run_id']} finished; results in {summary.get('results_dir')}")
    if summary.get("error"):
        print(f"  error: {summary['error']}")
    if summary.get("aborted"):
        print(f"  aborted by guard: {summary['aborted']}")
    if summary.get("capacity_note"):
        print(f"  {summary['capacity_note']}")
    for name, result in (summary.get("throughput") or {}).items():
        print(f"  {name}: max sustained {result['max_sustained_cameras']} cameras "
              f"({result['max_sustained_frames_per_s']} frames/s), saturated={result['saturated']}")
    for stage in summary.get("stages", []):
        cells = ", ".join(
            f"{name} p95={_num(entry.get('staleness_p95'))}s gpu={entry.get('gpu_workers_expected') or '?'}"
            + (f" [{entry['verdict']['reason']}]" if "verdict" in entry else "")
            for name, entry in stage["workers"].items()
        )
        print(f"  {stage['name']:<22} {cells}")
    teardown = summary.get("teardown") or {}
    for worker, keys in (teardown.get("manual_review") or {}).items():
        if keys:
            print(f"  MANUAL REVIEW {worker}: {len(keys)} rows also touched by real cameras: {', '.join(keys)}")
    if teardown.get("errors"):
        print("  teardown errors (run `cleanup --run-id` to retry):")
        for error in teardown["errors"]:
            print(f"    - {error}")


def _num(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def _cleanup(app: AppConfig, args) -> int:
    results_dir = Path(app.output.results_dir)
    if args.run_id:
        registry = Registry.load(results_dir / args.run_id / REGISTRY_NAME)
        report = Registrar(app, list(registry.workers), registry).teardown(args.keep_events)
        print(json.dumps(report, indent=2, default=str))
        return 1 if report["errors"] else 0

    orphans = find_orphans(app)
    if not orphans:
        print("No leftover synthetic cameras or regions")
        return 0
    for run_id, found in orphans.items():
        has_registry = (results_dir / run_id / REGISTRY_NAME).is_file()
        print(f"{run_id}: region={found['region_id'] or '-'} cameras={len(found['camera_ids'])} "
              f"assignments={len(found['assignments'])} registry={'yes' if has_registry else 'no'}")
    if not args.yes:
        print("Re-run with --yes to remove them")
        return 0
    exit_code = 0
    for run_id, found in orphans.items():
        path = results_dir / run_id / REGISTRY_NAME
        if path.is_file():
            registry = Registry.load(path)
            registrar = Registrar(app, list(registry.workers), registry)
        else:
            registry = registry_for_orphan(app, str(results_dir), run_id, found)
            registrar = Registrar(app, app.configured_workers(), registry)
        report = registrar.teardown(args.keep_events)
        print(f"{run_id}: {'clean' if not report['errors'] else report['errors']}")
        exit_code = exit_code or (1 if report["errors"] else 0)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
