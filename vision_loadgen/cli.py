from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from vision_loadgen import db
from vision_loadgen.config import AppConfig, ConfigError, load_scenario
from vision_loadgen.environment import DEFAULT_WORKER_SETTINGS, build_app_config
from vision_loadgen.kafka_io import LagReader
from vision_loadgen.registrar import Registrar, find_orphans, registry_for_orphan
from vision_loadgen.registry import REGISTRY_NAME, Registry
from vision_loadgen.runner import RunOptions, Runner, check_gates, plan_capacities, template_camera_id
from vision_loadgen.scenarios import build_stages, cap_stages
from vision_loadgen.sources import LiveTapSource, capture_corpus
from vision_loadgen.workers import WorkerClient, resolve_deployment

log = logging.getLogger("vision_loadgen")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vision_loadgen",
        description="Kafka frame load generator. Run it beside a worker (a one-off container of the "
                    "worker's service) to test that worker with its own settings, or standalone with env vars.",
    )
    parser.add_argument("--config", help="Optional YAML/JSON file merged over the built-in presets")
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

    run = commands.add_parser("run", help="Run a load test scenario")
    scenario_args(run)
    run.add_argument("--no-guard", action="store_true", help="Disable the real-camera guard (staging throughput only)")
    run.add_argument("--keep-events", action="store_true", help="Keep events created by synthetic cameras")
    run.add_argument("--allow-production", action="store_true")

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
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("kafka").setLevel(logging.WARNING)
    try:
        app = build_app_config(args.config, args.worker_settings, _overrides(args))
        if app.local_worker:
            log.info("Running beside the %s worker (%s)", app.local_worker, app.workers[app.local_worker].function_key)
        if args.command == "check":
            return _check(app, args)
        if args.command == "capture":
            return _capture(app, args)
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

    print("Preflight passed" if not problems else f"Preflight found {len(problems)} problem(s)")
    return 0 if not problems else 1


def _capture(app: AppConfig, args) -> int:
    cameras = args.camera or ([app.registration.template_camera_id] if app.registration.template_camera_id else [])
    if not cameras:
        raise ConfigError("Pass --camera or set LOADGEN_TEMPLATE_CAMERA_ID")
    result = capture_corpus(app.kafka, app.frames, args.name, args.duration, cameras, args.max_frames)
    print(f"Captured {result.frames} frames into {result.directory} ({result.missing_images} images already gone)")
    return 0 if result.frames else 1


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
