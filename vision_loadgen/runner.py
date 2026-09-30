from __future__ import annotations

import json
import logging
import queue
import secrets
import signal
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

from vision_loadgen.analysis import (
    RealCameraGuard,
    StageVerdict,
    WorkerSample,
    max_sustained,
    soak_summary,
    stage_stats,
    stage_verdict,
)
from vision_loadgen.capacity import Capacity, plan_capacity, probe_free_mb
from vision_loadgen.config import AppConfig, ConfigError, GuardConfig, ScenarioConfig
from vision_loadgen.exporter import MetricsExporter
from vision_loadgen.frames import build_message
from vision_loadgen.kafka_io import FrameProducer, LagReader
from vision_loadgen.metrics import ActiveCameras, Recorder, Sampler
from vision_loadgen.pacer import Pacer
from vision_loadgen.registrar import Registrar
from vision_loadgen.registry import Registry
from vision_loadgen.scenarios import Stage, build_stages, cap_stages
from vision_loadgen.sources import CorpusSource, FrameSource, LiveTapSource

log = logging.getLogger(__name__)


@dataclass
class RunOptions:
    no_guard: bool = False
    keep_events: bool = False
    allow_production: bool = False


def check_gates(app: AppConfig, scenario: ScenarioConfig, options: RunOptions) -> None:
    if app.environment == "production" and not options.allow_production:
        raise ConfigError(
            "Refusing to run against production without --allow-production "
            "(set LOADGEN_ENVIRONMENT=staging or pass --environment staging on staging)"
        )
    if options.no_guard:
        if app.environment != "staging":
            raise ConfigError("--no-guard is only allowed in staging")
        if scenario.type != "throughput":
            raise ConfigError("--no-guard is only allowed for throughput scenarios")


def template_camera_id(app: AppConfig, scenario: ScenarioConfig) -> str:
    """The configured template camera; "" means pick the freshest real camera of the first worker."""
    return app.registration.template_camera_id or next(iter(scenario.source.source_camera_ids), "")


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def plan_capacities(app: AppConfig, workers: list[str], real_cameras: dict[str, int]) -> dict[str, Capacity]:
    """GPU capacity per worker. nvidia-smi is only trusted for the worker this process runs beside."""
    probe = probe_free_mb() if app.local_worker in workers else None
    return {
        name: plan_capacity(name, app.workers[name], real_cameras.get(name, 0),
                            probe if name == app.local_worker else None)
        for name in workers
    }


class Runner:
    def __init__(self, app: AppConfig, scenario: ScenarioConfig, options: RunOptions) -> None:
        check_gates(app, scenario, options)
        self.app = app
        self.scenario = scenario
        self.options = options
        self.template_id = template_camera_id(app, scenario)
        self.stop_event = threading.Event()
        self.abort_reason: Optional[str] = None
        self.samples: dict[str, list[WorkerSample]] = {name: [] for name in scenario.workers}
        self.skipped_no_source = 0
        self._registrar: Optional[Registrar] = None
        self._pending_count: Optional[int] = None
        self._settled_at = 0.0
        self.exporter: Optional[MetricsExporter] = None

    def run(self) -> dict[str, Any]:
        registry = Registry.create(self.app.output.results_dir, new_run_id(), self.app.environment)
        registrar = Registrar(self.app, self.scenario.workers, registry)
        self._registrar = registrar
        self._install_signal_handlers()
        self._start_exporter(registry.run_id)
        self._phase("preparing")
        log.info("Run %s: %s (%s) on %s, sizing %s", registry.run_id, self.scenario.name, self.scenario.type,
                 ", ".join(self.scenario.workers), self.scenario.sizing)

        summary: dict[str, Any] = {
            "run_id": registry.run_id,
            "scenario": self.scenario.model_dump(),
            "environment": self.app.environment,
            "started_at": time.time(),
            "guard_enabled": self.scenario.guard.enabled and not self.options.no_guard,
            "results_dir": str(registry.directory),
            "stages": [],
        }
        source: Optional[FrameSource] = None
        producer: Optional[FrameProducer] = None
        lag_reader: Optional[LagReader] = None
        sampler: Optional[Sampler] = None
        recorder: Optional[Recorder] = None
        try:
            registrar.prepare(self.template_id)
            summary["template_camera_id"] = registry.template_camera_id
            self._export_run_info(registry.template_camera_id, summary["started_at"])
            capacities = plan_capacities(self.app, self.scenario.workers, registrar.real_camera_counts())
            summary["capacity"] = {name: capacity.as_dict() for name, capacity in capacities.items()}
            stages, cap_note = cap_stages(build_stages(self.scenario), capacities, self.scenario)
            if cap_note:
                log.warning("%s", cap_note)
                summary["capacity_note"] = cap_note
            summary["planned_stages"] = [asdict(stage) for stage in stages]

            camera_ids = registrar.create(max(stage.cameras for stage in stages))
            synthetic_ids = set(camera_ids)
            source = self._build_source(synthetic_ids, registry.template_camera_id)
            source.start()
            if not source.wait_for_first_frame(self.scenario.source.first_frame_timeout_s):
                raise RuntimeError("No frames arrived from the source camera(s); is the frame router publishing?")

            producer = FrameProducer(self.app.kafka)
            try:
                lag_reader = LagReader(self.app.kafka)
            except Exception as exc:
                log.warning("Consumer lag unavailable: %s", exc)
            active = ActiveCameras(camera_ids)
            sampler = Sampler(
                clients=registrar.clients,
                group_prefixes={name: self.app.workers[name].group_prefix for name in self.scenario.workers},
                lag_reader=lag_reader,
                active=active,
                synthetic_ids=synthetic_ids,
                interval_s=self.app.output.sample_interval_s,
                cameras_per_worker={name: self.app.workers[name].cameras_per_worker for name in self.scenario.workers},
            )
            # Baseline before anything is enabled: the workers' real cameras as they normally run.
            self._phase("baseline")
            guard = self._calibrated_guard(sampler)
            summary["guard_baseline_real_cameras"] = {worker: len(cams) for worker, cams in guard.baseline.items()}

            recorder = Recorder(registry.directory, self.scenario.workers, self.exporter)
            registry.run_started_at = time.time()
            registry.set_status("running")
            self._phase("running")
            sampler.start()
            pacer = Pacer(self.scenario.fps_per_camera)
            if self.scenario.sizing == "fixed":
                self._resize(max(stage.cameras for stage in stages), pacer, active, guard, publish=False)
            summary["stages"] = self._run_stages(stages, capacities, camera_ids, registry.run_id, source, producer,
                                                 pacer, active, sampler, recorder, guard)
        except Exception as exc:
            log.exception("Run failed")
            summary["error"] = str(exc)
        finally:
            if sampler:
                sampler.stop()
            if source:
                source.stop()
            if producer:
                summary["producer"] = asdict(producer.stats())
                producer.close()
            if lag_reader:
                lag_reader.close()
            if recorder:
                recorder.close()
            log.info("Tearing down run %s", registry.run_id)
            self._phase("teardown")
            self._metric("loadgen_active_cameras", 0)
            summary["teardown"] = registrar.teardown(self.options.keep_events)

        summary["ended_at"] = time.time()
        summary["clock_offset"] = {name: client.clock.as_dict() for name, client in registrar.clients.items()}
        summary["aborted"] = self.abort_reason
        summary["skipped_no_source_frame"] = self.skipped_no_source
        self._add_results(summary, registry)
        with (registry.directory / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        self._export_clock_offsets()
        self._metric("loadgen_run_end_time_seconds", summary["ended_at"])
        self._phase("done")
        if self.exporter is not None:
            self.exporter.stop(self.app.output.metrics_linger_s)
        return summary

    def _start_exporter(self, run_id: str) -> None:
        output = self.app.output
        if output.metrics_port is None:
            return
        exporter = MetricsExporter(run_id, output.metrics_addr, output.metrics_port)
        if exporter.start():
            self.exporter = exporter

    def _metric(self, name: str, value: Optional[float], **labels: Any) -> None:
        if self.exporter is not None:
            self.exporter.set(name, value, **labels)

    def _phase(self, phase: str) -> None:
        if self.exporter is not None:
            self.exporter.clear("loadgen_phase")
            self.exporter.set("loadgen_phase", 1, phase=phase)

    def _export_clock_offsets(self) -> None:
        if self.exporter is None or self._registrar is None:
            return
        for name, client in self._registrar.clients.items():
            if client.clock.known:
                self.exporter.set("loadgen_clock_offset_seconds", client.clock.seconds, worker=name)

    def _export_run_info(self, template_camera_id: str, started_at: float) -> None:
        scenario = self.scenario
        labels = {"scenario": scenario.name, "scenario_type": scenario.type, "environment": self.app.environment}
        self._metric("loadgen_run_info", 1, template_camera=template_camera_id or "", **labels)
        self._metric("loadgen_run_start_time_seconds", started_at, **labels)
        self._metric("loadgen_threshold_seconds", scenario.saturation.max_staleness_s, kind="max_staleness")
        self._metric("loadgen_threshold_seconds", scenario.guard.max_real_staleness_increase_s, kind="guard")

    def _build_source(self, synthetic_ids: set[str], template_id: str) -> FrameSource:
        cfg = self.scenario.source
        if cfg.mode == "corpus":
            return CorpusSource(self.app.frames, cfg.corpus_name)
        return LiveTapSource(self.app.kafka, cfg.source_camera_ids or [template_id], synthetic_ids)

    def _calibrated_guard(self, sampler: Sampler) -> RealCameraGuard:
        cfg = self.scenario.guard
        if self.options.no_guard:
            cfg = GuardConfig(**{**cfg.model_dump(), "enabled": False})
        guard = RealCameraGuard(cfg)
        baseline: list[WorkerSample] = []
        for index in range(max(1, cfg.baseline_samples)):
            if index:
                time.sleep(self.app.output.sample_interval_s)
            baseline.extend(sample for sample in sampler.sample_once() if sample.ok)
        guard.calibrate(baseline)
        for worker in self.scenario.workers:
            if cfg.enabled and not guard.baseline.get(worker):
                log.warning("%s has no live real cameras; the guard cannot protect it", worker)
        return guard

    def _resize(self, count: int, pacer: Pacer, active: ActiveCameras, guard: RealCameraGuard,
                publish: bool = True) -> None:
        """Enable exactly `count` synthetic cameras and re-sync; publishing pauses during the restart."""
        log.info("Enabling %d synthetic cameras and re-syncing the workers", count)
        pacer.set_active(0, time.monotonic())
        self._registrar.activate(count)
        now = time.time()
        active.set_count(count, now, restart=True)
        self._settled_at = now + self.scenario.settle_s
        guard.hold(self._settled_at)
        if publish:
            pacer.set_active(count, time.monotonic())

    def _run_stages(self, stages: list[Stage], capacities: dict[str, Capacity], camera_ids: list[str], run_id: str,
                    source: FrameSource, producer: FrameProducer, pacer: Pacer, active: ActiveCameras,
                    sampler: Sampler, recorder: Recorder, guard: RealCameraGuard) -> list[dict]:
        results: list[dict] = []
        failed: set[str] = set()
        for number, stage in enumerate(stages, start=1):
            if self.stop_event.is_set():
                break
            log.info("Stage: %s for %.0fs", stage.name, stage.duration_s)
            stage_index = f"{number:02d}"
            if self.exporter is not None:
                self.exporter.clear("loadgen_stage_info")
            self._metric("loadgen_stage_info", 1, stage=stage.name, stage_index=stage_index)
            self._metric("loadgen_stage_index", number)
            self._metric("loadgen_planned_cameras", stage.cameras)
            if self.scenario.sizing == "per_stage":
                self._resize(stage.cameras, pacer, active, guard)
            else:
                pacer.set_active(stage.cameras, time.monotonic())
                active.set_count(stage.cameras, time.time())
            start_index = {name: len(samples) for name, samples in self.samples.items()}
            stage_started = time.time()
            self._publish_until(stage, camera_ids, run_id, source, producer, pacer, active, sampler, recorder, guard)
            stage_ended = time.time()

            record: dict[str, Any] = {
                "name": stage.name, "cameras": active.count, "planned_cameras": stage.cameras,
                "duration_s": round(stage_ended - stage_started, 1), "workers": {},
            }
            measured_from = max(stage_started, self._settled_at)
            completed = not self.stop_event.is_set()
            for name, samples in self.samples.items():
                stage_samples = [sample for sample in samples[start_index[name]:] if sample.t >= measured_from]
                entry: dict[str, Any] = stage_stats(stage_samples)
                entry["gpu_workers_expected"] = capacities[name].gpu_workers(active.count)
                self._metric("loadgen_gpu_workers_expected", entry["gpu_workers_expected"], worker=name)
                if stage.evaluate and completed:
                    window_start = max(measured_from, stage_ended - self.scenario.saturation.window_s)
                    verdict = stage_verdict(stage_samples, window_start, self.scenario.saturation)
                    entry["verdict"] = asdict(verdict)
                    labels = {"worker": name, "stage": stage.name, "stage_index": stage_index}
                    self._metric("loadgen_stage_kept_up", 1 if verdict.kept_up else 0, **labels)
                    self._metric("loadgen_stage_staleness_p95_seconds", verdict.staleness_p95, **labels)
                    if not verdict.kept_up:
                        failed.add(name)
                record["workers"][name] = entry
            results.append(record)
            if self.scenario.type == "throughput" and failed >= set(self.scenario.workers):
                log.info("Every worker has fallen behind; stopping the ramp")
                break
        return results

    def _publish_until(self, stage: Stage, camera_ids: list[str], run_id: str, source: FrameSource,
                       producer: FrameProducer, pacer: Pacer, active: ActiveCameras, sampler: Sampler,
                       recorder: Recorder, guard: RealCameraGuard) -> None:
        end = time.monotonic() + stage.duration_s
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now >= end:
                return
            for index in pacer.due(now):
                frame = source.frame_for(index)
                if frame is None:
                    self.skipped_no_source += 1
                    continue
                producer.send(build_message(frame, camera_ids[index], time.time(), run_id,
                                            self.app.frames, source.passthrough_dates))
            self._drain_samples(stage, pacer, active, sampler, recorder, producer, guard)
            if self._pending_count is not None:
                count, self._pending_count = self._pending_count, None
                self._resize(count, pacer, active, guard)
            next_due = pacer.next_due()
            wake = min(end, next_due if next_due is not None else end, time.monotonic() + 0.05)
            time.sleep(max(0.0, wake - time.monotonic()))

    def _drain_samples(self, stage: Stage, pacer: Pacer, active: ActiveCameras, sampler: Sampler,
                       recorder: Recorder, producer: FrameProducer, guard: RealCameraGuard) -> None:
        while True:
            try:
                batch = sampler.batches.get_nowait()
            except queue.Empty:
                return
            for sample in batch:
                self.samples[sample.worker].append(sample)
            recorder.record(stage.name, active.count, producer.stats(), batch,
                            {sample.worker: guard.excess(sample) for sample in batch if sample.ok})
            self._export_clock_offsets()
            for sample in batch:
                reason = guard.check(sample)
                if reason:
                    self._on_guard_trip(reason, sample.worker, pacer, active, guard)

    def _on_guard_trip(self, reason: str, worker: str, pacer: Pacer, active: ActiveCameras,
                       guard: RealCameraGuard) -> None:
        cfg = self.scenario.guard
        remaining = active.count - cfg.backoff_step
        if cfg.action == "backoff" and remaining > 0:
            log.warning("Guard: %s; backing off to %d cameras", reason, remaining)
            guard.reset(worker)
            if self.scenario.sizing == "per_stage":
                self._pending_count = remaining
            else:
                pacer.set_active(remaining, time.monotonic())
                active.set_count(remaining, time.time())
            return
        log.error("Guard: %s; aborting", reason)
        self.abort_reason = reason
        self.stop_event.set()

    def _add_results(self, summary: dict[str, Any], registry: Registry) -> None:
        fps = self.scenario.fps_per_camera
        if self.scenario.type == "throughput":
            summary["throughput"] = {}
            for name in self.scenario.workers:
                results = [
                    (stage["cameras"], stage["workers"][name]["verdict"])
                    for stage in summary["stages"]
                    if "verdict" in stage["workers"].get(name, {})
                ]
                best = max_sustained([(cameras, StageVerdict(**verdict)) for cameras, verdict in results])
                capacity = (summary.get("capacity") or {}).get(name) or {}
                summary["throughput"][name] = {
                    "max_sustained_cameras": best,
                    "max_sustained_frames_per_s": best * fps if best else None,
                    "saturated": any(not verdict["kept_up"] for _, verdict in results),
                    "gpu_capacity_cap": capacity.get("max_synthetic_cameras"),
                }
                self._metric("loadgen_max_sustained_cameras", best or 0, worker=name)
        if self.scenario.type == "soak":
            start = registry.run_started_at or summary["started_at"]
            summary["soak"] = {name: soak_summary(samples, start) for name, samples in self.samples.items()}

    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            log.warning("Signal %s received; stopping and tearing down", signum)
            self.stop_event.set()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
