# Load Generator — Design

Date: 2026-09-29
Status: approved in conversation; implemented as the `vision_loadgen` package (`vision-loadgen` repository)

## Purpose

Find out how much camera load the Kafka-fed GPU workers can take, by replaying real
frames as many synthetic "stacked" copies of one camera. Three questions:

1. **Throughput** — how many cameras (at a given FPS) can each worker sustain before it
   falls behind?
2. **Latency** — how far behind real time is processing at fixed load levels?
3. **Soak** — does a fixed load stay stable for hours (no drift, leaks, restarts)?

Target workers (v1): `crowd-monitoring`, `sentiment-analysis` (emotion), `ai-attendance`.
Any worker built on `vision_shared`'s `KafkaFramePipeline` can be added by config alone.

## How the pipeline works today (what the design relies on)

- Frames reach workers only through Kafka topic `exelixi.frames.raw` (`kafka:7010`).
  Messages are JSON **pointers**: `camera_id`, `timestamp`, `image_path`, `frame_date`,
  `frame_hour`. The JPEG lives on a shared volume; the worker reads it with `cv2.imread`.
- Every worker consumes the whole topic in its own consumer group
  (`{FUNCTION_NAME}_consumer_group_{unix_ts}`, `auto_offset_reset=latest`) and drops
  messages whose `camera_id` is not in its enabled set (`kafka_pipeline.py:119-126`).
- The enabled set is two layers:
  1. **Platform** — `vision_main.get_cameras()`: cameras whose `camera_region` is assigned
     to the worker's `server_function` in `function_camera_regions`.
  2. **Module** — the module's enabled settings row: `crowd_gathering_settings.selected_cameras`,
     `emotion_settings.selected_cameras`, `frs_settings.check_in_cameras`.
- `POST /api/v1/worker/sync` (service JWT) reloads that set without a process restart,
  but it restarts the Kafka pipeline, which re-subscribes at the *latest* offset.
- `GET /api/v1/worker/status` returns `active_cameras` and `processed_buffer_info`
  (`{source_id: {timestamp, has_frame, ...}}` — the latest processed frame per camera).
- Under overload the worker **drops** frames (`gpu_engine.submit` returns `None`) rather
  than queueing them, so Kafka lag alone does not reveal saturation.

## Architecture

```
            ┌──────────── live tap (no consumer group) ◄──── exelixi.frames.raw ◄── frame router
            │                     or                                   ▲
 source ────┤                                                          │ N synthetic cameras
            └──────────── corpus (captured JPEGs + manifest)           │ (same image_path,
                                   │                                   │  fresh timestamp)
                                   ▼                                   │
                  fan-out ──► pacer (fps × active cams) ──► producer ──┘
                                                                        │
             crowd / sentiment / attendance workers ◄───────────────────┘
                     │  /worker/status  +  Kafka consumer lag
                     ▼
             sampler thread ──► guard (real cameras)  +  stage verdicts  ──► CSV / JSON / console
```

Units (`vision_loadgen/`; `docker/`, `docs/` and `tests/` sit alongside at the repository root
and are not installed):

| Module | Responsibility |
|---|---|
| `config.py` | App, worker and scenario models; YAML/JSON override files with `${ENV}` expansion |
| `presets.py` | Built-in worker definitions and scenarios, so no config file is required |
| `environment.py` | Builds the config from the worker's own `Settings`, env vars and `vision_shared/.env` (when installed beside it) |
| `capacity.py` | GPU model-copy math mirroring `GpuInferenceEngine`, capacity cap, `nvidia-smi` probe |
| `media.py` | Deletes synthetic events' image/video files inside `EVENTS_DIR` |
| `frames.py` | Build synthetic messages from a source frame |
| `pacer.py` | Per-camera send schedule with phase spreading and no catch-up bursts |
| `kafka_io.py` | Producer with ack/error counters, tap consumer, consumer-lag reader |
| `sources.py` | Live-tap source, corpus capture and replay |
| `db.py` | Postgres helpers: column introspection, row cloning, JSON-array edits |
| `workers.py` | Service token, worker endpoint resolution, sync/status client |
| `registry.py` | Write-ahead run manifest (everything created, so cleanup can undo it) |
| `registrar.py` | Setup and teardown of synthetic cameras across both DB layers |
| `analysis.py` | Pure logic: percentiles, lag slope, stage verdicts, real-camera guard |
| `scenarios.py` | Turns a scenario into stages |
| `metrics.py` | Sampler thread and recorder (console, `timeseries.csv`, `summary.json`) |
| `runner.py` | Orchestrates one run end to end |
| `cli.py` | `check`, `capture`, `run`, `cleanup` |

## Frame sources

- **Live tap** reads the topic with no consumer group (nothing to clean up, no effect on
  other groups), keeps the latest frame per real source camera, and ignores anything
  tagged `loadgen` or whose `camera_id` is synthetic — so it never re-reads its own output.
  `frame_date`/`frame_hour` pass through from the real producer.
- **Corpus**: `capture` taps for N seconds, copies each JPEG into
  `<corpus_dir>/<name>/` on the shared volume and writes `manifest.jsonl`. Replay loops the
  manifest; each synthetic camera starts at a different offset. `frame_date`/`frame_hour`
  are recomputed from the new timestamp. Repeatable across runs and immune to the frame
  router deleting old JPEGs.

Synthetic messages carry only the five known fields plus `"loadgen": <run_id>`.

## Registration lifecycle

Setup (every created id is written to `results/<run_id>/registry.json` as it is created):

1. Clone the **template camera's** region as `loadtest-<run_id>`.
2. For each target worker, resolve its single active `server_function` (optionally pinned by
   `server_ip`) and clone one of its `function_camera_regions` rows onto the new region.
3. Clone the template camera N times into the region: name `loadtest-<run_id>-NNN`,
   `ip=127.0.0.1`, unique port, `is_active=true`.
4. Clone the template camera's per-camera config rows for each synthetic camera
   (`crowd_gathering_camera_lines`, `emotion_camera_zones`) so processing is realistic.
5. Snapshot rows that need restoring later (`frs_attendance`, see below).
6. Per stage, set the first enabled settings row of each module to hold exactly that stage's
   synthetic ids, `POST /worker/sync` each worker and wait until `/worker/status` lists them
   and reports `running`. If a module has no enabled row the run stops: the tool never enables
   a module itself.

Cameras are cloned once for the largest stage. With `sizing: per_stage` (default) each stage
enables only its own count and re-syncs, because the GPU engine sizes its model copies from
the enabled cameras at sync time (next section). Each re-sync restarts the worker's Kafka
pipeline with a new consumer group at the latest offset: publishing pauses during it, the
guard ignores samples until `settle_s` after it, and stage statistics start there.
`sizing: fixed` enables the largest count once, as the first version did.

## GPU model copies

`GpuInferenceEngine` starts `ceil(cameras / CAMERAS_PER_WORKER)` model copies (plus one API
model when the module has one), each with one executor thread, and gives camera *i* of the
sorted camera ids to copy *i // CAMERAS_PER_WORKER*. If copies × `VRAM_PER_WORKER_MB` exceeds
the VRAM that was free when the worker process started, `start()` raises: the sync fails and
the worker processes **nothing**, real cameras included, until the next successful sync.

- Registering the largest count up front (the first version) would give every stage the model
  copies of the largest stage, so small stages would look better than production, and a
  too-large count would fail at setup.
- The tool reads `CAMERAS_PER_WORKER` and `VRAM_PER_WORKER_MB` from the worker's settings when
  run beside it (modules may override the shared defaults), else from the environment.
- The VRAM budget is `gpu_free_vram_mb` from a config file or, beside the worker with the GPU
  visible, `nvidia-smi` free memory plus what the worker's current copies hold (an estimate).
  With a budget, stages that do not fit are skipped and a throughput ramp ends at the cap.
  Without one nothing is capped; a failing sync stops the run and teardown re-syncs.
- Each stage reports `gpu_workers_expected` and synthetic staleness per model copy, computed from
  `/worker/status` `active_cameras` with the engine's own mapping, so one overloaded copy is
  visible even when the average is fine.

Rows are cloned with column introspection (`pg_attribute`), so unseen NOT NULL columns in
vision-main tables keep the template's values.

Teardown (normal exit, Ctrl+C, SIGTERM, or `cleanup`), each step best effort and recorded:

1. Remove the ids from the settings rows, then sync workers.
2. Delete the cloned per-camera config rows.
3. Plan the `frs_attendance` restore (below).
4. Delete events created by synthetic cameras (default; `--keep-events` skips this):
   `crowd_gathering_events`, `emotion_events`, `emotion_rollups`, `frs_recognition_events`,
   then the files their `image_path` / `video_path` columns point at (relative to `EVENTS_DIR`,
   deleted only if they resolve inside it) and any folder under `EVENTS_DIR/<function_key>`
   named exactly after a synthetic camera id (workers write `.../<camera_id>/<date>/`).
5. Apply the attendance restore.
6. Delete the `function_camera_regions` rows, the cameras and the region. If a foreign key
   blocks a hard delete, fall back to soft delete (`deleted_at = now()`).

`cleanup --run-id <id>` replays teardown from a registry. `cleanup --orphans` finds
`loadtest-*` cameras and regions in vision-main (e.g. after a crash between commit and
registry write) and removes them with `--yes`.

### AI attendance restore

Frames are real, so a synthetic check-in camera can recognise a real employee and write
their `frs_attendance` row, which has no camera column. Before the run the tool snapshots
recent `frs_attendance` rows. At teardown it finds attendance rows referenced by
recognition events from synthetic cameras:

- touched **only** by synthetic cameras → restored to the snapshot (or deleted if new);
- also touched by a real camera during the run → left alone and listed in
  `summary.json` under `manual_review`.

## Scenarios, metrics and verdicts

Per sample (default every 5 s), for each worker:

- **Synthetic staleness** per active synthetic camera = now − latest processed frame
  timestamp (or time since the camera was activated if never processed). Because the
  generator sets the frame timestamps, this is end-to-end latency with no clock skew.
- **Real staleness** — the same for the worker's real cameras. Their timestamps come from the
  ingestion side, so "now" is shifted by the worker's clock offset, estimated from the HTTP
  `Date` headers of its responses (1 s resolution, narrowed by intersecting successive
  responses). The same correction applies to the template pick; `summary.json` records the
  offset. The guard compares against its own baseline, so an offset cancels out there anyway.
- **Consumer lag** of the worker's newest consumer group, and its growth rate.
- Status reachability and `running`.

| Scenario | Stages | Verdict |
|---|---|---|
| throughput | `start → max` cameras in `step`s of `step_duration_s` | A worker "keeps up" at a stage if, over the last `window_s`, synthetic staleness p95 ≤ `max_staleness_s` and lag growth ≤ `max_lag_growth_per_s`. Reports the highest camera count each worker kept up with; stops when every worker has failed. |
| latency | fixed `levels`, each `level_duration_s` | staleness p50/p95/max and lag per level |
| soak | fixed `cameras` for `duration_s` | hourly staleness p95, overall lag slope, status failures, drift between first and last hour |

Outputs in `results/<run_id>/`: `registry.json`, `timeseries.csv`, `summary.json`, plus the
attendance snapshot.

### Video corpus

Without a streaming camera, a recorded video stands in. `corpus from-video` samples it with
OpenCV into JPEGs plus a manifest of paths relative to `image_root` (recorded in `corpus.json`),
the folder the workers read them from once copied there (default `/app/events/loadgen_corpus/<name>`,
the shared events bind mount; the frames tmpfs deletes old files). `CorpusSource` joins the paths
to that root and checks the files only when this machine can see it. The frames still travel the
production route from Kafka on: pointer messages on `exelixi.frames.raw`, consumed by each
worker's `KafkaFramePipeline`. The modules' video-upload API was not usable: it decodes uploads
inside the worker and never publishes to Kafka. `video-camera add` registers the video's
perspective as an inactive `loadgen-video-*` camera cloned from an existing one (region,
timezone; stream blanked), used as the template so each run's synthetic cameras are cloned from
it; it is never enabled itself and survives `cleanup --orphans`.

`vision-loadgen ui` is a stdlib-only local page over the same CLI: it starts allowlisted commands
as child processes (stopping a run through `LOADGEN_STOP_FILE`, since signals cannot reach a
console-less child on Windows), polls the run's metrics endpoint for live charts, follows worker
containers' `docker logs` over ssh (the workers have no log endpoint), and reads past
runs from the results folder.

### Observability

With `--metrics-port` set (the web UI always sets it), `run` serves `/metrics` in the Prometheus
text format on 127.0.0.1 (`vision_loadgen/exporter.py`, stdlib only so worker images need nothing
new). The `Recorder` publishes each sample batch (the same p50/p95/max staleness, lag, real-camera
excess and publish rate that go to `timeseries.csv`, plus staleness p95 per GPU model copy and
the worker's own stage times and processed fps) and the runner publishes run info, phase, current
stage, thresholds, per-stage verdicts, the abort reason, the throughput result and clock offsets.
Every series carries `run_id`. After a run the endpoint lingers (`metrics_linger_s`, 15 s) so the
UI reads the final verdicts. The CSV/JSON outputs remain the record of a run. (A Prometheus +
Grafana stack was tried and dropped: the UI covers it.)

Worker internals: a preset may name the worker's own metrics endpoint (`metrics.path`), a per-stage
time histogram and a per-camera frames counter. The sampler reads it with each status sample and
keeps the previous cumulative values, so each sample carries stage p50/p95 and processed fps for
the interval since the previous one (a counter reset skips an interval). `metrics.gpu_stage_limits`
turns a slow stage into a verdict hint, not a failure: staging's emotion worker once ran SCRFD on the
CPU because onnxruntime could not load its CUDA provider, which staleness alone did not explain.

## Safety

- **Real-camera guard** (on by default): the tool records each worker's real-camera
  staleness before publishing anything, then trips if real p95 exceeds the baseline by
  `max_real_staleness_increase_s` for `grace_s`. The guard either aborts the run or backs off
  by removing cameras.
- **Worker-health guard** (on by default, also with `--no-guard`): aborts when a worker's status
  endpoint has failed for `unreachable_s`, or its consumer backlog exceeds `max_backlog_s` of
  publishing and is still growing after `grace_s`. The real-camera guard skips failed samples
  and has nothing to compare in a corpus run, so without this a dying worker kept receiving
  load (staging: the sentiment worker at 5000% CPU and 150 GiB).
- `--no-guard` is accepted only for **throughput** scenarios in **staging**.
- Production runs require `--allow-production`, and the guard cannot be disabled there. An
  unset `LOADGEN_ENVIRONMENT` counts as production, since worker images run in both.
- Every re-sync restarts the worker's pipeline for all of its cameras, as adding a camera in
  the UI does; `sizing: per_stage` does that once per stage.
- Synthetic cameras use `ip=127.0.0.1` so nothing opens extra streams to real cameras.

## Deployment

The code is the `vision_loadgen` package in its own repository (`vision-loadgen`), versioned
and released independently of `vision_shared`. It never imports `vision_shared`; the only link is
reading `vision_shared/.env` defaults, located with `importlib.util.find_spec` (no import), when
`vision_shared` happens to be installed beside it.

The repository's `Dockerfile` builds one image, `ghcr.io/exelixi-ai/vision-loadgen:<version>`,
with two uses:

- `/dist/vision_loadgen`: the package alone (`pip install --no-deps --target /dist`), at a path
  that does not depend on the image's Python version. Worker Dockerfiles copy it next to
  `vision_shared`, the same way they copy `vision_shared` from the base image:

  ```dockerfile
  ARG VISION_LOADGEN_IMAGE=ghcr.io/exelixi-ai/vision-loadgen:0.1.0
  FROM ${VISION_LOADGEN_IMAGE} AS vision_loadgen
  ...
  COPY --from=vision_loadgen /dist/vision_loadgen       /opt/conda/lib/python3.11/site-packages/vision_loadgen
  ```

  Nothing is pip-installed into the worker: every library it needs (`kafka-python`, `pydantic`,
  `psycopg2`, `python-jose`, `python-dotenv`) is already in worker images, presets are built in,
  YAML files are optional (JSON always works) and HTTP uses `urllib`. Removing those lines drops
  it; bumping `VISION_LOADGEN_IMAGE` upgrades it without touching `VISION_BASE_IMAGE`.
- The standalone runner: the same package installed with its dependencies on `python:3.10-slim`,
  `ENTRYPOINT python -m vision_loadgen`.

Two ways to run it:

- **Beside a worker**: `docker compose run --rm --no-deps <worker-service> python -m
  vision_loadgen ...` starts a one-off container from the worker's image with its env,
  volumes and network, without touching the running worker. The tool imports the worker's
  `app.core.config:settings` (configurable) and takes its database URLs, Kafka, JWT secret,
  `FUNCTION_NAME`, `PUBLIC_BASE_IP`, GPU sizing, `EVENTS_DIR` and `TIMEZONE`. Frames and events
  are already mounted at the worker's paths.
- **Standalone** (several workers at once): `docker/docker-compose.yml` builds the root
  `Dockerfile`, joins the external network, mounts frames and events at the workers' paths,
  mounts `./results` and `./config`, and gives teardown a 3-minute stop grace period.

`kafka-python`: the tool uses only APIs present in 2.0.2 (the workers' pin) and is tested on
2.0.2 (Python 3.11) and 2.3.2 (Python 3.12). 3.x removed `KafkaAdminClient.list_consumer_groups`,
so `pyproject.toml` caps it below 3; with 3.x the tool runs without the lag metric.

## Verification

- Unit tests cover config sources and precedence, presets, message building, pacing, staleness,
  verdicts, the guard (including the re-sync hold), GPU capacity and stage capping, per-copy
  grouping, event file deletion, SQL builders, the registry and the attendance restore plan.
- `LOADGEN_IT=1 pytest` runs Postgres and Kafka in Docker with fake crowd and attendance workers
  that follow the real contract (including refusing a sync that exceeds GPU capacity), and
  checks a full per-stage run (re-sync per stage, GPU cap, sampling, verdicts, event rows and
  files, attendance restore, teardown), `check`, `capture` + corpus replay, and orphan cleanup
  after a simulated crash.
- The `compat` service runs the same suite on Python 3.11 with the crowd worker's pins
  (kafka-python 2.0.2, pydantic 2.5.3, psycopg2-binary 2.9.9, python-dotenv 1.0.0).
- Not yet run against the real staging workers.

## Open items to verify (flagged)

1. **Frames volume name and mount path** — configurable via `FRAMES_SOURCE` /
   `FRAMES_MOUNT_PATH`; must match the workers' mount exactly.
2. **Docker network name** — `DOCKER_NETWORK`.
3. **Timezone for corpus replay** — `frame_hour` is `%H` (confirmed: crowd falls back to
   `strftime("%H")`); the timezone is `frames.timezone` and should match the workers' `TIMEZONE`.
   Live tap passes the real values through.
4. **Consumer-group prefix for AI attendance** — `{function_key}_consumer_group_`. Confirmed for
   crowd and sentiment analysis, which force `FUNCTION_NAME` to their function key; unverified
   for attendance. Override per worker if different.
5. **`processed_buffer_info.timestamp`** — the Kafka message timestamp: confirmed for crowd
   (`KafkaFramePipeline` passes `payload["timestamp"]` through to `FrameStore.push_frame`), and
   assumed for the other workers built on the same pipeline.
6. **AI attendance worker** was not available to review. Assumed to expose the same
   `/api/v1/worker/sync` and `/status` and to process frames through `KafkaFramePipeline`.
   Its processing hours may restrict when load is applied.
7. **Module database URLs** — each module has its own database on the shared Postgres server
   (`crowd_gathering_db`, `emotion_detection_db`, `frs_db`, per vision-module-backend
   `MODULE_DATABASES`), derived from `POSTGRES_URL` the way the crowd worker's `Settings` does;
   `CROWD_/SENTIMENT_/ATTENDANCE_DATABASE_URL` override. Beside a worker, its own
   `DATABASE_URL` is used.
8. **First enabled settings row** — workers use the first enabled row; the tool orders by
   `id` (configurable `order_by`). Emotion's ordering not confirmed.
9. **vision-main event copies** — workers also `POST /api/v1/worker/events` to vision-main.
   That table was not available to review, so it is not purged by default; add it under a
   worker's `purge` with `db: main` once known. Event files are deleted when `EVENTS_DIR` is
   mounted; the attendance file columns (`event_image_path`, `event_video_path`) and folder layout
   are assumed from the vision-module-backend model.
10. **Frame router behaviour** — synthetic cameras are active in `cameras`; if the frame
    router polls every active camera it will log connection failures to `127.0.0.1`.
11. **`emotion_face_identities`** are not purged (they may be real people).
12. **Dropped frames** are only logged by workers, not exposed; staleness and lag are the
    saturation signals.
13. **Emotion processing window** — sentiment analysis skips frames outside its active hours
    and working days; `check` warns, but tests should run inside that window.
14. **Worker settings module** — assumed at `app.core.config:settings` (true for crowd);
    override with `--worker-settings` if a worker differs.
15. **GPU settings per module** — `api_worker` is `false` in every preset (confirmed for crowd,
    unverified for emotion and attendance); a module that overrides `CAMERAS_PER_WORKER` /
    `VRAM_PER_WORKER_MB` is only seen correctly when run beside it or configured.
16. **Worker images** — the tool is inside a worker image only after its Dockerfile gains the
    `vision_loadgen` stage and COPY (see Deployment) and the image is rebuilt; the
    `vision-loadgen` image must be pushed to the registry first.
17. **Results location beside a worker** — `$LOGS_DIR/loadgen` survives only if the worker's
    logs directory is a volume; otherwise mount one or set `LOADGEN_RESULTS_DIR`.
