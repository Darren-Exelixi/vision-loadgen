# vision-loadgen

Stress-tests the Kafka-fed GPU workers by publishing one real camera's frames as N synthetic
"stacked" cameras, then measuring how far behind each worker falls. It is its own package
(`vision_loadgen`) and image, versioned separately from `vision_shared`, and is added to a worker
image next to `vision_shared` when that image is built, so any worker can load-test itself.
Design, assumptions and open items: [docs/load-generator-design.md](docs/load-generator-design.md).

Layout: `vision_loadgen/` is the package. `Dockerfile` builds the image; `docker/` (compose file,
`.env.example`, example `--config` file), `monitoring/` (Prometheus + Grafana), `docs/` and
`tests/` are never installed.

Presets exist for crowd monitoring, sentiment analysis (emotion) and AI attendance. Any worker
built on `vision_shared`'s `KafkaFramePipeline` can be added with a `--config` file (see
`vision_loadgen/presets.py`).

## Image

```bash
docker build -t ghcr.io/exelixi-ai/vision-loadgen:0.1.0 .
docker push ghcr.io/exelixi-ai/vision-loadgen:0.1.0
```

One image, two uses: the standalone runner (`ENTRYPOINT python -m vision_loadgen`), and
`/dist/vision_loadgen`, the bare package, at a path that does not depend on Python version, for
worker images to copy.

## Adding it to a worker image

Next to the existing `vision_shared` lines in the worker's `Dockerfile` (crowd shown):

```dockerfile
ARG VISION_BASE_IMAGE=ghcr.io/exelixi-ai/vision-shared-base:1.0.6
ARG VISION_LOADGEN_IMAGE=ghcr.io/exelixi-ai/vision-loadgen:0.1.0
FROM ${VISION_BASE_IMAGE} AS vision_shared
FROM ${VISION_LOADGEN_IMAGE} AS vision_loadgen
...
COPY --from=vision_shared /usr/local/lib/python3.10/site-packages/vision_shared \
    /opt/conda/lib/python3.11/site-packages/vision_shared
COPY --from=vision_loadgen /dist/vision_loadgen \
    /opt/conda/lib/python3.11/site-packages/vision_loadgen
```

Nothing is pip-installed into the worker. Everything it needs (`kafka-python` 2.x, `pydantic`,
`psycopg2`, `python-jose`, `python-dotenv`) is already in the worker images, and it never imports
`vision_shared`, so any base image tag works. Bump `VISION_LOADGEN_IMAGE` to upgrade it; delete
the two lines to drop it.

## Two ways to run it

**Beside a worker (recommended for testing one worker).** Run a one-off container of the
worker's own compose service (the key under `services:`, not the function key). It gets the
worker's image, env, volumes and network, reads the worker's own `Settings`
(`app.core.config:settings`) and does not touch the running worker's process:

```bash
docker compose run --rm --no-deps -e LOADGEN_ENVIRONMENT=staging \
  -v "$PWD/loadgen_results:/app/loadgen_results" \
  <worker-service> python -m vision_loadgen check --scenario latency
docker compose run --rm --no-deps -e LOADGEN_ENVIRONMENT=staging \
  -v "$PWD/loadgen_results:/app/loadgen_results" \
  <worker-service> python -m vision_loadgen run --scenario latency
```

If the service defines an `entrypoint`, add `--entrypoint python` and drop `python` from the
command. Results go to `LOADGEN_RESULTS_DIR`, else `$LOGS_DIR/loadgen`, else `./loadgen_results`.
From the worker's settings it takes: the module and vision-main database URLs, Kafka, the JWT
secret, `FUNCTION_NAME` (which preset), `PUBLIC_BASE_IP` (which server's deployment),
`CAMERAS_PER_WORKER` / `VRAM_PER_WORKER_MB` (GPU sizing), `EVENTS_DIR` (event file cleanup) and
`TIMEZONE`, plus defaults from `vision_shared/.env`. When the container can see the GPU, it also
estimates free VRAM with `nvidia-smi`.

**Standalone (several workers in one run).** From `docker/`: copy `.env.example` to `.env`, then
`docker compose run --rm load-generator run --scenario latency --worker crowd --worker emotion`.
`config/example.yaml` there shows `--config` overrides (mounted at `/loadgen/config`).

**From your own machine**, when the staging Postgres, Kafka and worker ports are reachable by IP:
put the variables from `docker/.env.example` (IPs instead of container names) in `.env` at the
repository root, then

```powershell
.venv\Scripts\python -m vision_loadgen check --worker crowd --server-ip <worker's server IP> --scenario latency
.venv\Scripts\python -m vision_loadgen run   --worker crowd --server-ip <worker's server IP> --scenario latency
```

`./.env` is loaded automatically (or `--env-file` / `LOADGEN_ENV_FILE`); variables already set in
the shell win. Frames need not be reachable: only pointers are republished and the worker reads
the JPEGs from its own disk. Kafka must advertise an address this machine can resolve (else add
the broker's name to the hosts file). Without `EVENTS_DIR` locally, event rows are deleted but
their files stay on the server (the report says so). There is no `nvidia-smi` estimate from here,
so set `gpu_free_vram_mb` with `--config` to cap stages at GPU capacity.

## Benchmark from a video (no live cameras)

When no camera is streaming, replay a recorded video instead. The frames still go through Kafka:
every synthetic camera's frames are published to `exelixi.frames.raw` and the workers consume
them with their normal `KafkaFramePipeline`, reading each JPEG from the message's `image_path`.
Only the frames router's RTSP capture is left out, as in live mode. (The modules' video-upload
API cannot be used: it decodes the file inside the worker and never touches Kafka.)

```powershell
pip install -e ".[standalone,video]"          # OpenCV, only for `corpus from-video`
# 1. Video -> JPEGs + manifest in LOADGEN_CORPUS_DIR (./corpora), sampled at the run's fps
.venv\Scripts\python -m vision_loadgen corpus from-video --video lobby.mp4 --name lobby --fps 5
# 2. Copy it to where the workers read it (default /app/events/loadgen_corpus/<name>, i.e. the
#    modules host's events folder); the command above prints the scp line
scp -r corpora\lobby admin1@10.10.10.22:<compose folder>/events/loadgen_corpus/
# 3. Register the video's perspective as a camera (once), cloned from an existing camera
.venv\Scripts\python -m vision_loadgen video-camera add --name "Lobby video" --like <camera id> --corpus lobby
# 4. Check and run with it as the template
.venv\Scripts\python -m vision_loadgen check --worker emotion --worker attendance --server-ip 10.10.10.22 `
  --template-camera <video camera id> --set source.mode=corpus --set source.corpus_name=lobby
.venv\Scripts\python -m vision_loadgen run   --worker emotion --worker attendance --server-ip 10.10.10.22 `
  --template-camera <video camera id> --set source.mode=corpus --set source.corpus_name=lobby
```

- **Corpus:** `corpus.json` records the frame rate and `image_root`, the folder the workers read
  the JPEGs from. Manifest paths are relative to it. Override it with `--image-root`,
  `LOADGEN_CORPUS_IMAGE_ROOT` or `--set source.image_root=...`. The events folder is used because
  the frames mount is a tmpfs whose old files are deleted.
  - The workers must see the same folder. Check with
    `docker inspect <container> --format '{{json .Mounts}}'`.
  - From your machine the files can't be checked. A worker that can't read them shows no
    synthetic staleness, and the stage verdict says so.
- **Frame rate:** keep the corpus fps equal to `fps_per_camera` (default 5) so the video plays in
  real time. Each synthetic camera starts at a different point and loops.
- **Video camera:** a clone of `--like`'s row named `loadgen-video-...`, with its stream blanked
  and `is_active` off, so nothing tries to open it.
  - It isn't enabled in any module. Each run clones it into synthetic cameras and removes those
    afterwards.
  - `--zones-from <camera>` copies per-camera rows such as emotion zones.
  - `video-camera list` / `video-camera remove <id>` manage it. `cleanup --orphans` leaves it alone.
- **Attendance:** recognition load is realistic only if the video shows enrolled faces. Otherwise
  it measures detection alone.

## Web UI

`python -m vision_loadgen ui` serves a small page on http://127.0.0.1:8765 that runs everything
above: upload a video and build a corpus, add or remove video cameras, check, run and stop
(teardown still runs), and clean up leftovers.

- **Live view:** charts the run as it goes (staleness p95 against the keep-up limit, consumer lag,
  real-camera excess against the guard, cameras and publish rate) plus the stage verdicts.
  Runs always serve metrics on 9464 (or the next free port), so Grafana sees them too.
- **History:** shows any finished run from the results folder.
- **Command output:** each command's output streams into the page. Logs are kept in
  `<results>/ui-jobs/`.
- **Access:** the page only listens on localhost and accepts no arbitrary commands. Production
  runs need `ui --allow-production`.

## Commands

```bash
python -m vision_loadgen check   --scenario latency            # read-only preflight + plan
python -m vision_loadgen run     --scenario throughput --no-guard   # staging only
python -m vision_loadgen run     --scenario latency --set levels=[5,10,20,40]
python -m vision_loadgen capture --name soak-baseline --duration 600 --camera <id>
python -m vision_loadgen run     --scenario soak --set source.mode=corpus --set source.corpus_name=soak-baseline
python -m vision_loadgen cleanup --run-id 20260929-101500-ab12
python -m vision_loadgen cleanup --orphans [--yes]
python -m vision_loadgen corpus from-video --video clip.mp4 --name lobby [--fps 5] [--image-root /app/...]
python -m vision_loadgen video-camera add --name "Lobby video" --like <camera id> | list | remove <id>
python -m vision_loadgen ui [--port 8765]
```

`vision-loadgen` is the same CLI when installed with pip.

- `--scenario`: `throughput`, `latency`, `soak`, or a scenario file (YAML or JSON).
- `--worker`: preset name (`crowd`, `emotion`, `attendance`) or function key (`crowd-monitoring`), not
  a container name; the worker's URL comes from `vision_main_db`. Default: the worker it runs beside.
- `--template-camera`: the real camera to copy; default: the worker's freshest live camera.
- `--server-ip`: pick a deployment when a module runs on several servers.
- `--set key=value`: scenario overrides (dotted keys, JSON values).
- `--config file`: YAML/JSON merged over the presets (new workers, `gpu_free_vram_mb`, table names).
- `--env-file file`: `KEY=value` file loaded first (default `LOADGEN_ENV_FILE`, else `./.env`).
- `--environment staging|production`: overrides `LOADGEN_ENVIRONMENT`. **Unset means production**,
  which needs `--allow-production`.
- `--keep-events`: keep events (rows and files) created by synthetic cameras.
- `--metrics-port N` (`run`): serve Prometheus metrics on this port during the run (see Grafana).

## GPU model copies

`GpuInferenceEngine` runs one model copy per `CAMERAS_PER_WORKER` cameras (plus an API model
when it has one) and gives camera *i* of the sorted camera ids to copy *i // CAMERAS_PER_WORKER*.
It refuses to start when the copies need more than the VRAM that was free when the worker
started, and that stops its real cameras too. So the load generator:

- **Sizes each stage like production** (`sizing: per_stage`, default). Every stage enables
  exactly its camera count in the module's settings and re-syncs the worker, so the engine runs
  `ceil((real + synthetic) / CAMERAS_PER_WORKER)` copies, as it would for that many real cameras.
  Publishing pauses during the restart, the guard ignores it, and samples count only after
  `settle_s`. `sizing: fixed` enables the largest count once (one sync, copies sized for the
  largest stage).
- **Caps the run at GPU capacity** when the budget is known: `gpu_free_vram_mb` from a config
  file, or estimated from `nvidia-smi` when running beside the worker. Stages above the cap are
  skipped; a throughput ramp ends at the cap. Unknown capacity does not cap; a sync that fails
  anyway stops the run and tears down.
- **Reports per copy**: each stage lists `gpu_workers_expected` and staleness per copy
  (`gpu_workers`), so one overloaded copy shows up even when the average looks fine.

## What a run does

1. Resolves each worker's deployment, picks the template camera, checks the enabled settings rows,
   reads each worker's real camera count and plans GPU capacity.
2. Creates a `loadtest-<run_id>` region, assigns it to each worker, and clones the template
   camera into it (plus its counting lines / zones). Nothing is enabled yet.
3. Measures the real cameras' baseline for the guard.
4. Per stage: enables that many synthetic cameras, re-syncs, publishes their frame pointers to
   `exelixi.frames.raw` (reusing the real camera's JPEGs), and samples `/worker/status` and
   consumer lag every few seconds.
5. Tears everything down: settings, cloned rows, events and their image/video files, cameras,
   region, assignments. Attendance rows changed by synthetic cameras are restored from a
   pre-run snapshot.

A worker "keeps up" at a stage when, over the last `saturation.window_s`, synthetic-camera
staleness p95 stays under `max_staleness_s` and consumer lag grows no faster than
`max_lag_growth_per_s`. Workers drop frames under overload instead of queueing, so staleness is
the main signal.

Clocks: the worker reports the timestamp carried in each Kafka message. Synthetic messages are
stamped by the load generator, so synthetic staleness never depends on clock differences between
machines. Real cameras are stamped on the ingestion side; their staleness (report and template
pick) is corrected by the worker's clock offset, estimated from its HTTP `Date` headers. `check`
prints the offset and `summary.json` records it under `clock_offset`.

## Safety

- **Real-camera guard** (default on): aborts (or backs off, per scenario) if real cameras fall more
  than `max_real_staleness_increase_s` behind their baseline for `grace_s`. `--no-guard` is
  accepted only for throughput in staging.
- Every re-sync restarts the worker's pipeline for all its cameras, as adding a camera in the UI
  does. With `sizing: per_stage` that happens once per stage.
- Synthetic cameras get `ip=127.0.0.1` and blank stream URLs, so nothing opens extra streams.
- The tool never enables a module; a module without an enabled settings row stops the run.
- Everything created is written to `<results>/<run_id>/registry.json` first, so `cleanup` can
  undo it; `cleanup --orphans` finds leftovers by name even without the registry.
- Event files are deleted only when they resolve inside `EVENTS_DIR`, plus folders named exactly
  after a synthetic camera id. Without `EVENTS_DIR` mounted, files are left and the report says so.
- AI attendance: rows touched only by synthetic cameras are restored or deleted; rows also touched
  by a real camera are left alone and listed under `manual_review` in `summary.json`.

## Output

`<results>/<run_id>/`: `timeseries.csv` (per sample), `summary.json` (capacity, stages with
verdicts and per-copy staleness, throughput result, soak drift, teardown report), `registry.json`
and `snapshots/`.

## Grafana

`run` can also serve its metrics to Prometheus, live, for a Grafana dashboard. `monitoring/` holds
a Prometheus + Grafana stack for your own machine, with the datasource and dashboard provisioned:

```powershell
docker compose -f monitoring/docker-compose.yml up -d     # once; keeps running
# in .env (or the shell): LOADGEN_METRICS_PORT=9464
.venv\Scripts\python -m vision_loadgen run --worker emotion --server-ip 10.10.10.22 --scenario latency
```

Open http://localhost:3000 (`admin` / `admin`, or `GRAFANA_ADMIN_PASSWORD`): the home dashboard is
**vision-loadgen**. Pick the run in **Run** (newest first), or click a run id in the **Runs** table
to jump to its time window; widen the time range to find older runs (Prometheus keeps 90 days,
`PROMETHEUS_RETENTION`). Panels: synthetic cameras and publish rate; synthetic staleness p50/p95/max
against the keep-up limit; consumer lag; real-camera staleness above baseline against the guard
limit; worker health; staleness per GPU model copy and copies expected; stage verdicts and the
throughput result. Stages are shaded on every graph.

- The exporter is off unless `LOADGEN_METRICS_PORT` or `--metrics-port` is set, needs no extra
  package (worker images included), and never fails a run: a busy port only logs a warning.
  `LOADGEN_METRICS_ADDR` sets the bind address (default `0.0.0.0`). After the run it keeps serving
  for 15 s (`output.metrics_linger_s`) so the verdicts get scraped.
- Prometheus scrapes `host.docker.internal:9464` (this machine) every 5 s. To watch a run somewhere
  else, e.g. beside a worker on a server, add its address to `monitoring/prometheus/targets/`
  (picked up within 30 s, no restart) and publish the port:
  `docker compose run --rm --no-deps -p 9464:9464 -e LOADGEN_METRICS_PORT=9464 <worker-service> python -m vision_loadgen run ...`.
  For the standalone compose in `docker/`, use `docker compose run --rm --service-ports load-generator run ...`.
- Windows may ask to allow Python through the firewall on the first run; allow it for private
  networks, or set `LOADGEN_METRICS_ADDR=127.0.0.1` if Docker Desktop reaches it that way.
- Metric names and labels are listed in `vision_loadgen/exporter.py` (`METRICS`); every series
  carries `run_id`. Dashboard edits made in Grafana last until it restarts: export the JSON into
  `monitoring/grafana/dashboards/vision-loadgen.json` to keep them.

## Tests

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[standalone,dev]"   # bin/ on Linux/macOS
.venv/Scripts/python -m pytest                     # unit tests
LOADGEN_IT=1 .venv/Scripts/python -m pytest        # + end-to-end against Postgres/Kafka in Docker
```

`tests/integration/docker-compose.it.yml` also has a `compat` service that runs the whole suite
on Python 3.11 with the workers' pinned kafka-python 2.0.2 and pydantic 2.5.3 (commands in the file).
