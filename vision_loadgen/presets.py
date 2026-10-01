"""Built-in worker definitions and scenarios, so a run needs no config files.

A `--config` file (YAML or JSON) is merged over these; use it to add a worker or change a
table name. Worker keys are short names; `--worker` also accepts the function key.

  function_key           functions.key in vision-main (and the worker's FUNCTION_NAME)
  database_name          the module database, appended to POSTGRES_URL like the worker does
  db_url                 explicit module database URL (overrides database_name)
  server_ip              pin a deployment when the function runs on several servers
  base_url               override the worker URL resolved from servers/server_functions
  consumer_group_prefix  defaults to "<function_key>_consumer_group_"
  settings               where the module keeps its enabled camera list
  camera_rows            per-camera config rows cloned from the template camera
  purge                  events deleted at teardown (db: module | main), with their files
  restore                rows without a camera column that synthetic cameras can change
  cameras_per_worker     GPU sizing; read from the worker's settings when run beside it
  vram_per_worker_mb       "
  api_worker             the engine also holds an API model (counts against VRAM)
  gpu_free_vram_mb       VRAM the worker had free at start; enables the capacity cap
  metrics                the worker's own Prometheus endpoint: per-stage time and processed fps
  container              its Docker container (docker-compose-modules.yml), for the UI's Worker logs
"""

from __future__ import annotations

WORKER_PRESETS: dict[str, dict] = {
    "crowd": {
        "function_key": "crowd-monitoring",
        "container": "crowd_monitoring_backend",
        "database_name": "crowd_gathering_db",
        "db_url": "${CROWD_DATABASE_URL}",
        "settings": {"table": "crowd_gathering_settings", "camera_columns": ["selected_cameras"]},
        "camera_rows": [{"table": "crowd_gathering_camera_lines"}],
        "purge": [{"table": "crowd_gathering_events", "file_columns": ["image_path", "video_path"]}],
    },
    "emotion": {
        "function_key": "sentiment-analysis",
        "container": "sentiment_analysis_backend",
        "database_name": "emotion_detection_db",
        "db_url": "${SENTIMENT_DATABASE_URL}",
        "settings": {"table": "emotion_settings", "camera_columns": ["selected_cameras"]},
        "camera_rows": [{"table": "emotion_camera_zones"}],
        "purge": [
            {"table": "emotion_events", "file_columns": ["image_path", "video_path"]},
            {"table": "emotion_rollups"},
        ],
        "metrics": {
            "path": "/metrics",
            "stage_histogram": "emotion_stage_seconds",
            "frames_counter": "emotion_frames_processed_total",
            "stages": ["frame_total", "detect", "classify", "identify", "pose"],
            # SCRFD face detection takes milliseconds on the GPU and hundreds on the CPU.
            "gpu_stage_limits": {"detect": 0.15},
        },
        "note": "Frames outside emotion_settings active hours / working days are skipped; run inside that window.",
    },
    "attendance": {
        "function_key": "ai-attendance",
        "container": "frs_backend",
        "database_name": "frs_db",
        "db_url": "${ATTENDANCE_DATABASE_URL}",
        "settings": {"table": "frs_settings", "camera_columns": ["check_in_cameras"]},
        "purge": [
            {"table": "frs_recognition_events", "file_columns": ["event_image_path", "event_video_path"]},
        ],
        "restore": {
            "table": "frs_attendance",
            "snapshot_where": "date >= CURRENT_DATE - 1",
            "events_table": "frs_recognition_events",
            "events_link_column": "attendance_id",
            "events_camera_column": "camera_id",
            "events_time_column": "created_at",
        },
        "note": "Worker not reviewed locally; confirm it exposes /api/v1/worker/sync and /worker/status.",
    },
}

SCENARIO_PRESETS: dict[str, dict] = {
    # Ramp the camera count until each worker falls behind. Staging may use --no-guard.
    "throughput": {
        "name": "throughput-ramp",
        "type": "throughput",
        "workers": [],
        "fps_per_camera": 5,
        "source": {"mode": "live", "source_camera_ids": []},
        "start_cameras": 5,
        "step_cameras": 5,
        "max_cameras": 60,
        "step_duration_s": 120,
        "saturation": {"window_s": 45, "max_staleness_s": 5, "max_lag_growth_per_s": 1},
        "guard": {"enabled": True, "max_real_staleness_increase_s": 10, "grace_s": 15, "action": "abort"},
    },
    # Processing delay at fixed load levels.
    "latency": {
        "name": "latency-levels",
        "type": "latency",
        "workers": [],
        "fps_per_camera": 5,
        "source": {"mode": "live", "source_camera_ids": []},
        "levels": [5, 10, 20],
        "level_duration_s": 180,
        "saturation": {"window_s": 60, "max_staleness_s": 5, "max_lag_growth_per_s": 1},
        "guard": {"enabled": True, "max_real_staleness_increase_s": 10, "grace_s": 15,
                  "action": "backoff", "backoff_step": 5},
    },
    # Fixed load for hours. For identical input across runs, record a corpus and pass
    # --set source.mode=corpus --set source.corpus_name=<name>.
    "soak": {
        "name": "soak-4h",
        "type": "soak",
        "workers": [],
        "fps_per_camera": 5,
        "source": {"mode": "live", "source_camera_ids": []},
        "cameras": 15,
        "duration_s": 14400,
        "guard": {"enabled": True, "max_real_staleness_increase_s": 10, "grace_s": 30,
                  "action": "backoff", "backoff_step": 3},
    },
}
