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
        "worker_api": "attendance",
        # Recognition only matches employees of departments linked to the camera's region.
        "copy_department_links": True,
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
        "note": "Cameras show as active only once frames are processed; the template camera's region "
                "needs departments for recognitions to happen.",
    },
    # Tables below follow vision-module-backend's models for each module; the workers themselves
    # were not reviewed locally, so the sync/status contract is unconfirmed for all of them.
    "ppe": {
        "function_key": "ppe-detection",
        "container": "ppe_backend",
        "database_name": "ppe_db",
        "db_url": "${PPE_DATABASE_URL}",
        # Cameras are assigned per configuration; the first live configuration takes the synthetic ones.
        "settings": {"table": "ppe_configurations", "camera_columns": ["camera_ids"],
                     "enabled_where": "is_deleted = FALSE AND deleted_at IS NULL"},
        "camera_rows": [{"table": "ppe_camera_zones"}],
        "purge": [{"table": "ppe_violations", "file_columns": ["image_path", "video_path"]}],
        "note": "Worker not reviewed locally. Frames outside the configuration's start/end time are likely skipped.",
    },
    "intrusion": {
        "function_key": "intrusion-detection",
        "container": "intrusion_backend",
        "database_name": "intrusion_db",
        "db_url": "${INTRUSION_DATABASE_URL}",
        "settings": {"table": "intrusion_settings", "camera_columns": ["selected_cameras"], "enabled_where": "is_enabled = TRUE"},
        "camera_rows": [{"table": "intrusion_camera_zones"}],
        # intrusion_events has no camera_id; it keeps the camera id in camera_name.
        "purge": [{"table": "intrusion_events", "camera_column": "camera_name", "file_columns": ["image_path", "video_path"]}],
        "note": "Worker not reviewed locally; confirm it exposes /api/v1/worker/sync and /worker/status.",
    },
    "fire_smoke": {
        "function_key": "fire-smoke-detection",
        "container": "fire_smoke_backend",
        "database_name": "fire_smoke_db",
        "db_url": "${FIRE_SMOKE_DATABASE_URL}",
        "settings": {"table": "fire_smoke_settings", "camera_columns": ["selected_cameras"], "enabled_where": "is_enabled = TRUE"},
        "purge": [{"table": "fire_smoke_events", "file_columns": ["image_path", "video_path"]}],
        "note": "Worker not reviewed locally; confirm it exposes /api/v1/worker/sync and /worker/status.",
    },
    "obstacle": {
        "function_key": "obstacle-detection",
        "container": "obstacle_detection_backend",
        "database_name": "obstacle_detection_db",
        "db_url": "${OBSTACLE_DATABASE_URL}",
        "settings": {"table": "obstacle_detection_settings", "camera_columns": ["selected_cameras"],
                     "enabled_where": "is_enabled = TRUE"},
        "camera_rows": [{"table": "obstacle_detection_camera_zones"}],
        "purge": [{"table": "obstacle_detection_events", "file_columns": ["image_path", "video_path", "exit_video_path"]}],
        "note": "Worker not reviewed locally; confirm it exposes /api/v1/worker/sync and /worker/status.",
    },
    "productivity": {
        "function_key": "productivity-monitoring",
        "container": "productivity_monitoring_backend",
        "database_name": "productivity_monitoring_db",
        "db_url": "${PRODUCTIVITY_DATABASE_URL}",
        "settings": {"table": "productivity_settings", "camera_columns": ["selected_cameras"]},
        "camera_rows": [{"table": "productivity_camera_zones"}],
        "purge": [{"table": "productivity_events", "file_columns": ["image_path", "video_path", "exit_video_path"]}],
        "note": "Worker not reviewed locally; confirm it exposes /api/v1/worker/sync and /worker/status.",
    },
    "fall": {
        "function_key": "fall-detection",
        "container": "fall_detection_backend",
        "database_name": "fall_detection_db",
        "db_url": "${FALL_DATABASE_URL}",
        "settings": {"table": "fall_detection_settings", "camera_columns": ["selected_cameras"], "enabled_where": "is_enabled = TRUE"},
        "purge": [{"table": "fall_detection_events", "file_columns": ["image_path", "video_path"]}],
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
