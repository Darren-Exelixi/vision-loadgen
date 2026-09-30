"""Minimal copies of the vision-main and module tables the load generator touches."""

MAIN = """
CREATE TABLE servers (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), server_ip text NOT NULL, server_protocol text,
    is_active boolean, order_index int DEFAULT 0, created_at timestamptz DEFAULT now(), deleted_at timestamptz);
CREATE TABLE functions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), key text NOT NULL, container_port int,
    is_exelixi_activated boolean, deleted_at timestamptz);
CREATE TABLE server_functions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), server_id uuid REFERENCES servers(id),
    function_id uuid REFERENCES functions(id), container_port int, is_active boolean,
    created_at timestamptz DEFAULT now(), modified_at timestamptz, deleted_at timestamptz);
CREATE TABLE camera_regions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text NOT NULL UNIQUE, description text,
    created_at timestamptz NOT NULL DEFAULT now(), deleted_at timestamptz);
CREATE TABLE cameras (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text NOT NULL,
    region_id uuid REFERENCES camera_regions(id), type text, ip varchar(64), port int, "user" text,
    password text, model text, timezone text NOT NULL, description text, rtsp_url text NOT NULL,
    is_active boolean, created_at timestamptz NOT NULL, modified_at timestamptz, deleted_at timestamptz);
CREATE TABLE function_camera_regions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), server_function_id uuid NOT NULL REFERENCES server_functions(id),
    camera_region_id uuid NOT NULL REFERENCES camera_regions(id), status text,
    created_at timestamptz DEFAULT now(), deleted_at timestamptz);
"""

CROWD = """
CREATE TABLE crowd_gathering_settings (
    id serial PRIMARY KEY, name text, selected_cameras json NOT NULL DEFAULT '[]',
    is_enabled boolean NOT NULL DEFAULT true, deleted_at timestamptz);
CREATE TABLE crowd_gathering_camera_lines (
    id serial PRIMARY KEY, camera_id varchar NOT NULL, setting_id int REFERENCES crowd_gathering_settings(id),
    line_start json NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), deleted_at timestamptz,
    UNIQUE (camera_id, setting_id));
CREATE TABLE crowd_gathering_events (
    id serial PRIMARY KEY, camera_id varchar NOT NULL, event_time timestamptz NOT NULL DEFAULT now(),
    image_path varchar, video_path varchar);
"""

FRS = """
CREATE TABLE frs_settings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text, check_in_cameras jsonb NOT NULL DEFAULT '[]',
    check_out_cameras jsonb NOT NULL DEFAULT '[]', is_enabled boolean NOT NULL, deleted_at timestamptz);
CREATE TABLE frs_attendance (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), employee_id uuid NOT NULL, date date NOT NULL,
    check_in_at timestamptz, is_late boolean NOT NULL DEFAULT false, shift_working_days jsonb,
    created_at timestamptz NOT NULL DEFAULT now(), modified_at timestamptz,
    UNIQUE (employee_id, date));
CREATE TABLE frs_recognition_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), employee_id uuid NOT NULL, attendance_id uuid,
    camera_id uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
"""
