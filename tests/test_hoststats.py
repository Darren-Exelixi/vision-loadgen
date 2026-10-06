"""Host CPU/RAM/GPU/VRAM readings: parsing, the remote command, the background reader and the Recorder columns."""

import csv
import sys
import time

import pytest

from vision_loadgen import dockerhost, hoststats
from vision_loadgen.analysis import WorkerSample
from vision_loadgen.config import HostStatsConfig, WorkerLogsConfig
from vision_loadgen.exporter import MetricsExporter
from vision_loadgen.hoststats import ContainerSample, GpuSample, HostSample, HostStats
from vision_loadgen.kafka_io import ProducerStats
from vision_loadgen.metrics import Recorder

BLOCK_1 = """cpu  1000 0 1000 8000 0 0 0 0 0 0
MemTotal:       16384000 kB
MemAvailable:    8192000 kB
GPU 0, 40, 2000, 16000
GPU 1, [N/A], 100, 16000
CTR crowd-1 150.50% 1.5GiB / 62GiB
END
"""
# 1000 more jiffies of work, 3000 more idle: 25% busy.
BLOCK_2 = """cpu  1500 0 1500 11000 0 0 0 0 0 0
MemTotal:       16384000 kB
MemAvailable:    4096000 kB
GPU 0, 95, 9000, 16000
CTR crowd-1 --  512MiB / 62GiB
END
"""


def _feed(parser, text, now=1000.0):
    sample = None
    for line in text.splitlines():
        sample = parser.feed(line, now) or sample
    return sample


def test_parser_reads_a_block_and_derives_cpu_from_the_previous_one():
    parser = hoststats._Parser()
    first = _feed(parser, BLOCK_1)
    assert first.cpu_pct is None  # needs a previous block
    assert first.ram_total_mb == pytest.approx(16000) and first.ram_used_mb == pytest.approx(8000)
    assert first.gpus == [GpuSample(0, 40.0, 2000.0, 16000.0), GpuSample(1, None, 100.0, 16000.0)]
    assert first.containers["crowd-1"] == ContainerSample(150.5, pytest.approx(1536))

    second = _feed(parser, BLOCK_2)
    assert second.cpu_pct == pytest.approx(25.0)
    assert second.ram_used_mb == pytest.approx(12000)
    assert [gpu.util_pct for gpu in second.gpus] == [95.0]  # no GPU 1 line this time
    assert second.containers["crowd-1"].cpu_pct is None and second.containers["crowd-1"].ram_mb == pytest.approx(512)


def test_parser_skips_garbled_lines_and_flags_unknown_ones():
    parser = hoststats._Parser()
    assert parser.feed("GPU x, y", 1.0) is None and parser.recognised
    assert parser.feed("bash: nvidia-smi: not found", 1.0) is None and not parser.recognised
    sample = parser.feed("END", 1.0)
    assert sample.gpus == [] and sample.ram_total_mb is None and sample.cpu_pct is None


@pytest.mark.parametrize("text, mb", [("1.5GiB", 1536), ("512MiB", 512), ("2GB", 2e9 / 1048576), ("1024KiB", 1), ("zz", None)])
def test_size_units(text, mb):
    assert hoststats._size_mb(text) == (pytest.approx(mb) if mb is not None else None)


def test_script_quotes_container_names_and_skips_docker_stats_without_containers():
    script = hoststats.build_script("docker", ["crowd-1", "odd name"], 2)
    assert "docker stats --no-stream --format 'CTR {{.Name}} {{.CPUPerc}} {{.MemUsage}}' crowd-1 'odd name'" in script
    assert "sleep 2" in script and "nvidia-smi" in script
    assert "docker stats" not in hoststats.build_script("docker", [], 2)


def test_remote_command_runs_under_the_sudo_prefix_of_docker_command():
    plain = hoststats.remote_command(WorkerLogsConfig(), ["c1"], 2)
    assert plain[:2] == ["sh", "-c"] and "docker stats" in plain[2]
    sudo = hoststats.remote_command(WorkerLogsConfig(docker_command="sudo -n docker"), ["c1"], 2)
    assert sudo[:4] == ["sudo", "-n", "sh", "-c"] and "\ndocker stats" in sudo[4]
    # With password login, argv() turns the leading sudo into `sudo -S` for the ssh helper.
    keyed = WorkerLogsConfig(ssh_target="admin1@10.0.0.2", password="pw", docker_command="sudo docker")
    argv = dockerhost.argv(keyed, hoststats.remote_command(keyed, [], 2))
    assert argv[:3] == [sys.executable, "-m", "vision_loadgen.ssh_follow"] and argv[4:7] == ["sudo", "-S", "-p"]


def test_not_on_a_posix_host_without_ssh_there_is_nothing_to_read(monkeypatch):
    monkeypatch.setattr(hoststats.os, "name", "nt")
    host = HostStats(WorkerLogsConfig(), HostStatsConfig())
    assert "LOADGEN_WORKER_LOGS_SSH" in host.problem
    assert host.wait_first(0.1) is None
    assert HostStats(WorkerLogsConfig(ssh_target="a@b"), HostStatsConfig()).problem is None


def _fake_host(monkeypatch, program):
    """A HostStats whose 'remote' command is a local python program."""
    monkeypatch.setattr(dockerhost, "argv", lambda cfg, remote: [sys.executable, "-u", "-c", program])
    return HostStats(WorkerLogsConfig(ssh_target="a@b"), HostStatsConfig(), {"crowd": "crowd-1", "emotion": ""})


def test_reader_follows_the_command_and_exposes_the_latest_reading(monkeypatch):
    program = f"import time\nprint({BLOCK_1!r}, end='')\nprint({BLOCK_2!r}, end='')\ntime.sleep(30)"
    host = _fake_host(monkeypatch, program)
    assert host.containers == {"crowd": "crowd-1"}  # a worker without a container has no container stats
    host.start()
    try:
        assert host.wait_first(10) is not None
        deadline = time.time() + 10
        while time.time() < deadline and (host.latest() is None or host.latest().cpu_pct is None):
            time.sleep(0.05)
        assert host.latest().cpu_pct == pytest.approx(25.0)
        assert host.gpu_count == 2
        assert host.problem is None
    finally:
        host.stop()


def test_reader_reports_what_the_host_printed_when_it_never_answers(monkeypatch):
    host = _fake_host(monkeypatch, "print('ssh: cannot connect to a@b: timed out')")
    host.start()
    try:
        assert host.wait_first(10) is None
        assert "cannot connect" in host.problem
    finally:
        host.stop()


def test_stale_readings_are_not_current(monkeypatch):
    host = HostStats(WorkerLogsConfig(ssh_target="a@b"), HostStatsConfig())
    host._latest = HostSample(time.time() - hoststats.MAX_AGE_S - 1, 5.0, 1.0, 2.0)
    assert host.latest() is None


class FakeHost:
    containers = {"crowd": "crowd-1"}
    gpu_count = 1
    source = "this machine"

    def __init__(self):
        self.sample = None

    def latest(self):
        return self.sample


def _batch():
    return [WorkerSample(worker="crowd", t=time.time(), ok=True, running=True, lag=0, synthetic_staleness=[1.0])]


def test_recorder_writes_host_columns_exports_them_and_summarises_each_stage(tmp_path):
    exporter = MetricsExporter("r")
    host = FakeHost()
    recorder = Recorder(tmp_path, ["crowd"], exporter, host=host)

    host.sample = HostSample(time.time(), 30.0, 8192.0, 16384.0, [GpuSample(0, 80.0, 6000.0, 16000.0)],
                             {"crowd-1": ContainerSample(150.0, 1024.0)})
    recorder.record("5 cameras", 5, ProducerStats(), _batch(), {})
    host.sample = HostSample(time.time(), 50.0, 9216.0, 16384.0, [GpuSample(0, 100.0, 7000.0, 16000.0)],
                             {"crowd-1": ContainerSample(250.0, 2048.0)})
    recorder.record("5 cameras", 5, ProducerStats(), _batch(), {})
    host.sample = None  # the host stopped answering: blanks, and the series go away
    recorder.record("10 cameras", 10, ProducerStats(), _batch(), {})

    assert exporter.value("loadgen_host_cpu_percent") is None
    assert exporter.value("loadgen_gpu_utilization_percent", gpu=0) is None
    recorder.close()

    rows = list(csv.DictReader((tmp_path / "timeseries.csv").open(encoding="utf-8")))
    assert [r["host_cpu_pct"] for r in rows] == ["30.0", "50.0", ""]
    assert rows[1]["gpu0_util_pct"] == "100.0" and rows[1]["gpu0_vram_used_mb"] == "7000.0"
    assert rows[1]["gpu0_vram_total_mb"] == "16000.0" and rows[1]["host_ram_used_mb"] == "9216.0"
    assert rows[1]["crowd_container_cpu_pct"] == "250.0" and rows[1]["crowd_container_ram_mb"] == "2048.0"

    resources = recorder.stage_resources("5 cameras")
    assert resources["readings"] == 2
    assert resources["host"]["cpu_pct"] == {"max": 50.0, "avg": 40.0}
    assert resources["gpus"]["0"]["util_pct"] == {"max": 100.0, "avg": 90.0}
    assert resources["gpus"]["0"]["vram_used_mb"] == {"max": 7000.0} and resources["gpus"]["0"]["vram_total_mb"] == 16000.0
    assert resources["containers"]["crowd"]["ram_mb"] == {"max": 2048.0}
    assert recorder.stage_resources("10 cameras") == {}


def test_live_metrics_carry_the_latest_host_reading(tmp_path):
    exporter = MetricsExporter("r")
    host = FakeHost()
    recorder = Recorder(tmp_path, ["crowd"], exporter, host=host)
    host.sample = HostSample(time.time(), 30.0, 8192.0, 16384.0, [GpuSample(0, 80.0, 6000.0, 16000.0)],
                             {"crowd-1": ContainerSample(150.0, 1024.0)})
    recorder.record("5 cameras", 5, ProducerStats(), _batch(), {})
    mb = 1024 * 1024
    assert exporter.value("loadgen_host_cpu_percent") == 30.0
    assert exporter.value("loadgen_host_memory_used_bytes") == 8192 * mb
    assert exporter.value("loadgen_gpu_utilization_percent", gpu=0) == 80.0
    assert exporter.value("loadgen_gpu_memory_used_bytes", gpu=0) == 6000 * mb
    assert exporter.value("loadgen_gpu_memory_total_bytes", gpu=0) == 16000 * mb
    assert exporter.value("loadgen_container_cpu_percent", worker="crowd") == 150.0
    assert exporter.value("loadgen_container_memory_bytes", worker="crowd") == 1024 * mb
    recorder.close()


def test_without_a_host_the_csv_has_no_host_columns(tmp_path):
    recorder = Recorder(tmp_path, ["crowd"])
    recorder.record("5 cameras", 5, ProducerStats(), _batch(), {})
    recorder.close()
    header = (tmp_path / "timeseries.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "host_cpu_pct" not in header and "gpu0" not in header
    assert recorder.stage_resources("5 cameras") == {}
