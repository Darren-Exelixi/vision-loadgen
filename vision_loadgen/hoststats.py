"""CPU, RAM, GPU utilization and VRAM of the workers' Docker host, sampled while a run is going.

One long-lived process on the host (this machine, or over ssh like the Worker logs tab) loops a
small shell script and prints one block per tick. Keeping it open avoids a new ssh login per
sample. Each block holds the host's CPU jiffies and memory from /proc, one line per GPU from
`nvidia-smi`, and one line per worker container from `docker stats`. Anything the host lacks
(no GPU, no docker access) is left out of the block and shows as a gap, never as a failed run.

Host numbers cover everything on that machine, not only the workers under test; the container
numbers are the worker's own (docker's CPU % counts one core as 100).
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from vision_loadgen import dockerhost
from vision_loadgen.config import HostStatsConfig, WorkerLogsConfig

log = logging.getLogger(__name__)

# A reading older than this is not shown as the current one (the connection dropped, say).
MAX_AGE_S = 15.0
RETRY_S = 10.0

_UNITS_MB = {"b": 1 / 1048576, "kb": 1000 / 1048576, "kib": 1 / 1024, "mb": 1e6 / 1048576, "mib": 1.0,
             "gb": 1e9 / 1048576, "gib": 1024.0, "tb": 1e12 / 1048576, "tib": 1048576.0}
_SIZE = re.compile(r"^([0-9.]+)\s*([A-Za-z]+)$")


@dataclass(frozen=True)
class GpuSample:
    index: int
    util_pct: Optional[float]
    vram_used_mb: Optional[float]
    vram_total_mb: Optional[float]


@dataclass(frozen=True)
class ContainerSample:
    cpu_pct: Optional[float]
    ram_mb: Optional[float]


@dataclass(frozen=True)
class HostSample:
    t: float
    cpu_pct: Optional[float]
    ram_used_mb: Optional[float]
    ram_total_mb: Optional[float]
    gpus: list[GpuSample] = field(default_factory=list)
    containers: dict[str, ContainerSample] = field(default_factory=dict)  # by container name


def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None  # nvidia-smi prints "[N/A]" for what a GPU does not report


def _size_mb(text: str) -> Optional[float]:
    match = _SIZE.match(text.strip())
    unit = _UNITS_MB.get(match.group(2).lower()) if match else None
    return float(match.group(1)) * unit if match and unit else None


class _Parser:
    """Turns the script's lines into a HostSample at each END line; CPU % needs the previous block."""

    def __init__(self) -> None:
        self._previous_cpu: Optional[list[int]] = None
        self.recognised = False
        self._reset()

    def _reset(self) -> None:
        self._cpu: Optional[list[int]] = None
        self._memory: dict[str, float] = {}
        self._gpus: list[GpuSample] = []
        self._containers: dict[str, ContainerSample] = {}

    def feed(self, line: str, now: float) -> Optional[HostSample]:
        """The finished sample at an END line; `recognised` says whether `line` was one of ours."""
        line = line.strip()
        self.recognised = line.startswith(("cpu ", "MemTotal:", "MemAvailable:", "GPU ", "CTR ")) or line == "END"
        try:
            if line.startswith("cpu "):
                self._cpu = [int(field) for field in line.split()[1:9]]
            elif line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                self._memory[key] = int(value.split()[0]) / 1024  # kB -> MB
            elif line.startswith("GPU "):
                parts = [part.strip() for part in line[4:].split(",")]
                self._gpus.append(GpuSample(int(parts[0]), _number(parts[1]), _number(parts[2]), _number(parts[3])))
            elif line.startswith("CTR "):
                parts = line.split()
                self._containers[parts[1]] = ContainerSample(_number(parts[2].rstrip("%")), _size_mb(parts[3]))
            elif line == "END":
                sample = self._sample(now)
                self._reset()
                return sample
        except (ValueError, IndexError):
            pass  # a garbled line costs that reading, nothing else
        return None

    def _sample(self, now: float) -> HostSample:
        cpu_pct = None
        if self._cpu is not None and len(self._cpu) >= 5:
            if self._previous_cpu is not None:
                total = sum(self._cpu) - sum(self._previous_cpu)
                idle = (self._cpu[3] + self._cpu[4]) - (self._previous_cpu[3] + self._previous_cpu[4])
                if total > 0:
                    cpu_pct = max(0.0, min(100.0, 100.0 * (1 - idle / total)))
            self._previous_cpu = self._cpu
        total_mb = self._memory.get("MemTotal")
        available_mb = self._memory.get("MemAvailable")
        used_mb = total_mb - available_mb if total_mb is not None and available_mb is not None else None
        return HostSample(now, cpu_pct, used_mb, total_mb, self._gpus, self._containers)


def build_script(docker: str, container_names: list[str], interval_s: float) -> str:
    lines = [
        "while :; do",
        "head -n1 /proc/stat",
        "grep -E '^Mem(Total|Available):' /proc/meminfo",
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        " 2>/dev/null | sed 's/^/GPU /'",
    ]
    if container_names:
        lines.append(f"{docker} stats --no-stream --format 'CTR {{{{.Name}}}} {{{{.CPUPerc}}}} {{{{.MemUsage}}}}' "
                     f"{' '.join(shlex.quote(name) for name in container_names)} 2>/dev/null")
    lines += ["echo END", f"sleep {interval_s:g}", "done"]
    return "\n".join(lines)


def remote_command(cfg: WorkerLogsConfig, container_names: list[str], interval_s: float) -> list[str]:
    """`sh -c <script>` on the host, under the same sudo prefix as docker_command when it has one."""
    words = dockerhost.docker_command(cfg)
    prefix: list[str] = []
    if words[0] == "sudo":
        end = 1
        while end < len(words) and words[end].startswith("-"):
            end += 1
        prefix, words = words[:end], words[end:]
    return [*prefix, "sh", "-c", build_script(" ".join(words), container_names, interval_s)]


class HostStats:
    """Background reader of the host's resource use; `latest()` is the newest reading."""

    def __init__(self, logs: WorkerLogsConfig, config: HostStatsConfig, containers: Optional[dict[str, str]] = None):
        self._logs = logs
        self._interval_s = config.interval_s
        self.containers: dict[str, str] = {}  # worker -> container name, those docker accepts
        for worker, container in (containers or {}).items():
            try:
                self.containers[worker] = dockerhost.check_container(container) if container else ""
            except dockerhost.DockerHostError as exc:
                log.warning("Host stats: no container stats for %s (%s)", worker, exc)
            if not self.containers.get(worker):
                self.containers.pop(worker, None)
        self.problem: Optional[str] = self.unavailable_reason(logs)
        self.gpu_count = 0
        self._parser = _Parser()
        self._lock = threading.Lock()
        self._latest: Optional[HostSample] = None
        self._first = threading.Event()
        self._stop = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def unavailable_reason(logs: WorkerLogsConfig) -> Optional[str]:
        """Why nothing can be read, when that is known up front."""
        if not logs.ssh_target and os.name != "posix":
            return ("not running on the workers' host; set LOADGEN_WORKER_LOGS_SSH (user@host) to read "
                    "its CPU, RAM and GPU over ssh")
        return None

    @property
    def source(self) -> str:
        return self._logs.ssh_target or "this machine"

    def start(self) -> None:
        if self.problem or self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="loadgen-host-stats", daemon=True)
        self._thread.start()

    def wait_first(self, timeout_s: float) -> Optional[HostSample]:
        """The first reading, or None (see `problem`) when none arrived in time."""
        if self.problem:
            return None
        if not self._first.wait(timeout_s) and not self.problem:
            self.problem = "no reading from the host yet (is ssh reachable?)"
        return self.latest()

    def latest(self) -> Optional[HostSample]:
        with self._lock:
            sample = self._latest
        return sample if sample is not None and time.time() - sample.t <= MAX_AGE_S else None

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        warned = False
        while not self._stop.is_set():
            try:
                tail = self._follow()
            except (OSError, dockerhost.DockerHostError) as exc:
                tail = str(exc)
            if self._stop.is_set():
                return
            if not warned:
                warned = True
                log.warning("Host stats stopped (%s); retrying every %.0fs", tail or "no output", RETRY_S)
            if not self._first.is_set():
                self.problem = tail or "the host command ended without output"
                self._first.set()
                return
            self._stop.wait(RETRY_S)

    def _follow(self) -> str:
        names = list(self.containers.values())
        remote = remote_command(self._logs, names, self._interval_s)
        self._process = subprocess.Popen(dockerhost.argv(self._logs, remote), env=dockerhost.env(self._logs),
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, errors="replace")
        recent: deque[str] = deque(maxlen=4)  # what a failed login or missing command printed
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            sample = self._parser.feed(line, time.time())
            if sample is None:
                if not self._parser.recognised and line.strip():
                    recent.append(line.strip())
                continue
            with self._lock:
                self._latest = sample
            self.gpu_count = max(self.gpu_count, len(sample.gpus))
            self._first.set()
        self._process.wait()
        return " | ".join(recent)
