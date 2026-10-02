"""Docker commands on the workers' Docker host: on this machine, or over ssh (key or password login).

Used by the UI's Worker logs tab (`docker logs`) and by teardown, which removes synthetic cameras'
event folders inside the worker container when the events folder is not mounted here.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from typing import Optional

from vision_loadgen.config import WorkerLogsConfig
from vision_loadgen.ssh_follow import PASSWORD_ENV

CONTAINER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
SSH_TARGET = re.compile(r"^[A-Za-z0-9_.\-]+@[A-Za-z0-9.\-]+$")
DOCKER_WORD = re.compile(r"^[A-Za-z0-9_./\-]+$")


class DockerHostError(ValueError):
    pass


def docker_command(cfg: WorkerLogsConfig) -> list[str]:
    words = cfg.docker_command.split()
    if not words or not all(DOCKER_WORD.match(word) for word in words):
        raise DockerHostError(f"Invalid worker_logs.docker_command {cfg.docker_command!r}")
    return words


def check_container(container: str) -> str:
    if not CONTAINER.match(container or ""):
        raise DockerHostError(f"Invalid container name {container!r}")
    return container


def argv(cfg: WorkerLogsConfig, remote: list[str]) -> list[str]:
    """The local command that runs `remote` on the Docker host."""
    if not cfg.ssh_target:
        return remote
    if not SSH_TARGET.match(cfg.ssh_target):
        raise DockerHostError(f"Invalid worker_logs.ssh_target {cfg.ssh_target!r} (expected user@host)")
    if cfg.password:
        # OpenSSH cannot take a password without a terminal; the helper logs in with paramiko and
        # feeds the same password to sudo.
        if remote[0] == "sudo":
            remote = ["sudo", "-S", "-p", "", *[word for word in remote[1:] if word != "-n"]]
        return [sys.executable, "-m", "vision_loadgen.ssh_follow", cfg.ssh_target, *remote]
    # BatchMode: fail at once without key login instead of waiting on a password prompt nobody sees.
    # The remote shell re-parses the command, so it is passed quoted.
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
            cfg.ssh_target, shlex.join(remote)]


def env(cfg: WorkerLogsConfig) -> Optional[dict[str, str]]:
    """Environment for argv(): carries the password to the helper, never on its command line."""
    if cfg.password and cfg.ssh_target:
        return {**os.environ, PASSWORD_ENV: cfg.password}
    return None


def run(cfg: WorkerLogsConfig, remote: list[str], timeout_s: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(argv(cfg, remote), env=env(cfg), stdin=subprocess.DEVNULL, capture_output=True,
                          timeout=timeout_s, check=False)
