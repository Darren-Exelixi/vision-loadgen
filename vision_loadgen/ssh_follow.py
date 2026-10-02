"""Runs one command on the Docker host over ssh with password login and copies its output to stdout.

The Worker logs tab uses this in place of `ssh` when LOADGEN_WORKER_LOGS_PASSWORD is set, since
OpenSSH cannot take a password without a terminal. The password arrives in PASSWORD_ENV, never in
argv, and also answers `sudo -S` when the command starts with it.

    python -m vision_loadgen.ssh_follow user@host docker logs --follow <container>
"""

from __future__ import annotations

import os
import shlex
import sys

PASSWORD_ENV = "LOADGEN_SSH_FOLLOW_PASSWORD"


def main(argv: list[str]) -> int:
    if len(argv) < 2 or "@" not in argv[0]:
        print("usage: ssh_follow user@host command...", file=sys.stderr)
        return 2
    try:
        import paramiko
    except ImportError:
        print("Password login needs paramiko: pip install paramiko", flush=True)
        return 2
    user, host = argv[0].split("@", 1)
    command = argv[1:]
    password = os.environ.pop(PASSWORD_ENV, "")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Unknown hosts are accepted like `StrictHostKeyChecking=accept-new`; a changed known key still fails.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=user, password=password, timeout=10, allow_agent=False, look_for_keys=False)
    except paramiko.AuthenticationException:
        print(f"ssh: password login to {argv[0]} was refused; check LOADGEN_WORKER_LOGS_PASSWORD", flush=True)
        return 255
    except Exception as exc:  # noqa: BLE001 - every connect failure is reported the same way
        print(f"ssh: cannot connect to {argv[0]}: {exc}", flush=True)
        return 255
    try:
        transport = client.get_transport()
        transport.set_keepalive(15)
        channel = transport.open_session()
        channel.set_combine_stderr(True)
        channel.exec_command(shlex.join(command))
        if command[:2] == ["sudo", "-S"]:
            channel.sendall((password + "\n").encode())
        out = sys.stdout.buffer
        while True:
            data = channel.recv(65536)
            if not data:
                break
            out.write(data)
            out.flush()
        return channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
