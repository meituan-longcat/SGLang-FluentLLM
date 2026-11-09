#!/usr/bin/env python3
"""Run a decode trace command and persist its complete stdout/stderr."""

import argparse
import errno
import os
import pty
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    master, slave = pty.openpty()
    process = subprocess.Popen(command, stdout=slave, stderr=slave)
    os.close(slave)
    chunks = []
    while True:
        try:
            chunk = os.read(master, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        chunks.append(chunk)
        print(chunk.decode(errors="replace"), end="", flush=True)
    os.close(master)
    returncode = process.wait()
    args.output.write_bytes(b"".join(chunks))
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
