"""Small PyInstaller helper that activates a verified staged MonStim update."""

from __future__ import annotations

import argparse
import ctypes
import subprocess

from monstim_gui.updates import activate_update, staged_executable


def _wait_for_pid(pid: int) -> None:
    synchronize = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
    if handle:
        try:
            ctypes.windll.kernel32.WaitForSingleObject(handle, 120000)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    _wait_for_pid(args.wait_pid)
    activate_update(args.version)
    if args.restart:
        subprocess.Popen([str(staged_executable(args.version))], close_fds=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
