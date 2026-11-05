"""Cross-platform advisory file lock for AndroidWorld research runs.

One-byte nonblocking lock held by the process for the entire study. Native
Windows uses msvcrt; POSIX uses fcntl. This prevents concurrent experiments
from silently overwriting a shared frozen result root.
"""
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def exclusive_run_lock(filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        # Windows locking requires a byte in the file; POSIX accepts it too.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another process owns AndroidWorld study: {path}") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another process owns AndroidWorld study: {path}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
