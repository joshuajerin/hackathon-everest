from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO


def acquire_isaac_process_lock(path: str | Path | None = None) -> IO[str]:
    """Acquire the host-wide Everest Isaac application lock for this process."""

    selected = Path(path or os.environ.get("EVEREST_ISAAC_APP_LOCK", "/tmp/everest_isaac_app.lock"))
    selected.parent.mkdir(parents=True, exist_ok=True)
    handle = selected.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"Another Everest Isaac application owns {selected}: {owner}. "
            "Wait for it to finish; concurrent Isaac applications are disabled."
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} command={' '.join(os.sys.argv)}\n")
    handle.flush()
    return handle
