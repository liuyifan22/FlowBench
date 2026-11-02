"""Race-safe JSONL append with batched flushes (fcntl lock)."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional


class LockedJsonlWriter:
    """Buffer records and append under an exclusive flock every ``flush_every``.

    Multiple AI2-THOR worker processes can share the same ``path``; each holds
    its own buffer and only contends briefly on flush.
    """

    def __init__(
        self,
        path: Path,
        *,
        flush_every: int = 10,
        lock_path: Optional[Path] = None,
    ) -> None:
        self.path = Path(path)
        self.flush_every = max(1, int(flush_every))
        self.lock_path = Path(lock_path) if lock_path else self.path.with_suffix(
            self.path.suffix + ".lock"
        )
        self._buf: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure files exist for append/lock.
        if not self.path.exists():
            self.path.touch()
        if not self.lock_path.exists():
            self.lock_path.touch()

    def add(self, record: dict[str, Any]) -> None:
        self._buf.append(record)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def extend(self, records: Iterable[dict[str, Any]]) -> None:
        for r in records:
            self.add(r)

    def flush(self) -> int:
        if not self._buf:
            return 0
        n = len(self._buf)
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self._buf)
        with self.lock_path.open("a+") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        self._buf.clear()
        return n

    def close(self) -> int:
        return self.flush()

    def __enter__(self) -> "LockedJsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()
