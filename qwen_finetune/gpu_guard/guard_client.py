"""GPU guard client for non-vLLM jobs (e.g. qwen_finetune/train.py).

With ``GPU_GUARD_DIR`` set and an external holder process holding the GPUs (not part of this release):

  take_over(local_idx, weights_gib)  # holder frees GPU local_idx; we keep it full with a placeholder
  ...load the model on CPU...
  drop()                             # free the placeholder right before the model moves to the GPU

Without ``GPU_GUARD_DIR`` both calls are no-ops.
"""

from __future__ import annotations

import os
import sys
import time

GIB = 1 << 30
_held: list = []


def _log(msg: str) -> None:
    print(f"[gpu-guard pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def take_over(idx: int, weights_gib: float, headroom_gib: float = 4.0) -> None:
    d = os.environ.get("GPU_GUARD_DIR")
    if not d:
        return
    import torch

    if os.path.exists(os.path.join(d, "ready")):
        open(os.path.join(d, f"release_{idx}"), "w").close()
        t0 = time.time()
        while not os.path.exists(os.path.join(d, f"released_{idx}")) and time.time() - t0 < 60:
            time.sleep(0.02)
        total = torch.cuda.mem_get_info(idx)[1]
        while torch.cuda.mem_get_info(idx)[0] < 0.8 * total and time.time() - t0 < 20:
            time.sleep(0.05)
        _log(f"holder released gpu {idx} after {time.time() - t0:.2f} s; "
             f"free {torch.cuda.mem_get_info(idx)[0] / GIB:.1f} GiB")
    size = int(torch.cuda.mem_get_info(idx)[0] - (weights_gib + headroom_gib) * GIB)
    while size > GIB:  # may exceed a per-process memory-fraction cap: shrink until it fits
        try:
            _held.append(torch.empty(size, dtype=torch.uint8, device=f"cuda:{idx}"))
            _log(f"placeholder {size / GIB:.1f} GiB on gpu {idx} until the model moves to the GPU")
            return
        except torch.OutOfMemoryError:
            size = int(size * 0.9)


def drop() -> None:
    if not _held:
        return
    import torch

    _held.clear()
    torch.cuda.empty_cache()
    _log("placeholder freed")
