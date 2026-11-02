"""Optical flow between rgb0/rgb1 via AllTracker or SEA-RAFT.

Visualization helpers follow ``ccwm.utils.flow.flow_to_image`` and the arrow
overlay style in ``inv/audits/audit_interleved_flows.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np

DEFAULT_CCWM_ROOT = Path(os.environ.get("CCWM_ROOT", "ccwm"))
DEFAULT_ALLTRACKER_PATH = os.environ.get("ALLTRACKER_PATH", "alltracker")
DEFAULT_SEARAFT_CONFIG = str(
    DEFAULT_CCWM_ROOT / "ccwm" / "sea_raft" / "config" / "eval" / "spring-M.json"
)


class FlowExtractor(Protocol):
    def extract(self, path0: Path, path1: Path, **kwargs: Any) -> np.ndarray: ...


class AllTrackerFlowExtractor:
    """Dense optical flow rgb0→rgb1 using AllTracker (no flow quantizer needed)."""

    backend = "alltracker"

    def __init__(
        self,
        *,
        device: str = "cuda",
        alltracker_path: str = DEFAULT_ALLTRACKER_PATH,
        alltracker_checkpoint: Optional[str] = None,
        inference_iters: int = 4,
        ccwm_root: Path = DEFAULT_CCWM_ROOT,
    ) -> None:
        import torch

        self.device = device
        self.inference_iters = inference_iters
        self.torch = torch

        if str(ccwm_root) not in sys.path:
            sys.path.insert(0, str(ccwm_root))
        if alltracker_path not in sys.path:
            sys.path.insert(0, alltracker_path)

        from nets.alltracker import Net

        self.model = Net(16)
        if alltracker_checkpoint is None:
            url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        else:
            state_dict = torch.load(alltracker_checkpoint, map_location="cpu", weights_only=False)

        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"], strict=True)
        elif "state_dict" in state_dict:
            self.model.load_state_dict(state_dict["state_dict"], strict=True)
        else:
            self.model.load_state_dict(state_dict, strict=True)

        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _load_pair(
        path0: Path,
        path1: Path,
        *,
        short_side: int = 1024,
        device: str = "cuda",
    ):
        import torch
        import torchvision
        from PIL import Image

        to_tensor = torchvision.transforms.ToTensor()
        img0 = Image.open(path0).convert("RGB")
        img1 = Image.open(path1).convert("RGB")
        w, h = img0.size
        # Short-side resize (multiple of 16). Higher short_side (e.g. 1024 vs 512)
        # preserves small-object motion better for AllTracker.
        scale = short_side / float(min(h, w))
        th = int(round(h * scale / 16.0) * 16)
        tw = int(round(w * scale / 16.0) * 16)
        th = max(16, th)
        tw = max(16, tw)
        resize = torchvision.transforms.Resize((th, tw))
        f0 = resize(to_tensor(img0)).to(device)
        f1 = resize(to_tensor(img1)).to(device)
        return f0, f1, (h, w), (th, tw)

    def extract(self, path0: Path, path1: Path, *, short_side: int = 1024, **_: Any) -> np.ndarray:
        """Return flow HxWx2 in the *resized* model resolution (pixels)."""
        torch = self.torch
        with torch.no_grad():
            frame0, frame1, _orig_hw, _model_hw = self._load_pair(
                path0, path1, short_side=short_side, device=self.device
            )
            # AllTracker expects (B, 2, C, H, W) in [0, 255]
            frame_pairs = torch.stack([frame0, frame1], dim=0).unsqueeze(0)
            frame_pairs_255 = frame_pairs * 255.0
            model_dtype = next(self.model.parameters()).dtype
            if frame_pairs_255.dtype != model_dtype:
                frame_pairs_255 = frame_pairs_255.to(model_dtype)

            flows_e, _, _, _ = self.model.forward_sliding(
                frame_pairs_255, iters=self.inference_iters, sw=None, is_training=False
            )
            flows_e = flows_e.float()
            if flows_e.dim() == 5:
                flow = flows_e[:, -1]  # (1, 2, H, W)
            elif flows_e.dim() == 4:
                if flows_e.shape[1] == 2:
                    flow = flows_e
                elif flows_e.shape[-1] == 2:
                    flow = flows_e.permute(0, 3, 1, 2)
                else:
                    raise ValueError(f"Unexpected flow shape {flows_e.shape}")
            else:
                raise ValueError(f"Unexpected flow dim {flows_e.dim()}")

            # (H, W, 2)
            return flow[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


class SeaRAFTFlowExtractor:
    """Dense optical flow rgb0→rgb1 using Princeton SEA-RAFT (via ccwm wrapper)."""

    backend = "searaft"

    def __init__(
        self,
        *,
        device: str = "cuda",
        config_path: str = DEFAULT_SEARAFT_CONFIG,
        iters: int = 4,
        ccwm_root: Path = DEFAULT_CCWM_ROOT,
    ) -> None:
        import torch

        self.device = device
        self.iters = iters
        self.torch = torch

        if str(ccwm_root) not in sys.path:
            sys.path.insert(0, str(ccwm_root))

        from ccwm.sea_raft.custom import SeaRAFT

        self.model = SeaRAFT(config_path).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _round8(n: int) -> int:
        return max(8, int(round(n / 8.0) * 8))

    def extract(
        self,
        path0: Path,
        path1: Path,
        *,
        width: int = 800,
        height: int = 600,
        short_side: Optional[int] = None,
        **_: Any,
    ) -> np.ndarray:
        """Return flow HxWx2 at ``height×width`` (defaults match iTHOR 800×600).

        If ``short_side`` is set, resize so the short side matches (multiple of 8),
        preserving aspect ratio — useful when images are not 4:3.
        """
        import torchvision
        from PIL import Image

        torch = self.torch
        img0 = Image.open(path0).convert("RGB")
        img1 = Image.open(path1).convert("RGB")
        ow, oh = img0.size
        if short_side is not None:
            scale = float(short_side) / float(min(oh, ow))
            th = self._round8(int(round(oh * scale)))
            tw = self._round8(int(round(ow * scale)))
        else:
            th = self._round8(int(height))
            tw = self._round8(int(width))

        resize = torchvision.transforms.Resize((th, tw))
        to_tensor = torchvision.transforms.ToTensor()
        # SEA-RAFT expects [0, 255] RGB tensors (B, C, H, W).
        t0 = (resize(to_tensor(img0)) * 255.0).unsqueeze(0).to(self.device)
        t1 = (resize(to_tensor(img1)) * 255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda" if "cuda" in str(self.device) else "cpu",
                enabled="cuda" in str(self.device),
            ):
                out = self.model.model.forward(
                    t0, t1, iters=self.iters, test_mode=True
                )
            flow = out["flow"][-1].float()  # (1, 2, H, W)
            return flow[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def _make_colorwheel() -> np.ndarray:
    """Middlebury color wheel (same as ccwm.utils.flow.make_colorwheel)."""
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3))
    col = 0
    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(0, ry) / ry)
    col += ry
    colorwheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(0, yg) / yg)
    colorwheel[col : col + yg, 1] = 255
    col += yg
    colorwheel[col : col + gc, 1] = 255
    colorwheel[col : col + gc, 2] = np.floor(255 * np.arange(0, gc) / gc)
    col += gc
    colorwheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[col : col + cb, 2] = 255
    col += cb
    colorwheel[col : col + bm, 2] = 255
    colorwheel[col : col + bm, 0] = np.floor(255 * np.arange(0, bm) / bm)
    col += bm
    colorwheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[col : col + mr, 0] = 255
    return colorwheel


def flow_color_image(flow_hw2: np.ndarray) -> np.ndarray:
    """Middlebury-style RGB uint8 visualization (no ccwm/h5py dependency)."""
    assert flow_hw2.ndim == 3 and flow_hw2.shape[2] == 2
    u = flow_hw2[:, :, 0].astype(np.float64)
    v = flow_hw2[:, :, 1].astype(np.float64)
    rad = np.sqrt(u * u + v * v)
    rad_max = float(np.max(rad)) + 1e-5
    u = u / rad_max
    v = v / rad_max
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = _make_colorwheel()
    ncols = colorwheel.shape[0]
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad / rad_max <= 1
        col[idx] = 1 - (rad[idx] / rad_max) * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        flow_image[:, :, i] = np.floor(255 * col)
    return flow_image


def overlay_flow_arrows(
    flow_hw2: np.ndarray,
    img: np.ndarray | None = None,
    *,
    stride: int = 64,
) -> np.ndarray:
    """Overlay subsampled quiver arrows on a color flow image (PSI2 audit style).

    Uses matplotlib quiver when available (matches audit); falls back to cv2.
    """
    flow_hw2 = np.asarray(flow_hw2, dtype=np.float32)
    if img is None:
        img = flow_color_image(flow_hw2)
    else:
        img = np.asarray(img, dtype=np.uint8).copy()

    try:
        return _overlay_flow_arrows_mpl(flow_hw2, img, stride=stride)
    except ImportError:
        return _overlay_flow_arrows_cv2(flow_hw2, img, stride=stride)


def _overlay_flow_arrows_mpl(
    flow_hw2: np.ndarray, img: np.ndarray, *, stride: int = 64
) -> np.ndarray:
    import matplotlib.pyplot as plt

    u = flow_hw2[:, :, 0]
    v = flow_hw2[:, :, 1]
    h, w = img.shape[:2]

    ys = np.arange(stride // 2, h, stride)
    xs = np.arange(stride // 2, w, stride)
    xg, yg = np.meshgrid(xs, ys)
    u_s = u[np.ix_(ys, xs)]
    v_s = v[np.ix_(ys, xs)]

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img, interpolation="nearest")
    mag = np.sqrt(u_s**2 + v_s**2)
    ax.quiver(
        xg,
        yg,
        u_s,
        -v_s,
        mag,
        cmap="cool",
        clim=(0, mag.max() + 1e-6),
        scale_units="xy",
        scale=1,
        width=0.004,
        headwidth=4,
        headlength=5,
        alpha=0.85,
    )
    ax.axis("off")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    out = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return out


def _overlay_flow_arrows_cv2(
    flow_hw2: np.ndarray, img: np.ndarray, *, stride: int = 64
) -> np.ndarray:
    import cv2

    out = img.copy()
    u = flow_hw2[:, :, 0]
    v = flow_hw2[:, :, 1]
    h, w = out.shape[:2]
    mag_all = np.sqrt(u**2 + v**2)
    mag_max = float(mag_all.max()) + 1e-6

    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            ux, vy = float(u[y, x]), float(v[y, x])
            mag = np.sqrt(ux * ux + vy * vy)
            if mag < 0.5:
                continue
            t = mag / mag_max
            # approximate matplotlib "cool": blue -> cyan -> yellow
            b = int(255 * (1 - t))
            g = int(255 * min(1.0, 2 * t))
            r = int(255 * max(0.0, 2 * t - 1))
            x1, y1 = int(round(x + ux)), int(round(y + vy))
            cv2.arrowedLine(out, (x, y), (x1, y1), (r, g, b), 1, tipLength=0.35)
    return out


def flow_color_with_arrows(flow_hw2: np.ndarray, *, stride: int = 64) -> np.ndarray:
    """Middlebury color flow + audit-style quiver arrows."""
    base = flow_color_image(flow_hw2)
    return overlay_flow_arrows(flow_hw2, base, stride=stride)


def flow_arrow_overlay(
    rgb_path: Path,
    flow_hw2: np.ndarray,
    *,
    max_arrows: int = 64,
    percentile: float = 70.0,
) -> np.ndarray:
    """Draw significant flow arrows on rgb0 (audit_interleved_flows style)."""
    from PIL import Image, ImageDraw

    rgb = Image.open(rgb_path).convert("RGB")
    # Resize rgb to flow resolution for aligned arrows, then we keep that size.
    fh, fw = flow_hw2.shape[:2]
    rgb = rgb.resize((fw, fh), Image.BILINEAR)
    draw = ImageDraw.Draw(rgb)

    mag = np.sqrt(flow_hw2[..., 0] ** 2 + flow_hw2[..., 1] ** 2)
    thr = float(np.percentile(mag, percentile))
    ys, xs = np.where((mag > thr) & (mag > 0.5))
    if len(ys) > max_arrows:
        idx = np.random.choice(len(ys), max_arrows, replace=False)
        ys, xs = ys[idx], xs[idx]

    for y, x in zip(ys.tolist(), xs.tolist()):
        fx = float(flow_hw2[y, x, 0])
        fy = float(flow_hw2[y, x, 1])
        x0, y0 = int(x), int(y)
        x1, y1 = int(x + fx), int(y + fy)
        draw.line([(x0, y0), (x1, y1)], fill=(220, 30, 30), width=2)
        # Small arrow head
        draw.ellipse([x1 - 2, y1 - 2, x1 + 2, y1 + 2], fill=(255, 220, 40))
    return np.asarray(rgb)


def save_flow_pair(
    extractor: Any,
    rgb0: Path,
    rgb1: Path,
    out_flow: Path,
    out_arrows: Path,
    *,
    short_side: int = 1024,
    width: int = 800,
    height: int = 600,
) -> dict[str, Any]:
    """Extract flow and write ``*_flow.png`` + ``*_flow_arrows.png``."""
    from PIL import Image

    backend = getattr(extractor, "backend", "alltracker")
    if backend == "searaft":
        flow = extractor.extract(rgb0, rgb1, width=width, height=height)
        size_meta: dict[str, Any] = {"width": int(width), "height": int(height)}
    else:
        flow = extractor.extract(rgb0, rgb1, short_side=short_side)
        size_meta = {"short_side": int(short_side)}

    color = flow_color_image(flow)
    arrows = flow_arrow_overlay(rgb0, flow)
    Image.fromarray(color).save(out_flow)
    Image.fromarray(arrows).save(out_arrows)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return {
        "flow_png": out_flow.name,
        "flow_arrows_png": out_arrows.name,
        "flow_h": int(flow.shape[0]),
        "flow_w": int(flow.shape[1]),
        "backend": backend,
        "mean_mag_px": float(mag.mean()),
        "max_mag_px": float(mag.max()),
        **size_meta,
    }
