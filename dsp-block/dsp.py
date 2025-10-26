# dsp.py
from typing import Any, Dict, List, Tuple
import numpy as np

# Lazy import so the server can start and give a clear error if OpenCV is missing
try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError(
        "OpenCV is not available in the container. "
        "Ensure 'opencv-python-headless' is installed in the Docker image."
    ) from e


# ---------- Utilities ----------


def _as_int(name: str, value: Any, lo: int = 1, hi: int = 8192) -> int:
    try:
        iv = int(value)
    except Exception:
        raise ValueError(f"Parameter '{name}' must be an integer, got: {value!r}")
    if not (lo <= iv <= hi):
        raise ValueError(f"Parameter '{name}' out of range [{lo},{hi}]: {iv}")
    return iv


def _infer_channels(axes: List[str] | None, raw_len: int) -> int:
    if axes:
        a = [str(x).lower() for x in axes]
        if {"r", "g", "b"}.issubset(set(a)):
            return 3
        if any(x in {"px", "gray", "grey", "grayscale"} for x in a) and len(a) == 1:
            return 1
        if len(a) in (1, 3):
            return len(a)
    if raw_len % 3 == 0:
        return 3
    return 1


def _best_wh_from_pixels(pixels: int) -> Tuple[int, int]:
    pairs: List[Tuple[int, int]] = []
    r = int(np.sqrt(pixels))
    for h in range(1, r + 1):
        if pixels % h == 0:
            w = pixels // h
            pairs.append((h, w))
    if not pairs:
        s = int(np.sqrt(pixels))
        return max(1, s), max(1, s)
    targets = [1.0, 4 / 3, 16 / 9, 3 / 4, 9 / 16]

    def score(hw: Tuple[int, int]) -> float:
        h, w = hw
        ratio = w / h
        aspect_err = min(abs(ratio - t) for t in targets)
        return aspect_err + 1e-6 * (1 / h)

    best = min(pairs, key=score)
    return best


def _ensure_float01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


# ---------- Main hook called by EI server ----------


def generate_features(
    implementation_version: int,
    draw_graphs: bool,
    raw_data: List[float] | np.ndarray,
    axes: List[str] | None,
    sampling_freq: float | None,
    img_width: int | None = None,
    img_height: int | None = None,
    **kwargs,
) -> Dict[str, Any]:

    # ---- Read target size from parameters.json ----
    Wt = _as_int("img_width", kwargs.get("img-width", 96))
    Ht = _as_int("img_height", kwargs.get("img-height", 96))

    # ---- Flatten & basic info ----
    raw = np.asarray(raw_data)
    flat = raw.reshape(-1) if raw.ndim > 1 else raw

    # ---- Determine input channels (prefer explicit) ----
    Cin = (
        kwargs.get("channels")
        or kwargs.get("input_channels")
        or _infer_channels(axes, flat.size)
    )
    if Cin not in (1, 3):
        Cin = 3 if flat.size % 3 == 0 else 1

    # ---- Determine input W,H (prefer explicit) ----
    Win = kwargs.get("width") or kwargs.get("input_width")
    Hin = kwargs.get("height") or kwargs.get("input_height")

    if Win is None or Hin is None:
        if flat.size % Cin != 0:
            raise ValueError(
                f"Incoming data length {flat.size} not divisible by channels={Cin}"
            )
        pixels = flat.size // Cin
        Hin, Win = _best_wh_from_pixels(pixels)

    Win = _as_int("input_width", Win)
    Hin = _as_int("input_height", Hin)

    # ---- Reshape to HxWxC ----
    try:
        img = flat.astype(np.float32).reshape(Hin, Win, Cin)
    except Exception as e:
        raise ValueError(
            f"Could not reshape raw data to (H={Hin}, W={Win}, C={Cin}): {e}"
        )

    # ---- Resize (OpenCV expects (width, height)) ----
    if Cin == 1:
        resized = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[..., None]  # keep shape (H, W, 1)
    else:
        resized = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_AREA)

    # ---- Scale to [0,1] ----
    resized = _ensure_float01(resized)

    # Return the 3D features as a nested list.
    features = resized.astype(np.float32)

    # ----- Build output config -----
    output_config = {
        "type": "image",
        "shape": {"width": Wt, "height": Ht, "channels": Cin},
    }

    return {
        "features": features,
        "graphs": [],
        "fft_used": [],
        "output_config": output_config,
    }
