# dsp.py
from typing import Any, Dict, List, Tuple
import numpy as np

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
    return 3 if (raw_len % 3 == 0) else 1


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

    return min(pairs, key=score)


def _ensure_float01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    if a.size == 0:
        return a
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


# Try a candidate (Cin, Hin, Win) and return resized image and its std
def _try_candidate(
    flat: np.ndarray, Cin: int, Hin: int, Win: int, Wt: int, Ht: int
) -> Tuple[np.ndarray, float]:
    try:
        img = flat.astype(np.float32).reshape(Hin, Win, Cin)
    except Exception:
        return np.empty((0,)), -1.0
    res = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_AREA)
    if res.ndim == 2:
        res = res[..., None]
    std = float(np.std(res))
    return res, std


# ---------- Main hook ----------


def generate_features(
    implementation_version: int,
    draw_graphs: bool,
    raw_data: List[float] | np.ndarray,
    axes: List[str] | None,
    sampling_freq: float | None,
    img_width: int | None = None,  # EI may pass these; prefer when present
    img_height: int | None = None,
    **kwargs,
) -> Dict[str, Any]:

    # Target output size
    Wt = _as_int("img_width", kwargs.get("img-width", 96))
    Ht = _as_int("img_height", kwargs.get("img-height", 96))

    # Desired OUTPUT channels (guaranteed)
    Cout = 3 if int(kwargs.get("out_channels", 3)) == 3 else 1

    # Flatten input
    flat = np.asarray(raw_data).reshape(-1)
    N = int(flat.size)

    # INPUT channels preference
    Cin_pref = kwargs.get("channels") or kwargs.get("input_channels")
    if Cin_pref in (1, 3):
        Cin_pref = int(Cin_pref)
    else:
        Cin_pref = _infer_channels(axes, N)

    # INPUT size (prefer EI-provided function args first)
    Win_kw = kwargs.get("width") or kwargs.get("input_width")
    Hin_kw = kwargs.get("height") or kwargs.get("input_height")
    Win_in = Win_kw or img_width
    Hin_in = Hin_kw or img_height

    # Build candidate list for Cin ∈ {1,3} that divide N
    candidates = []
    for Cin_try in (Cin_pref, 3 if Cin_pref != 3 else 1):
        if Cin_try in (1, 3) and N % Cin_try == 0:
            pixels = N // Cin_try
            Hin_try = _as_int("input_height", Hin_in) if Hin_in else None
            Win_try = _as_int("input_width", Win_in) if Win_in else None
            if Hin_try is None or Win_try is None:
                Hin_try, Win_try = _best_wh_from_pixels(pixels)
            candidates.append((Cin_try, Hin_try, Win_try))

    # If nothing divisible, force Cin=1 as last resort
    if not candidates:
        Cin_try = 1
        pixels = N // Cin_try
        Hin_try, Win_try = _best_wh_from_pixels(pixels)
        candidates.append((Cin_try, Hin_try, Win_try))

    # Evaluate candidates; pick the one with highest variance after resize
    best_resized, best_std, best_cfg = None, -1.0, None
    for Cin_try, Hin_try, Win_try in candidates:
        res, s = _try_candidate(flat, Cin_try, Hin_try, Win_try, Wt, Ht)
        if s > best_std:
            best_resized, best_std, best_cfg = res, s, (Cin_try, Hin_try, Win_try)

    # Use best
    resized = (
        best_resized if best_resized is not None else np.zeros((Ht, Wt, 1), np.float32)
    )

    # Convert to requested OUTPUT channels
    if resized.shape[2] == 1 and Cout == 3:
        resized = np.repeat(resized, 3, axis=2)
    elif resized.shape[2] == 3 and Cout == 1:
        y = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
        resized = y.astype(np.float32)[..., None]

    # Normalize and flatten
    resized = _ensure_float01(resized)
    features = resized.astype(np.float32).reshape(-1).tolist()

    # Final length check (prevents UI product errors)
    expected = Wt * Ht * Cout
    if len(features) != expected:
        raise ValueError(
            f"Feature length mismatch: got {len(features)} vs expected {expected} "
            f"(width={Wt}, height={Ht}, channels={Cout}, cfg={best_cfg}, std={best_std:.6f})"
        )

    output_config = {
        "type": "image",
        "shape": {"width": Wt, "height": Ht, "channels": Cout},
    }

    return {
        "features": features,
        "graphs": [],
        "fft_used": [],
        "output_config": output_config,
    }
