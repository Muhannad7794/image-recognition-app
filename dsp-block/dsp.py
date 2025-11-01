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


def _factor_pairs(n: int, limit: int = 50) -> List[Tuple[int, int]]:
    """Return up to `limit` (H,W) factor pairs of n, biased toward near-square/aspect-y pairs."""
    pairs: List[Tuple[int, int]] = []
    r = int(np.sqrt(n))
    for h in range(1, r + 1):
        if n % h == 0:
            w = n // h
            pairs.append((h, w))
    if not pairs:
        return []
    # rank by aspect closeness to common targets & larger H
    targets = [1.0, 4 / 3, 16 / 9, 3 / 4, 9 / 16]

    def score(hw: Tuple[int, int]) -> float:
        h, w = hw
        ratio = w / h
        aspect_err = min(abs(ratio - t) for t in targets)
        return aspect_err + 1e-6 * (1 / h)

    pairs.sort(key=score)
    return pairs[:limit]


def _ensure_float01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    if a.size == 0:
        return a
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def _try_reshape_and_resize(
    flat: np.ndarray, Hin: int, Win: int, Cin: int, Wt: int, Ht: int
) -> Tuple[np.ndarray, float]:
    """Return resized image and its std; std<0 means failure."""
    try:
        img = flat.astype(np.float32).reshape(Hin, Win, Cin)
    except Exception:
        return np.empty((0,)), -1.0
    res = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_AREA)
    if res.ndim == 2:
        res = res[..., None]
    return res, float(np.nan_to_num(np.std(res), nan=-1.0))


# ---------- Main hook ----------


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

    # ---- Output size & channels (contract to model) ----
    # accept both snake_case and kebab-case keys
    Wt = _as_int("img_width", kwargs.get("img_width", kwargs.get("img-width", 160)))
    Ht = _as_int("img_height", kwargs.get("img_height", kwargs.get("img-height", 160)))
    Cout = 3 if int(kwargs.get("out_channels", 3)) == 3 else 1  # sanitize {1,3}

    # ---- Flatten incoming buffer ----
    flat = np.asarray(raw_data).reshape(-1)
    N = int(flat.size)

    # ---- NEW: Detect and unpack packed 24-bit RGB (0xRRGGBB per pixel) ----
    # Only do this if we are not already clearly 3-channel bytes (N % 3 == 0) and values look like big ints.
    if N > 0 and (N % 3 != 0):
        try:
            mx = float(np.nanmax(flat))
            mn = float(np.nanmin(flat))
        except Exception:
            mx = 0.0
            mn = 0.0

        # Heuristics: non-negative, reasonably large integers (typical for 0xRRGGBB),
        # and not already tiny byte-like values.
        if mn >= 0.0 and mx > 4096 and mx <= 0xFFFFFF and np.all(np.mod(flat, 1) == 0):
            u = flat.astype(np.uint32)
            r = ((u >> 16) & 0xFF).astype(np.float32)
            g = ((u >> 8) & 0xFF).astype(np.float32)
            b = (u & 0xFF).astype(np.float32)
            # Expand packed ints to interleaved RGB bytes
            flat = np.stack([r, g, b], axis=1).reshape(-1)
            N = int(flat.size)

    # ---- Input channels (robust) ----
    Cin_pref = kwargs.get("channels") or kwargs.get("input_channels")
    if Cin_pref in (1, 3):
        Cin_pref = int(Cin_pref)
        Cin_candidates = [Cin_pref] + ([3] if Cin_pref != 3 else [1])
    else:
        Cin_candidates = [_infer_channels(axes, N), 3, 1]

    # ---- Optional size hints (prefer EI function args) ----
    Win_hint = kwargs.get("width") or kwargs.get("input_width")
    Hin_hint = kwargs.get("height") or kwargs.get("input_height")

    # ---- Build candidate (Cin, Hin, Win) tuples ----
    tried = set()
    candidates: List[Tuple[int, int, int]] = []

    for Cin in Cin_candidates:
        if Cin not in (1, 3) or (N % Cin != 0):
            continue
        pixels = N // Cin

        # 1) If we have explicit hints, try them first
        if Hin_hint and Win_hint:
            Hin_t = _as_int("input_height", Hin_hint)
            Win_t = _as_int("input_width", Win_hint)
            key = (Cin, Hin_t, Win_t)
            if key not in tried:
                candidates.append(key)
                tried.add(key)

        # 2) Add up to K factor pairs as fallbacks
        for Hin_t, Win_t in _factor_pairs(pixels, limit=40):
            key = (Cin, Hin_t, Win_t)
            if key not in tried:
                candidates.append(key)
                tried.add(key)

    if not candidates:
        raise ValueError(
            f"Cannot determine image shape: len={N} not factorable with Cin∈{{1,3}} "
            f"(channels param={kwargs.get('channels')!r}, hints W×H={Win_hint}×{Hin_hint})"
        )

    # ---- Evaluate candidates; pick highest-variance reconstruction ----
    best_res, best_std, best_cfg = None, -1.0, None
    for Cin, Hin, Win in candidates:
        res, s = _try_reshape_and_resize(flat, Hin, Win, Cin, Wt, Ht)
        if s > best_std:
            best_res, best_std, best_cfg = res, s, (Cin, Hin, Win)
        # Early exit if variance is clearly non-degenerate
        if s > 1e-3:
            break

    if best_res is None or best_std < 0:
        raise ValueError(
            f"All reshape candidates failed for N={N}. Tried {len(candidates)} options; "
            f"hints W×H={Win_hint}×{Hin_hint}, Cin candidates={Cin_candidates}"
        )

    # ---- Convert channels to match OUTPUT contract ----
    resized = best_res
    if resized.shape[2] == 1 and Cout == 3:
        resized = np.repeat(resized, 3, axis=2)
    elif resized.shape[2] == 3 and Cout == 1:
        y = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
        resized = y.astype(np.float32)[..., None]

    # ---- Normalize & flatten ----
    resized = _ensure_float01(resized)
    features = resized.astype(np.float32).reshape(-1).tolist()

    # ---- Final length check ----
    expected = Wt * Ht * Cout
    if len(features) != expected:
        raise ValueError(
            f"Feature length mismatch: got {len(features)} vs expected {expected} "
            f"(width={Wt}, height={Ht}, channels={Cout}, picked_cfg={best_cfg}, std={best_std:.6f})"
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
