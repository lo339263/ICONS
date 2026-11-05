"""CIELAB Gaussian naming and compact appearance labels."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def _model():
    data = json.loads((Path(__file__).parent / "models/gaussian.json").read_text())["lab"]
    cov = np.asarray(data["covariances"])
    return (data["classes"], np.asarray(data["means"]), np.linalg.inv(cov),
            np.log(data["priors"]) - 0.5 * np.linalg.slogdet(cov)[1])


def classify_cielab_colors(lab_data: list[float] | tuple[float, ...]) -> str:
    """Return the class with highest prior-weighted Gaussian density."""
    lab = np.asarray(lab_data, dtype=float)
    if lab.shape != (3,) or not np.isfinite(lab).all():
        raise ValueError("lab_data must contain three finite values")
    names, means, precision, bias = _model()
    delta = lab - means
    scores = bias - 0.5 * np.einsum("ij,ijk,ik->i", delta, precision, delta)
    return names[int(np.argmax(scores))]


def get_depth_description(D: float) -> str:
    for threshold, name in zip((20, 40, 60, 80), ("Very pale", "Pale", "Mid depth", "Deep")):
        if D < threshold:
            return name
    return "Very Deep"


def get_ncs_hue_name(H: float) -> str:
    H %= 400
    quadrant = int(H // 100)
    if H % 100 == 0:
        return "RYGB"[quadrant]
    return f"{'YGBR'[quadrant]}{round(100 - H % 100)}{'RYGB'[quadrant]}"
