"""Load saved simulation arrays and metadata."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

import numpy as np

from .pulse import HSPulse
from .simulate import BlochSimResult


@dataclass
class LoadedSim:
    """
    Container for a loaded simulation file.

    - result: BlochSimResult(raw_mat, t_s, used_workers, path)
    - arrays: raw arrays from the npz
    - meta:   dict from JSON if present; {} otherwise
    - pulse:  HSPulse if it can be reconstructed; else None
    - pulse_arrays: dict of pulse arrays saved under "pulse__*" (always provided; may be empty)
    """
    result: BlochSimResult
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any]
    pulse: Optional[HSPulse]
    pulse_arrays: dict[str, np.ndarray]
    path: str
    filename: str
    path_stem: str
    
    # Convenience accessors
    @property
    def raw_mat(self) -> np.ndarray:
        return self.result.raw_mat

    @property
    def t_s(self) -> np.ndarray:
        return self.result.t_s


def _load_json(npz_path: str) -> dict[str, Any]:
    json_path = os.path.splitext(npz_path)[0] + ".json"
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Do not fail loading the sim because JSON is missing/corrupt
        return {}


def _extract_pulse_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return dict of pulse arrays saved as pulse__<name> (name stripped)."""
    out: dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        if k.startswith("pulse__"):
            out[k[len("pulse__"):]] = v
    return out


def _try_make_pulse(pulse_arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> Optional[HSPulse]:
    """
    Best-effort reconstruction of HSPulse.
    We only construct it if the required fields are present.
    """
    # Required array fields for HSPulse dataclass
    required_arrays = [
        "t_s",
        "B1xy_G",
        "x_rf_rad_s",
        "fn",
        "omega1_rad_s",
        "omegaRF_rad_s",
        "phi_rad",
    ]
    if not all(k in pulse_arrays for k in required_arrays):
        return None

    # Scalars: try pulse_scalars from meta first; fall back to meta keys; then reasonable defaults.
    pulse_scalars = {}
    if isinstance(meta.get("pulse_scalars", None), dict):
        pulse_scalars.update(meta["pulse_scalars"])

    def _pick(name: str, default: Any) -> Any:
        if name in pulse_scalars:
            return pulse_scalars[name]
        if name in meta:
            return meta[name]
        return default

    try:
        return HSPulse(
            t_s=np.asarray(pulse_arrays["t_s"], dtype=float),
            B1xy_G=np.asarray(pulse_arrays["B1xy_G"], dtype=np.complex128),
            x_rf_rad_s=np.asarray(pulse_arrays["x_rf_rad_s"], dtype=np.complex128),
            fn=np.asarray(pulse_arrays["fn"], dtype=float),
            omega1_rad_s=np.asarray(pulse_arrays["omega1_rad_s"], dtype=float),
            omegaRF_rad_s=np.asarray(pulse_arrays["omegaRF_rad_s"], dtype=float),
            phi_rad=np.asarray(pulse_arrays["phi_rad"], dtype=float),
            flip_angle_deg=float(_pick("flip_angle_deg", np.nan)),
            rel_rf_power=float(_pick("rel_rf_power", np.nan)),
            duration_s=float(_pick("duration_s", np.nan)),
            n_points=int(_pick("n_points", int(pulse_arrays["t_s"].size))),
            lhs_rad_s2=float(_pick("lhs_rad_s2", np.nan)),
            rhs_Hz2=float(_pick("rhs_Hz2", np.nan)),
            adiabatic_ratio=float(_pick("adiabatic_ratio", np.nan)),
        )
    except Exception:
        # If anything is inconsistent, just don't construct the object
        return None


def load_sim_npz(path: str) -> LoadedSim:
    """
    Load a Bloch simulation .npz saved by simulate.simulate_radial_bloch(save=True).

    Pulse arrays and JSON metadata are loaded when present.

    Returns a LoadedSim wrapper with:
      - result: BlochSimResult
      - arrays/meta: raw data
      - pulse: HSPulse if reconstructable
      - pulse_arrays: dict of pulse arrays (phi_rad, omegaRF_rad_s, etc. if saved)
    """
    p = Path(path)
    full_path = str(p)
    filename = p.name
    stem = p.stem

    npz_path = os.fspath(path)

    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    # Required simulation arrays
    if "raw_mat" not in arrays or "t_s" not in arrays:
        raise ValueError(f"{npz_path} does not look like a simulation file (missing raw_mat/t_s).")

    meta = _load_json(npz_path)

    # Read the saved worker count, or use 1 when it was not recorded.
    used_workers = int(meta.get("used_workers", 1)) if isinstance(meta.get("used_workers", 1), (int, float)) else 1

    result = BlochSimResult(
        raw_mat=np.asarray(arrays["raw_mat"], dtype=np.complex128),
        t_s=np.asarray(arrays["t_s"], dtype=float),
        used_workers=used_workers,
        path=npz_path,
    )

    pulse_arrays = _extract_pulse_arrays(arrays)
    pulse = _try_make_pulse(pulse_arrays, meta)

    return LoadedSim(
        path=full_path,
        filename=filename,
        path_stem=stem,
        result=result,
        arrays=arrays,
        meta=meta,
        pulse=pulse,
        pulse_arrays=pulse_arrays,
    )
