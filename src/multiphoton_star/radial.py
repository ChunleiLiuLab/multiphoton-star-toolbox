"""Radial gradient and k-space trajectory construction."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

try:
    # Optional import; only needed if you want to pass a pulse object directly
    from .pulse import HSPulse
except Exception:
    HSPulse = None  # type: ignore

@dataclass
class RadialTrajectory:
    # Setup / metadata
    dims: int                          # 2 or 3
    FOV_cm: Tuple[float, ...]          # (FOVx, FOVy[, FOVz]) in cm
    bw_hz: float                       # acquisition (readout) bandwidth [Hz]
    Tp_s: float                        # readout duration [s]
    dt_s: float                        # time step [s]
    M: int                             # samples per spoke
    N_spokes: int
    Gmax_G_per_cm: float               # gradient amplitude per axis along spoke [G/cm]
    Wk_cycles_per_cm: float            # encoded k-space diameter = gamma_bar*Gmax*Tp [cycles/cm]
    resolution_cm: float               # nominal resolution = 1/Wk [cm]

    # Directions and waveforms
    directions: np.ndarray             # (N_spokes, 3) unit vectors
    grads_G_per_cm: np.ndarray         # (N_spokes, 3, M) gradient waveform [G/cm]
    k_traj_cyc_per_cm: np.ndarray      # (N_spokes, 3, M) k-space trajectory [cycles/cm]

    # Time base (convenience)
    t_s: np.ndarray                    # (M,)

    # Samples per spoke used for reconstruction (t < Tp/2); None uses all M
    encoded_samples: Optional[int] = None

def _normalize_fov(FOV_cm, dims: int) -> Tuple[float, ...]:
    if np.isscalar(FOV_cm):
        FOV_cm = (float(FOV_cm),) * dims
    FOV_cm = tuple(float(x) for x in FOV_cm)
    assert len(FOV_cm) == dims, f"FOV must have {dims} values (cm)."
    return FOV_cm

def _phi_uniform(N: int) -> np.ndarray:
    return np.linspace(0.0, 2.0*np.pi, N, endpoint=False)

def generate_radial_gradients(
    *,
    bw_hz: float,                         # acquisition bandwidth [Hz]
    FOV_cm: float | Tuple[float, ...],    # cm; scalar -> square/cube
    dims: Literal[2, 3] = 2,              # 2D radial; 3D requires supplied directions
    pulse: Optional["HSPulse"] = None,# optional: infer t_s, Tp from your RF pulse
    t_s: Optional[np.ndarray] = None,     # or provide timebase directly
    dt_s: Optional[float] = None,         # (required if t_s not given)
    Tp_s: Optional[float] = None,         # (required if t_s not given)
    N_spokes: Optional[int] = None,       # override number of spokes
    directions: Optional[np.ndarray] = None,  # optional (N_spokes, dims) unit vectors
    Gmax_G_per_cm: Optional[float] = None,     # override gradient amplitude
    gamma_bar_Hz_per_G: float = 4258.0,
    scale_Gmax: float = 1.0,              # scale factor applied to the default Gmax
    verbose: bool = True,
) -> RadialTrajectory:
    """
    Build constant-amplitude radial gradient spokes and k-space trajectories.

    Defaults:
    - Timebase is taken from `pulse.t_s` if an HSPulse is provided.
      Otherwise, provide `t_s` OR (`dt_s` and `Tp_s`).
    - Gmax is chosen so that the swept bandwidth spans twice FOV_min, keeping
      the object within the central half of the frequency sweep (as in
      continuous SWIFT):
        Gmax = bw / (2 * gamma_bar * FOV_min)
      You can override with Gmax_G_per_cm (e.g., to hardware-limit or scale down).
    - Each isochromat is encoded only from the time the sweep reaches it until
      the end of the pulse, so a line through k-space spans
      Wk = gamma_bar * Gmax * Tp and the resolution is dx = 1/Wk
      (= 2 * FOV_min / R for the default Gmax). Reconstruction uses the first
      half of each spoke (t < Tp/2, `encoded_samples`).
    - Default 2D spokes are uniformly spaced over a full circle, with
      N = ceil(PI * FOV_min / dx) (= ceil(PI * R / 2) for the default Gmax).
    - For 3D, provide unit direction vectors with shape (N_spokes, 3).
      Optional 2D direction vectors have shape (N_spokes, 2).
        """
    # --- timebase ---
    if pulse is not None:
        t_s = pulse.t_s
        dt_s = float(np.median(np.diff(t_s)))
        Tp_s = float(t_s[-1] + dt_s) if t_s.size > 0 else 0.0
    elif t_s is not None:
        t_s = np.asarray(t_s, float)
        if dt_s is None:
            dt_s = float(np.median(np.diff(t_s)))
        if Tp_s is None:
            Tp_s = float(t_s[-1] + dt_s) if t_s.size > 0 else 0.0
    else:
        assert dt_s is not None and Tp_s is not None, \
            "Provide either `pulse`, or `t_s`, or both `dt_s` and `Tp_s`."
        t_s = np.arange(0.0, Tp_s, dt_s, dtype=float)

    dt_s = float(dt_s)
    Tp_s = float(Tp_s)
    M = int(t_s.size)

    # --- FOV handling ---
    dims = int(dims)
    assert dims in (2, 3), "dims must be 2 or 3."
    FOV_cm = _normalize_fov(FOV_cm, dims)
    FOV_min = float(min(FOV_cm))

    # --- pick Gmax (or use override) ---
    if Gmax_G_per_cm is None:
        # Swept bandwidth spans 2*FOV_min (object in the central half of the sweep):
        #   Gmax = bw / (2 * gamma_bar * FOV_min)
        Gmax = (bw_hz) / (2.0 * gamma_bar_Hz_per_G * FOV_min)
        Gmax *= float(scale_Gmax)
    else:
        Gmax = float(Gmax_G_per_cm)

    # --- encoded k-space diameter and nominal resolution ---
    # Each isochromat is encoded from its sweep-passage time to Tp; opposite
    # spokes together span gamma_bar*Gmax*Tp along any line through k = 0.
    Wk = gamma_bar_Hz_per_G * Gmax * Tp_s                           # [cycles/cm]
    dx_cm = 1.0 / Wk if Wk > 0 else np.inf
    encoded_samples = M // 2                                        # t < Tp/2

    # Use supplied unit vectors, or generate uniform 2D directions.
    if directions is not None:
        dirs = np.asarray(directions, dtype=float)
        if dirs.ndim != 2 or dirs.shape[1] != dims or dirs.shape[0] == 0:
            raise ValueError("directions must have shape (N_spokes, dims).")
        if not np.all(np.isfinite(dirs)) or not np.allclose(np.linalg.norm(dirs, axis=1), 1.0):
            raise ValueError("directions must be finite unit vectors.")
        if N_spokes is not None and N_spokes != dirs.shape[0]:
            raise ValueError("N_spokes must match the number of supplied directions.")
        N_spokes = dirs.shape[0]
        if dims == 2:
            dirs = np.column_stack([dirs, np.zeros(N_spokes)])
    else:
        if dims == 3:
            raise ValueError("Provide directions for a 3D trajectory.")
        if N_spokes is None:
            central_angle = max(2.0 * dx_cm / FOV_min, 1e-6)
            N_spokes = int(np.ceil(2.0 * np.pi / central_angle))
        N_spokes = int(max(1, N_spokes))
        phis = _phi_uniform(N_spokes)
        dirs = np.stack([np.cos(phis), np.sin(phis), np.zeros_like(phis)], axis=1)

    # --- gradient waveforms: constant along each spoke ---
    grads = np.zeros((N_spokes, 3, M), dtype=float)
    for i in range(N_spokes):
        grads[i] = Gmax * dirs[i][:, None]   # [G/cm]

    # --- integrate to get k-space trajectory (cycles/cm) ---
    #   k(t) = ∫ γ_bar * G(t) dt
    k_traj = np.cumsum(grads * gamma_bar_Hz_per_G * dt_s, axis=2)

    if verbose:
        print(f"[Radial] dims={dims}D, FOV={FOV_cm} cm, bw={bw_hz:.3g} Hz")
        print(f"         Tp={Tp_s*1e3:.3f} ms, dt={dt_s*1e3:.3f} ms, M={M}")
        print(f"         Gmax={Gmax:.6g} G/cm, Wk={Wk:.6g} cycles/cm, res≈{dx_cm:.6g} cm")
        print(f"         N_spokes={N_spokes}")

    return RadialTrajectory(
        dims=dims,
        FOV_cm=FOV_cm,
        bw_hz=float(bw_hz),
        Tp_s=Tp_s,
        dt_s=dt_s,
        M=M,
        N_spokes=N_spokes,
        Gmax_G_per_cm=Gmax,
        Wk_cycles_per_cm=Wk,
        resolution_cm=dx_cm,
        directions=dirs,
        grads_G_per_cm=grads,
        k_traj_cyc_per_cm=k_traj,
        t_s=t_s,
        encoded_samples=encoded_samples,
    )
