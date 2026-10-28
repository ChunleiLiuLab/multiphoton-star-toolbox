"""Small analytic phantom helpers for simulation examples."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Phantom2D:
    # Image/grid view
    img: np.ndarray                  # (Ny, Nx), float
    X_cm: np.ndarray                 # (Ny, Nx)
    Y_cm: np.ndarray                 # (Ny, Nx)
    FOV_cm: Tuple[float, float]      # (FOVx, FOVy)

    # Bloch-ready flattened view (masked)
    positions_cm: np.ndarray         # (2, P)
    values: np.ndarray               # (P,)
    mx0: np.ndarray                  # (P,)
    my0: np.ndarray                  # (P,)
    mz0: np.ndarray                  # (P,)

# ---------- Analytic Shepp–Logan ----------
def _shepp_logan_2d(Ny: int, Nx: Optional[int] = None, *, dtype=np.float32) -> np.ndarray:
    """Analytic Shepp–Logan phantom on [-1,1]×[-1,1]."""
    if Nx is None:
        Nx = Ny

    # Ellipse parameters from P. A. Toft, The Radon Transform - Theory and
    # Implementation, PhD thesis, Technical University of Denmark, 1996.
    ellipses = [
        [ 1.0,   .69, .92,   0.0,    0.0,   0.0],
        [-0.8,   .6624, .8740, 0.0,  -0.0184,  0.0],
        [-0.2,   .1100, .3100, 0.22,  0.0,  -18.0],
        [-0.2,   .1600, .4100,-0.22,  0.0,   18.0],
        [ 0.1,   .2100, .2500, 0.0,   0.35,  0.0],
        [ 0.1,   .0460, .0460, 0.0,   0.1,   0.0],
        [ 0.1,   .0460, .0460, 0.0,  -0.1,   0.0],
        [ 0.1,   .0460, .0230,-0.08, -0.605, 0.0],
        [ 0.1,   .0230, .0230, 0.0,  -0.606, 0.0],
        [ 0.1,   .0230, .0460, 0.06, -0.605, 0.0],
    ]

    y = np.linspace(-1, 1, Ny, dtype=dtype)
    x = np.linspace(-1, 1, Nx, dtype=dtype)
    X, Y = np.meshgrid(x, y, indexing='xy')
    P = np.zeros((Ny, Nx), dtype=dtype)

    for I, a, b, x0, y0, phi in ellipses:
        th = np.deg2rad(phi)
        c, s = np.cos(th), np.sin(th)
        xr =  (X - x0) * c + (Y - y0) * s
        yr = -(X - x0) * s + (Y - y0) * c
        mask = (xr / a)**2 + (yr / b)**2 <= 1.0
        P[mask] += I

    # Normalize to [0,1] for convenience
    P -= P.min()
    if P.max() > 0:
        P /= P.max()
    return P

def shepp_logan_bloch(
    Ny: int,
    Nx: Optional[int] = None,
    *,
    FOV_cm: Tuple[float, float] = (4.0, 4.0),
    threshold: float = 1e-6,
) -> Phantom2D:
    """
    Build a Shepp–Logan phantom image and Bloch-ready flattened arrays.
    - Ny, Nx: image size in pixels
    - FOV_cm: (FOVx, FOVy) in cm mapping the image to physical coords
    - threshold: drop pixels <= threshold from the Bloch particle set
    """
    if Nx is None:
        Nx = Ny

    img = _shepp_logan_2d(Ny, Nx, dtype=np.float32)

    FOVx, FOVy = float(FOV_cm[0]), float(FOV_cm[1])
    x = np.linspace(-FOVx/2, FOVx/2, Nx)
    y = np.linspace(-FOVy/2, FOVy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Flatten & mask
    positions = np.vstack([X.ravel(), Y.ravel()])      # (2, P_all)
    values = img.ravel().astype(float)                 # (P_all,)
    keep = values > float(threshold)
    positions = positions[:, keep]
    values = values[keep]

    mx0 = np.zeros_like(values)
    my0 = np.zeros_like(values)
    mz0 = values.copy()

    return Phantom2D(
        img=img, X_cm=X, Y_cm=Y, FOV_cm=(FOVx, FOVy),
        positions_cm=positions, values=values, mx0=mx0, my0=my0, mz0=mz0
    )

# ---------- Circle phantom ----------
def circle_bloch(
    center_cm: Tuple[float, float] = (0.0, 0.0),
    radius_cm: float = 2.0,
    *,
    step_cm: float = 0.03,
    edge_width_cm: float = 0.0,         # 0 --> hard edge; >0 --> cosine taper
    FOV_pad_cm: float = 0.0,            # extra pad beyond bounding box for the output image
) -> Phantom2D:
    """
    Generate a filled circle phantom with optional cosine edge roll-off (reduces ringing artifacts).
    Returns both an image on a Cartesian grid and the Bloch-ready (positions, values).

    - Grid is built with spacing `step_cm` over the minimal bounding square (± FOV_pad_cm).
    - values = 1.0 inside, smoothly tapering to 0 over `edge_width_cm` near the boundary.
    """
    cx, cy = float(center_cm[0]), float(center_cm[1])
    R = float(radius_cm)
    h = float(step_cm)

    # Bounding box (optionally padded)
    x = np.arange(cx - R - FOV_pad_cm, cx + R + FOV_pad_cm + h/2, h)
    y = np.arange(cy - R - FOV_pad_cm, cy + R + FOV_pad_cm + h/2, h)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Radial distance and mask
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    inside = r <= R

    # Values: 1 in interior; optional cosine taper across last edge_width_cm
    vals = np.zeros_like(r, dtype=float)
    if edge_width_cm <= 0.0:
        vals[inside] = 1.0
    else:
        r0 = max(0.0, R - edge_width_cm)
        t = np.clip((r - r0) / max(edge_width_cm, 1e-12), 0.0, 1.0)
        smooth = 0.5 * (1.0 + np.cos(np.pi * t))   # 1 --> 0
        vals = np.where(r <= r0, 1.0, np.where(r <= R, smooth, 0.0))

    # Build image (vals already on grid)
    img = vals.astype(np.float32)

    # Flatten to Bloch
    positions = np.vstack([X.ravel(), Y.ravel()])   # (2,P_all)
    values = img.ravel().astype(float)              # (P_all,)
    keep = values > 0.0
    positions = positions[:, keep]
    values = values[keep]

    mx0 = np.zeros_like(values)
    my0 = np.zeros_like(values)
    mz0 = values.copy()

    # FOV for image display
    FOVx = float((x.max() - x.min()))
    FOVy = float((y.max() - y.min()))

    return Phantom2D(
        img=img, X_cm=X, Y_cm=Y, FOV_cm=(FOVx, FOVy),
        positions_cm=positions, values=values, mx0=mx0, my0=my0, mz0=mz0
    )
