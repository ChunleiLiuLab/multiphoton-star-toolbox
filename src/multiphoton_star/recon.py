"""Replica isolation, spoke deconvolution, and radial reconstruction."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Sequence, List
import finufft
from scipy.special import jv
import os
from datetime import datetime

from .pulse import HSPulse, MultiphotonWaveforms
from .radial import RadialTrajectory


DEFAULT_REPLICA_BANDWIDTH_GUARD = 1.1


# ---------- Deconvolution ----------
@dataclass
class DeconvolutionResult:
    Xf: np.ndarray        # (N_spokes, nfft) freq-domain estimate
    x_time: np.ndarray    # (N_spokes, T)   time-domain (cropped)
    nfft: int
    dt_s: float

def deconvolve_spokes(
    raw_mat: np.ndarray,          # (N_spokes, T)
    x: np.ndarray,                # (T,) complex drive (same convention as simulator)
    dt_s: float,
    pad_factor: float = 2.0,
    eps_rel: float = 1e-3,
    conj_x: bool = False,         # set True if you previously passed np.conj(x_rf)
) -> DeconvolutionResult:
    """
    Matched-filter deconvolution per spoke.
    If you used to call: deconvolve_spokes(raw_mat, np.conj(x_rf), dt, ...),
    then here use: conj_x=True and pass x_rf directly.
    """
    raw_mat = np.asarray(raw_mat)
    x = np.asarray(x)
    N_spokes, T = raw_mat.shape
    if x.shape != (T,):
        raise ValueError(f"x must have shape (T,) matching raw_mat.shape[1]={T}, got {x.shape}")

    if conj_x:
        x = np.conj(x)

    nfft = int(2 ** np.ceil(np.log2(pad_factor * T)))

    # Excitation spectrum H(f)
    H = dt_s * np.fft.fft(x, nfft)
    H2 = np.abs(H) ** 2
    eps = float(eps_rel) * float(H2.max() if H2.size else 1.0)

    # Spoke spectra S_i(f)
    S = dt_s * np.fft.fft(raw_mat, nfft, axis=1)

    # Matched-filter product Y = S * conj(H)  = X * |H|^2
    Y = S * np.conj(H)[None, :]

    # Stabilized division
    Xf = Y / (H2[None, :] + eps)

    # Back to time (first T samples)
    x_time = np.fft.ifft(Xf, axis=1) / dt_s
    x_time = x_time[:, :T]

    return DeconvolutionResult(Xf=Xf, x_time=x_time, nfft=nfft, dt_s=dt_s)


def deconvolve_with_pulse(
    raw_mat: np.ndarray,
    pulse: HSPulse,
    pad_factor: float = 2.0,
    eps_rel: float = 1e-3,
    conj_x: bool = True,   # matches your notebook usage of np.conj(x_rf)
) -> DeconvolutionResult:
    # Accept either an array or a BlochSimResult-like object
    if hasattr(raw_mat, "raw_mat"):   # e.g., BlochSimResult
        raw_mat = raw_mat.raw_mat
        
    t = pulse.t_s
    dt_s = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    return deconvolve_spokes(
        raw_mat=raw_mat,
        x=pulse.x_rf_rad_s,
        dt_s=dt_s,
        pad_factor=pad_factor,
        eps_rel=eps_rel,
        conj_x=conj_x,
    )

def deconvolve_spokes_two_photon(
    raw_mat: np.ndarray,        # (N_spokes, T)
    x_rf: np.ndarray,           # (T,)
    dt_s: float,
    f_z_Hz: float,
    *,
    m: int = 1,
    bw_Hz: float | None = None,
    bandlimit_guard: float = DEFAULT_REPLICA_BANDWIDTH_GUARD,
    pad_factor: float = 2.0,
    eps_rel: float = 1e-3,
    conj_x: bool = True,
    scale: float = 1.71872,  # <- two-photon scale
):
    """
    Deconvolve a constant-B1z two-photon replica.

    Replica ``m`` is centered at ``m*f_z_Hz`` on the NumPy FFT frequency
    axis. When ``bw_Hz`` is supplied, the raw signal is first isolated with
    the default total width ``1.1*bw_Hz``. The 10% guard avoids clipping the
    small finite-pulse tails observed at the nominal CW bandwidth. The default
    ``m=+1`` is demodulated with
    ``exp(-i*2*pi*f_z_Hz*t)`` before ordinary one-photon deconvolution.
    Omitting ``bw_Hz`` preserves the pre-isolated-input behavior.
    Returns a DeconvolutionResult (same type as deconvolve_spokes).
    """
    raw_mat = np.asarray(raw_mat)
    if raw_mat.ndim != 2:
        raise ValueError(f"raw_mat must be (N_spokes,T). Got {raw_mat.shape}")
    N_spokes, T = raw_mat.shape
    t_center = (np.arange(T) + 0.5) * dt_s
    mm = int(m)

    if bw_Hz is not None:
        fmin, fmax = two_photon_cw_band_edges(
            f_z_Hz=f_z_Hz,
            m=mm,
            bw_Hz=bw_Hz,
            guard_factor=bandlimit_guard,
        )
        raw_mat = bandlimit_raw_mat(raw_mat, dt_s, fmin, fmax)

    demod = np.exp(-1j * 2.0 * np.pi * mm * f_z_Hz * t_center)[None, :]
    raw_demod = raw_mat * demod

    res = deconvolve_spokes(
        raw_demod,
        np.conj(x_rf) if conj_x else x_rf,
        dt_s,
        pad_factor=pad_factor,
        eps_rel=eps_rel,
    )

    # Return a copy with scaled fields
    return replace(res, Xf=res.Xf * scale, x_time=res.x_time * scale)

# ---------- Radial gridding (2D) ----------
@dataclass
class ReconResult2D:
    F_cart: np.ndarray            # (Ny, Nx) complex image (k-space FFT grid result)
    kx_norm: np.ndarray           # (N_spokes*M,)
    ky_norm: np.ndarray           # (N_spokes*M,)
    data_flat: np.ndarray         # (N_spokes*M,)
    Nx: int
    Ny: int
    FOVx_cm: float
    FOVy_cm: float

def reconstruct_radial_image_2d(
    x_time: np.ndarray,                 # (N_spokes, M)
    traj: RadialTrajectory,             # dims=2
    FOV_cm: Optional[Tuple[float, float]] = None,
    resolution_cm: Optional[float] = None,
    dcf: str | np.ndarray = "ramp",     # "ramp", "none", or array
    eps: float = 1e-6,
    grid_oversamp: float = 2.0,
    edge_taper_r0: Optional[float] = None,   # e.g., 0.85 to taper only outer 15%
    edge_taper_kind: str = "tukey",
) -> ReconResult2D:
    """
    NUFFT gridding with optional DCF and *outer-radius* taper.

    x_time is the full deconvolved spoke (N_spokes, M). Only the first
    traj.encoded_samples samples (t < Tp/2) are reconstructed, because each
    isochromat is encoded only from the time the sweep reaches it; per-sample
    DCF arrays refer to these samples.

    dcf:
      - "ramp" --> standard r in [0,1), with DC protected.
      - "none" --> no DCF.
      - ndarray --> either (M,) per-spoke weights (tiled to all spokes) or (N_spokes*M,) flattened.

    edge_taper_r0:
      - None --> no edge taper.
      - float in (0,1) --> multiply each spoke by a taper that is 1 up to r0, then smoothly to 0 at kmax.
                         Use in addition to ramp; reduces circular ringing from hard k-space cutoff.
    """
    if traj.dims != 2:
        raise NotImplementedError("reconstruct_radial_image_2d only supports dims=2.")

    k_traj = np.asarray(traj.k_traj_cyc_per_cm)      # (N_spokes, 3, M)
    N_spokes, _, M = k_traj.shape
    if x_time.shape != (N_spokes, M):
        raise ValueError(f"x_time must be (N_spokes, M) = {(N_spokes, M)}, got {x_time.shape}")

    # Use only the encoded part of each spoke (t < Tp/2 for generated trajectories)
    encoded = getattr(traj, "encoded_samples", None)
    if encoded is not None:
        M = int(encoded)
        k_traj = k_traj[:, :, :M]
        x_time = np.asarray(x_time)[:, :M]

    # FOV / resolution
    if FOV_cm is None:
        if getattr(traj, "FOV_cm", None) is None:
            raise ValueError("FOV_cm not provided and traj has no FOV metadata.")
        if isinstance(traj.FOV_cm, Sequence) and len(traj.FOV_cm) >= 2:
            FOVx_cm, FOVy_cm = float(traj.FOV_cm[0]), float(traj.FOV_cm[1])
        else:
            FOVx_cm = FOVy_cm = float(traj.FOV_cm)
    else:
        FOVx_cm, FOVy_cm = (float(FOV_cm[0]), float(FOV_cm[1]))

    if resolution_cm is None:
        res = getattr(traj, "resolution_cm", None)
        if res is None or not np.isfinite(res):
            raise ValueError("resolution_cm not provided and traj has no resolution metadata.")
        resolution_cm = float(res)
    else:
        resolution_cm = float(resolution_cm)

    trajectory_resolution = getattr(traj, "resolution_cm", None)
    if trajectory_resolution is not None and np.isfinite(trajectory_resolution):
        trajectory_resolution = float(trajectory_resolution)
        if not np.isclose(resolution_cm, trajectory_resolution, rtol=1e-6, atol=1e-12):
            raise ValueError(
                "resolution_cm must match the trajectory resolution. "
                "Change the trajectory to request a different resolution."
            )

    Nx = int(round(FOVx_cm / resolution_cm))
    Ny = int(round(FOVy_cm / resolution_cm))
    Nx_os = int(np.ceil(Nx * grid_oversamp))
    Ny_os = int(np.ceil(Ny * grid_oversamp))

    # Flatten trajectories & data
    kx_flat = k_traj[:, 0, :].ravel()   # cycles/cm
    ky_flat = k_traj[:, 1, :].ravel()
    data_flat = x_time.ravel().astype(np.complex128, copy=False)

    # ---- Build DCF ----
    if isinstance(dcf, str):
        if dcf.lower() == "ramp":
            r = np.linspace(0.0, 1.0, M, endpoint=False)
            if M > 1:
                r[0] = r[1]     # keep DC nonzero
            w = r.copy()
        elif dcf.lower() == "none":
            w = np.ones(M, dtype=np.float64)
        else:
            raise ValueError("dcf must be 'ramp', 'none', or a numpy array.")
    else:
        dcf_arr = np.asarray(dcf)
        if dcf_arr.ndim != 1:
            raise ValueError("dcf array must be 1-D.")
        if dcf_arr.size == M:
            w = dcf_arr.astype(np.float64, copy=False)
        elif dcf_arr.size == N_spokes * M:
            data_flat *= dcf_arr.astype(np.float64, copy=False)
            w = None  # already applied fully
        else:
            raise ValueError(f"dcf array length must be M ({M}) or N_spokes*M ({N_spokes*M}).")

    # Optional outer-radius taper (per spoke)
    if edge_taper_r0 is not None:
        taper = radial_edge_taper(M, r0=float(edge_taper_r0),
                                  kind=edge_taper_kind)
        if "w" in locals() and w is not None:
            w = w * taper
        else:
            # If a full-length (N_spokes*M,) DCF was already applied, multiply data_flat directly
            data_flat *= np.tile(taper, N_spokes)

    # Apply per-spoke weights if present
    if "w" in locals() and w is not None:
        data_flat = data_flat * np.tile(w, N_spokes)

    # Normalize sample locations to [-pi, pi]
    kx_max = float(np.max(np.abs(kx_flat))) if kx_flat.size else 1.0
    ky_max = float(np.max(np.abs(ky_flat))) if ky_flat.size else 1.0
    kx_norm = (kx_flat / kx_max) * np.pi
    ky_norm = (ky_flat / ky_max) * np.pi

    # NUFFT type-1
    F_os = finufft.nufft2d1(
        ky_norm, kx_norm,
        data_flat,
        (Ny_os, Nx_os),
        eps=eps
    )

    # Center-crop back to (Ny, Nx)
    sy, sx = F_os.shape
    y0 = (sy - Ny)//2; x0 = (sx - Nx)//2
    F_cart = F_os[y0:y0+Ny, x0:x0+Nx]

    return ReconResult2D(
        F_cart=F_cart,
        kx_norm=kx_norm,
        ky_norm=ky_norm,
        data_flat=data_flat,
        Nx=Nx, Ny=Ny,
        FOVx_cm=FOVx_cm, FOVy_cm=FOVy_cm,
    )

def radial_edge_taper(M: int, r0: float = 0.85, kind: str = "tukey") -> np.ndarray:
    """
    Outer-radius taper: 1.0 up to r0, then smoothly rolls to 0 over r in (r0,1].
    r = 0..1 maps DC..kmax along a spoke. Use in addition to ramp DCF.
    kind: 'tukey' (raised-cosine outer-edge taper)
    """
    if kind.lower() != "tukey":
        raise ValueError("Only the 'tukey' outer-edge taper is supported.")
    r = np.linspace(0.0, 1.0, M, endpoint=False)
    if M > 1:
        r[0] = r[1]  # keep DC from zeroing

    w = np.ones_like(r)
    mask = (r > r0)
    if not np.any(mask):
        return w

    x = (r[mask] - r0) / (1.0 - r0)  # 0..1 over outer band
    w[mask] = 0.5 * (1.0 + np.cos(np.pi * x))

    w = np.clip(w, 0.0, 1.0)
    return w.astype(np.float64)

def bandlimit_raw_mat(
    raw_mat: np.ndarray,
    dt_s: float,
    fmin_Hz: float,
    fmax_Hz: float,
    *,
    smooth: bool = False,
    transition_Hz: float = 0.0,
) -> np.ndarray:
    """
    Bandlimit raw_mat along time axis (axis=1).

    Default behavior (smooth=False) preserves the current implementation exactly:
      zero spectrum outside [fmin_Hz, fmax_Hz].

    Optional smooth behavior (smooth=True):
      apply a cosine-tapered bandpass with transition width `transition_Hz`
      at each edge, instead of a hard rectangular cutoff.

    Parameters
    ----------
    raw_mat : np.ndarray
        Shape (N_spokes, T), complex.
    dt_s : float
        Time step in seconds.
    fmin_Hz, fmax_Hz : float
        Passband edges in Hz.
    smooth : bool, optional
        If False, use the original hard mask.
        If True, use a tapered mask.
    transition_Hz : float, optional
        Width of each taper edge in Hz. Ignored when smooth=False.
        If <= 0 and smooth=True, a default of 5% of bandwidth is used.

    Returns
    -------
    x_band : np.ndarray
        Shape (N_spokes, T), complex, back in time domain.
    """
    raw_mat = np.asarray(raw_mat)
    if raw_mat.ndim != 2:
        raise ValueError(f"raw_mat must be 2D (N_spokes,T). Got {raw_mat.shape}.")
    N_spokes, T = raw_mat.shape
    if T < 1:
        return raw_mat

    fmin_Hz = float(fmin_Hz)
    fmax_Hz = float(fmax_Hz)
    if fmax_Hz < fmin_Hz:
        raise ValueError(f"Require fmax_Hz >= fmin_Hz, got {fmin_Hz}, {fmax_Hz}")

    X = np.fft.fft(raw_mat, axis=1)
    X = np.fft.fftshift(X, axes=1)

    f = np.fft.fftshift(np.fft.fftfreq(T, d=dt_s))

    if not smooth:
        # Original behavior: hard rectangular mask
        mask = (f >= fmin_Hz) & (f <= fmax_Hz)
        X = X * mask[None, :]
    else:
        bw_Hz = fmax_Hz - fmin_Hz
        if bw_Hz <= 0:
            # degenerate case: keep nearest bin only
            mask = np.zeros_like(f, dtype=float)
            idx = int(np.argmin(np.abs(f - fmin_Hz)))
            mask[idx] = 1.0
            X = X * mask[None, :]
        else:
            # pick a default transition width if not supplied
            if transition_Hz <= 0.0:
                transition_Hz = 0.05 * bw_Hz  # 5% of passband width

            # prevent transition from exceeding half the band
            transition_Hz = min(float(transition_Hz), 0.5 * bw_Hz)

            if transition_Hz <= 0.0:
                mask = ((f >= fmin_Hz) & (f <= fmax_Hz)).astype(float)
            else:
                mask = np.zeros_like(f, dtype=float)

                left0 = fmin_Hz
                left1 = fmin_Hz + transition_Hz
                right0 = fmax_Hz - transition_Hz
                right1 = fmax_Hz

                # flat passband
                core = (f >= left1) & (f <= right0)
                mask[core] = 1.0

                # left cosine ramp: 0 -> 1
                left = (f >= left0) & (f < left1)
                if np.any(left):
                    x = (f[left] - left0) / transition_Hz
                    mask[left] = 0.5 * (1.0 - np.cos(np.pi * x))

                # right cosine ramp: 1 -> 0
                right = (f > right0) & (f <= right1)
                if np.any(right):
                    x = (f[right] - right0) / transition_Hz
                    mask[right] = 0.5 * (1.0 + np.cos(np.pi * x))

            X = X * mask[None, :]

    X = np.fft.ifftshift(X, axes=1)
    x_band = np.fft.ifft(X, axis=1)
    return x_band


def two_photon_band_edges(
    *,
    f_z_Hz: float,
    m: int,
    n: int = -1,
    bw_Hz: float,
    guard_factor: float = DEFAULT_REPLICA_BANDWIDTH_GUARD,
) -> tuple[float, float]:
    """
    Return the guarded phase-modulated window centered at ``m*f_z``.

    The analytical total bandwidth is ``(1 + abs(m/n))*bw``. The practical
    finite-pulse guard multiplies that complete nominal width and defaults to
    1.1; it is not part of the analytical formula.
    """
    mm = int(m)
    nn = int(n)
    if nn == 0 or nn != n:
        raise ValueError(f"n must be a nonzero integer, got {n}")
    bw = float(bw_Hz)
    guard = float(guard_factor)
    if bw <= 0.0:
        raise ValueError(f"bw_Hz must be positive, got {bw_Hz}")
    if guard < 1.0:
        raise ValueError(f"guard_factor must be at least 1, got {guard_factor}")
    nominal_width = (1.0 + abs(mm / nn)) * bw
    half_width = 0.5 * guard * nominal_width
    center = float(f_z_Hz) * mm
    return center - half_width, center + half_width


def two_photon_cw_band_edges(
    *,
    f_z_Hz: float,
    m: int,
    bw_Hz: float,
    guard_factor: float = DEFAULT_REPLICA_BANDWIDTH_GUARD,
) -> tuple[float, float]:
    """Return the guarded CW window centered at physical replica ``m*f_z``.

    The default total width is ``1.1*bw``: 5% of nominal bandwidth
    is added at each edge to retain finite-pulse spectral tails.
    """
    bw = float(bw_Hz)
    guard = float(guard_factor)
    if bw <= 0.0:
        raise ValueError(f"bw_Hz must be positive, got {bw_Hz}")
    if guard < 1.0:
        raise ValueError(f"guard_factor must be at least 1, got {guard_factor}")
    center = float(int(m)) * float(f_z_Hz)
    half_width = 0.5 * guard * bw
    return center - half_width, center + half_width


def deconvolve_spokes_two_photon_phase_modulated(
    *,
    raw_mat: np.ndarray,
    pulse: HSPulse,
    dt_s: float,
    f_z_Hz: float,
    n: int = -1,
    m: int = 1,
    bw_Hz: float,
    bandlimit_guard: float = DEFAULT_REPLICA_BANDWIDTH_GUARD,
    pad_factor: float = 2.0,
    eps_rel: float = 1e-3,
    conj_x: bool = True,
    scale: float = 1.71872,
    apply_bandlimit: bool = True,
    smooth_bandlimit: bool = False,
    bandlimit_transition_Hz: float = 0.0,
) -> DeconvolutionResult:
    """
    Deconvolve a phase-modulated-B1z two-photon replica.

      1) Bandlimit around physical center ``m*f_z`` using analytical total
         width ``(1 + abs(m/n))*bw`` times the practical guard factor
      2) Demodulate to baseband using centered time axis:
             raw *= exp(-i 2PI (m f_z) t_center)
      3) Apply the general phase correction exp(-i*(m/n)*phi(t))
      4) Deconvolve using the SAME kernel as always: pulse.x_rf_rad_s

    The paper defaults ``n=-1, m=+1`` therefore select the positive-frequency
    replica, use total width ``2.2*bw``, and apply ``exp(+i*phi(t))``.

    Returns DeconvolutionResult with Xf/x_time scaled by `scale`.
    """
    raw_mat = np.asarray(raw_mat, dtype=np.complex128)
    if raw_mat.ndim != 2:
        raise ValueError(f"raw_mat must be (N_spokes,T). Got {raw_mat.shape}")
    N_spokes, T = raw_mat.shape
    nn = int(n)
    if nn == 0 or nn != n:
        raise ValueError(f"n must be a nonzero integer, got {n}")

    if pulse.x_rf_rad_s.shape != (T,):
        raise ValueError(f"pulse.x_rf_rad_s must be (T,) matching raw_mat T={T}, got {pulse.x_rf_rad_s.shape}")
    if pulse.phi_rad.shape != (T,):
        raise ValueError(f"pulse.phi_rad must be (T,) matching raw_mat T={T}, got {pulse.phi_rad.shape}")

    # 1) optional bandlimit
    if apply_bandlimit:
        fmin, fmax = two_photon_band_edges(
            f_z_Hz=f_z_Hz,
            m=m,
            n=nn,
            bw_Hz=bw_Hz,
            guard_factor=bandlimit_guard,
        )
        raw_filt = bandlimit_raw_mat(
            raw_mat,
            dt_s,
            fmin,
            fmax,
            smooth=smooth_bandlimit,
            transition_Hz=bandlimit_transition_Hz,
        )
    else:
        raw_filt = raw_mat

    # 2) demodulate physical replica m from m*f_z to baseband
    t_center = (np.arange(T) + 0.5) * dt_s
    mm = int(m)
    demod_freq = float(f_z_Hz) * float(mm)
    demod = np.exp(-1j * 2.0 * np.pi * demod_freq * t_center)[None, :]
    raw_demod = raw_filt * demod

    # 3) general phase correction exp[-i*(m/n)*phi]
    corr = np.exp(-1j * (mm / nn) * pulse.phi_rad)[None, :]
    raw_corr = raw_demod * corr

    # 4) deconvolution with the *same* kernel as always
    x_rf = pulse.x_rf_rad_s
    res = deconvolve_spokes(
        raw_corr,
        np.conj(x_rf) if conj_x else x_rf,
        dt_s,
        pad_factor=pad_factor,
        eps_rel=eps_rel,
    )

    return replace(res, Xf=res.Xf * scale, x_time=res.x_time * scale)


def analytical_replica_normalization(
    waveforms: MultiphotonWaveforms,
    *,
    m: int,
) -> float:
    """Return the signed normalization for analytical replica ``m``.

    The generated transverse field contains ``B1xy_scale * J_n(a)`` and the
    selected received replica contains ``J_-m(a)``. Dividing by their product
    recovers the corresponding one-photon spoke, including the coefficient
    signs. This normalization applies to constant-amplitude and phase-modulated B1z pulses.
    """
    mm = int(m)
    if mm != m:
        raise ValueError(f"m must be an integer, got {m}")
    if waveforms.B1xy_is_custom or waveforms.B1z_is_custom:
        raise ValueError("Analytical normalization is unavailable for custom waveforms.")
    if (
        waveforms.a is None
        or waveforms.excitation_coefficient is None
        or waveforms.B1xy_scale is None
    ):
        raise ValueError("Waveforms do not contain analytical Bessel metadata.")
    replica_coefficient = float(jv(-mm, waveforms.a))
    denominator = (
        waveforms.B1xy_scale
        * waveforms.excitation_coefficient
        * replica_coefficient
    )
    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError(
            f"Replica m={mm} has zero or numerically negligible analytical scaling."
        )
    return 1.0 / denominator


def deconvolve_replica(
    *,
    raw_mat: np.ndarray,
    pulse: HSPulse,
    waveforms: MultiphotonWaveforms,
    m: int = 1,
    bw_Hz: float,
    bandlimit_guard: float = DEFAULT_REPLICA_BANDWIDTH_GUARD,
    pad_factor: float = 2.0,
    eps_rel: float = 1e-3,
    conj_x: bool = True,
) -> DeconvolutionResult:
    """Isolate and deconvolve any analytical multiphoton replica.

    Excitation order, carrier frequency, longitudinal mode, Bessel argument,
    and transverse scaling come from ``waveforms``. ``m`` independently
    selects the physical received replica centered at ``m*f_z_Hz``.
    """
    scale = analytical_replica_normalization(waveforms, m=m)
    if waveforms.mode == "constant":
        return deconvolve_spokes_two_photon(
            raw_mat=raw_mat,
            x_rf=pulse.x_rf_rad_s,
            dt_s=pulse.dt_s,
            f_z_Hz=waveforms.f_z_Hz,
            m=m,
            bw_Hz=bw_Hz,
            bandlimit_guard=bandlimit_guard,
            pad_factor=pad_factor,
            eps_rel=eps_rel,
            conj_x=conj_x,
            scale=scale,
        )
    if waveforms.mode == "phase_modulated":
        return deconvolve_spokes_two_photon_phase_modulated(
            raw_mat=raw_mat,
            pulse=pulse,
            dt_s=pulse.dt_s,
            f_z_Hz=waveforms.f_z_Hz,
            n=waveforms.n,
            m=m,
            bw_Hz=bw_Hz,
            bandlimit_guard=bandlimit_guard,
            pad_factor=pad_factor,
            eps_rel=eps_rel,
            conj_x=conj_x,
            scale=scale,
        )
    raise ValueError(f"Unsupported analytical waveform mode {waveforms.mode!r}.")

def save_recon2d_npz(
    recon: "ReconResult2D | np.ndarray",
    *,
    filename: str,
    directory: str = ".",
    overwrite: bool = False,
) -> str:
    """
    Save complex 2D recon result to NPZ.

    Stores:
      - img2d_cplx : complex64 (Ny, Nx)  [F_cart]
      - (optional, if ReconResult2D) Nx, Ny, FOVx_cm, FOVy_cm

    Returns the full path written.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"File exists: {path} (set overwrite=True to replace)")

    if hasattr(recon, "F_cart"):
        img = np.asarray(recon.F_cart)
        saved_arrays = {
            "img2d_cplx": img.astype(np.complex64, copy=False),
            "Nx": int(getattr(recon, "Nx")),
            "Ny": int(getattr(recon, "Ny")),
            "FOVx_cm": float(getattr(recon, "FOVx_cm")),
            "FOVy_cm": float(getattr(recon, "FOVy_cm")),
        }
    else:
        img = np.asarray(recon)
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {img.shape}")
        saved_arrays = {
            "img2d_cplx": img.astype(np.complex64, copy=False),
        }

    np.savez_compressed(path, **saved_arrays)
    return path
