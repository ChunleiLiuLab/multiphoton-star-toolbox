"""Compact plotting helpers for pulses, spokes, and reconstructed images.

These functions accept arrays or reconstruction results and do not load data,
run simulations, or impose manuscript-specific typography.
"""

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np


Domain = Literal["time", "frequency", "both"]
ImageDisplay = Literal["magnitude", "phase", "both"]


def _domains(domain: Domain) -> tuple[str, ...]:
    if domain == "both":
        return ("time", "frequency")
    if domain in ("time", "frequency"):
        return (domain,)
    raise ValueError("domain must be 'time', 'frequency', or 'both'")


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1D array")
    return array


def _spectrum(values: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(values.size, d=dt_s))
    spectrum = dt_s * np.fft.fftshift(np.fft.fft(values))
    return frequency_hz, spectrum


def _frequency_scale(unit: str) -> tuple[float, str]:
    scales = {"Hz": 1.0, "kHz": 1e-3, "MHz": 1e-6}
    if unit not in scales:
        raise ValueError("frequency_unit must be 'Hz', 'kHz', or 'MHz'")
    return scales[unit], unit


def plot_waveforms(
    t_s: np.ndarray,
    B1xy_G: np.ndarray,
    B1z_G: np.ndarray | None = None,
    *,
    domain: Domain = "both",
    frequency_unit: str = "kHz",
    normalize_spectra: bool = False,
    figsize: tuple[float, float] | None = None,
):
    """Plot transverse and optional longitudinal excitation waveforms.

    Time-domain B1xy panels show real, imaginary, and magnitude components.
    Frequency panels show spectral magnitude. B1z is plotted in a second row
    when supplied. Returns ``(figure, axes)``.
    """
    time = _as_1d(t_s, "t_s").astype(float, copy=False)
    b1xy = _as_1d(B1xy_G, "B1xy_G")
    if b1xy.shape != time.shape:
        raise ValueError("B1xy_G must have the same shape as t_s")
    b1z = None if B1z_G is None else _as_1d(B1z_G, "B1z_G")
    if b1z is not None and b1z.shape != time.shape:
        raise ValueError("B1z_G must have the same shape as t_s")
    if time.size < 2:
        raise ValueError("at least two time samples are required")

    domains = _domains(domain)
    rows = 2 if b1z is not None else 1
    if figsize is None:
        figsize = (5.2 * len(domains), 3.2 * rows)
    figure, axes = plt.subplots(
        rows,
        len(domains),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    dt_s = float(np.median(np.diff(time)))
    frequency_scale, frequency_label = _frequency_scale(frequency_unit)

    signals = (("B1xy", b1xy),) if b1z is None else (("B1xy", b1xy), ("B1z", b1z))
    for row, (label, signal) in enumerate(signals):
        for column, selected_domain in enumerate(domains):
            axis = axes[row, column]
            if selected_domain == "time":
                if np.iscomplexobj(signal):
                    axis.plot(1e3 * time, signal.real, label="real")
                    axis.plot(1e3 * time, signal.imag, label="imaginary")
                    axis.plot(1e3 * time, np.abs(signal), label="magnitude")
                    axis.legend()
                else:
                    axis.plot(1e3 * time, signal)
                axis.set_xlabel("Time (ms)")
                axis.set_ylabel(f"{label} (G)")
            else:
                frequency_hz, spectrum = _spectrum(signal, dt_s)
                magnitude = np.abs(spectrum)
                if normalize_spectra and magnitude.max() > 0.0:
                    magnitude = magnitude / magnitude.max()
                axis.plot(frequency_scale * frequency_hz, magnitude)
                axis.set_xlabel(f"Frequency ({frequency_label})")
                ylabel = "Normalized magnitude" if normalize_spectra else "Magnitude (G s)"
                axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.set_title(f"{label} {selected_domain}")
    return figure, axes


def plot_spoke(
    spoke,
    dt_s: float | None = None,
    *,
    spoke_index: int = 0,
    domain: Domain = "both",
    frequency_unit: str = "kHz",
    normalize_spectrum: bool = False,
    figsize: tuple[float, float] | None = None,
):
    """Plot one complex receive spoke in the time and/or frequency domain.

    ``spoke`` may be a selected 1D array, a 2D ``(N_spokes, M)`` array, or an
    object such as ``BlochSimResult``/``LoadedSim`` with a ``raw_mat``
    attribute. For 2D inputs, ``spoke_index`` selects the displayed row. The
    sample interval is inferred from a result's ``dt_s`` or ``t_s`` when
    possible; array inputs require an explicit ``dt_s``.
    """
    source = spoke
    if hasattr(source, "raw_mat"):
        source = source.raw_mat
    elif hasattr(source, "x_time"):
        source = source.x_time

    array = np.asarray(source)
    if isinstance(spoke_index, (bool, np.bool_)) or not isinstance(
        spoke_index, (int, np.integer)
    ):
        raise TypeError("spoke_index must be an integer")
    index = int(spoke_index)
    if array.ndim == 2:
        if not 0 <= index < array.shape[0]:
            raise IndexError(
                f"spoke_index {index} is outside the available range "
                f"0..{array.shape[0] - 1}"
            )
        values = _as_1d(array[index], "selected spoke")
        title_prefix = f"Spoke {index}"
    elif array.ndim == 1:
        if index != 0:
            raise ValueError("spoke_index applies only to 2D spoke collections")
        values = _as_1d(array, "spoke")
        title_prefix = "Spoke"
    else:
        raise ValueError("spoke must be a 1D array or a 2D (N_spokes, M) collection")

    if dt_s is None:
        inferred_dt = getattr(spoke, "dt_s", None)
        if inferred_dt is None and hasattr(spoke, "t_s"):
            time = _as_1d(spoke.t_s, "t_s").astype(float, copy=False)
            inferred_dt = float(np.median(np.diff(time))) if time.size > 1 else None
        if inferred_dt is None or float(inferred_dt) <= 0.0:
            raise ValueError("dt_s is required when it cannot be inferred from the input")
        dt_s = float(inferred_dt)
    elif float(dt_s) <= 0.0:
        raise ValueError("dt_s must be positive")
    else:
        dt_s = float(dt_s)

    domains = _domains(domain)
    if figsize is None:
        figsize = (5.2 * len(domains), 3.5)
    figure, axes = plt.subplots(
        1,
        len(domains),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    frequency_scale, frequency_label = _frequency_scale(frequency_unit)

    for column, selected_domain in enumerate(domains):
        axis = axes[0, column]
        if selected_domain == "time":
            time_ms = 1e3 * (np.arange(values.size) + 0.5) * dt_s
            axis.plot(time_ms, values.real, label="real")
            axis.plot(time_ms, values.imag, label="imaginary")
            axis.plot(time_ms, np.abs(values), label="magnitude")
            axis.set_xlabel("Time (ms)")
            axis.set_ylabel("Signal (a.u.)")
            axis.legend()
        else:
            frequency_hz, spectrum = _spectrum(values, dt_s)
            magnitude = np.abs(spectrum)
            if normalize_spectrum and magnitude.max() > 0.0:
                magnitude = magnitude / magnitude.max()
            axis.plot(frequency_scale * frequency_hz, magnitude)
            axis.set_xlabel(f"Frequency ({frequency_label})")
            ylabel = "Normalized magnitude" if normalize_spectrum else "Magnitude (a.u. s)"
            axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.set_title(f"{title_prefix} {selected_domain}")
    return figure, axes


def plot_image(
    image: np.ndarray,
    *,
    display: ImageDisplay = "both",
    mask: np.ndarray | None = None,
    normalize_magnitude: bool = False,
    extent_cm: Sequence[float] | None = None,
    origin: str = "lower",
    colorbar: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """Plot magnitude and/or phase of a complex 2D reconstruction."""
    values = np.asarray(image)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("image must be a nonempty 2D array")
    if display == "both":
        panels = ("magnitude", "phase")
    elif display in ("magnitude", "phase"):
        panels = (display,)
    else:
        raise ValueError("display must be 'magnitude', 'phase', or 'both'")
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != values.shape:
            raise ValueError("mask must have the same shape as image")
    else:
        mask_array = np.ones(values.shape, dtype=bool)

    if figsize is None:
        figsize = (4.8 * len(panels), 4.2)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    extent = None if extent_cm is None else tuple(float(value) for value in extent_cm)
    if extent is not None and len(extent) != 4:
        raise ValueError("extent_cm must contain (xmin, xmax, ymin, ymax)")

    for column, panel in enumerate(panels):
        axis = axes[0, column]
        axis.set_facecolor("black")  # masked (NaN) pixels show as black, not white
        if panel == "magnitude":
            shown = np.abs(values)
            if normalize_magnitude and shown.max() > 0.0:
                shown = shown / shown.max()
            shown = np.where(mask_array, shown, np.nan)
            artist = axis.imshow(shown, cmap="gray", origin=origin, extent=extent)
            title = "Magnitude"
        else:
            shown = np.where(mask_array, np.angle(values), np.nan)
            artist = axis.imshow(
                shown,
                cmap="twilight",
                origin=origin,
                extent=extent,
                vmin=-np.pi,
                vmax=np.pi,
            )
            title = "Phase (rad)"
        axis.set_title(title)
        if extent is None:
            axis.set_axis_off()
        else:
            axis.set_xlabel("x (cm)")
            axis.set_ylabel("y (cm)")
        if colorbar:
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    return figure, axes


def plot_image2d(reconstruction, **kwargs):
    """Compatibility wrapper accepting a 2D array or ``ReconResult2D``."""
    image = reconstruction.F_cart if hasattr(reconstruction, "F_cart") else reconstruction
    if "extent_cm" not in kwargs and hasattr(reconstruction, "FOVx_cm"):
        half_x = 0.5 * float(reconstruction.FOVx_cm)
        half_y = 0.5 * float(reconstruction.FOVy_cm)
        kwargs["extent_cm"] = (-half_x, half_x, -half_y, half_y)
    return plot_image(image, **kwargs)


__all__ = ["plot_waveforms", "plot_spoke", "plot_image", "plot_image2d"]
