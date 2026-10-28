"""HS pulse construction and arbitrary-order multiphoton waveforms."""

from dataclasses import dataclass

import numpy as np
from scipy.special import jnp_zeros, jv

@dataclass
class HSPulse:
    t_s: np.ndarray
    B1xy_G: np.ndarray
    x_rf_rad_s: np.ndarray
    fn: np.ndarray
    omega1_rad_s: np.ndarray
    omegaRF_rad_s: np.ndarray
    phi_rad: np.ndarray
    flip_angle_deg: float
    rel_rf_power: float
    duration_s: float
    n_points: int
    lhs_rad_s2: float
    rhs_Hz2: float
    adiabatic_ratio: float

    # --- convenience properties ---
    @property
    def dt_s(self) -> float:
        """Median time step derived from t_s."""
        t = self.t_s
        return float(np.median(np.diff(t))) if t.size > 1 else 0.0

    @property
    def fs_Hz(self) -> float:
        """Sampling rate derived from t_s."""
        dt = self.dt_s
        return (1.0 / dt) if dt > 0 else 0.0

    @property
    def M(self) -> int:
        """Number of time samples."""
        return int(self.t_s.size)

    def make_multiphoton(self, *args, **kwargs) -> "MultiphotonWaveforms":
        """Construct constant- or phase-modulated-B1z multiphoton fields."""
        return make_multiphoton(self, *args, **kwargs)

@dataclass
class MultiphotonWaveforms:
    n: int
    f_z_Hz: float
    mode: str
    B1xy_G: np.ndarray
    B1z_G: np.ndarray
    a: float | None
    excitation_coefficient: float | None
    B1xy_scale: float | None
    B1z_amplitude_G: float | None
    samples_per_period: float
    phi_rad: np.ndarray
    B1xy_is_custom: bool = False
    B1z_is_custom: bool = False

    def to_sim_kwargs(self) -> dict:
        """Return waveform arguments for ``simulate_radial_bloch``."""
        return {
            "B1xy_override_G": self.B1xy_G,
            "B1z_G": self.B1z_G,
            "phi_rad": self.phi_rad,
        }

def generate_hs_pulse(
    beta: float,                 # truncation factor (dimensionless)
    q: int,                      # HS shape order (dimensionless, typical small integer)
    bw_hz: float,                # target excitation bandwidth [Hz]
    R: float,                    # time-bandwidth product (dimensionless)
    dt_s: float,                 # time step [s]
    B1xy_max_G: float,           # max transverse RF field [G]
    gamma_bar_Hz_per_G: float = 4258.0,  # gyromagnetic ratio [Hz/G]
    verbose: bool = True
) -> HSPulse:
    """
    Generate a cSWIFT-like hyperbolic-secant (HS-q) adiabatic RF pulse.

    Inputs are unit-explicit. Outputs include complex RF envelope in Gauss (B1xy_G)
    and the complex drive in rad/s (x_rf_rad_s = omega1 * exp(-i*phi)).

    """
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)):
        raise TypeError("q must be a positive integer HS shape order.")
    q = int(q)
    if q <= 0:
        raise ValueError("q must be a positive integer HS shape order.")

    # Constants / conversions
    gamma_rad_s_per_G = 2 * np.pi * gamma_bar_Hz_per_G   # [rad/s/G]
    A_rad_s = bw_hz * np.pi                              # FM amplitude [rad/s]; (half BW in rad/s)
    Tp_s = R / bw_hz                                     # total pulse duration [s]
    t_s = np.arange(0.0, Tp_s, dt_s)

    # Driving function (HS-q)
    tau = 2.0 * t_s / Tp_s - 1.0
    fn = 1.0 / np.cosh(beta * (tau ** q))

    # RF amplitude modulation (omega1), capped by B1xy_max_G
    omega1_max = gamma_rad_s_per_G * B1xy_max_G          # [rad/s]
    omega1 = omega1_max * fn                             # [rad/s]

    # Frequency modulation (centered, no carrier added)
    numRF = np.cumsum(fn**2) * dt_s
    denRF = np.trapz(fn**2, t_s)
    omegaRF = 2.0 * A_rad_s * (numRF / denRF - 0.5)      # [rad/s]
    phi = np.cumsum(omegaRF) * dt_s                      # [rad]

    # Complex RF (Gauss) and complex drive (rad/s)
    B1xy_G = (omega1 / gamma_rad_s_per_G) * np.exp(-1j * phi)  # [G]
    x_rf_rad_s = omega1 * np.exp(-1j * phi)                     # [rad/s]

    # Relative RF power (matches your definition)
    rel_rf_power = (1.0 / beta) ** (1.0 / q)

    # Flip angle estimate (your original closed-form; unit: degrees)
    flip_angle_deg = (
        omega1_max * (180.0 / np.pi) * (beta ** (-1.0 / (2.0 * q))) * (R ** 0.5) / bw_hz
    )

    # Adiabatic condition (rapid passage, linear regime)
    lhs_rad_s2 = A_rad_s / (np.pi * denRF)              # [rad/s^2]
    rhs_Hz2 = (omega1_max / (2.0 * np.pi)) ** 2         # [(Hz)^2]
    adiabatic_ratio = lhs_rad_s2 / rhs_Hz2

    # Sanity checks
    if verbose:
        print(f"HS-{q} pulse with maximum B1xy = {B1xy_max_G:.5g} G")
        print(f"Flip angle = {flip_angle_deg:.3f} degrees")
        print(f"Bandwidth = {bw_hz/1e3:.3f} kHz")
        print(f"Pulse duration = {Tp_s*1e3:.3f} ms")
        print(f"Time step = {dt_s*1e3:.3f} ms")
        print(f"Relative RF power = {rel_rf_power:.5f} (want this below 1)")
        print(f"# of points = {t_s.size}")

        print("Adiabatic condition (rapid passage, linear regime):")
        print(f"  LHS [rad/s^2] = {lhs_rad_s2:.6g}")
        print(f"  RHS [(Hz)^2] = {rhs_Hz2:.6g}")
        print(f"  Ratio (LHS/RHS, should be >> 1) = {adiabatic_ratio:.6g}")

        points_per_Tp = Tp_s / dt_s
        if points_per_Tp < 500:
            print(f"WARNING: Only {int(points_per_Tp)} points across Tp; consider reducing dt_s.")
        if dt_s > 0.1 / A_rad_s:
            print("WARNING: dt_s may be large relative to 1/A; consider smaller dt_s for stable phase integration.")

        if flip_angle_deg >= 90:
            print("NOTE: Flip angle >= 90 degrees. The small-tip approximation used in some derivations may not hold.")

    return HSPulse(
        t_s=t_s,
        B1xy_G=B1xy_G,
        x_rf_rad_s=x_rf_rad_s,
        fn=fn,
        omega1_rad_s=omega1,
        omegaRF_rad_s=omegaRF,
        phi_rad=phi,
        flip_angle_deg=flip_angle_deg,
        rel_rf_power=rel_rf_power,
        duration_s=Tp_s,
        n_points=t_s.size,
        lhs_rad_s2=lhs_rad_s2,
        rhs_Hz2=rhs_Hz2,
        adiabatic_ratio=adiabatic_ratio
    )

def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "constant": "constant",
        "cw": "constant",
        "continuous": "constant",
        "continuous_wave": "constant",
        "phase_modulated": "phase_modulated",
        "pm": "phase_modulated",
        "phasemod": "phase_modulated",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown mode={mode!r}. Use 'constant' or 'phase_modulated'."
        ) from exc


def _validate_override(name: str, values, shape: tuple[int, ...], *, real: bool):
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if real and np.iscomplexobj(array) and np.any(np.imag(array) != 0):
        raise ValueError(f"{name} must be real-valued.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(np.real(array) if real else array, dtype=float if real else complex)


def make_multiphoton(
    pulse: HSPulse,
    n: int,
    f_z_Hz: float,
    *,
    mode: str = "constant",
    a: float | None = None,
    gamma_bar_Hz_per_G: float = 4257.75,
    B1xy_scale: float | None = None,
    B1xy_override_G=None,
    B1z_override_G=None,
    min_samples_per_period: float = 5.0,
    verbose: bool = True,
) -> MultiphotonWaveforms:
    """Construct fields for signed multiphoton excitation order ``n``.

    The standard sinusoidal designs use transverse carrier
    ``exp(+1j*n*omega_z*t)``. Constant-B1z mode retains the HS sweep in
    ``B1xy``. Phase-modulated-B1z mode uses an unswept transverse envelope and
    ``B1z(t) = B1z_amplitude*cos(omega_z*t + phi(t)/n)``.

    If ``a`` is omitted, the first positive maximum of ``abs(J_n(a))`` is
    used. The default transverse scale is then ``1/abs(J_n(a))``. A custom
    B1z array has no general single-Bessel analytical scaling, so the user must
    also provide either ``B1xy_scale`` or the complete ``B1xy_override_G``.
    Overrides replace the corresponding final physical waveform exactly.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be a nonzero signed integer.")
    n = int(n)
    if n == 0:
        raise ValueError("n=0 is the separate one-photon path; use nonzero n.")
    if not np.isfinite(f_z_Hz) or f_z_Hz <= 0:
        raise ValueError("f_z_Hz must be finite and > 0.")
    if not np.isfinite(gamma_bar_Hz_per_G) or gamma_bar_Hz_per_G <= 0:
        raise ValueError("gamma_bar_Hz_per_G must be finite and > 0.")

    mode = _normalize_mode(mode)
    t = np.asarray(pulse.t_s, dtype=float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    min_samples_per_period = float(min_samples_per_period)
    if not np.isfinite(min_samples_per_period) or min_samples_per_period < 2.0:
        raise ValueError("min_samples_per_period must be finite and at least 2.")
    samples_per_period = (1.0 / f_z_Hz) / dt if dt > 0 else np.inf
    if samples_per_period < min_samples_per_period:
        maximum_dt = 1.0 / (min_samples_per_period * f_z_Hz)
        raise ValueError(
            f"dt_s={dt:.6g} gives only {samples_per_period:.3g} samples per "
            f"f_z cycle; require at least {min_samples_per_period:g} "
            f"(dt_s <= {maximum_dt:.6g} s for f_z_Hz={f_z_Hz:g})."
        )
    if pulse.phi_rad.shape != t.shape or pulse.B1xy_G.shape != t.shape:
        raise ValueError("Pulse fields and phase must match pulse.t_s.")

    custom_xy = B1xy_override_G is not None
    custom_z = B1z_override_G is not None
    if custom_xy and B1xy_scale is not None:
        raise ValueError("B1xy_scale cannot be combined with B1xy_override_G.")

    a_value: float | None
    coefficient: float | None
    b1z_amplitude: float | None
    if custom_z and a is None:
        a_value = coefficient = b1z_amplitude = None
    else:
        a_value = float(jnp_zeros(abs(n), 1)[0] if a is None else a)
        if not np.isfinite(a_value) or a_value <= 0:
            raise ValueError("a must be finite and > 0.")
        coefficient = float(jv(n, a_value))
        b1z_amplitude = a_value * float(f_z_Hz) / float(gamma_bar_Hz_per_G)

    if not custom_xy:
        if B1xy_scale is None:
            if coefficient is None:
                raise ValueError(
                    "Custom B1z requires an explicit B1xy_scale or B1xy_override_G."
                )
            B1xy_scale = 1.0 / abs(coefficient)
        B1xy_scale = float(B1xy_scale)
        if not np.isfinite(B1xy_scale) or B1xy_scale <= 0:
            raise ValueError("B1xy_scale must be finite and > 0.")

    omega_z = 2.0 * np.pi * f_z_Hz
    carrier = np.exp(1j * n * omega_z * t)
    if custom_xy:
        B1xy = _validate_override("B1xy_override_G", B1xy_override_G, t.shape, real=False)
    elif mode == "constant":
        B1xy = B1xy_scale * pulse.B1xy_G * carrier
    else:
        B1xy = B1xy_scale * np.abs(pulse.B1xy_G) * carrier

    if custom_z:
        B1z = _validate_override("B1z_override_G", B1z_override_G, t.shape, real=True)
    elif mode == "constant":
        B1z = b1z_amplitude * np.cos(omega_z * t)
    else:
        B1z = b1z_amplitude * np.cos(omega_z * t + pulse.phi_rad / n)

    spp = samples_per_period

    if verbose:
        print(f"[Multiphoton] n = {n}, mode = {mode}")
        print(f"[Multiphoton] f_z = {f_z_Hz:.3f} Hz")
        if a_value is not None:
            print(
                f"[Multiphoton] a = {a_value:.8g}, J_n(a) = {coefficient:.8g}, "
                f"B1xy scale = {B1xy_scale if B1xy_scale is not None else 'custom'}"
            )
        print(f"[Multiphoton] samples per period ≈ {spp:.2f} (dt={dt:.3g} s)")

    return MultiphotonWaveforms(
        n=n,
        f_z_Hz=f_z_Hz,
        mode=mode,
        B1xy_G=np.asarray(B1xy, dtype=np.complex128),
        B1z_G=np.asarray(B1z, dtype=float),
        B1xy_scale=B1xy_scale,
        a=a_value,
        excitation_coefficient=coefficient,
        B1z_amplitude_G=b1z_amplitude,
        samples_per_period=spp,
        phi_rad=np.asarray(pulse.phi_rad, dtype=float),
        B1xy_is_custom=custom_xy,
        B1z_is_custom=custom_z,
    )
