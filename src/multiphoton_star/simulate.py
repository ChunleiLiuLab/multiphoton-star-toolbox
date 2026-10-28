"""Bloch-simulation for multiphoton STAR sequences."""
from __future__ import annotations
import os, json
from datetime import datetime
import numpy as np
from typing import Optional, Sequence
from dataclasses import dataclass
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback
import gc

from .pulse import HSPulse
from .radial import RadialTrajectory
from bloch.bloch import bloch

@dataclass
class BlochSimResult:
    raw_mat: np.ndarray   # (N_spokes, M) complex
    t_s: np.ndarray       # (M,)
    used_workers: int
    path: Optional[str] = None  # path to .npz if saved

    @property
    def dt_s(self) -> float:
        t = self.t_s
        return float(np.median(np.diff(t))) if t.size > 1 else 0.0

    @property
    def M(self) -> int:
        return int(self.t_s.size)

    @property
    def filename(self) -> Optional[str]:
        return Path(self.path).name if self.path else None

    @property
    def path_stem(self) -> Optional[str]:
        return Path(self.path).stem if self.path else None

    @property
    def directory(self) -> Optional[str]:
        return str(Path(self.path).parent) if self.path else None

@dataclass
class IsochromatSimResult:
    t_s: np.ndarray                 # (M,)
    positions_cm: np.ndarray        # (1, P)
    df_Hz: float
    mx: np.ndarray                  # (P, M)
    my: np.ndarray                  # (P, M)
    mz: np.ndarray                  # (P, M)
    signal: np.ndarray              # (M,) sum over P of mx+imy
    used_workers: int = 1
    path: Optional[str] = None      # path to .npz if saved

    @property
    def dt_s(self) -> float:
        t = self.t_s
        return float(np.median(np.diff(t))) if t.size > 1 else 0.0

    @property
    def M(self) -> int:
        return int(self.t_s.size)

    @property
    def filename(self) -> Optional[str]:
        return Path(self.path).name if self.path else None

    @property
    def path_stem(self) -> Optional[str]:
        return Path(self.path).stem if self.path else None

    @property
    def directory(self) -> Optional[str]:
        return str(Path(self.path).parent) if self.path else None

# ---------- helpers ----------
def _init_or_resume_checkpoint(
    checkpoint_dir: str,
    *,
    N_spokes: int,
    M: int,
    dtype=np.complex128,
    resume: bool = True,
):
    """
    Creates or resumes:
      raw_mat.npy  (memmap, shape (N_spokes, M))
      done.npy     (bool, shape (N_spokes,))
      errors.jsonl (append-only)
      checkpoint_meta.json
    """
    ckpt = Path(checkpoint_dir).expanduser().resolve()
    ckpt.mkdir(parents=True, exist_ok=True)

    raw_path   = (ckpt / "raw_mat.npy").resolve()
    done_path  = (ckpt / "done.npy").resolve()
    err_path   = (ckpt / "errors.jsonl").resolve()
    meta_path  = (ckpt / "checkpoint_meta.json").resolve()

    # raw memmap (use open_memmap in both create + resume to avoid accidental full loads)
    if resume and raw_path.exists():
        raw_mm = np.lib.format.open_memmap(raw_path, mode="r+")
        if raw_mm.shape != (N_spokes, M):
            raise ValueError(f"Checkpoint raw_mat.npy shape {raw_mm.shape} != {(N_spokes, M)}")
    else:
        raw_mm = np.lib.format.open_memmap(raw_path, mode="w+", dtype=dtype, shape=(N_spokes, M))
        raw_mm[:] = 0
        raw_mm.flush()

    # done mask
    if resume and done_path.exists():
        done = np.load(done_path)
        if done.shape != (N_spokes,):
            raise ValueError(f"Checkpoint done.npy shape {done.shape} != {(N_spokes,)}")
        done = done.astype(bool, copy=False)
    else:
        done = np.zeros((N_spokes,), dtype=bool)
        np.save(done_path, done)

    # ensure error log exists (optional but nice)
    if not err_path.exists():
        err_path.write_text("", encoding="utf-8")

    # write meta once (non-critical)
    if not meta_path.exists():
        meta = {
            "checkpoint_dir": str(ckpt),
            "N_spokes": int(N_spokes),
            "M": int(M),
            "dtype": str(np.dtype(dtype)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return raw_mm, done, str(raw_path), str(done_path), str(err_path)


def _append_error(err_path: str, error_record: dict):
    error_record = dict(error_record)
    error_record["time"] = datetime.now().isoformat(timespec="seconds")
    with open(err_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(error_record) + "\n")


def _flush_checkpoint(raw_mat, done: np.ndarray, done_path: str):
    raw_mat.flush()
    np.save(done_path, done)

    
def _sum_except_last(arr: np.ndarray) -> np.ndarray:
    """Sum over all axes except the last (assumed time), return (M,)."""
    a = np.asarray(arr)
    while a.ndim > 1:
        a = a.sum(axis=0)
    return np.asarray(a, dtype=np.complex128)


def _select_preview_spokes(N: int, preview_spokes: Optional[Sequence[int]]) -> list[int]:
    if preview_spokes is None:
        cand = [0, N // 4, N // 2]
        return sorted({i for i in cand if 0 <= i < N})
    return [i for i in preview_spokes if 0 <= i < N]


def _coerce_positions(pos: np.ndarray, dims: int) -> np.ndarray:
    """Accept (dims,P) or (P,dims); return (dims,P) float array."""
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or dims not in pos.shape:
        raise ValueError(f"positions must be (dims,P) or (P,dims) with dims={dims}. Got {pos.shape}.")
    if pos.shape[0] == dims:
        return pos
    if pos.shape[1] == dims:
        return pos.T
    raise ValueError("Invalid positions shape.")


def _coerce_init(m: Optional[np.ndarray], Nf: int, P: int, default: float, name: str) -> np.ndarray:
    """
    Accept (P,) or (Nf,P) arrays for mx0/my0/mz0. Fill default if None.
    Returns float array of shape (Nf, P).
    """
    if m is None:
        print("Warning: no initial magnetization was provided. Assuming default.")
        return np.full((Nf, P), default, dtype=float)
    m = np.asarray(m, dtype=float)
    if m.shape == (P,):
        return np.broadcast_to(m[None, :], (Nf, P)).copy()
    if m.shape == (Nf, P):
        return m
    raise ValueError(f"{name} must be shape (P,) or (Nf,P). Got {m.shape}.")


def _fmt_num_for_fname(x: Optional[float | int], kind: str = "float") -> str:
    if x is None:
        return "na"
    try:
        if kind == "int":
            return str(int(round(float(x))))
        # float-ish: avoid weird decimals; keep compact
        v = float(x)
        if np.isfinite(v):
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            # keep a few sig figs, strip trailing zeros
            s = f"{v:.6g}"
            s = s.replace(".", "p").replace("-", "m")
            return s
        return "na"
    except Exception:
        return "na"


def _default_out_path_v2(
    *,
    sim_type: Optional[str],
    bw_Hz: Optional[float],
    f_z_Hz: Optional[float],
    R: Optional[float],
) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st = (sim_type or "unknown").replace(" ", "").replace("-", "").replace("/", "")
    bw_s = _fmt_num_for_fname(bw_Hz, kind="int")
    fz_s = _fmt_num_for_fname(f_z_Hz, kind="int")
    R_s  = _fmt_num_for_fname(R, kind="int")
    return f"cswift_{st}_bw{bw_s}Hz_fz{fz_s}Hz_R{R_s}_{stamp}.npz"


def _save_npz_with_meta(
    out_path: str,
    arrays: dict[str, np.ndarray],
    meta: dict,
    compressed: bool = True,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if compressed:
        np.savez_compressed(out_path, **arrays)
    else:
        np.savez(out_path, **arrays)
    # write JSON metadata with small metadata (readable, avoids pickle)
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_path


def _jsonable_scalar(v):
    # Convert numpy scalar -> Python scalar, guard unsupported types
    if isinstance(v, (str, bool, int, float)):
        return v
    if np.isscalar(v):
        try:
            vv = v.item()
            if isinstance(vv, (str, bool, int, float)):
                return vv
        except Exception:
            return None
    return None
    

def _extract_pulse_arrays(
    pulse: HSPulse,
    *,
    max_elems: int = 2_000_000,   # guard against accidental huge arrays
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Additive pulse capture:
      - arrays_out: np.ndarray attrs saved into NPZ as pulse__<name>
      - meta_out: JSON-friendly scalars under pulse_scalars, plus list of saved array keys
    """
    arrays_out: dict[str, np.ndarray] = {}
    scalars_out: dict[str, object] = {}
    saved_array_keys: list[str] = []

    # Iterate public attributes; skip callables and private
    for name in dir(pulse):
        if name.startswith("_"):
            continue
        try:
            val = getattr(pulse, name)
        except Exception:
            continue
        if callable(val):
            continue

        # Save arrays (including complex) into NPZ
        if isinstance(val, np.ndarray):
            if val.size == 0:
                continue
            if val.size > max_elems:
                # too large, skip
                continue
            key = f"pulse__{name}"
            arrays_out[key] = val
            saved_array_keys.append(key)
            continue

        # Save JSON-safe scalars into meta
        s = _jsonable_scalar(val)
        if s is not None:
            scalars_out[name] = s

    meta_out = {
        "pulse_class": pulse.__class__.__name__,
        "pulse_scalars": scalars_out,
        "pulse_saved_arrays": saved_array_keys,
    }
    return arrays_out, meta_out


# ---------- parallel worker globals ----------
_WORK = {}


def _worker_init(B1xy_G, G_all, dt_s, T1_s, T2_s, df, positions_cm, mode, mx0, my0, mz0, B1z_G):
    _WORK.update(dict(
        B1xy_G=B1xy_G, G_all=G_all, dt_s=dt_s, T1_s=T1_s, T2_s=T2_s,
        df=df, positions_cm=positions_cm, mode=mode,
        mx0=mx0, my0=my0, mz0=mz0, B1z_G=B1z_G
    ))


def _run_spoke(i: int) -> tuple[int, np.ndarray]:
    g = _WORK["G_all"][i]  # (dims, M)
    mx, my, mz = bloch(
        _WORK["B1xy_G"], g, _WORK["dt_s"], _WORK["T1_s"], _WORK["T2_s"],
        _WORK["df"], _WORK["positions_cm"], _WORK["mode"],
        _WORK["mx0"], _WORK["my0"], _WORK["mz0"], b1z=_WORK["B1z_G"]
    )

    # IMPORTANT: avoid mx + 1j*my (huge complex temp)
    raw = _sum_except_last(mx) + 1j * _sum_except_last(my)

    # help allocator reuse memory inside the persistent worker
    del mx, my, mz
    return i, raw


# ---------- main API ----------
def simulate_radial_bloch(
    *,
    pulse: HSPulse,
    traj: RadialTrajectory,
    positions_cm: np.ndarray,            # (dims,P) or (P,dims)
    df_Hz: np.ndarray | float = 0.0,     # (Nf,) or scalar
    T1_s: float = 100.0,
    T2_s: float = 100.0,
    mode: int = 2,                       # 2 = record timepoints
    mx0: Optional[np.ndarray] = None,    # (P,) or (Nf,P)
    my0: Optional[np.ndarray] = None,    # (P,) or (Nf,P)
    mz0: Optional[np.ndarray] = None,    # (P,) or (Nf,P)
    B1z_G: Optional[np.ndarray] = None,  # (M,) optional longitudinal field [G]
    B1xy_override_G: Optional[np.ndarray] = None,
    workers: int = 1,                    # >1 enables process pool
    show_previews: bool = False,
    preview_spokes: Optional[Sequence[int]] = None,
    progress: bool = True,               # tqdm progress bar
    tqdm_desc: str = "Bloch spokes",
    tqdm_leave: bool = False,

    # ---- naming / pulse meta ----
    sim_type: Optional[str] = None,      # "1photon", "2photonCW", "2photonFM", etc.
    bw_Hz: Optional[float] = None,       # bandwidth used to design gradients/pulse
    f_z_Hz: Optional[float] = None,      # longitudinal-field frequency metadata [Hz]
    R: Optional[float] = None,           # time-bandwidth product

    # ---- saving options ----
    save: bool = False,
    save_path: Optional[str] = None,     # ".npz" path; defaults to auto name
    save_compressed: bool = True,

    # ---- FM-specific recon helpers ----
    phi_rad: Optional[np.ndarray] = None,   # (M,) phase modulation used for FM correction
    m: Optional[int] = None,                # store sideband index (defaults to 1)

    # ---- checkpointing ----
    checkpoint_dir: Optional[str] = None,   # if set, write raw_mat incrementally here
    checkpoint_resume: bool = True,         # skip already done spokes
    checkpoint_flush_every: int = 10,       # flush every N spokes completed
    checkpoint_abort_on_error: bool = True, # stop immediately on first spoke failure

) -> BlochSimResult:
    """
    Bloch-simulate all spokes in a RadialTrajectory.

    NOTE
    ----
    - This function does NOT alter or modulate the RF. It uses the RF
      exactly as provided in `pulse.B1xy_G` (or B1xy_override_G).
    - If `B1z_G` is provided with shape `(M,)`, it is passed unchanged to
      `bloch` as a time-varying longitudinal field in gauss. Its presence does
      not by itself imply a particular multiphoton excitation order.
    - Sample the input object at least twice as finely as the requested output
      image in each spatial dimension. A coarse Bloch particle grid can create
      discretization artifacts that are not reconstruction or pulse errors.

    Saving
    ------
    If `save=True`, writes a compressed .npz (plus a small .json metadata).
    """
    # Pulse time base and RF
    B1xy_G = pulse.B1xy_G if B1xy_override_G is None else np.asarray(B1xy_override_G, dtype=complex)
    t_s = pulse.t_s
    dt_s = float(np.median(np.diff(t_s))) if t_s.size > 1 else 0.0
    M = t_s.size

    # Length consistency
    if traj.M != M:
        raise ValueError(f"Pulse length (M={M}) and gradient length (M={traj.M}) must match.")

    # Positions / dims
    dims = traj.dims
    positions_cm = _coerce_positions(np.asarray(positions_cm), dims)  # (dims, P)
    P = positions_cm.shape[1]

    # Off-resonance and initial magnetization
    df = np.atleast_1d(np.asarray(df_Hz, dtype=float))
    Nf = df.size
    mx0 = _coerce_init(mx0, Nf, P, default=0.0, name="mx0")
    my0 = _coerce_init(my0, Nf, P, default=0.0, name="my0")
    mz0 = _coerce_init(mz0, Nf, P, default=1.0, name="mz0")

    # Optional time-varying longitudinal field
    if B1z_G is not None:
        B1z_G = np.asarray(B1z_G, dtype=float)
        if B1z_G.shape != (M,):
            raise ValueError("B1z_G must be shape (M,) if provided.")

    # Infer sim_type if not provided
    if sim_type is None:
        sim_type = "1photon" if B1z_G is None else "2photon"

    # Gradients
    G = traj.grads_G_per_cm[:, :dims, :]  # (N_spokes, dims, M)
    N_spokes = traj.N_spokes

    # Optional previews (serial only)
    if show_previews and workers == 1:
        import matplotlib.pyplot as plt
        for i in _select_preview_spokes(N_spokes, preview_spokes):
            mx, my, mz = bloch(B1xy_G, G[i], dt_s, T1_s, T2_s, df, positions_cm, mode,
                               mx0, my0, mz0, b1z=B1z_G)
            #raw = _sum_except_last(mx + 1j * my)
            raw = _sum_except_last(mx) + 1j * _sum_except_last(my)
            del mx, my, mz

            plt.figure()
            plt.plot(t_s, raw.real, "-k", label="Re{raw}")
            plt.plot(t_s, np.abs(raw), label="|raw|")
            plt.title(f"Spoke {i} Raw Signal")
            plt.xlabel("time (s)")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # ---- checkpoint init (optional) ----
    done = None
    done_path = None
    err_path = None

    if checkpoint_dir is not None:
        raw_mat, done, raw_path, done_path, err_path = _init_or_resume_checkpoint(
            checkpoint_dir,
            N_spokes=N_spokes,
            M=M,
            dtype=np.complex128,
            resume=checkpoint_resume,
        )
        if progress:
            n_done = int(done.sum())
            print(f"[ckpt] {n_done}/{N_spokes} spokes already complete in {checkpoint_dir}")
    else:
        raw_mat = np.zeros((N_spokes, M), dtype=np.complex128)

    # Which spokes to run
    if done is None:
        spokes_to_run = list(range(N_spokes))
    else:
        spokes_to_run = [i for i in range(N_spokes) if not done[i]]

    # Handle "nothing left to do"
    if len(spokes_to_run) == 0:
        used = 1 if workers <= 1 else int(workers)
    else:
        # Serial or parallel execution with tqdm
        if workers <= 1:
            n_written = 0

            for i in tqdm(spokes_to_run, disable=not progress, desc=tqdm_desc, unit="spoke", leave=tqdm_leave):
                try:
                    mx, my, mz = bloch(
                        B1xy_G, G[i], dt_s, T1_s, T2_s, df, positions_cm, mode,
                        mx0, my0, mz0, b1z=B1z_G
                    )
                    #raw = _sum_except_last(mx + 1j * my)
                    raw = _sum_except_last(mx) + 1j * _sum_except_last(my)
                    del mx, my, mz

                    raw_mat[i, :] = raw
                    
                    if (i % 5) == 0:
                        gc.collect()
                        
                    if done is not None:
                        done[i] = True
                        n_written += 1
                        if checkpoint_flush_every and (n_written % int(checkpoint_flush_every) == 0):
                            _flush_checkpoint(raw_mat, done, done_path)

                except Exception as e:
                    if err_path is not None:
                        _append_error(err_path, {
                            "spoke": int(i),
                            "where": "serial",
                            "error": repr(e),
                            "traceback": traceback.format_exc(),
                        })
                    if done is not None:
                        _flush_checkpoint(raw_mat, done, done_path)
                    if checkpoint_abort_on_error:
                        raise
                    else:
                        print(f"[warn] spoke {i} failed: {e!r}")
                        continue

            if done is not None:
                _flush_checkpoint(raw_mat, done, done_path)

            used = 1

        else:
            init_args = (B1xy_G, G, dt_s, T1_s, T2_s, df, positions_cm, mode, mx0, my0, mz0, B1z_G)

            with ProcessPoolExecutor(
                max_workers=int(workers),
                initializer=_worker_init,
                initargs=init_args
            ) as ex:

                # Only submit missing spokes
                futures = {ex.submit(_run_spoke, i): i for i in spokes_to_run}

                n_written = 0
                for fut in tqdm(as_completed(futures), total=len(futures), disable=not progress,
                                desc=tqdm_desc, unit="spoke", leave=tqdm_leave):
                    i = futures[fut]
                    try:
                        ii, raw = fut.result()
                        raw_mat[ii, :] = raw
                        
                        print(f"[done] spoke {ii} finished")

                        if done is not None:
                            done[ii] = True
                            n_written += 1
                            if checkpoint_flush_every and (n_written % int(checkpoint_flush_every) == 0):
                                _flush_checkpoint(raw_mat, done, done_path)

                    except Exception as e:
                        if err_path is not None:
                            _append_error(err_path, {
                                "spoke": int(i),
                                "where": "parallel",
                                "error": repr(e),
                                "traceback": traceback.format_exc(),
                            })
                        if done is not None:
                            _flush_checkpoint(raw_mat, done, done_path)
                        if checkpoint_abort_on_error:
                            raise
                        else:
                            print(f"[warn] spoke {i} failed: {e!r}")
                            continue

            if done is not None:
                _flush_checkpoint(raw_mat, done, done_path)

            used = int(workers)

            # Optional previews after parallel run
            if show_previews:
                import matplotlib.pyplot as plt
                for i in _select_preview_spokes(N_spokes, preview_spokes):
                    plt.figure()
                    plt.plot(t_s, raw_mat[i].real, "-k", label="Re{raw}")
                    plt.plot(t_s, np.abs(raw_mat[i]), label="|raw|")
                    plt.title(f"Spoke {i} Raw Signal")
                    plt.xlabel("time (s)")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

    # ---- optional save ----
    out_path_final = None
    if save:
        out_path_final = save_path or _default_out_path_v2(
            sim_type=sim_type,
            bw_Hz=bw_Hz,
            f_z_Hz=f_z_Hz,
            R=R,
        )
        out_path_final = os.path.abspath(out_path_final)

        arrays: dict[str, np.ndarray] = {
            "raw_mat": raw_mat,
            "t_s": t_s,
            "k_traj_cyc_per_cm": traj.k_traj_cyc_per_cm,
            "positions_cm": positions_cm,
            "df_Hz": df,
            "B1xy_G": B1xy_G,
            "B1z_G": (B1z_G if B1z_G is not None else np.array([], dtype=float)),
        }

        pulse_arrays, pulse_meta = _extract_pulse_arrays(pulse)
        arrays.update(pulse_arrays)

        try:
            x_rf_attr = getattr(pulse, "x_rf_rad_s", None)
        except Exception:
            x_rf_attr = None

        if isinstance(x_rf_attr, np.ndarray) and x_rf_attr.shape == (M,):
            arrays["x_rf_rad_s"] = np.asarray(x_rf_attr)
            x_rf_source = "pulse.x_rf_rad_s"
        else:
            GAMMA_RAD_PER_S_PER_G = 2.0 * np.pi * (42.57747892e6 / 1.0e4)  # rad/s/G
            arrays["x_rf_rad_s"] = np.asarray(GAMMA_RAD_PER_S_PER_G * B1xy_G, dtype=float)
            x_rf_source = "gamma*B1xy_G"

        phi_saved = False
        phi_source = None

        if phi_rad is not None:
            phi_arr = np.asarray(phi_rad, dtype=float)
            if phi_arr.shape != (M,):
                raise ValueError(f"phi_rad must be shape (M,), got {phi_arr.shape}")
            arrays["phi_rad"] = phi_arr
            phi_saved = True
            phi_source = "arg(phi_rad)"
        else:
            try:
                phi_attr = getattr(pulse, "phi_rad", None)
            except Exception:
                phi_attr = None
            if isinstance(phi_attr, np.ndarray) and phi_attr.shape == (M,):
                arrays["phi_rad"] = np.asarray(phi_attr, dtype=float)
                phi_saved = True
                phi_source = "pulse.phi_rad"

        meta: dict = {
            "dims": int(traj.dims),
            "N_spokes": int(traj.N_spokes),
            "M": int(M),
            "T1_s": float(T1_s),
            "T2_s": float(T2_s),
            "mode": int(mode),
            "used_workers": int(used),
            "FOV_cm": [float(x) for x in getattr(traj, "FOV_cm", (np.nan,) * max(1, traj.dims))],
            "resolution_cm": float(getattr(traj, "resolution_cm", np.nan)),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "two_photon": bool(B1z_G is not None),

            "sim_type": str(sim_type),
            "bw_Hz": (float(bw_Hz) if bw_Hz is not None else None),
            "f_z_Hz": (float(f_z_Hz) if f_z_Hz is not None else None),
            "R": (float(R) if R is not None else None),

            "m": int(m) if m is not None else 1,
            "has_x_rf_rad_s": True,
            "has_phi_rad": bool(phi_saved),
            "x_rf_rad_s_source": str(x_rf_source),
            "phi_rad_source": (str(phi_source) if phi_source is not None else None),
        }

        for k, attr in [
            ("flip_angle_deg", "flip_angle_deg"),
            ("rel_rf_power", "rel_rf_power"),
            ("duration_s", "duration_s"),
            ("adiabatic_ratio", "adiabatic_ratio"),
        ]:
            v = getattr(pulse, attr, None)
            if v is not None and _jsonable_scalar(v) is not None:
                meta[k] = _jsonable_scalar(v)

        meta.update({
            "Gmax_G_per_cm": float(getattr(traj, "Gmax_G_per_cm", np.nan)),
            "scale_Gmax": float(getattr(traj, "scale_Gmax", 1.0)),
        })

        meta.update(pulse_meta)

        out_path_final = _save_npz_with_meta(out_path_final, arrays, meta, compressed=save_compressed)

    return BlochSimResult(raw_mat=raw_mat, t_s=t_s, used_workers=used, path=out_path_final)

# ---------- 1D isochromat sim for paper ----------
def simulate_isochromat_1d(
    *,
    pulse: HSPulse,
    positions_cm_1d: np.ndarray | float,             # (P,) or scalar
    g_1d_G_per_cm: np.ndarray | float | None = None, # (M,) gradient vs time or scalar or None -> 0
    df_Hz: float = 0.0,
    T1_s: float = 100.0,
    T2_s: float = 100.0,
    mode: int = 2,                                   # 2 = record timepoints
    mx0: Optional[np.ndarray | float] = None,        # (P,) or scalar; default 0
    my0: Optional[np.ndarray | float] = None,        # (P,) or scalar; default 0
    mz0: Optional[np.ndarray | float] = None,        # (P,) or scalar; default 1
    B1z_G: Optional[np.ndarray] = None,              # (M,) longitudinal field [G]
    B1xy_override_G: Optional[np.ndarray] = None,    # (M,)
    show_preview: bool = False,

    # ---- naming / pulse meta ----
    sim_type: Optional[str] = None,                  # "1photon", "2photonCW", "2photonFM"
    bw_Hz: Optional[float] = None,
    f_z_Hz: Optional[float] = None,
    R: Optional[float] = None,

    # ---- FM-specific metadata helpers ----
    phi_rad: Optional[np.ndarray] = None,            # (M,) if you want to store explicit FM phase
    m: Optional[int] = None,                         # sideband index; defaults to 1

    # ---- saving ----
    save: bool = False,
    save_path: Optional[str] = None,
    save_compressed: bool = False,
) -> IsochromatSimResult:
    """
    Simulate a 1D isochromat line under an arbitrary 1D gradient.

    This is intentionally lightweight and single-core. It mirrors the updated
    naming/saving conventions of simulate_radial_bloch, while keeping the simple
    behavior of the old isochromat function.

    Notes
    -----
    - positions_cm_1d : scalar or (P,) positions along the active gradient axis [cm]
    - g_1d_G_per_cm   : scalar, (M,), or None -> zero gradient
    - Without B1z: use pulse.B1xy_G (or B1xy_override_G if supplied).
    - With B1z: pass the length-M longitudinal field to the Bloch simulator.
      Current two-photon helpers provide this through ``to_sim_kwargs()``.
    - FM-specific behavior is not altered here; we simulate exactly the applied fields,
      matching the updated radial Bloch function.
    """
    # --- time base ---
    t_s = pulse.t_s
    M = t_s.size
    dt_s = float(np.median(np.diff(t_s))) if M > 1 else 0.0

    # --- RF ---
    B1xy_G = pulse.B1xy_G if B1xy_override_G is None else np.asarray(B1xy_override_G, dtype=complex)
    if B1xy_G.shape != (M,):
        raise ValueError(f"B1xy_G must have shape (M,), got {B1xy_G.shape}")

    # --- optional time-varying longitudinal field ---
    if B1z_G is not None:
        B1z_G = np.asarray(B1z_G, dtype=float)
        if B1z_G.shape != (M,):
            raise ValueError(f"B1z_G must have shape (M,), got {B1z_G.shape}")

    # --- infer sim_type if omitted ---
    if sim_type is None:
        sim_type = "1photon" if B1z_G is None else "2photon"

    # --- positions ---
    pos1d = np.atleast_1d(np.asarray(positions_cm_1d, dtype=float))   # (P,)
    P = pos1d.size
    positions_cm = pos1d[None, :]                                     # (1, P)

    # --- defaults for initial magnetization ---
    def _coerce_iso_init(arr, default: float, name: str) -> np.ndarray:
        if arr is None:
            return np.full(P, default, dtype=float)
        out = np.atleast_1d(np.asarray(arr, dtype=float))
        if out.size == 1:
            return np.full(P, float(out.item()), dtype=float)
        if out.shape == (P,):
            return out
        raise ValueError(f"{name} must be scalar or shape (P,), got {out.shape}")

    mx0_arr = _coerce_iso_init(mx0, 0.0, "mx0")
    my0_arr = _coerce_iso_init(my0, 0.0, "my0")
    mz0_arr = _coerce_iso_init(mz0, 1.0, "mz0")

    # bloch API expects (Nf,P); here Nf=1
    df = np.atleast_1d(np.asarray(df_Hz, dtype=float))
    if df.shape != (1,):
        raise ValueError(f"For simulate_isochromat_1d, df_Hz must be scalar. Got shape {df.shape}")

    # --- gradient (dims=1) ---
    if g_1d_G_per_cm is None:
        g_1d = np.zeros(M, dtype=float)
    elif np.isscalar(g_1d_G_per_cm):
        g_1d = np.full(M, float(g_1d_G_per_cm), dtype=float)
    else:
        g_1d = np.asarray(g_1d_G_per_cm, dtype=float)
        if g_1d.shape != (M,):
            raise ValueError(f"g_1d_G_per_cm must be scalar or shape (M,), got {g_1d.shape}")

    G = g_1d[None, :]   # (1, M)

    # --- Bloch simulation ---
    mx, my, mz = bloch(
        B1xy_G,
        G,
        dt_s,
        T1_s,
        T2_s,
        df,
        positions_cm,          # (1, P)
        mode,
        mx0_arr[None, :],      # (1, P)
        my0_arr[None, :],      # (1, P)
        mz0_arr[None, :],      # (1, P)
        b1z=B1z_G
    )

    # avoid forming a full complex temp if arrays are large
    signal = _sum_except_last(mx) + 1j * _sum_except_last(my)

    # --- preview ---
    if show_preview:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 3))
        plt.plot(t_s, signal.real, "-k", label="Re{signal}")
        plt.plot(t_s, np.abs(signal), label="|signal|")
        plt.xlabel("time (s)")
        plt.title("Isochromat signal (sum over positions)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- optional save ---
    out_path_final = None
    if save:
        out_path_final = save_path or _default_out_path_v2(
            sim_type=sim_type,
            bw_Hz=bw_Hz,
            f_z_Hz=f_z_Hz,
            R=R,
        )
        out_path_final = os.path.abspath(out_path_final)

        arrays: dict[str, np.ndarray] = {
            "t_s": t_s,
            "positions_cm": positions_cm,
            "df_Hz": df,
            "mx": mx,
            "my": my,
            "mz": mz,
            "signal": signal,
            "B1xy_G": B1xy_G,
            "B1z_G": (B1z_G if B1z_G is not None else np.array([], dtype=float)),
            "g_1d_G_per_cm": g_1d,
        }

        pulse_arrays, pulse_meta = _extract_pulse_arrays(pulse)
        arrays.update(pulse_arrays)

        try:
            x_rf_attr = getattr(pulse, "x_rf_rad_s", None)
        except Exception:
            x_rf_attr = None

        if isinstance(x_rf_attr, np.ndarray) and x_rf_attr.shape == (M,):
            arrays["x_rf_rad_s"] = np.asarray(x_rf_attr)
            x_rf_source = "pulse.x_rf_rad_s"
        else:
            GAMMA_RAD_PER_S_PER_G = 2.0 * np.pi * (42.57747892e6 / 1.0e4)  # rad/s/G
            arrays["x_rf_rad_s"] = np.asarray(GAMMA_RAD_PER_S_PER_G * B1xy_G, dtype=float)
            x_rf_source = "gamma*B1xy_G"

        phi_saved = False
        phi_source = None

        if phi_rad is not None:
            phi_arr = np.asarray(phi_rad, dtype=float)
            if phi_arr.shape != (M,):
                raise ValueError(f"phi_rad must be shape (M,), got {phi_arr.shape}")
            arrays["phi_rad"] = phi_arr
            phi_saved = True
            phi_source = "arg(phi_rad)"
        else:
            try:
                phi_attr = getattr(pulse, "phi_rad", None)
            except Exception:
                phi_attr = None
            if isinstance(phi_attr, np.ndarray) and phi_attr.shape == (M,):
                arrays["phi_rad"] = np.asarray(phi_attr, dtype=float)
                phi_saved = True
                phi_source = "pulse.phi_rad"

        meta: dict = {
            "dims": 1,
            "P": int(P),
            "M": int(M),
            "T1_s": float(T1_s),
            "T2_s": float(T2_s),
            "mode": int(mode),
            "used_workers": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "two_photon": bool(B1z_G is not None),

            "sim_type": str(sim_type),
            "bw_Hz": (float(bw_Hz) if bw_Hz is not None else None),
            "f_z_Hz": (float(f_z_Hz) if f_z_Hz is not None else None),
            "R": (float(R) if R is not None else None),

            "m": int(m) if m is not None else 1,
            "has_x_rf_rad_s": True,
            "has_phi_rad": bool(phi_saved),
            "x_rf_rad_s_source": str(x_rf_source),
            "phi_rad_source": (str(phi_source) if phi_source is not None else None),
        }

        for k, attr in [
            ("flip_angle_deg", "flip_angle_deg"),
            ("rel_rf_power", "rel_rf_power"),
            ("duration_s", "duration_s"),
            ("adiabatic_ratio", "adiabatic_ratio"),
        ]:
            v = getattr(pulse, attr, None)
            if v is not None and _jsonable_scalar(v) is not None:
                meta[k] = _jsonable_scalar(v)

        meta.update(pulse_meta)

        out_path_final = _save_npz_with_meta(
            out_path_final,
            arrays,
            meta,
            compressed=save_compressed
        )

    return IsochromatSimResult(
        t_s=t_s,
        positions_cm=positions_cm,
        df_Hz=float(df_Hz),
        mx=mx,
        my=my,
        mz=mz,
        signal=signal,
        used_workers=1,
        path=out_path_final,
    )
