# Multiphoton STAR Toolbox

Python tools for multiphoton simultaneous transmission and reception (STAR) with frequency-swept RF
pulses. This toolbox provides pulse generation, multiphoton Bloch
simulation, reconstruction, and compact plotting utilities. The theory implemented in this toolbox is described by the publication:
"Multiphoton simultaneous transmission and reception MRI: Theory and validation." Please see citation details below.

The implementation supports arbitrary nonzero signed excitation order `n` and
independent reconstruction of replica `m`. The demonstrated paper sequence is
the two-photon case `n=-1`, reconstructed from physical replica `m=+1`. 

## Features

- Hyperbolic-secant pulse generation with configurable shape order `q`.
- Multiphoton waveforms with constant-amplitude or phase-modulated `B1z`.
- Order-dependent Bessel scaling for arbitrary nonzero signed `n`.
- Two-dimensional radial trajectory generation from pulse bandwidth, duration,
  and field of view.
- Bloch simulation of radial spokes (for imaging) and isochromats.
- Isolation and deconvolution of an independently selected replica `m`.
- NUFFT-based radial image reconstruction.
- Plots for excitation waveforms, individual spokes, and complex images.
- Analytic Shepp–Logan and circle phantoms for simulations.

Multiphoton excitation combines transverse RF excitation with an oscillating
longitudinal field, `B1z`. The two waveform modes differ in whether the frequency
sweep is applied to the transverse RF waveform or through phase modulation of
`B1z`; “constant” refers to the amplitude of the oscillating `B1z` field.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib
- FINUFFT
- tqdm
- The separate [Multiphoton Bloch simulator](https://github.com/ChunleiLiuLab/multiphoton-bloch-simulator)

The simulator repository should be installed first. Its README describes the
required C build environment. 

Then install this toolbox from its repository directory:

```bash
python -m pip install -e .
```

## Tutorials

- `01_two_photon_image_workflow.ipynb` compares one-photon excitation with the
  constant- and phase-modulated-`B1z` two-photon implementations and
  reconstructs physical replica `m=+1`.
- `02_arbitrary_order_and_replica.ipynb` exposes excitation order `n`,
  longitudinal-field mode, and reconstruction replica `m`; it defaults to a
  three-photon example.

The radial trajectory determines the image
resolution and spoke count from the time-bandwidth product `R` and the FOV.
The default gradient makes the swept bandwidth span twice the FOV, keeping the
object within the central half of the frequency sweep.
Each isochromat is encoded only from the time the sweep reaches it, so the
resolution is `2*FOV/R`, reconstruction uses the first half of each spoke, and
the default spoke count is `ceil(pi*R/2)`. The Bloch object is
sampled on a finer grid to well-approximate a continuous object. 

Parameters in the tutorials are chosen such that the simulations are reasonable to run 
on a local computer, but do not necessarily reflect optimal pulse values and can yield higher error. 
With more realistic pulse parameters, these are substantive Bloch simulations and may require considerable memory and
runtime. Multiple workers can be specified in the simulation function. The current multicore implementation is
embarrassingly parallel and could be further optimized, particularly for memory-heavy simulations.

## Conventions

- `q` is the hyperbolic-secant shape order.
- `n` is the signed multiphoton excitation order.
- `m` is the physical received-replica index; replica `m` is centered at
  `m*f_z` on the NumPy FFT frequency axis.
- Reconstruction defaults to `m=+1`.
- The phase-modulated correction is `exp[-i*(m/n)*phi(t)]`.
- The nominal constant-`B1z` replica bandwidth is `bw`.
- The nominal phase-modulated bandwidth is `(1+abs(m/n))*bw`.
- Replica isolation applies a configurable finite-pulse guard factor, default
  `1.1`, to the complete nominal bandwidth.

For the paper case `n=-1`, `m=+1`, the default total isolation widths are
`1.1*bw` for constant `B1z` and `2.2*bw` for phase-modulated `B1z`.

## Repository scope

This repository contains the general toolbox and tutorials. The publication simulations were run with
an earlier version of this toolbox from command-line scripts rather than notebooks, at higher
time-bandwidth products and finer temporal and spatial sampling. These scripts are available from the
authors upon reasonable request. The tutorials use `R=200` and three workers by default; increasing
temporal or spatial sampling increases memory and runtime.

The analytic Shepp--Logan phantom used in the tutorials is generated
locally from standard conventions. The associated manuscript cites P. A. Toft, *The Radon
Transform—Theory and Implementation*, Technical University of Denmark, 1996. 

## Acknowledgments

Bloch simulation is provided by a separately maintained extension of the
Python Bloch simulator derived from Brian Hargreaves's original MATLAB/MEX
implementation. Full attribution is provided in the simulator
repository.

## Citation

If you use this toolbox or multiphoton Bloch simulator in published work, please cite the associated publication and the original simulator authors (detailed in the simulator repository). The complete citation will be added after publication.
