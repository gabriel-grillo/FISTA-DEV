# FISTA-DEV: learned deviations and step sizes for an inexact FISTA

This repository contains the code for the numerical experiments of the paper

> **[Title of the paper]**, by [authors]. [Journal / arXiv reference].

FISTA-DEV is an accelerated proximal gradient method (FISTA type) for convex composite problems
$\min_x f(x) + g(x)$, in which, during the first $K$ iterations, the step sizes and the deviations
$d_k$ (gradient), $e_k$ (proximal output) and $v_k$ (extrapolation) are given by trained neural networks. The
proximal operator of $g$ is computed inexactly by an inner solver stopped by a duality gap certificate. The
deviations are norm-bounded by summable sequences and the step sizes lie in a fixed interval, so that the
convergence of the iterates and the convergence rates proved in the paper hold for **any** parameters of the
learned model.

The experiments solve total variation regularized least squares problems with a non-negativity constraint,
arising in computed tomography (CT), with images from the AAPM Low Dose CT Grand Challenge.

---

## Contents

- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running a single configuration](#running-a-single-configuration)
- [Reproducing the experiments of the paper](#reproducing-the-experiments-of-the-paper)
- [Performance measures](#performance-measures)
- [Output folders](#output-folders)
- [Implemented algorithms and problems](#implemented-algorithms-and-problems)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Repository structure

```
.
├── Dockerfile              # Image with CUDA, TensorFlow, PyTorch, LaTeX (for the plots) and the ray tracing library
├── ray_tracing/            # Fork of elias-helou/ray_tracing: CUDA ray transform (Radon) used for CT
└── code/
    ├── data/mayo_clinic_image/   # Place the CT images here (see "Data" below)
    ├── data.py               # Simulation of the sinograms, train/val/test split, reference objective values
    ├── operators.py          # Ray transform, discrete gradient, wavelets, TensorFlow/PyTorch integration
    ├── proxTV.py             # Inexact proximal operator of the TV (dual FISTA with duality gap stopping criterion)
    ├── optimization.py       # FISTA-DEV, FISTA, ISTA and DeepOpt
    ├── traintest.py          # Training and testing routines
    ├── paths.py              # Paths where models, histories and plots are saved
    ├── plots_and_metrics.py  # Loading of histories, plots, tables and metrics
    ├── paper_config.py       # Shared configuration of the experiments of the paper
    ├── paper_experiments.py  # Runs the experiments of the paper
    ├── paper_results.py      # Processes the results of the experiments of the paper (figures and tables)
    ├── estimate_tau.py       # Estimates the accuracy of the reference optimal values and the threshold TAU
    ├── measure_costs.py      # Measures the cost of inner iterations and networks relative to a gradient evaluation
    ├── single_experiment.py  # Runs a single configuration (for development)
    └── results.py            # Processes the results of single_experiment.py
```

## Requirements

- An NVIDIA GPU with CUDA support. The scripts limit TensorFlow to about 13 GiB of GPU memory
  (`VRAM_MB` in `paper_experiments.py` and `single_experiment.py`); adjust this value to your GPU.
- [Docker](https://docs.docker.com/) with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), recommended.
  The `Dockerfile` installs all the dependencies with fixed versions (Python packages, LaTeX for the figures, and
  the CUDA ray tracing library, compiled for the compute capability of your GPU).

Without Docker, you need Python 3 with the packages listed in the `Dockerfile` (TensorFlow, PyTorch, PyWavelets,
pydicom, matplotlib, pandas), a LaTeX installation (the figures are rendered with `usetex`), and the ray tracing
library compiled with `make -C ray_tracing COMP_CAP=<compute capability>` and added to the `PYTHONPATH`
(`ray_tracing/tmp`).

## Setup

### 1. Data

The images are taken from the *AAPM Low Dose CT Grand Challenge* dataset. Download `full_3mm.zip` and extract it
in `code/data/mayo_clinic_image/`, so that the DICOM files are under `code/data/mayo_clinic_image/full_3mm/`.
See [`code/data/mayo_clinic_image/README.md`](code/data/mayo_clinic_image/README.md) for the download link and
references.

Nothing else needs to be prepared by hand: in the first run, the code

1. splits the images into training (70%), validation (10%) and test (20%) sets, with a fixed seed;
2. simulates the sinograms with the ray transform (1000 angles and 1000 detectors for `mayo_clinic_512`) and
   additive Gaussian noise (5% of the mean absolute value of the sinogram), and saves them in
   `code/data/<dataset>/observed/`;
3. runs FISTA on every problem and saves the objective values in `code/data/<dataset>/F_ref/<problem>/`. These
   values are used to normalize the training loss and as a reference for the optimal values.

These steps may take a while, but are done only once. If you change the problem, the noise or the reference
method, delete the corresponding folder in `code/data/<dataset>/` so that it is generated again.

### 2. Docker image

From the root of the repository, build the image (the compute capability of the first GPU is detected
automatically):

```bash
docker build --build-arg COMP_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.') -t fista-dev .
```

and start a container with the `code` folder mounted at `/code`:

```bash
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --ipc=host -p 8888:8888 -v $(pwd)/code:/code fista-dev
```

All the commands below are run **inside the container, from `/code`**: the scripts save and read data, models
and results relative to the current directory.

## Running a single configuration

`single_experiment.py` trains and tests a set of configurations defined in its `EDIT HERE` block (dataset,
problem, number of iterations, parameters of FISTA, FISTA-DEV and DeepOpt, ablation, fraction of the data, ...).
It is meant for development and quick tests, for instance with a small fraction of the data:

```bash
python3 single_experiment.py
```

The results are processed by `results.py`, whose `EDIT HERE` block must match the configurations of
`single_experiment.py` (in particular, the `DATE` of the run):

```bash
python3 results.py
```

The figures and tables are saved in `code/plots/`.

## Reproducing the experiments of the paper

The experiments of the paper are run by `paper_experiments.py` and processed by `paper_results.py`. Both scripts
read the grids of parameters, the training budgets and the **selected parameters** from `paper_config.py`, so that
the results are always processed with the configurations that were actually trained and tested.

The experiments are:

| Flag | Experiment | Data | What is run |
|---|---|---|---|
| `RUN_EXPERIMENT_1`  | Parameter selection for FISTA | validation | ISTA and FISTA (no training) for a grid of $(\gamma, \alpha)$ |
| `RUN_EXPERIMENT_2A` | Parameter selection for FISTA-DEV and DeepOpt (screening) | validation | FISTA-DEV for a grid of $(\gamma,\alpha)$ and DeepOpt for a grid of its parameter, trained with a fraction of the training data and few epochs |
| `RUN_EXPERIMENT_2B` | Parameter selection for FISTA-DEV and DeepOpt (full budget) | validation | The finalists of FISTA-DEV and of DeepOpt, trained with all the training data |
| `RUN_EXPERIMENT_3`  | Ablation study | validation | FISTA-DEV with the selected parameters, learning only some of its components ($d_k$, $e_k$, $v_k$, all the deviations, or only the step sizes) |
| `RUN_EXPERIMENT_4`  | Comparison with the baselines | test | ISTA, FISTA ($\alpha=0$ and the selected $\alpha$), FISTA-DEV and DeepOpt; no training (the models come from Experiment 2B) |

Each experiment depends on the parameters selected in the previous ones. The workflow is:

1. **Experiment 1.** In `paper_experiments.py`, set `RUN_EXPERIMENT_1 = True` and the other flags to `False`, and
   run `python3 paper_experiments.py`. Then set the same flags in `paper_results.py` and run
   `python3 paper_results.py`. Using the tables of cost to reach each relative gap (in
   `paper_results_data/exp1/`, see [Performance measures](#performance-measures)), choose the parameters of FISTA
   and set `BEST_FISTA` in `paper_config.py`.
2. **Experiment 2A.** Run both scripts with `RUN_EXPERIMENT_2A = True`. Using the validation loss and the tables
   in `paper_results_data/exp2a/`, set `FINALISTS_FISTA_DEV` and `FINALISTS_DO` in `paper_config.py`.
3. **Experiment 2B.** Run both scripts with `RUN_EXPERIMENT_2B = True`. Using the results in
   `paper_results_data/exp2b/`, set `BEST_FISTA_DEV` and `BEST_ALPHA_DO` in `paper_config.py`.
4. **Experiment 3.** Run both scripts with `RUN_EXPERIMENT_3 = True`.
5. **Experiment 4.** Run both scripts with `RUN_EXPERIMENT_4 = True`.

To reproduce the results of the paper without repeating the selection, use the selected parameters of the paper in
`paper_config.py` (`BEST_FISTA`, `FINALISTS_FISTA_DEV`, `BEST_FISTA_DEV`, `FINALISTS_DO` and `BEST_ALPHA_DO`) and run all the
experiments at once, with all the flags set to `True` in both scripts:

```bash
python3 paper_experiments.py   # trains and tests all the configurations (takes many hours)
python3 paper_results.py       # produces all the figures and tables
```

Some remarks:

- The training of the learned methods is the expensive part. Experiment 1 involves no training and is fast in
  comparison.
- The runs are deterministic (fixed seeds and `enable_op_determinism`), up to differences between hardware and
  library versions.
- Training a configuration with `LOAD_SAVED_MODELS = False` **deletes** previously saved models with the same
  parameters. Experiments 3 and 4 only test the full FISTA-DEV model trained in Experiment 2B, so they do not
  overwrite it.
- `paper_results.py` reads the constants `TAU`, `R_COST`, `R_NET` and `R_NET_DO`, described in
  [Performance measures](#performance-measures).

## Performance measures

For each problem $b$, `paper_results.py` reports the relative optimality gap $(F_b(\tilde x_k) - F_b^\ast)/F_b^\ast$,
in which $F_b^\ast$ is the smallest value of $F_b$ attained by all the methods and by the reference run saved in
`data/<dataset>/F_ref/` (FISTA with $\alpha = 0$ and $\gamma = 2$ after 500 iterations). The figures show the
median and the interquartile range over the problems, against the iterations and against the cost.

**Cost.** The cost of a run up to iteration $k$ is measured in gradient evaluations (one application of the ray
transform and one of its transpose),

$$\mathrm{cost}(k) = k + r \cdot (\text{inner iterations of the proximal solver up to } k) + r_{\mathrm{net}} \cdot \min(k, K),$$

in which $r$ (`R_COST`) is the cost of one inner iteration and $r_{\mathrm{net}}$ is the cost of the networks
applied in each of the first $K$ iterations (`R_NET` for FISTA-DEV, depending on the ablation, and `R_NET_DO` for
DeepOpt; zero for ISTA and FISTA). These ratios depend on the GPU and are measured by

```bash
python3 measure_costs.py              # batches of 32 problems, as in the tests
```

which prints the time of each operation and the ratios to be copied to `paper_results.py`.

**Tables.** For each tolerance in `TOLS` ($10^{-2}$ and $10^{-5}$), the tables show the median cost to reach the
relative gap and, in parentheses, the percentage of problems that reach it, as in `192.0 (100%)`. When fewer than
half of the problems reach the tolerance, the median is shown as `-`.

**Threshold `TAU`.** Since the reference run is still decreasing at its last iterations, $F_b^\ast$ approximates the
optimal value only up to a small relative error, and the relative gaps are also limited by the resolution of single
precision. Relative gaps below `TAU` are not shown. The accuracy of $F_b^\ast$ and the corresponding values of `TAU`
are estimated by

```bash
python3 estimate_tau.py               # validation set
python3 estimate_tau.py --mode test   # test set
```

which extrapolates the decrease rate of the gap of the reference run. For the data of the paper, the relative error
of $F_b^\ast$ is below $10^{-7}$ for 90% of the problems, and `TAU = 1e-6`.

## Output folders

All folders below are created inside `code/` and are ignored by git.

| Folder | Content |
|---|---|
| `data/<dataset>/observed/` | Simulated sinograms (train, val, test) |
| `data/<dataset>/F_ref/<problem>/` | Reference objective values of FISTA, used in the training loss and as reference optimal values |
| `models/` | Trained models (networks and step size parameters), one folder per configuration and per saved epoch, with the training loss history |
| `hist/` | Histories saved in the tests: objective values, tolerances and number of inner iterations of the proximal solver, norms of the deviations and their bounds, step sizes |
| `plots/` | Figures and tables produced by `results.py` |
| `paper_results_data/` | Figures (`.pdf`), data of the curves (`.txt`) and tables (`.txt` and `.tex`) produced by `paper_results.py`, one subfolder per experiment |

## Implemented algorithms and problems

Algorithms (`optimization.py`):

- **FISTA-DEV** (`fista_dev`): FISTA with learned deviations and step sizes in the first $K$ iterations, and the
  step size $\mu/L$ afterwards. The networks are small convolutional networks (`traintest.py`), and the
  ablation option selects which components are learned.
- **FISTA** (`fista`) and **ISTA** (`ista`): classical baselines, with the inexact proximal operator.
- **DeepOpt** (`deepopt_nonsmooth`): the learned forward-backward method of Banert et al., *Accelerated
  Forward-Backward Optimization Using Deep Learning*, SIAM J. Optim. (2024).

Problems (`PROBLEM`): `lstv` (least squares with total variation and non-negativity, used in the paper), and also
`nnls`, `lasso` (wavelets), `slasso` and `nnslasso`.

Datasets (`DATASET`): `mayo_clinic_512` (512×512 images, used in the paper) and `mayo_clinic_128` (128×128
images, faster, useful for tests).

## Troubleshooting

- **Out of memory on the GPU.** Reduce `VRAM_MB` only if other processes use the GPU; otherwise, reduce the batch
  sizes (`TEST_BATCH_SIZE`) or test with `mayo_clinic_128`. The gradients are not propagated through the inner
  iterations of the proximal solver (`jfb = True` in `proxTV.py`), which keeps the memory used in training small.
- **`There is no data in .../full_3mm`.** The images were not extracted to the expected folder; see
  [Data](#1-data).
- **Results inconsistent with a code change.** The sinograms and the reference objective values are generated only
  once. Delete `code/data/<dataset>/F_ref/` (or `observed/`) after changing the problem, the reference method or
  the noise.
- **LaTeX errors when plotting.** The figures use `usetex`; inside the Docker image LaTeX is already installed.

## Acknowledgements

The CUDA ray transform is a fork of [elias-helou/ray_tracing](https://github.com/elias-helou/ray_tracing), adapted
for TensorFlow. The CT images are from the AAPM Low Dose CT Grand Challenge (McCollough et al., *Med. Phys.*
44(10), 2017).

## Citation

If you use this code, please cite:

```bibtex
@article{[key],
  title   = {[Title of the paper]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]}
}
```