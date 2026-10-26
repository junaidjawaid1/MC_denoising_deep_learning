# Deep learning for fast Monte Carlo proton dose denoising

Code for the paper **"Deep learning for fast Monte-Carlo proton dose denoising"**
(J. Jawaid, M. Spezialetti, G. Stilo, F. Mignosi, A. Langiu, M. Schwarz), 2 mm isotropic experiments.

A 3D U-Net refines a noisy Monte Carlo proton dose simulated with 5×10³ primary histories, together with
the patient CT, into a dose comparable to a 10⁶-history simulation (200 times fewer histories). The network
works on full 256×32×256 volumes (2 mm voxels, 512 × 64 × 512 mm) with beams at arbitrary angles in the
X–Z plane, without rotating or cropping the volume around the beam. It is trained with a weighted MSE
normalized by its moving average plus a differentiable surrogate of the gamma passing rate (GPR).

The TOPAS code that generates the training data is in a separate repository:
[TOPAS_sim](https://github.com/junaidjawaid1/TOPAS_sim).

## Results

Mean gamma passing rates (%) on absolute doses at 2 mm, 20% lower-dose cutoff, global normalization
(PyMedPhys), 89 test volumes. *Direct*: the 10⁶-history dose is the reference. *Reverse*: the dose under
evaluation is the reference; this direction is the one that detects the spurious hot spots of
low-statistics MC noise.

| Test | Criterion | 5×10³ input | Network |
|---|---|---|---|
| direct | 2%/2 mm | 99.83 | 99.83 |
| direct | 1%/1 mm | 83.57 | 98.68 |
| reverse | 2%/2 mm | 79.92 | 99.92 |
| reverse | 1%/1 mm | 52.46 | 98.70 |

Inference takes 0.464 s per volume (8.27 GB peak GPU memory) on an NVIDIA A100. The full pipeline
(5×10³-history simulation + network) is 5.84× faster than the 10⁶-history simulation.

## Repository structure

```
config.yaml              all hyperparameters used in the paper
dose_denoising/          Python package
  blocks.py              Multiscale Elaboration, Reduction and Expansion blocks (Fig. 1)
  model.py               DoseDenoisingUNet with the stepwise 2 -> 1 channel output head
  losses.py              differentiable gamma loss, weighted MSE, combined loss, beta schedule (Eq. 3-11)
  data.py                dataset (normalization and 90-degree rotation augmentation)
  evaluation.py          absolute dose recovery (Eq. 12), direct / reverse gamma with PyMedPhys
  config.py              config loading
scripts/
  train.py               training
  evaluate.py            gamma evaluation of the network or of the 5e3-history input (Table 2)
  benchmark_inference.py inference time and GPU memory (Table 4)
  predict.py             denoise a single dose volume
slurm/                   example SLURM job scripts (train.sbatch, evaluate.sbatch)
requirements.txt         pinned Python dependencies
environment.yml          conda environment (Python 3.10 + requirements.txt)
CITATION.cff, LICENSE
```

## Installation

Python 3.10 and a CUDA GPU with bfloat16 support (Ampere or newer) are needed for training. Evaluation and
`predict.py` also run on the CPU, slowly.

```bash
git clone https://github.com/junaidjawaid1/MC_denoising_deep_learning.git
cd MC_denoising_deep_learning
conda env create -f environment.yml
conda activate proton-dose-denoising
```

or with pip:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

`requirements.txt` pins the versions used for the paper (PyTorch 2.7.1 with CUDA 11.8, PyMedPhys 0.41.0).

## Data

The CTs are public: head and neck from
[Head-Neck-PET-CT](https://doi.org/10.7937/K9/TCIA.2017.8oje5q00) (TCIA) and thorax from the
[COVID-19 rural U.S. population chest imaging collection](https://doi.org/10.1038/s41597-020-00741-6).
The doses are simulated with TOPAS using
[TOPAS_sim](https://github.com/junaidjawaid1/TOPAS_sim), which writes the
layout below directly:

```
dataset_2mm/
  train/        ct/CT_<id>.npy   dose_5k/<id>_5k.npy   dose_1m/<id>_1M.npy
  validation/   (same)
  test/         (same)
```

* `ct/CT_<id>.npy`: CT in Hounsfield units.
* `dose_5k/<id>_5k.npy`: 5×10³-history dose, **already multiplied by 200** so that it is on the scale of
  the 10⁶-history dose.
* `dose_1m/<id>_1M.npy`: 10⁶-history dose.

All arrays have shape 256×32×256. Sample IDs are read from the file names in `dose_1m/`. The paper used
787 / 77 / 89 volumes (train / validation / test); no CT is shared between splits.

Inputs are normalized as in the paper: CT → (HU + 1000) / 3000, both doses min–max. Every volume is used
four times, rotated by 0, 90, 180 and 270 degrees in the X–Z plane.

## Training

```bash
python scripts/train.py --data-root /path/to/dataset_2mm --output-dir runs
```

The run directory (`runs/<timestamp>/`) gets a copy of the config, `training_log.csv` with per-epoch losses
and GPR, and `best_model.pth`, the weights with the highest validation GPR. Add `--wandb` to log to
Weights & Biases. All settings are in `config.yaml`:

| | |
|---|---|
| Loss | normalized WMSE (α = 1, EMA momentum 0.99) + differentiable GPR, 2%/2 mm, 20% cutoff, at 1 mm³ |
| Optimizer | Adam, learning rate 10⁻⁵, ×0.7 when validation GPR stalls 10 epochs (minimum 10⁻⁶) |
| Epochs | 150, bfloat16 mixed precision |
| β schedule | 0.1 for 10 epochs, linear to 2.0 at epoch 50, to 5.0 at epoch 100, updated every 5 epochs |
| Batch size | 2 |

The gamma loss is memory-hungry: one training step with batch size 2 peaks at about 58 GB of GPU memory.
The original run did not fix a random seed; set `training.seed` for repeatable runs.

## Evaluation

```bash
# network prediction: every test volume under the 4 rotations (n = 356)
python scripts/evaluate.py --data-root /path/to/dataset_2mm \
    --checkpoint runs/<timestamp>/best_model.pth --output-dir results/prediction

# 5e3-history input baseline: one evaluation per test volume (n = 89)
python scripts/evaluate.py --data-root /path/to/dataset_2mm --baseline --output-dir results/baseline
```

`results.csv` has one row per evaluated volume and `summary.csv` the n / min / max / mean / median / std
reported in Table 2. All criteria and directions in `config.yaml` (`evaluation.criteria`,
`evaluation.directions`) are computed in one run. PyMedPhys is slow: expect several hours for the test
set. An interrupted run resumes from `results.csv`.

How the evaluation works:

* **Absolute dose (Eq. 12).** The network output is max-normalized, then rescaled so that its integral
  matches that of the 5×10³-history input: `s = Σ D_5k / Σ p`, summed over the voxels where the
  prediction is ≥ 10% of its maximum. No 10⁶-history information is used. The 5×10³-history input is
  evaluated as is.
* **Gamma.** `pymedphys.gamma` with global normalization to the 10⁶-history maximum, 20% lower-dose cutoff,
  `interp_fraction=10`, pass if γ < 1. The passing rate is computed over the voxels where the 10⁶-history
  dose is ≥ 20% of its maximum. In the reverse test the evaluated dose is the PyMedPhys reference, so a
  voxel must also be above the cutoff in the evaluated dose.

## Inference on a new volume

```bash
python scripts/predict.py --checkpoint runs/<timestamp>/best_model.pth --ct CT_001.npy --dose-5k 001_5k.npy \
    --output 001_denoised.npy
```

The output is the denoised dose in the units of the input dose. Each dimension of the volume must be a
multiple of 8.

## Inference time and memory

```bash
python scripts/benchmark_inference.py
```

Batch size 1, bfloat16, 5 warm-up and 20 timed forward passes. Reports the time per volume and the peak
allocated (and reserved) GPU memory, as in Table 4. The Monte Carlo timings come from the
`simulations.csv` written by the TOPAS repository.

## Citation

```bibtex
@article{jawaid2026proton,
  title   = {Deep learning for fast Monte-Carlo proton dose denoising},
  author  = {Jawaid, Junaid and Spezialetti, Matteo and Stilo, Giovanni and Mignosi, Filippo and
             Langiu, Alessio and Schwarz, Marco},
  year    = {2026},
  note    = {TODO: journal, volume, DOI}
}
```

The network builds on Spezialetti et al., *Using deep learning for fast dose refinement in proton therapy*,
IEEE SMC 2021, and the gamma loss on Martinot et al., *Differentiable gamma index-based loss functions*,
IPMI 2023.

## License

MIT; see [LICENSE](LICENSE).
