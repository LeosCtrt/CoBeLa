# CoBELa: Concept Bottlenecks on Energy Landscapes

**Unofficial implementation** of *"Steering Transparent Generation via Concept Bottlenecks on Energy Landscapes"* (Kim et al., 2026) — [arXiv:2507.08334](https://arxiv.org/abs/2507.08334)

This repository was developed as a course project for the **Explainable AI (XAI)** class of the [MVA Master's program](https://www.master-mva.com/) (ENS Paris-Saclay, 2025–2026).

> **Disclaimer:** This is not the official implementation by the authors. It was built from scratch for educational and reproducibility purposes, reusing only the vendor utilities (`dnnlib/`, `torch_utils/`) and pretrained classifiers from the [CB-AE repository](https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm).

---

## Overview

CoBELa is a decoder-free, energy-based concept bottleneck framework for interpretable image generation. It operates on the latent space of a frozen pretrained StyleGAN2 generator: a concept-conditioned energy network learns per-concept energies whose gradients guide DDIM denoising, enabling compositional concept interventions (conjunction and negation) without retraining the generator.



---

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- A GPU with at least 12 GB VRAM (16 GB recommended for full latent mode)

### Installation

```bash
git clone https://github.com/LeosCtrt/CoBeLa.git
cd CoBeLa
pip install -r requirements.txt
python setup.py
```

This will install dependencies, clone vendor files from CB-AE, download pretrained StyleGAN2 weights and concept classifiers, and verify the environment.

```bash
python setup.py --check       # verify only (no downloads)
python setup.py --skip-cub    # skip CUB weights (CelebA-HQ only)
```

---

## Training

```bash
python train.py --dataset celebahq
python train.py --dataset cub
```

### Hyperparameters

All hyperparameters have defaults from the config files (`configs/celebahq.yaml`, `configs/cub.yaml`). CLI flags override the config values.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | `celebahq` or `cub` |
| `--latent-mode` | `single` | Latent space parameterization: `single`, `subset`, or `full` |
| `--batch-size` | 16 | Training batch size |
| `--lr` | 1e-4 | Learning rate (Adam) |
| `--epochs` | 50 | Number of training epochs |
| `--lambda-score` | 1.0 | Weight for the score-matching loss |
| `--lambda-concept` | 1e-3 | Weight for the concept classification loss |
| `--resume` | — | Path to a checkpoint to resume training from |
| `--quick` | — | Quick test run (2 epochs, 50 steps) |
| `--device` | `cuda` | Device to train on |

### Latent modes

- **`single`** — Uses a single style vector `w[0]` (dim 512). Cheapest, matches CB-AE's default.
- **`subset`** — Uses style slots `[8, 9, 10, 11, 12, 13]` (dim 3072). Middle ground.
- **`full`** — Uses the entire W+ tensor (dim 7168). Most expressive, hardest to optimize.

### Example: two-phase training

We found that a two-phase schedule produces the best CA/FID tradeoff. First train with default config, then resume with higher concept weight:

```bash
# Phase 1: default config
python train.py --dataset cub --latent-mode single --batch-size 128 --epochs 20

# Phase 2: push concept accuracy
python train.py --dataset cub --latent-mode single --batch-size 128 \
    --resume checkpoints/cobela/cub_final.pt \
    --lr 5e-5 --lambda-concept 1.0 --epochs 30
```

---

## Evaluation

### Concept accuracy

```bash
python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt
```

### Concept accuracy + FID

```bash
python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt \
    --fid --num-samples 5000
```

### Concept interventions (qualitative figures)

```bash
python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt \
    --intervene --intervene-num-samples 4 --intervene-random
```

---

## Results

Results from our reproduction (see the full report for discussion):

| Method | CelebA-HQ CA (%) | CelebA-HQ FID | CUB CA (%) | CUB FID |
|--------|:-:|:-:|:-:|:-:|
| CB-AE (Kulkarni et al.) | 74.38 | 9.77 | 75.56 | 8.37 |
| CoBELa (paper) | 75.70 | 6.47 | **82.42** | 5.37 |
| Ours — single latent | 69.05 | **5.12** | 78.75 | **4.59** |
| Ours — full latent | **79.35** | 9.79 | 61.50 | 4.97 |

CelebA-HQ uses ResNet-18 pseudo-labelers, CUB uses ResNet-50 (matching CB-AE's available checkpoints).

---

## Project structure

```
CoBeLa/
├── cobela/                  # Core CoBELa implementation
│   ├── __init__.py          # Patches and imports
│   ├── energy_network.py    # Energy network Eθ
│   ├── losses.py            # Score-matching + concept loss
│   ├── noise_schedule.py    # Cosine noise schedule
│   ├── ddim_sampler.py      # Concept-guided DDIM sampling
│   ├── latent_space.py      # Latent mode selection (single/subset/full)
│   ├── pseudolabeler.py     # Pseudo-labeler wrapper
│   └── stylegan2_wrapper.py # StyleGAN2 g1/g2 split
├── configs/                 # Dataset configs (YAML)
├── dnnlib/                  # StyleGAN2 vendor (from CB-AE)
├── torch_utils/             # StyleGAN2 vendor (from CB-AE)
├── train.py                 # Training script
├── evaluate.py              # Evaluation script (CA, FID, interventions)
└── setup.py                 # Environment setup and weight download
```

---

## References

- **CoBELa paper:** Kim et al., *"Steering Transparent Generation via Concept Bottlenecks on Energy Landscapes"*, 2026. [arXiv:2507.08334](https://arxiv.org/abs/2507.08334)
- **CB-AE:** Kulkarni et al., *"Interpretable Generative Models through Post-hoc Concept Bottlenecks"*, CVPR 2025. [GitHub](https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm)
- **StyleGAN2:** Karras et al., *"Analyzing and Improving the Image Quality of StyleGAN"*, CVPR 2020.

---

## Authors

- Yannis Kolodziej
- Léos Coutrot

MVA Master's program, ENS Paris-Saclay — XAI course, April 2026.
