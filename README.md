# CoBELa: Concept Bottlenecks on Energy Landscapes

Reproduction of "Steering Transparent Generation via Concept Bottlenecks on Energy Landscapes" (Kim et al., 2026).

## Project Structure

```
cobela_project/
├── setup.py               # Install deps + download weights
├── train.py               # Train CoBELa energy network
├── evaluate.py            # Evaluate CA + FID on CelebA-HQ / CUB
├── configs/
│   ├── celebahq.yaml      # CelebA-HQ experiment config
│   └── cub.yaml           # CUB experiment config
├── cobela/                # CoBELa source code (written from scratch)
│   ├── energy_network.py  # Eθ: FiLM-conditioned residual blocks
│   ├── noise_schedule.py  # Cosine diffusion schedule
│   ├── losses.py          # Score-matching + concept loss
│   ├── ddim_sampler.py    # Algorithm 1: concept-guided DDIM
│   ├── stylegan2_wrapper.py  # g1/g2 split wrapper
│   └── pseudolabeler.py   # Pseudo-labeler M wrapper
├── dnnlib/                # ← COPY from CB-AE repo (see below)
├── torch_utils/           # ← COPY from CB-AE repo (see below)
└── checkpoints/           # Downloaded by setup.py
```

## Quick Start

### Step 1: Copy StyleGAN2 dependencies from CB-AE

Clone the CB-AE repo somewhere temporary, then copy two directories:

```bash
git clone --depth 1 https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm.git /tmp/cbae

# Copy these two directories into the project root:
cp -r /tmp/cbae/dnnlib/*        ./dnnlib/
cp -r /tmp/cbae/torch_utils/*   ./torch_utils/

rm -rf /tmp/cbae
```

**Exact files required** (see below for the full list):
- `dnnlib/__init__.py`, `dnnlib/util.py`
- `torch_utils/__init__.py`, `custom_ops.py`, `misc.py`, `persistence.py`, `training_stats.py`
- `torch_utils/ops/` — ALL files (Python + CUDA sources)

### Step 2: Setup environment and download weights

```bash
python setup.py                      # full setup (deps + weights)
python setup.py --check              # verify only
python setup.py --skip-cub           # CelebA-HQ only
```

### Step 3: Train

```bash
python train.py --dataset celebahq                  # full training (50 epochs)
python train.py --dataset celebahq --quick           # quick test (2 epochs)
python train.py --dataset cub                        # CUB dataset
```

### Step 4: Evaluate

```bash
python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt
python evaluate.py --dataset cub --checkpoint checkpoints/cobela/cub_final.pt
python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt --fid
```

---

## Files to copy from CB-AE repo

Source repo: `https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm`

### dnnlib/ (2 files)

| Source path              | Copy to            |
|--------------------------|--------------------|
| `dnnlib/__init__.py`     | `dnnlib/__init__.py` |
| `dnnlib/util.py`         | `dnnlib/util.py`    |

### torch_utils/ (5 files + ops/)

| Source path                      | Copy to                        |
|----------------------------------|--------------------------------|
| `torch_utils/__init__.py`        | `torch_utils/__init__.py`      |
| `torch_utils/custom_ops.py`      | `torch_utils/custom_ops.py`    |
| `torch_utils/misc.py`            | `torch_utils/misc.py`          |
| `torch_utils/persistence.py`     | `torch_utils/persistence.py`   |
| `torch_utils/training_stats.py`  | `torch_utils/training_stats.py`|

### torch_utils/ops/ (20 files — copy ALL)

| Source path                              | Copy to                                |
|------------------------------------------|----------------------------------------|
| `torch_utils/ops/__init__.py`            | `torch_utils/ops/__init__.py`          |
| `torch_utils/ops/bias_act.py`            | `torch_utils/ops/bias_act.py`          |
| `torch_utils/ops/bias_act.cpp`           | `torch_utils/ops/bias_act.cpp`         |
| `torch_utils/ops/bias_act.cu`            | `torch_utils/ops/bias_act.cu`          |
| `torch_utils/ops/bias_act.h`             | `torch_utils/ops/bias_act.h`           |
| `torch_utils/ops/conv2d_gradfix.py`      | `torch_utils/ops/conv2d_gradfix.py`    |
| `torch_utils/ops/conv2d_resample.py`     | `torch_utils/ops/conv2d_resample.py`   |
| `torch_utils/ops/filtered_lrelu.py`      | `torch_utils/ops/filtered_lrelu.py`    |
| `torch_utils/ops/filtered_lrelu.cpp`     | `torch_utils/ops/filtered_lrelu.cpp`   |
| `torch_utils/ops/filtered_lrelu.cu`      | `torch_utils/ops/filtered_lrelu.cu`    |
| `torch_utils/ops/filtered_lrelu.h`       | `torch_utils/ops/filtered_lrelu.h`     |
| `torch_utils/ops/filtered_lrelu_ns.cu`   | `torch_utils/ops/filtered_lrelu_ns.cu` |
| `torch_utils/ops/filtered_lrelu_rd.cu`   | `torch_utils/ops/filtered_lrelu_rd.cu` |
| `torch_utils/ops/filtered_lrelu_wr.cu`   | `torch_utils/ops/filtered_lrelu_wr.cu` |
| `torch_utils/ops/fma.py`                 | `torch_utils/ops/fma.py`              |
| `torch_utils/ops/grid_sample_gradfix.py` | `torch_utils/ops/grid_sample_gradfix.py`|
| `torch_utils/ops/upfirdn2d.py`           | `torch_utils/ops/upfirdn2d.py`        |
| `torch_utils/ops/upfirdn2d.cpp`          | `torch_utils/ops/upfirdn2d.cpp`       |
| `torch_utils/ops/upfirdn2d.cu`           | `torch_utils/ops/upfirdn2d.cu`        |
| `torch_utils/ops/upfirdn2d.h`            | `torch_utils/ops/upfirdn2d.h`         |

**Total: 27 files** from CB-AE. Everything else is written from scratch.
