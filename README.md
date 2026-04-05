# CoBELa: Concept Bottlenecks on Energy Landscapes

Reproduction of "Steering Transparent Generation via Concept Bottlenecks on Energy Landscapes" (Kim et al., 2026).

### Step 1: Setup environment and download weights

```bash
python setup.py                      # full setup (deps + weights)
python setup.py --check              # verify only
python setup.py --skip-cub           # CelebA-HQ only
```

### Step 2: Train

```bash
python train.py --dataset celebahq                  # full training (50 epochs)
python train.py --dataset celebahq --quick           # quick test (2 epochs)
python train.py --dataset cub                        # CUB dataset
```

### Step 3: Evaluate

```bash
python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt
python evaluate.py --dataset cub --checkpoint checkpoints/cobela/cub_final.pt
python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt --fid
python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt --intervene --intervene-num-samples 4 --intervene-random
```
