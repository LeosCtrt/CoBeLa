#!/usr/bin/env python3
"""
CoBELa Evaluation Script
==========================
Evaluate concept accuracy (CA) and optionally FID on CelebA-HQ or CUB.

Usage:
    # Concept accuracy only
    python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt

    # With FID computation
    python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt --fid

    # Quick test (100 samples)
    python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt --quick

    # With concept intervention demo (saves visualizations)
    python evaluate.py --dataset celebahq --checkpoint checkpoints/cobela/celebahq_final.pt --intervene
"""

import os
import sys
import argparse
import subprocess
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cobela
cobela.patch_stylegan2_ops()

from cobela.energy_network import EnergyNetwork
from cobela.latent_space import resolve_latent_config
from cobela.noise_schedule import CosineNoiseSchedule
from cobela.ddim_sampler import concept_guided_sample, generate_with_negation
from cobela.stylegan2_wrapper import load_stylegan2, MappingWrapper, SynthesisWrapper
from cobela.pseudolabeler import PseudoLabeler


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _tensor_to_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(-1, 1)
    image = ((image + 1.0) / 2.0).permute(1, 2, 0).numpy()
    return image


def _plot_image_and_scores(image: torch.Tensor, scores: torch.Tensor, concept_names, title: str, output_path: str):
    scores = scores.detach().cpu().numpy()
    image_np = _tensor_to_image(image)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), gridspec_kw={"width_ratios": [1.0, 1.15]})
    axes[0].imshow(image_np)
    axes[0].axis("off")
    axes[0].set_title(title)

    y = np.arange(len(concept_names))
    axes[1].barh(y, scores, color=plt.cm.Pastel1(np.linspace(0, 1, len(concept_names))))
    axes[1].set_yticks(y, concept_names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("concept scores")
    axes[1].set_title("Scores")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_intervention_comparison(
    original_image: torch.Tensor,
    original_scores: torch.Tensor,
    intervened_image: torch.Tensor,
    intervened_scores: torch.Tensor,
    concept_names,
    intervention_label: str,
    output_path: str,
):
    original_scores = original_scores.detach().cpu().numpy()
    intervened_scores = intervened_scores.detach().cpu().numpy()
    original_np = _tensor_to_image(original_image)
    intervened_np = _tensor_to_image(intervened_image)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1.0, 1.1, 1.0, 1.1]})

    axes[0].imshow(original_np)
    axes[0].axis("off")
    axes[0].set_title("Original")

    y = np.arange(len(concept_names))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(concept_names)))
    axes[1].barh(y, original_scores, color=colors)
    axes[1].set_yticks(y, concept_names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("concept scores")
    axes[1].set_title("Original scores")

    axes[2].imshow(intervened_np)
    axes[2].axis("off")
    axes[2].set_title(intervention_label)

    axes[3].barh(y, intervened_scores, color=colors)
    axes[3].set_yticks(y, concept_names)
    axes[3].invert_yaxis()
    axes[3].set_xlim(0.0, 1.0)
    axes[3].set_xlabel("concept scores")
    axes[3].set_title("Intervened scores")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_style_overview(
    energy_net,
    g1,
    g2,
    noise_schedule,
    latent_config,
    concept_names,
    output_path: str,
    device: str,
    n_rows: int = 2,
    n_interventions: int = 2,
):
    """
    Create a compact figure with one original image per row, the associated
    concept scores, and several intervention results from the same latent.
    """
    selected_indices = list(range(min(n_interventions, len(concept_names))))
    if len(selected_indices) == 0:
        return

    fig, axes = plt.subplots(
        n_rows,
        2 + len(selected_indices),
        figsize=(3.2 * (2 + len(selected_indices)), 3.3 * n_rows),
        gridspec_kw={"width_ratios": [1.0, 1.15] + [1.0] * len(selected_indices)},
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(concept_names)))

    for row in range(n_rows):
        z = torch.randn(1, g1.z_dim, device=device)
        image_orig, _, scores_orig = concept_guided_sample(
            energy_net,
            g1,
            g2,
            noise_schedule,
            z=z,
            latent_config=latent_config,
            device=device,
        )

        axes[row, 0].imshow(_tensor_to_image(image_orig[0]))
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Original")

        orig_scores_np = scores_orig[0].detach().cpu().numpy()
        y = np.arange(len(concept_names))
        axes[row, 1].barh(y, orig_scores_np, color=colors)
        axes[row, 1].set_yticks(y, concept_names)
        axes[row, 1].invert_yaxis()
        axes[row, 1].set_xlim(0.0, 1.0)
        axes[row, 1].set_xlabel("concept scores")
        if row == 0:
            axes[row, 1].set_title("Concepts")

        for col_offset, concept_idx in enumerate(selected_indices, start=2):
            image_neg, _, scores_neg = generate_with_negation(
                energy_net,
                g1,
                g2,
                noise_schedule,
                negate_concepts=[concept_idx],
                z=z,
                latent_config=latent_config,
                device=device,
            )
            axes[row, col_offset].imshow(_tensor_to_image(image_neg[0]))
            axes[row, col_offset].axis("off")
            if row == 0:
                axes[row, col_offset].set_title(f"Negate {concept_names[concept_idx]}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Load model from checkpoint ────────────────────────────────────────

def load_cobela(ckpt_path, gen_info, device="cuda"):
    """Load a trained CoBELa energy network."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    en_cfg = cfg.get("energy_network", {})
    latent_cfg = ckpt.get("latent_space", None)
    if latent_cfg is None:
        latent_cfg = resolve_latent_config(
            cfg.get("latent_space", None),
            num_ws=gen_info["num_ws"],
            w_dim=gen_info["w_dim"],
        )

    energy_net = EnergyNetwork(
        latent_dim=latent_cfg["latent_dim"],
        num_concepts=cfg.get("concepts", {}).get("num_concepts", 8),
        concept_embed_dim=en_cfg.get("concept_embed_dim", 128),
        time_embed_dim=en_cfg.get("time_embed_dim", 128),
        hidden_dim=en_cfg.get("hidden_dim", 512),
        num_res_blocks=en_cfg.get("num_res_blocks", 2),
    ).to(device)

    energy_net.load_state_dict(ckpt["model_state_dict"])
    energy_net.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"[model] Loaded from {ckpt_path} (epoch {epoch})")
    print(
        f"[model] latent mode={latent_cfg['mode']} "
        f"indices={latent_cfg['selected_indices']} "
        f"latent_dim={latent_cfg['latent_dim']}"
    )
    return energy_net, latent_cfg


# ── Concept Accuracy evaluation ───────────────────────────────────────

@torch.no_grad()
def evaluate_concept_accuracy(
    energy_net, g1, g2, pseudolabeler, noise_schedule, latent_config,
    num_samples, batch_size, Ts, ddim_steps, seed, save_dir, device,
):
    """Generate images and compute concept accuracy."""
    torch.manual_seed(seed)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_scores, all_labels = [], []
    img_count = 0
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        current_batch = min(batch_size, num_samples - img_count)
        z = torch.randn(current_batch, g1.z_dim, device=device)

        images, _, scores = concept_guided_sample(
            energy_net, g1, g2, noise_schedule,
            z=z, Ts=Ts, ddim_steps=ddim_steps, latent_config=latent_config, device=device,
        )
        labels = pseudolabeler(images)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

        if save_dir:
            for j in range(current_batch):
                save_image((images[j:j+1] + 1) / 2, f"{save_dir}/{img_count:05d}.png")
                img_count += 1
        else:
            img_count += current_batch

        if (i + 1) % 50 == 0 or (i + 1) == num_batches:
            print(f"  [{img_count}/{num_samples}] generated...")

    all_scores = torch.cat(all_scores)[:num_samples]
    all_labels = torch.cat(all_labels)[:num_samples]

    # Overall CA
    pred_binary = (all_scores >= 0.5).long()
    ca = (pred_binary == all_labels.long()).float().mean().item() * 100.0

    # Per-concept CA
    K = all_scores.shape[1]
    per_concept = []
    for k in range(K):
        acc = ((all_scores[:, k] >= 0.5).long() == all_labels[:, k].long()).float().mean().item() * 100.0
        per_concept.append(acc)

    return ca, per_concept


# ── FID computation ───────────────────────────────────────────────────

def compute_fid(gen_img_dir, ref_img_dir, G, num_samples, device):
    """Generate reference images from StyleGAN2 and compute FID."""
    os.makedirs(ref_img_dir, exist_ok=True)

    # Generate reference images (same seeds as evaluation)
    existing = len([f for f in os.listdir(ref_img_dir) if f.endswith(".png")])
    if existing < num_samples:
        print(f"[fid] Generating {num_samples} StyleGAN2 reference images...")
        torch.manual_seed(42)
        count = 0
        while count < num_samples:
            batch = min(16, num_samples - count)
            z = torch.randn(batch, G.z_dim, device=device)
            with torch.no_grad():
                w = G.mapping(z, None)
                imgs = G.synthesis(w, noise_mode="const")
            for j in range(batch):
                save_image((imgs[j:j+1] + 1) / 2, f"{ref_img_dir}/{count:05d}.png")
                count += 1
            if count % 1000 == 0:
                print(f"  [{count}/{num_samples}]")

    print("[fid] Computing FID...")
    result = subprocess.run(
        [sys.executable, "-m", "pytorch_fid", ref_img_dir, gen_img_dir],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    return result.stdout.strip()


# ── Intervention visualization ────────────────────────────────────────

def run_interventions(energy_net, g1, g2, noise_schedule, latent_config, concept_names, output_dir, device):
    """Generate intervention visualization (Fig. 3 style)."""
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(123)

    K = len(concept_names)
    n_samples = 4
    summaries = []

    for negate_k in range(min(K, 4)):  # show first 4 concepts
        name = concept_names[negate_k]
        print(f"  Intervention: negating '{name}'...")

        all_imgs = []
        for i in range(n_samples):
            z = torch.randn(1, g1.z_dim, device=device)

            # Original
            img_orig, _, scores_orig = concept_guided_sample(
                energy_net, g1, g2, noise_schedule, z=z, latent_config=latent_config, device=device,
            )
            # Negated
            img_neg, _, scores_neg = generate_with_negation(
                energy_net, g1, g2, noise_schedule,
                negate_concepts=[negate_k], z=z, latent_config=latent_config, device=device,
            )
            all_imgs.extend([img_orig[0], img_neg[0]])

            sample_stem = f"negate_{_slugify(name)}_sample_{i:02d}"
            sample_dir = os.path.join(output_dir, sample_stem)
            os.makedirs(sample_dir, exist_ok=True)
            _plot_image_and_scores(
                img_orig[0],
                scores_orig[0],
                concept_names,
                title="Original",
                output_path=os.path.join(sample_dir, "original_scores.png"),
            )
            _plot_intervention_comparison(
                img_orig[0],
                scores_orig[0],
                img_neg[0],
                scores_neg[0],
                concept_names,
                intervention_label=f"Negate {name}",
                output_path=os.path.join(sample_dir, "comparison.png"),
            )
            save_image((img_orig + 1) / 2, os.path.join(sample_dir, "original.png"))
            save_image((img_neg + 1) / 2, os.path.join(sample_dir, "intervened.png"))

            summaries.append(
                {
                    "sample": sample_stem,
                    "sample_dir": sample_dir,
                    "negated_concepts": [name],
                    "original_scores": {
                        concept_names[idx]: float(scores_orig[0, idx].item()) for idx in range(K)
                    },
                    "intervened_scores": {
                        concept_names[idx]: float(scores_neg[0, idx].item()) for idx in range(K)
                    },
                }
            )

        grid = make_grid([(img + 1) / 2 for img in all_imgs], nrow=2)
        save_image(grid, os.path.join(output_dir, f"intervene_negate_{name}.png"))

    with open(os.path.join(output_dir, "intervention_scores.json"), "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    _plot_paper_style_overview(
        energy_net,
        g1,
        g2,
        noise_schedule,
        latent_config,
        concept_names,
        output_path=os.path.join(output_dir, "paper_style_overview.png"),
        device=device,
        n_rows=2,
        n_interventions=2,
    )
    print(f"  Saved to {output_dir}/")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate CoBELa")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["celebahq", "cub"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained CoBELa checkpoint")
    parser.add_argument("--fid", action="store_true",
                        help="Also compute FID")
    parser.add_argument("--intervene", action="store_true",
                        help="Generate intervention visualizations")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (100 samples)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override number of eval samples")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load config
    config_path = os.path.join(PROJECT_ROOT, "configs", f"{args.dataset}.yaml")
    cfg = OmegaConf.load(config_path)

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

    # Load generator
    G, g1, g2, gen_info = load_stylegan2(resolve(cfg.generator.weights), device=device)

    # Load pseudo-labeler
    M = PseudoLabeler(
        weights_dir=resolve(cfg.pseudolabeler.weights_dir),
        concept_names=list(cfg.concepts.names),
        arch=cfg.pseudolabeler.arch,
        dataset_prefix=cfg.dataset.name,
        device=device,
    )

    # Load CoBELa
    energy_net, latent_cfg = load_cobela(args.checkpoint, gen_info, device=device)
    noise_schedule = CosineNoiseSchedule(max_timesteps=cfg.noise_schedule.max_timesteps)

    # Eval params
    num_samples = args.num_samples or (100 if args.quick else cfg.evaluation.num_samples)
    batch_size = cfg.training.batch_size
    Ts = cfg.sampling.Ts
    ddim_steps = cfg.sampling.ddim_steps
    seed = cfg.evaluation.seed

    concept_names = list(cfg.concepts.names)
    output_base = os.path.join(PROJECT_ROOT, "outputs", args.dataset)
    gen_img_dir = os.path.join(output_base, "cobela_images") if args.fid else None

    # ── Concept Accuracy ──
    print(f"\n{'='*60}")
    print(f"  Evaluating CoBELa on {args.dataset} ({num_samples} samples)")
    print(f"{'='*60}\n")

    ca, per_concept = evaluate_concept_accuracy(
        energy_net, g1, g2, M, noise_schedule,
        latent_config=latent_cfg,
        num_samples=num_samples,
        batch_size=batch_size,
        Ts=Ts,
        ddim_steps=ddim_steps,
        seed=seed,
        save_dir=gen_img_dir,
        device=device,
    )

    print(f"\n  Concept Accuracy: {ca:.2f}%")
    paper_target = {"celebahq": 75.70, "cub": 82.42}
    print(f"  Paper target:     {paper_target.get(args.dataset, '?')}%\n")
    print("  Per-concept:")
    for k, name in enumerate(concept_names):
        print(f"    {name}: {per_concept[k]:.1f}%")

    # ── FID ──
    if args.fid:
        print(f"\n{'='*60}")
        print(f"  Computing FID")
        print(f"{'='*60}\n")
        ref_img_dir = os.path.join(output_base, "stylegan2_reference")
        fid_result = compute_fid(gen_img_dir, ref_img_dir, G, num_samples, device)
        paper_fid = {"celebahq": 6.47, "cub": 5.37}
        print(f"  Paper target FID: {paper_fid.get(args.dataset, '?')}")

    # ── Interventions ──
    if args.intervene:
        print(f"\n{'='*60}")
        print(f"  Generating concept interventions")
        print(f"{'='*60}\n")
        intv_dir = os.path.join(output_base, "interventions")
        run_interventions(energy_net, g1, g2, noise_schedule, latent_cfg, concept_names, intv_dir, device)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Summary: {args.dataset}")
    print(f"{'='*60}")
    print(f"  CA:  {ca:.2f}% (target: {paper_target.get(args.dataset, '?')}%)")
    if args.fid:
        print(f"  FID: {fid_result}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
