import os
import sys
import argparse
import subprocess
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
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


def _plot_top3_intervention_comparison(
    images: list[torch.Tensor],
    scores: list[torch.Tensor],
    score_concept_indices: list[int],
    concept_names,
    titles: list[str],
    output_path: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), gridspec_kw={"height_ratios": [1.3, 1.0]})
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(score_concept_indices)))
    selected_names = [concept_names[index] for index in score_concept_indices]
    y = np.arange(len(score_concept_indices))

    for column in range(3):
        axes[0, column].imshow(_tensor_to_image(images[column]))
        axes[0, column].axis("off")
        axes[0, column].set_title(titles[column])

        score_values = scores[column].detach().cpu().numpy()[score_concept_indices]
        axes[1, column].barh(y, score_values, color=colors)
        axes[1, column].set_yticks(y, selected_names)
        axes[1, column].invert_yaxis()
        axes[1, column].set_xlim(0.0, 1.0)
        axes[1, column].set_xlabel("concept scores")

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

def run_interventions(
    energy_net,
    g1,
    g2,
    noise_schedule,
    latent_config,
    concept_names,
    output_dir,
    device,
    num_samples: int = 4,
    randomize: bool = False,
):
    """Generate intervention visualization (Fig. 3 style)."""
    os.makedirs(output_dir, exist_ok=True)
    if not randomize:
        torch.manual_seed(123)

    K = len(concept_names)
    summaries = []

    for i in range(num_samples):
        z = torch.randn(1, g1.z_dim, device=device)
        img_orig, _, scores_orig = concept_guided_sample(
            energy_net, g1, g2, noise_schedule, z=z, latent_config=latent_config, device=device,
        )

        topk = torch.topk(scores_orig[0], k=min(3, K))
        top3_indices = topk.indices.tolist()
        if randomize:
            random_order = torch.randperm(len(top3_indices)).tolist()
            single_idx = top3_indices[random_order[0]]
            pair_indices = [top3_indices[random_order[0]], top3_indices[random_order[1]]] if len(top3_indices) > 1 else [top3_indices[0]]
        else:
            single_idx = top3_indices[0]
            pair_indices = top3_indices[:2] if len(top3_indices) > 1 else [top3_indices[0]]

        img_single, _, scores_single = generate_with_negation(
            energy_net,
            g1,
            g2,
            noise_schedule,
            negate_concepts=[single_idx],
            z=z,
            latent_config=latent_config,
            device=device,
        )
        img_pair, _, scores_pair = generate_with_negation(
            energy_net,
            g1,
            g2,
            noise_schedule,
            negate_concepts=pair_indices,
            z=z,
            latent_config=latent_config,
            device=device,
        )

        sample_stem = f"sample_{i:02d}"
        single_slug = _slugify(concept_names[single_idx])
        pair_slug = "_".join(_slugify(concept_names[index]) for index in pair_indices)
        comparison_name = f"neg_{single_slug}_neg_{pair_slug}.png"

        _plot_top3_intervention_comparison(
            images=[img_orig[0], img_single[0], img_pair[0]],
            scores=[scores_orig[0], scores_single[0], scores_pair[0]],
            score_concept_indices=top3_indices,
            concept_names=concept_names,
            titles=[
                "Original",
                f"Negate {_slugify(concept_names[single_idx]).replace('_', ' ')}",
                "Negate " + " and ".join(
                    _slugify(concept_names[index]).replace("_", " ") for index in pair_indices
                ),
            ],
            output_path=os.path.join(output_dir, comparison_name),
        )

        summaries.append(
            {
                "sample": sample_stem,
                "comparison_path": os.path.join(output_dir, comparison_name),
                "top3_concepts": [concept_names[index] for index in top3_indices],
                "single_negation": concept_names[single_idx],
                "pair_negation": [concept_names[index] for index in pair_indices],
                "original_scores": {
                    concept_names[idx]: float(scores_orig[0, idx].item()) for idx in top3_indices
                },
                "single_negation_scores": {
                    concept_names[idx]: float(scores_single[0, idx].item()) for idx in top3_indices
                },
                "pair_negation_scores": {
                    concept_names[idx]: float(scores_pair[0, idx].item()) for idx in top3_indices
                },
            }
        )

    with open(os.path.join(output_dir, "intervention_scores.json"), "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
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
    parser.add_argument("--intervene-num-samples", type=int, default=4,
                        help="Number of intervention samples to generate when using --intervene")
    parser.add_argument("--intervene-random", action="store_true",
                        help="Randomly choose the single and pair negations among the top-3 concepts")
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
        run_interventions(
            energy_net,
            g1,
            g2,
            noise_schedule,
            latent_cfg,
            concept_names,
            intv_dir,
            device,
            num_samples=args.intervene_num_samples,
            randomize=args.intervene_random,
        )

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
