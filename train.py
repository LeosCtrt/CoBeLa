#!/usr/bin/env python3
"""
CoBELa Training Script
=======================
Train the energy network Eθ on synthetic data from a frozen StyleGAN2.

Usage:
    python train.py --dataset celebahq                    # full training
    python train.py --dataset celebahq --quick            # 2 epochs test
    python train.py --dataset cub                         # CUB dataset
    python train.py --dataset celebahq --resume checkpoints/cobela/celebahq_epoch10.pt
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from omegaconf import OmegaConf

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cobela  # applies numpy patches, sys.path, warnings
cobela.patch_stylegan2_ops()

from cobela.energy_network import EnergyNetwork
from cobela.latent_space import extract_energy_latent, resolve_latent_config
from cobela.noise_schedule import CosineNoiseSchedule
from cobela.losses import cobela_loss
from cobela.stylegan2_wrapper import load_stylegan2
from cobela.pseudolabeler import PseudoLabeler


def train(args):
    device = args.device if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # ── Load config ──
    config_path = os.path.join(PROJECT_ROOT, "configs", f"{args.dataset}.yaml")
    cfg = OmegaConf.load(config_path)
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.lambda_score is not None:
        cfg.training.lambda_score = args.lambda_score
    if args.lambda_concept is not None:
        cfg.training.lambda_concept = args.lambda_concept
    if args.latent_mode is not None:
        cfg.latent_space.mode = args.latent_mode
    print(f"[config] Loaded {config_path}")

    # ── Resolve paths ──
    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

    # ── Load frozen generator ──
    G, g1, g2, gen_info = load_stylegan2(resolve(cfg.generator.weights), device=device)

    # ── Load pseudo-labeler ──
    M = PseudoLabeler(
        weights_dir=resolve(cfg.pseudolabeler.weights_dir),
        concept_names=list(cfg.concepts.names),
        arch=cfg.pseudolabeler.arch,
        dataset_prefix=cfg.dataset.name,
        device=device,
    )

    # ── Build energy network ──
    latent_cfg = resolve_latent_config(
        cfg.get("latent_space", None),
        num_ws=gen_info["num_ws"],
        w_dim=gen_info["w_dim"],
    )
    energy_net = EnergyNetwork(
        latent_dim=latent_cfg["latent_dim"],
        num_concepts=cfg.concepts.num_concepts,
        concept_embed_dim=cfg.energy_network.concept_embed_dim,
        time_embed_dim=cfg.energy_network.time_embed_dim,
        hidden_dim=cfg.energy_network.hidden_dim,
        num_res_blocks=cfg.energy_network.num_res_blocks,
    ).to(device)

    num_params = sum(p.numel() for p in energy_net.parameters())
    print(f"[model] Energy network: {num_params:,} parameters")

    # ── Optimizer ──
    optimizer = Adam(energy_net.parameters(), lr=cfg.training.lr)
    start_epoch = 1

    # ── Resume from checkpoint ──
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        energy_net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] Loaded checkpoint from epoch {ckpt['epoch']}")

    # ── Training params ──
    noise_schedule = CosineNoiseSchedule(max_timesteps=cfg.noise_schedule.max_timesteps)
    batch_size = cfg.training.batch_size
    epochs = 2 if args.quick else cfg.training.epochs
    steps_per_epoch = 50 if args.quick else cfg.training.num_samples_per_epoch // batch_size
    T_max = cfg.noise_schedule.max_timesteps
    lambda_score = cfg.training.lambda_score
    lambda_concept = cfg.training.lambda_concept
    grad_clip = cfg.training.get("grad_clip", 1.0)
    log_every = 10 if args.quick else cfg.training.log_every
    save_every = 1 if args.quick else cfg.training.save_every

    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "cobela")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n[train] {args.dataset}: {epochs} epochs × {steps_per_epoch} steps, "
          f"batch_size={batch_size}, lr={cfg.training.lr}")
    print(f"[train] λ_score={lambda_score}, λ_concept={lambda_concept}")
    print(
        f"[train] latent mode={latent_cfg['mode']} "
        f"indices={latent_cfg['selected_indices']} "
        f"latent_dim={latent_cfg['latent_dim']}"
    )
    if args.quick:
        print("[train] ⚡ Quick mode enabled (2 epochs, 50 steps)")

    # ── Training loop ──
    all_losses = []

    for epoch in range(start_epoch, start_epoch + epochs):
        energy_net.train()
        ep_total, ep_score, ep_concept = 0.0, 0.0, 0.0
        t0 = time.time()

        for step in range(1, steps_per_epoch + 1):
            # Generate synthetic data
            z = torch.randn(batch_size, gen_info["z_dim"], device=device)
            with torch.no_grad():
                w = g1(z)
                v = extract_energy_latent(w, latent_cfg).clone()
                imgs = g2(w)
                pseudo_labels = M(imgs)

            # Noise the latent
            t = torch.randint(0, T_max + 1, (batch_size,), device=device)
            vt, noise = noise_schedule.noise_latent(v, t)
            vt = vt.detach().requires_grad_(True)

            # Loss
            losses = cobela_loss(
                energy_net, vt, t, noise, pseudo_labels,
                lambda_score=lambda_score, lambda_concept=lambda_concept,
            )

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(energy_net.parameters(), grad_clip)
            optimizer.step()

            ep_total += losses["total"].item()
            ep_score += losses["score"].item()
            ep_concept += losses["concept"].item()

            if step % log_every == 0:
                print(f"  [E{epoch} S{step}/{steps_per_epoch}] "
                      f"total={ep_total/step:.4f} score={ep_score/step:.4f} "
                      f"concept={ep_concept/step:.4f}")

        elapsed = time.time() - t0
        avg = {k: v / steps_per_epoch for k, v in
               [("total", ep_total), ("score", ep_score), ("concept", ep_concept)]}
        all_losses.append({"epoch": epoch, **avg})

        print(f"[Epoch {epoch}] total={avg['total']:.4f} score={avg['score']:.4f} "
              f"concept={avg['concept']:.4f} ({elapsed:.1f}s)")

        # Save checkpoint
        if epoch % save_every == 0 or epoch == start_epoch + epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"{args.dataset}_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": energy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg),
                "latent_space": latent_cfg,
                "losses": all_losses,
            }, ckpt_path)
            print(f"  → Saved {ckpt_path}")

    # Save final with a clean name
    final_path = os.path.join(ckpt_dir, f"{args.dataset}_final.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": energy_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg),
        "latent_space": latent_cfg,
        "losses": all_losses,
    }, final_path)
    print(f"\n[train] Done! Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CoBELa")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["celebahq", "cub"], help="Dataset to train on")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (2 epochs, 50 steps)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda-score", type=float, default=1.0)
    parser.add_argument("--lambda-concept", type=float, default=1e-3)
    parser.add_argument("--latent-mode", type=str, default=single, choices=["single", "subset", "full"])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
