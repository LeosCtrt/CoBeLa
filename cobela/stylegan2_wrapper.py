"""
StyleGAN2 loader and g1/g2 splitter for CoBELa.

Requires dnnlib/ and torch_utils/ at the project root (see README).

Usage:
    from cobela.stylegan2_wrapper import load_stylegan2
    G, g1, g2, info = load_stylegan2("checkpoints/stylegan2-celebahq-256x256.pkl")
"""

import os
import sys
import pickle
import torch
import torch.nn as nn


class MappingWrapper(nn.Module):
    """g1: z → w (W+ space)."""

    def __init__(self, G):
        super().__init__()
        self.mapping = G.mapping
        self.z_dim = G.z_dim
        self.w_dim = G.w_dim
        self.num_ws = G.mapping.num_ws

    @torch.no_grad()
    def forward(self, z, truncation_psi=1.0):
        return self.mapping(z, None, truncation_psi=truncation_psi)


class SynthesisWrapper(nn.Module):
    """g2: w → image."""

    def __init__(self, G):
        super().__init__()
        self.synthesis = G.synthesis

    @torch.no_grad()
    def forward(self, w):
        return self.synthesis(w, noise_mode="const")


def load_stylegan2(pkl_path, device="cuda"):
    """
    Load a pretrained StyleGAN2 and split into g1 (mapping) / g2 (synthesis).

    Args:
        pkl_path: path to .pkl weights
        device:   target device

    Returns:
        G:    full generator (frozen)
        g1:   MappingWrapper
        g2:   SynthesisWrapper
        info: dict with z_dim, w_dim, num_ws, img_resolution
    """
    # Ensure project root is on sys.path (needed for pickle to find modules)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"[stylegan2] Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    G = data["G_ema"].to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)

    g1 = MappingWrapper(G).to(device).eval()
    g2 = SynthesisWrapper(G).to(device).eval()

    info = {
        "z_dim": G.z_dim,
        "w_dim": G.w_dim,
        "num_ws": G.mapping.num_ws,
        "img_resolution": G.img_resolution,
    }
    print(f"[stylegan2] z_dim={info['z_dim']}, w_dim={info['w_dim']}, "
          f"num_ws={info['num_ws']}, res={info['img_resolution']}")

    return G, g1, g2, info
