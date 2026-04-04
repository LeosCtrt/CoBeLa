"""
Utilities for choosing which part of StyleGAN2's W+ latent is exposed to CoBELa.

Modes:
  - single: use one style vector w_i and broadcast it back to every W+ slot.
  - subset: use a selected subset of W+ slots and write them back in place.
  - full:   use the entire W+ tensor flattened as a single latent vector.
"""

from __future__ import annotations


def resolve_latent_config(cfg, num_ws: int, w_dim: int) -> dict:
    """
    Normalize the latent-space configuration against the generator shape.

    Args:
        cfg:   OmegaConf section or plain dict with keys mode/single_index/subset_indices.
        num_ws: number of StyleGAN synthesis style slots.
        w_dim: width of each style vector.
    Returns:
        dict containing mode, selected_indices, latent_dim, num_ws, and w_dim.
    """
    mode = "single"
    single_index = 0
    subset_indices = None

    if cfg is not None:
        mode = str(cfg.get("mode", mode))
        single_index = int(cfg.get("single_index", single_index))
        raw_subset = cfg.get("subset_indices", None)
        if raw_subset is not None:
            subset_indices = [int(index) for index in raw_subset]

    if mode not in {"single", "subset", "full"}:
        raise ValueError(f"Unsupported latent mode: {mode}")

    if mode == "single":
        if not 0 <= single_index < num_ws:
            raise ValueError(f"single_index must be in [0, {num_ws - 1}], got {single_index}")
        selected_indices = [single_index]
        latent_dim = w_dim
    elif mode == "subset":
        if subset_indices is None:
            start = num_ws // 2
            subset_indices = list(range(start, num_ws))
        if len(subset_indices) == 0:
            raise ValueError("subset mode requires at least one style index")
        if len(set(subset_indices)) != len(subset_indices):
            raise ValueError(f"subset_indices contains duplicates: {subset_indices}")
        for index in subset_indices:
            if not 0 <= index < num_ws:
                raise ValueError(f"subset index must be in [0, {num_ws - 1}], got {index}")
        selected_indices = list(subset_indices)
        latent_dim = len(selected_indices) * w_dim
    else:
        selected_indices = list(range(num_ws))
        latent_dim = num_ws * w_dim

    return {
        "mode": mode,
        "single_index": single_index,
        "selected_indices": selected_indices,
        "latent_dim": latent_dim,
        "num_ws": num_ws,
        "w_dim": w_dim,
    }


def extract_energy_latent(w_latent, latent_config: dict):
    """
    Extract the latent vector that CoBELa will model from a full W+ tensor.

    Args:
        w_latent: full StyleGAN W+ tensor of shape (batch, num_ws, w_dim)
        latent_config: output of resolve_latent_config()
    Returns:
        latent tensor of shape (batch, latent_dim)
    """
    mode = latent_config["mode"]
    if mode == "single":
        return w_latent[:, latent_config["selected_indices"][0], :]
    if mode == "subset":
        return w_latent[:, latent_config["selected_indices"], :].reshape(w_latent.shape[0], -1)
    return w_latent.reshape(w_latent.shape[0], -1)


def inject_energy_latent(base_w_latent, modeled_latent, latent_config: dict):
    """
    Reconstruct a full W+ tensor from CoBELa's modeled latent.

    single:
      The learned style vector is repeated across every synthesis slot to match
      the historical CB-AE/CoBeLa simplification.
    subset:
      Only the selected W+ slots are replaced; other slots keep the original
      clean latent from g1(z).
    full:
      The modeled latent is reshaped back into the complete W+ tensor.
    """
    mode = latent_config["mode"]
    batch_size = base_w_latent.shape[0]
    num_ws = latent_config["num_ws"]
    w_dim = latent_config["w_dim"]

    if mode == "single":
        return modeled_latent.unsqueeze(1).expand(-1, num_ws, -1)

    if mode == "subset":
        rebuilt = base_w_latent.clone()
        subset = modeled_latent.reshape(batch_size, len(latent_config["selected_indices"]), w_dim)
        rebuilt[:, latent_config["selected_indices"], :] = subset
        return rebuilt

    return modeled_latent.reshape(batch_size, num_ws, w_dim)
