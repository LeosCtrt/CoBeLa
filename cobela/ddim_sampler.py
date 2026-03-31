"""
CoBELa Concept-Guided Sampling (Algorithm 1).

DDIM-based denoising where the noise predictor is the energy gradient:
    epsilon_hat_t = grad_vt sum_k w_k * LogSumExp(E_theta(vt | ck))
"""

import torch
from cobela.noise_schedule import CosineNoiseSchedule


@torch.no_grad()
def concept_guided_sample(
    energy_net,
    g1,
    g2,
    noise_schedule: CosineNoiseSchedule,
    z: torch.Tensor = None,
    batch_size: int = 1,
    Ts: int = 400,
    ddim_steps: int = 50,
    w_pos: float = 1.0,
    w_neg: float = -0.001,
    intervene_concepts: dict = None,
    device: str = "cuda",
):
    """
    Algorithm 1: Concept-Guided Sampling.

    Args:
        energy_net:         trained EnergyNetwork
        g1:                 MappingWrapper  (z -> w)
        g2:                 SynthesisWrapper (w -> image)
        noise_schedule:     CosineNoiseSchedule
        z:                  optional input noise (batch, z_dim)
        batch_size:         images to generate (ignored if z given)
        Ts:                 starting noise level (default 400)
        ddim_steps:         DDIM denoising steps (default 50)
        w_pos:              positive weight (default 1.0)
        w_neg:              negative weight for negation (default -0.001)
        intervene_concepts: dict {concept_index: desired_value (0 or 1)}
        device:             target device

    Returns:
        images: (batch, 3, H, W) in [-1, 1]
        v0:     denoised latent (batch, latent_dim)
        scores: concept scores (batch, K) in [0, 1]
    """
    energy_net.eval()
    K = energy_net.num_concepts

    # Build intervention weights
    weights = torch.ones(K, device=device) * w_pos
    if intervene_concepts is not None:
        for k, val in intervene_concepts.items():
            if val == 0:
                weights[k] = w_neg

    # Get clean latent from frozen generator
    if z is None:
        z = torch.randn(batch_size, g1.z_dim, device=device)
    else:
        batch_size = z.shape[0]

    with torch.no_grad():
        w_latent = g1(z)           # (batch, num_ws, w_dim)
        v = w_latent[:, 0, :]     # (batch, 512) W space

    # Noise the clean latent at Ts (Eq. 9)
    ab_Ts = noise_schedule.get_alpha_bar(torch.tensor([Ts], device=device))
    eps_init = torch.randn_like(v)
    vt = torch.sqrt(ab_Ts) * v + torch.sqrt(1 - ab_Ts) * eps_init

    # DDIM schedule
    timesteps = noise_schedule.get_ddim_schedule(Ts, ddim_steps)

    # Denoising loop (Algorithm 1, lines 2-6)
    for i in range(len(timesteps) - 1):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]
        t_tensor = torch.full((batch_size,), t_now, device=device, dtype=torch.long)

        vt_grad = vt.detach().requires_grad_(True)

        # Line 3: eps_hat via energy gradient (enable_grad overrides outer no_grad)
        with torch.enable_grad():
            energies = energy_net.concept_energies(vt_grad, t_tensor)
            weighted_energy = (energies * weights.unsqueeze(0)).sum(dim=1)
            eps_hat = torch.autograd.grad(weighted_energy.sum(), vt_grad)[0]

        # Line 4: predict v0
        ab_t = noise_schedule.get_alpha_bar(t_tensor[:1]).squeeze()
        v0_pred = (vt - torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(ab_t)

        # Line 5: DDIM step
        if t_next > 0:
            ab_next = noise_schedule.get_alpha_bar(
                torch.tensor([t_next], device=device)
            ).squeeze()
            vt = torch.sqrt(ab_next) * v0_pred + torch.sqrt(1 - ab_next) * eps_hat
        else:
            vt = v0_pred

    v0_final = vt

    # Generate image: broadcast v0 to W+ space then run g2
    num_ws = w_latent.shape[1]
    w_final = v0_final.unsqueeze(1).expand(-1, num_ws, -1)
    with torch.no_grad():
        images = g2(w_final)

    # Concept scores at the denoised latent
    with torch.no_grad():
        t_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
        scores = energy_net.concept_scores(v0_final, t_zero)

    return images, v0_final, scores


@torch.no_grad()
def generate_with_negation(
    energy_net, g1, g2, noise_schedule,
    negate_concepts: list,
    z=None, batch_size=4, Ts=400, ddim_steps=50, device="cuda",
):
    """Generate while negating specified concept indices."""
    intervene = {k: 0 for k in negate_concepts}
    return concept_guided_sample(
        energy_net, g1, g2, noise_schedule,
        z=z, batch_size=batch_size, Ts=Ts, ddim_steps=ddim_steps,
        intervene_concepts=intervene, device=device,
    )
