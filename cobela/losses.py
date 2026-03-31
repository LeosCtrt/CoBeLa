"""
CoBELa Training Losses (Section III-C).

Two losses:
  1. Score-matching loss (Eq. 6):
     L_score = E[ 0.5 * || epsilon - grad_vt E(vt) ||^2 ]
     Aligns the energy gradient with the diffusion noise.

  2. Concept loss (Eq. 7):
     L_concept = - sum_k log softmax(E_theta(vt | ck))[s_hat_k]
     Cross-entropy on per-concept logits against pseudo-labels.

  Total: L = lambda1 * L_score + lambda2 * L_concept
"""

import torch
import torch.nn.functional as F


def score_matching_loss(
    energy_net,
    vt: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Score-matching loss (Eq. 6).

    The energy gradient grad_vt E(vt) should predict the noise epsilon
    that was added to the clean latent.

    Args:
        energy_net:  EnergyNetwork module
        vt:          noised latent, shape (batch, latent_dim), requires_grad=True
        t:           timesteps, shape (batch,)
        noise:       the noise that was added, shape (batch, latent_dim)
        weights:     per-concept weights for total energy, shape (K,)
    Returns:
        loss: scalar
    """
    # Compute total energy
    total_e = energy_net.total_energy(vt, t, weights)  # (batch,)

    # Compute gradient of energy w.r.t. noised latent
    grad_vt = torch.autograd.grad(
        outputs=total_e.sum(),
        inputs=vt,
        create_graph=True,   # need this for second-order through training
        retain_graph=True,
    )[0]  # (batch, latent_dim)

    # MSE between gradient and noise (Eq. 6)
    loss = 0.5 * F.mse_loss(grad_vt, noise, reduction="mean")
    return loss


def concept_loss(
    energy_net,
    vt: torch.Tensor,
    t: torch.Tensor,
    pseudo_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Concept loss (Eq. 7).

    Cross-entropy between per-concept logits and pseudo-labels.

    Args:
        energy_net:    EnergyNetwork module
        vt:            noised latent, shape (batch, latent_dim)
        t:             timesteps, shape (batch,)
        pseudo_labels: binary labels from pseudo-labeler, shape (batch, K)
    Returns:
        loss: scalar
    """
    K = energy_net.num_concepts
    total_ce = 0.0

    for k in range(K):
        k_idx = torch.tensor(k, device=vt.device)
        logits = energy_net.forward_single_concept(vt, k_idx, t)  # (batch, 2)
        targets = pseudo_labels[:, k].long()                       # (batch,)
        total_ce = total_ce + F.cross_entropy(logits, targets)

    return total_ce / K  # average over concepts


def cobela_loss(
    energy_net,
    vt: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    pseudo_labels: torch.Tensor,
    lambda_score: float = 1.0,
    lambda_concept: float = 1e-3,
    weights: torch.Tensor = None,
) -> dict:
    """
    Combined CoBELa training loss.

    L = lambda1 * L_score + lambda2 * L_concept

    Args:
        energy_net:     EnergyNetwork module
        vt:             noised latent (requires_grad=True), shape (batch, latent_dim)
        t:              timesteps, shape (batch,)
        noise:          noise added to clean latent, shape (batch, latent_dim)
        pseudo_labels:  binary pseudo-labels, shape (batch, K)
        lambda_score:   weight for score-matching loss (default 1.0)
        lambda_concept: weight for concept loss (default 1e-3)
        weights:        per-concept weights, shape (K,)
    Returns:
        dict with keys: 'total', 'score', 'concept'
    """
    l_score = score_matching_loss(energy_net, vt, t, noise, weights)
    l_concept = concept_loss(energy_net, vt, t, pseudo_labels)

    total = lambda_score * l_score + lambda_concept * l_concept

    return {
        "total": total,
        "score": l_score.detach(),
        "concept": l_concept.detach(),
    }
