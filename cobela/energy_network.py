"""
CoBELa Energy Network Eθ

Architecture (from paper Section III-B):
  - Input: noised latent vt (512-d) + concept embedding ck (128-d)
  - Two conditional residual blocks with FiLM conditioning on concept + timestep
  - Output: 2 logits per concept (Eq. 2)

From these logits we derive:
  - Concept score sk = softmax(logits)[1]          (Eq. 3)
  - Concept energy ek = LogSumExp(logits)           (Eq. 4)
  - Total energy E(vt) = sum_k ek                   (Eq. 5)

The energy gradient grad_vt E(vt) serves as the noise predictor
during DDIM sampling (Algorithm 1, line 3).
"""

import math
import torch
import torch.nn as nn


# ─── Sinusoidal timestep embedding ──────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: timestep indices, shape (batch,) or scalar
        Returns:
            embedding, shape (batch, dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (batch, dim)
        return emb


# ─── FiLM conditioning layer ────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al., 2018).
    Produces scale (gamma) and shift (beta) from a conditioning vector,
    then applies: output = gamma * input + beta
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    features, shape (batch, hidden_dim)
            cond: conditioning vector, shape (batch, cond_dim)
        Returns:
            modulated features, shape (batch, hidden_dim)
        """
        gamma_beta = self.proj(cond)       # (batch, hidden_dim * 2)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * x + beta


# ─── Conditional residual block ─────────────────────────────────────

class ConditionalResBlock(nn.Module):
    """
    Residual block with FiLM conditioning on concept + timestep embeddings.

    Architecture:
        x -> Linear -> LayerNorm -> FiLM(cond) -> SiLU -> Linear -> + x
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.film = FiLMLayer(cond_dim, hidden_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    input features, shape (batch, hidden_dim)
            cond: conditioning vector (concept_emb + time_emb), shape (batch, cond_dim)
        Returns:
            output features, shape (batch, hidden_dim)
        """
        residual = x
        h = self.linear1(x)
        h = self.norm1(h)
        h = self.film(h, cond)
        h = self.act(h)
        h = self.linear2(h)
        h = self.norm2(h)
        return h + residual


# ─── Energy Network Eθ ──────────────────────────────────────────────

class EnergyNetwork(nn.Module):
    """
    CoBELa's energy network Eθ.

    For each concept k, takes (vt, ck) and outputs 2 logits.
    The positive-class softmax probability is the concept score (Eq. 3).
    The LogSumExp of the logits is the scalar energy (Eq. 4).

    Args:
        latent_dim:        dimension of the latent space (512 for StyleGAN2 W space)
        num_concepts:      K, number of concepts
        concept_embed_dim: dimension of learnable concept embeddings
        time_embed_dim:    dimension of sinusoidal timestep embeddings
        hidden_dim:        hidden dimension of residual blocks
        num_res_blocks:    number of conditional residual blocks (paper says 2)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_concepts: int = 8,
        concept_embed_dim: int = 128,
        time_embed_dim: int = 128,
        hidden_dim: int = 512,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim

        # Learnable concept embedding table (Section III-B)
        self.concept_embeddings = nn.Embedding(num_concepts, concept_embed_dim)

        # Timestep embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        # Project latent vt to hidden dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Conditioning dimension = concept_embed + time_embed
        cond_dim = concept_embed_dim + time_embed_dim

        # Conditional residual blocks with FiLM
        self.res_blocks = nn.ModuleList([
            ConditionalResBlock(hidden_dim, cond_dim)
            for _ in range(num_res_blocks)
        ])

        # Output head: project to 2 logits (binary concept classification)
        self.output_head = nn.Linear(hidden_dim, 2)

    def _get_conditioning(
        self,
        concept_idx: torch.Tensor,
        t: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Build the combined conditioning vector [concept_emb; time_emb].

        Args:
            concept_idx: which concept, shape (1,) or scalar
            t: timestep, shape (batch,) or scalar
            batch_size: number of samples
        Returns:
            cond: shape (batch, concept_embed_dim + time_embed_dim)
        """
        # Concept embedding
        c_emb = self.concept_embeddings(concept_idx)  # (1, concept_embed_dim) or (concept_embed_dim,)
        if c_emb.dim() == 1:
            c_emb = c_emb.unsqueeze(0)
        c_emb = c_emb.expand(batch_size, -1)  # (batch, concept_embed_dim)

        # Time embedding
        t_emb = self.time_embed(t)         # (batch, time_embed_dim)
        t_emb = self.time_mlp(t_emb)       # (batch, time_embed_dim)

        return torch.cat([c_emb, t_emb], dim=-1)  # (batch, cond_dim)

    def forward_single_concept(
        self,
        vt: torch.Tensor,
        concept_idx: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for a single concept k.

        Args:
            vt:          noised latent, shape (batch, latent_dim)
            concept_idx: concept index, scalar or shape (1,)
            t:           timestep, shape (batch,)
        Returns:
            logits: shape (batch, 2)  — Eq. (2)
        """
        batch_size = vt.shape[0]
        cond = self._get_conditioning(concept_idx, t, batch_size)

        h = self.input_proj(vt)  # (batch, hidden_dim)
        for block in self.res_blocks:
            h = block(h, cond)
        logits = self.output_head(h)  # (batch, 2)
        return logits

    def concept_scores(
        self,
        vt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute concept scores for ALL K concepts (Eq. 3).

        Args:
            vt: noised latent, shape (batch, latent_dim)
            t:  timestep, shape (batch,)
        Returns:
            scores: shape (batch, K), values in [0, 1]
        """
        scores = []
        for k in range(self.num_concepts):
            k_idx = torch.tensor(k, device=vt.device)
            logits = self.forward_single_concept(vt, k_idx, t)  # (batch, 2)
            sk = torch.softmax(logits, dim=-1)[:, 1]             # P(positive)
            scores.append(sk)
        return torch.stack(scores, dim=1)  # (batch, K)

    def concept_energies(
        self,
        vt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-concept energies for ALL K concepts (Eq. 4).

        Args:
            vt: noised latent, shape (batch, latent_dim)
            t:  timestep, shape (batch,)
        Returns:
            energies: shape (batch, K)
        """
        energies = []
        for k in range(self.num_concepts):
            k_idx = torch.tensor(k, device=vt.device)
            logits = self.forward_single_concept(vt, k_idx, t)
            ek = torch.logsumexp(logits, dim=-1)  # (batch,)
            energies.append(ek)
        return torch.stack(energies, dim=1)  # (batch, K)

    def total_energy(
        self,
        vt: torch.Tensor,
        t: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute weighted total energy (Eq. 8).

        Args:
            vt:      noised latent, shape (batch, latent_dim)
            t:       timestep, shape (batch,)
            weights: per-concept weights, shape (K,). Default: all ones.
        Returns:
            total_energy: shape (batch,)
        """
        energies = self.concept_energies(vt, t)  # (batch, K)
        if weights is None:
            weights = torch.ones(self.num_concepts, device=vt.device)
        return (energies * weights.unsqueeze(0)).sum(dim=1)  # (batch,)
