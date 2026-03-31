"""
Cosine noise schedule for CoBELa (Nichol & Dhariwal, 2021).

Provides alpha_bar values and noising/denoising utilities for the
intermediate latent space of StyleGAN2.
"""

import torch
import numpy as np


class CosineNoiseSchedule:
    """
    Cosine noise schedule.

    alpha_bar_t = f(t) / f(0),  where  f(t) = cos((t/T + s) / (1+s) * pi/2)^2

    Args:
        max_timesteps: T, total number of timesteps (default 1000)
        s:             offset to prevent alpha_bar from being too small near t=0
    """

    def __init__(self, max_timesteps: int = 1000, s: float = 0.008):
        self.T = max_timesteps

        steps = np.arange(max_timesteps + 1, dtype=np.float64)
        f = np.cos((steps / max_timesteps + s) / (1 + s) * (np.pi / 2)) ** 2
        alpha_bar = f / f[0]

        # Clip to avoid numerical issues
        alpha_bar = np.clip(alpha_bar, 1e-5, 1.0)

        self.alpha_bar = torch.tensor(alpha_bar, dtype=torch.float32)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Look up alpha_bar for given timesteps.

        Args:
            t: timestep indices, shape (batch,), values in [0, T]
        Returns:
            alpha_bar_t: shape (batch,)
        """
        return self.alpha_bar.to(t.device)[t.long()]

    def noise_latent(
        self,
        v: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """
        Forward diffusion: noise a clean latent v at timestep t (Eq. 1).

            vt = sqrt(alpha_bar_t) * v + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            v:     clean latent, shape (batch, latent_dim)
            t:     timesteps, shape (batch,)
            noise: optional pre-sampled noise, shape (batch, latent_dim)
        Returns:
            vt:    noised latent, shape (batch, latent_dim)
            noise: the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(v)

        ab = self.get_alpha_bar(t)  # (batch,)

        # Broadcast to latent dimensions
        while ab.dim() < v.dim():
            ab = ab.unsqueeze(-1)

        vt = torch.sqrt(ab) * v + torch.sqrt(1 - ab) * noise
        return vt, noise

    def get_ddim_schedule(self, Ts: int, num_steps: int):
        """
        Create a DDIM sub-sequence of timesteps from Ts down to 0.

        Args:
            Ts:        starting timestep (e.g. 400)
            num_steps: number of DDIM steps (e.g. 50)
        Returns:
            timesteps: list of timesteps from Ts to 0, length num_steps+1
        """
        step_size = Ts / num_steps
        timesteps = [int(round(Ts - i * step_size)) for i in range(num_steps + 1)]
        # Ensure we end at 0
        timesteps[-1] = 0
        return timesteps
