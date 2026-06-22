"""
rnd_model.py - Random Network Distillation (Curiosity-Driven Exploration)
==========================================================================
Generates an intrinsic reward  R_i = ||predictor(s) - target(s)||^2
to push the bot into novel areas during EXPLORING, preventing the
"stand still to minimize penalties" collapse.

Architecture
------------
  Target    (f):   frozen random CNN  - never updated, defines the projection
  Predictor (f^):  trained CNN         - learns to match target output
  Normalizer:      Welford running mean/std - keeps reward scale stable
                   as raw MSE decays during learning

VRAM footprint:  ~3 MB  (two 133K-param networks + Adam state)
Per-step cost:   ~0.5 ms (single forward pass inside torch.no_grad)
Training cost:   ~1-2 s per rollout  (runs during PPO update gap, not the loop)

The predictor is ONLY trained in RNDCallback._on_rollout_end() - completely
outside the game loop - so it never affects env step throughput.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -- Running statistics (Welford online algorithm) -----------------------------

class RunningMeanStd:
    """
    Welford streaming mean + variance over scalar values.

    Without this, raw RND prediction error shrinks to near-zero as the
    predictor learns, making the curiosity signal vanish mid-training.
    The normaliser keeps it on a consistent N(0, 1)-ish scale throughout.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon      # seed count avoids div-by-zero on first call

    def update(self, values: np.ndarray) -> None:
        """Incorporate a new batch of scalar observations."""
        batch_mean  = float(np.mean(values))
        batch_var   = float(np.var(values))
        batch_count = len(values)

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean   = self.mean + delta * batch_count / tot_count
        m_2        = (self.var  * self.count
                      + batch_var * batch_count
                      + delta ** 2 * self.count * batch_count / tot_count)

        self.mean  = new_mean
        self.var   = m_2 / tot_count
        self.count = tot_count

    def normalize(self, value: float) -> float:
        return (value - self.mean) / (np.sqrt(self.var) + 1e-8)


# -- Tiny CNN encoder ----------------------------------------------------------

def _make_encoder(output_dim: int) -> nn.Sequential:
    """
    Lightweight CNN: (3, 224, 224) -> output_dim float vector.

    AvgPool(7) collapses 224 -> 32 before the conv stack, cutting
    computation 49x vs operating on full resolution.

    Layer shapes  (B = batch size):
        Input            (B,  3, 224, 224)
        AvgPool(7)       (B,  3,  32,  32)
        Conv(3->16, /2)   (B, 16,  16,  16)
        LeakyReLU
        Conv(16->32, /2)  (B, 32,   8,   8)
        LeakyReLU
        Flatten          (B, 2048)
        Linear           (B, output_dim)

    Params per network ~ 133 K  ->  ~530 KB VRAM each.
    """
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=7),
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Flatten(),
        nn.Linear(2048, output_dim),
    )


# -- RND Model -----------------------------------------------------------------

class RNDModel:
    """
    Random Network Distillation curiosity model.

    The target network defines a fixed random projection of every state.
    The predictor network tries to match it. Prediction error = novelty.
    States the bot has visited many times are easy to predict (low R_i).
    States the bot hasn't seen yet are hard to predict (high R_i).

    Typical usage
    -------------
    # Once, in train_rl.py before building the env:
        rnd = RNDModel(output_dim=64, lr=1e-4, device='cuda')

    # Every env step (game_env.py) - no gradients, ~0.5 ms:
        r_i = rnd.get_reward(obs_np)        # obs_np: (3, H, W) uint8

    # Every rollout end (RNDCallback) - trains predictor, ~1-2 s:
        rnd.update(obs_batch_np)            # (N, 3, H, W) uint8
    """

    def __init__(self,
                 output_dim: int   = 64,
                 lr:         float = 1e-4,
                 device:     str   = 'cuda') -> None:

        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )

        self.target    = _make_encoder(output_dim).to(self.device)
        self.predictor = _make_encoder(output_dim).to(self.device)

        # Target is frozen forever - random projection stays fixed
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

        self.optimizer   = optim.Adam(self.predictor.parameters(), lr=lr)
        self._normalizer = RunningMeanStd()

    # -- Inference - called every env step ------------------------------------

    @torch.no_grad()
    def get_reward(self, obs_np: np.ndarray) -> float:
        """
        Return the normalised intrinsic reward for one observation.
        Runs entirely under torch.no_grad() - no autograd overhead.

        obs_np  : (3, H, W) uint8 numpy array  (direct from env._frame_to_obs)
        returns : float, roughly N(0,1) after the normaliser warms up
                  positive -> novel state,  negative -> familiar state
        """
        obs = (
            torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            .unsqueeze(0)   # (1, 3, H, W)
            .div_(255.0)
        )
        t_feat = self.target(obs)
        p_feat = self.predictor(obs)
        raw    = float(((p_feat - t_feat) ** 2).mean())
        return self._normalizer.normalize(raw)

    # -- Training - called from RNDCallback._on_rollout_end -------------------

    def update(self, obs_batch_np: np.ndarray) -> float:
        """
        Train the predictor on a batch of rollout observations.
        Also updates the running normaliser with the per-sample errors
        from this batch so get_reward() stays well-calibrated.

        obs_batch_np : (N, 3, H, W) uint8 numpy array
        returns      : mean MSE loss for logging
        """
        self.predictor.train()
        obs = (
            torch.as_tensor(obs_batch_np, dtype=torch.float32,
                            device=self.device)
            .div_(255.0)
        )

        with torch.no_grad():
            t_feat = self.target(obs)

        p_feat = self.predictor(obs)
        loss   = nn.functional.mse_loss(p_feat, t_feat)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # Update normaliser with per-sample errors from this batch
        with torch.no_grad():
            per_sample = ((p_feat.detach() - t_feat) ** 2).mean(dim=1)
            self._normalizer.update(per_sample.cpu().numpy())

        return float(loss)
