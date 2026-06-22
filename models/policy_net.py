"""
Shared Policy Network
======================
Single architecture used across BOTH training phases.

  Phase 1 — IL  : ILPolicyNet         trains with BCEWithLogitsLoss
  Phase 2 — RL  : MobileNetV2Extractor plugs into SB3 PPO as features_extractor

Action space  — MultiDiscrete([2] * N_ACTIONS), one binary slot per input.
HOLD_KEYS are kept pressed while active; everything else is tapped once per step.

  idx  key     type   description
  ---  -----   ----   -----------
  0    w        hold   move forward
  1    a        hold   move left
  2    s        hold   move backward
  3    d        hold   move right
  4    shift    hold   sprint
  5    space    tap    jump
  6    q        tap    dash
  7    e        tap    skill / interact
  8    r        tap    skill
  9    f        tap    block / skill
  10   g        tap    boss lock-on
  11   z        tap    sword skill Z
  12   x        tap    sword skill X
  13   c        tap    skill C
  14   v        tap    skill V
  15   1        tap    ability slot 1
  16   2        tap    ability slot 2
  17   3        tap    ability slot 3
  18   4        tap    ability slot 4
  19   5        tap    ability slot 5
  20   m1       tap    basic attack (left click)
  21   m2       tap    right click / aim
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Action vocabulary ─────────────────────────────────────────────────────────
ACTION_KEYS: list[str] = [
    # ── HELD keys (sustained press while action == 1) ──
    'w', 'a', 's', 'd', 'shift',
    # ── TAPPED keys (single press per step when action == 1) ──
    'space', 'q',
    'e', 'r', 'f', 'g',
    'z', 'x', 'c', 'v',
    '1', '2', '3', '4', '5',
    'm1', 'm2',
]
N_ACTIONS:   int       = len(ACTION_KEYS)   # 22
FEATURE_DIM: int       = 512               # shared trunk output dimensionality

# Keys that are held continuously (vs. tapped once per step)
HOLD_KEYS: frozenset[str] = frozenset({'w', 'a', 's', 'd', 'shift'})

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── SB3 Feature Extractor (Phase 2) ──────────────────────────────────────────

class MobileNetV2Extractor(BaseFeaturesExtractor):
    """
    SB3-compatible features extractor.
    SB3 delivers uint8 images normalised to [0,1]; we apply ImageNet stats on top.
    Returns a 512-dim feature vector per observation.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = FEATURE_DIM) -> None:
        super().__init__(observation_space, features_dim)

        mv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = mv2.features           # (B, 1280, 7, 7) at 224-px input
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.trunk    = nn.Sequential(
            nn.Linear(1280, features_dim),
            nn.ReLU(inplace=True),
        )

        # ImageNet normalisation applied after SB3's /255 division
        self.register_buffer(
            'img_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            'img_std',  torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.img_mean) / self.img_std
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.trunk(x)

    def freeze_backbone(self) -> None:
        """Freeze backbone during early RL steps to protect IL features."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


# ── IL Policy Net (Phase 1) ───────────────────────────────────────────────────

class ILPolicyNet(nn.Module):
    """
    Full network for behavioural cloning.
    Identical backbone + trunk to MobileNetV2Extractor, plus a multi-label
    action head for simultaneous keypress prediction.

    forward() returns raw logits (B, N_ACTIONS) — apply sigmoid for probs,
    or pass directly to nn.BCEWithLogitsLoss.
    """

    def __init__(self, n_actions: int = N_ACTIONS,
                 feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        mv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = mv2.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.trunk    = nn.Sequential(
            nn.Linear(1280, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.action_head = nn.Linear(feature_dim, n_actions)

        self.register_buffer(
            'img_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            'img_std',  torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, 224, 224) float32 in [0, 1]."""
        x = (x - self.img_mean) / self.img_std
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.trunk(x)
        return self.action_head(x)   # (B, N_ACTIONS) raw logits

    # ── Weight transfer utilities ─────────────────────────────────────────────

    def extractor_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Return weights compatible with MobileNetV2Extractor's load_state_dict().
        Copies backbone + trunk (trunk[2]=Dropout is absent in extractor → strict=False).
        """
        sd: dict[str, torch.Tensor] = {}
        for k, v in self.backbone.state_dict().items():
            sd[f'backbone.{k}'] = v.clone()
        sd['trunk.0.weight'] = self.trunk[0].weight.clone()
        sd['trunk.0.bias']   = self.trunk[0].bias.clone()
        sd['img_mean']       = self.img_mean.clone()
        sd['img_std']        = self.img_std.clone()
        return sd

    def action_head_for_sb3(self) -> dict[str, torch.Tensor]:
        """
        Remap IL action_head  (Linear 512→10, single logit per action)
        to SB3's MultiCategorical action_net (Linear 512→20, two logits per action).

        Encoding: for action i,
            SB3 logit[2i]   = -IL logit[i]   (probability of NOT pressing)
            SB3 logit[2i+1] =  IL logit[i]   (probability of pressing)
        """
        w = self.action_head.weight   # (10, 512)
        b = self.action_head.bias     # (10,)
        n = w.shape[0]
        sb3_w = torch.zeros(n * 2, w.shape[1])
        sb3_b = torch.zeros(n * 2)
        for i in range(n):
            sb3_w[2 * i]     = -w[i]
            sb3_w[2 * i + 1] =  w[i]
            sb3_b[2 * i]     = -b[i]
            sb3_b[2 * i + 1] =  b[i]
        return {'weight': sb3_w, 'bias': sb3_b}
