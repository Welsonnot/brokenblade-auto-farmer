"""
CombatEngine  -  Layer 2: COMBAT handler
=========================================
Primary path  : MobileNetV2 on CUDA - frame in, key probabilities out.
Fallback path : Scripted combo loop used until a trained model exists.

On state entry the engine taps 'G' to engage Boss Lock-On,
then aggressively spams M1 slashes weaved with Z/X sword attacks
and F blocks based on either model predictions or the scripted combo.

Label order for model output (must match dataset.py):
  [m1, z, x, f, w, a, s, d]
"""

import os
import time

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

MODEL_PATH  = "models/game_bot.pth"
NUM_CLASSES = 8     # m1, z, x, f, w, a, s, d
THRESHOLD   = 0.50

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# Scripted fallback - aggressive M1 spam weaved with Z / X / F
_COMBO = [
    ('m1', 0.10), ('m1', 0.10), ('m1', 0.10),
    ('z',  0.30),
    ('m1', 0.10), ('m1', 0.10),
    ('x',  0.35),
    ('m1', 0.10),
    ('f',  0.20),   # block
    ('m1', 0.10), ('m1', 0.10),
]


def _build_model() -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(m.last_channel, NUM_CLASSES),
    )
    return m


class CombatEngine:
    def __init__(self, grabber, input_emu):
        self._grabber    = grabber
        self._input      = input_emu
        self._device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        self._model_ok      = False
        self._combo_idx     = 0
        self._combo_next    = 0.0    # timestamp when next scripted action fires
        self._last_lock     = 0.0    # timestamp of last G (BossLock) tap
        self._LOCK_INTERVAL = 3.0    # re-tap G every 3 s to maintain lock-on

        self._load_model()

    # -- Model loading --------------------------------------------------------

    def _load_model(self) -> None:
        if not os.path.isfile(MODEL_PATH):
            print("[Combat] No model found - scripted combo active until trained.")
            return
        try:
            m = _build_model().to(self._device)
            m.load_state_dict(torch.load(MODEL_PATH, map_location=self._device))
            m.eval()
            # Warm-up pass to pre-allocate CUDA kernels
            dummy = torch.zeros(1, 3, 224, 224, device=self._device)
            with torch.no_grad():
                _ = m(dummy)
            self._model    = m
            self._model_ok = True
            print(f"[Combat] Model loaded on {self._device}")
        except Exception as exc:
            print(f"[Combat] Model load failed ({exc}) - scripted combo active.")

    # -- Lifecycle callbacks ---------------------------------------------------

    def on_enter(self) -> None:
        """Tap G to engage Boss Lock-On the moment we enter COMBAT."""
        time.sleep(0.08)
        self._input.tap('g', 0.08)
        self._combo_idx  = 0
        self._combo_next = time.perf_counter()

    def on_exit(self) -> None:
        self._input.release_all()

    # -- Tick -----------------------------------------------------------------

    def tick(self) -> None:
        # Periodically re-tap G to keep BossLock active
        now = time.perf_counter()
        if now - self._last_lock >= self._LOCK_INTERVAL:
            self._input.tap('g', 0.08)
            self._last_lock = now

        if self._model_ok:
            self._infer_tick()
        else:
            self._scripted_tick()

    # -- Inference path --------------------------------------------------------

    def _infer_tick(self) -> None:
        frame = self._grabber.get_small_frame()
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = _TRANSFORM(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.sigmoid(self._model(inp)).squeeze().cpu().tolist()
        self._input.apply_predictions(probs, THRESHOLD)

    # -- Scripted fallback -----------------------------------------------------

    def _scripted_tick(self) -> None:
        now = time.perf_counter()
        if now < self._combo_next:
            return   # not yet time for next action

        action, hold = _COMBO[self._combo_idx % len(_COMBO)]
        self._combo_idx += 1
        self._combo_next = now + hold + 0.02   # small gap between actions

        if action == 'm1':
            self._input.click(hold=hold)
        else:
            self._input.tap(action, hold)
