# brokenblade-auto-farmer

> An RL agent that learns to auto-farm the Broken Blade boss in Roblox.

A reinforcement learning agent that learns to auto-farm the Broken Blade boss
in Roblox using PPO fine-tuned on top of an imitation-learning backbone.

The bot targets the "Katana Master" rank quest (500 boss kills) and runs
fully autonomously: it navigates to the boss, engages in combat, recovers
from deaths, and resets if it falls into the ocean.

> **Disclaimer:** This project is an independent open-source research tool.
> It is not affiliated with, associated with, authorized by, endorsed by, or
> in any way officially connected with Roblox Corporation or any of its
> subsidiaries. "Roblox" is a trademark of Roblox Corporation. This software
> uses only external screen capture and keyboard emulation -- it does not
> modify game memory, bypass anti-cheat, or distribute proprietary assets.
> Users are responsible for ensuring their use of this software complies
> with the Roblox Terms of Service.

---

## Demo

### RL Training (`launch_trainer.bat`)

https://github.com/Welsonnot/brokenblade-auto-farmer/raw/main/showcase/showcase1.mp4

### Ocean Detection (`python scripts/check_ocean.py`)

https://github.com/Welsonnot/brokenblade-auto-farmer/raw/main/showcase/showcase2.mp4

---

## Architecture

### Two-Layer Design

```
Layer 1 -- StateController  (core/state.py)
    Runs at 60 Hz in a background thread.
    Reads the screen and maintains the current GameState:
      EXPLORING  -- no boss detected, navigating
      ENGAGING   -- boss found, approaching
      COMBAT     -- actively fighting
      LOOTING    -- picking up drops

Layer 2 -- Action Handlers  (handlers/)
    Dispatched from the main loop at 30 Hz.
    CombatEngine      -- attack rotation, skill timing
    ExplorationManager -- movement toward boss spawn
    LootHandler        -- post-kill looting
```

### Reinforcement Learning Pipeline

```
1. Record gameplay    ->  python recorder.py
2. Train IL baseline  ->  python train_il.py    (MobileNetV2 backbone)
3. PPO fine-tuning    ->  python train_rl.py    (live in-game)
4. Resume training    ->  launch_trainer.bat
```

The IL (imitation learning) phase gives the PPO policy a warm start from
human demonstrations. The backbone (MobileNetV2, 512-dim features) is frozen
for the first 50,000 RL steps so only the action heads learn, then unfrozen
for full end-to-end tuning.

### RND Curiosity

A Random Network Distillation module provides intrinsic exploration rewards
during EXPLORING state. This prevents the policy from getting stuck in
repetitive loops before the boss spawns. Beta is annealed from 0.3 to 0.15
once the agent has mapped the environment.

---

## Prerequisites

- Windows 10/11 (uses DirectInput for keyboard/mouse emulation)
- Python 3.10+
- NVIDIA GPU with CUDA (tested on RTX 3060 12 GB)
- Roblox running in windowed fullscreen on a dedicated monitor
- Tesseract OCR (optional, for kill counter detection)

### Install Python dependencies

```
pip install -r requirements.txt
```

### Tesseract OCR (optional)

Download the Windows installer from:
https://github.com/UB-Mannheim/tesseract/wiki

The default path expected by the bot is:
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If your Tesseract is installed elsewhere, edit this line in `rl/game_env.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## Setup & Calibration

### 1. Set your monitor index

Roblox must be on **monitor index 2** (the second display).
If your setup is different, change `monitor_idx=2` in:
- `train_rl.py`
- `main.py`
- `rl/game_env.py` (the `_RewardSensor.__init__` method)

Check which monitor index Roblox is on:
```
python scripts/obs_ocean_check.py
```
This saves `obs_view.png` -- open it and verify it shows your Roblox window.

### 2. Calibrate the boss HP bar region

The bot detects the boss by scanning for a warm-colored (red/orange) HP bar
at a fixed screen region. Verify the fractional coordinates in `rl/game_env.py`:
```python
_BAR_TOP_F   = 85   / 1440
_BAR_BOT_F   = 116  / 1440
_BAR_LEFT_F  = 1137 / 2560
_BAR_RIGHT_F = 1812 / 2560
```
Run `python scripts/calibrate_hp_bar.py` while the boss is on screen to auto-detect
the correct region for your resolution.

### 3. Ocean / out-of-bounds detection

The bot uses pixel statistics on side strips of the 224x224 observation to
detect when the character has fallen into the ocean. Verify it works:
```
python scripts/obs_ocean_check.py   # run in the ocean  -> should print [OK] TRUE
python scripts/obs_ocean_check.py   # run on land       -> should print [FAIL] FALSE
```

Thresholds are tunable in `config/rl_params.json` without restarting.

---

## Pretrained Weights

Model weights are NOT included in the repo (they're large binaries).
Download them from the [Releases page](../../releases) and place them in
the `models/` folder before running:

```
models/
  il_baseline.pth            <- imitation learning baseline (download)
  rl_latest.zip              <- PPO checkpoint, latest (download)
  rl_ckpt_25000_steps.zip    <- PPO checkpoint snapshot (download)
```

You can also train from scratch without downloading anything -- see
"Option A" below.

---

## Running

### Option A -- From scratch (no downloads needed)

```
# 1. Record ~30 minutes of your gameplay for imitation learning
python recorder.py

# 2. Train the IL backbone (~30 min on RTX 3060)
python train_il.py

# 3. Start RL fine-tuning live in-game
python train_rl.py
```

### Option B -- Skip recording, train RL from pretrained IL

Download `il_baseline.pth` from Releases and place it in `models/`.

```
python train_rl.py
```

PPO starts from the pretrained backbone -- skip the ~30 min recording step
and the IL training time.

### Option C -- Continue training from the pretrained PPO checkpoint

Download both `il_baseline.pth` AND `rl_latest.zip` from Releases.

```
launch_trainer.bat
```

The launcher auto-detects `models/rl_latest.zip` and passes `--resume`.
You pick up at step 25,000 with the learned policy.

### Option D -- Scripted bot (no RL, rule-based fallback)

```
python main.py
```
Press **F1** to start/pause. Press **Esc** to quit cleanly.

---

### Crash recovery

If training crashes (Roblox disconnect, etc.), the watchdog atomically
saves the model and the launcher restarts within 15 seconds. All saves
use temp-file + atomic rename so the checkpoint is never half-written.

### Monitoring training

```
tensorboard --logdir logs/ppo
```

Open http://localhost:6006 in your browser.

---

## Tunable Parameters

All reward weights and detection thresholds are in `config/rl_params.json`.
Changes take effect on the next training run (no code edits needed).

| Parameter | Default | Description |
|---|---|---|
| `OCEAN_PENALTY` | 1.0 | Reward penalty per step in ocean |
| `OCEAN_FAIL_STEPS` | 30 | Steps in ocean before episode terminates |
| `OCEAN_FAIL_PENALTY` | 20.0 | One-time termination penalty |
| `rnd_beta` | 0.15 | RND curiosity signal scale |
| `AIR_ATTACK_GRACE` | 8 | Steps before air-attack penalty kicks in |
| `LOST_WINDOW_S` | 120.0 | Seconds without COMBAT before character reset |
| `ocean_bright_bri_min` | 85.0 | Ocean brightness lower bound (Branch A) |
| `ocean_bright_bias_min` | 38.0 | Ocean blue bias threshold (Branch A) |
| `ocean_bright_gr_min` | 15.0 | Ocean green-red threshold (Branch A) |
| `ocean_dark_bri_max` | 60.0 | Night ocean brightness upper bound (Branch B) |

The `rl/qwen_advisor.py` script can read training telemetry and suggest
parameter updates automatically (paste output here for review).

---

## File Structure

```
brokenblade-auto-farmer/
  main.py                 -- scripted bot entry point (rule-based)
  train_rl.py             -- PPO training loop
  train_il.py             -- imitation learning
  recorder.py             -- gameplay recorder for IL data
  launch_trainer.bat      -- auto-restart RL training launcher

  core/                   -- runtime systems
    screen.py             -- 60 Hz background screen capture
    state.py              -- GameState machine (EXPLORING/ENGAGING/COMBAT)
    input.py              -- keyboard + mouse emulation via DirectInput
    hp_tracker.py         -- player + boss HP tracking
    yolo_detect.py        -- YOLO-based boss label detector

  handlers/               -- state-based action handlers
    combat.py             -- attack rotation during COMBAT
    exploration.py        -- navigation during EXPLORING/ENGAGING
    loot.py               -- post-kill looting

  models/                 -- neural network architectures (weights gitignored)
    policy_net.py         -- MobileNetV2Extractor + ILPolicyNet

  rl/                     -- RL-specific runtime
    game_env.py           -- Gymnasium environment wrapping the live game
    rnd_model.py          -- Random Network Distillation (curiosity)
    recovery.py           -- stuck detector + character reset
    qwen_advisor.py       -- telemetry reader / parameter advisor

  scripts/                -- diagnostic + calibration tools
    setup_check.py        -- pre-flight system verification
    privacy_scan.py       -- find leaked personal data before publishing
    obs_ocean_check.py    -- ocean detection diagnostic
    bench_step.py         -- env step latency benchmark
    check_*.py            -- live HP / ocean / boss bar checks
    calibrate_*.py        -- auto-calibration helpers
    find_bar.py           -- locate boss HP bar position

  legacy/                 -- earlier pre-RL approach (still works)
    run_bot.py            -- CNN classifier-based bot entry point
    autoattack.py         -- simple auto-attack loop
    train_model.py        -- small CNN classifier trainer
    train_yolo.py         -- YOLO label-detector trainer
    collect_yolo_data.py  -- YOLO dataset collector
    ui_detector.py        -- pre-CNN UI detector
    dataset.py            -- classifier dataset wrapper

  config/
    rl_params.json        -- tunable reward weights and detection thresholds
```

---

## Action Space

22 binary actions (MultiDiscrete):

```
Movement : w  a  s  d  shift
Combat   : m1 (left click)  q  e  r  f  g  z  x  c  v
Skills   : 1  2  3  4  5
Misc     : space  m2
```

Hold keys (w, a, s, d, shift) stay pressed between steps.
All other keys are tapped for 20 ms per step.

---

## Notes

- The bot uses `pydirectinput` for keyboard/mouse input, which requires
  the Roblox window to be in focus. The recovery system calls
  `win32gui.SetForegroundWindow` after character resets to restore focus.
- Roblox must be on the monitor assigned to index 2 in the MSS monitor list.
  Use `scripts/obs_ocean_check.py` to verify this before training.
- Model weights (`.pth`, `.zip`) are excluded from git by `.gitignore`.
  Do not commit recordings or personal gameplay footage.

---

## Troubleshooting

**`PytorchStreamReader failed reading zip archive`**
Your `rl_latest.zip` is corrupt -- likely from a save interrupted by a
crash before the atomic-save fix. Replace it with the most recent
checkpoint:
```
copy models\rl_ckpt_25000_steps.zip models\rl_latest.zip
```

**Bot screen-captures the wrong monitor (e.g. your code editor)**
Roblox must be on monitor index 2. Check what each monitor is showing
by running `python scripts/obs_ocean_check.py` and inspecting `obs_view.png`.
If your monitors are swapped, edit `monitor_idx=2` to `monitor_idx=1`
in `train_rl.py`, `main.py`, and `rl/game_env.py`.

**`CUDA not available` or training is very slow**
Make sure you installed the CUDA-enabled torch build:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then re-run `python scripts/setup_check.py` to verify.

**OCR kill counter not detected**
Install Tesseract OCR (see Prerequisites). Check the install path matches
`C:\Program Files\Tesseract-OCR\tesseract.exe` or edit `rl/game_env.py`.
The bot still trains without OCR -- it just loses one reward signal.

**Bot keeps walking into the ocean**
This is normal early in training. Check `python scripts/obs_ocean_check.py` in
the ocean -- if it reports FALSE, the detection thresholds need tuning
in `config/rl_params.json`. See the comments in `rl/game_env.py` near
`_OCEAN_BRIGHT_BRI_MIN` for the two-branch detection logic.

**Watchdog keeps triggering "No env step for 120 s"**
Roblox is disconnecting during the PPO update window (15-30 s of no
inputs). Either reduce `n_epochs` in `train_rl.py` (faster updates) or
make sure your Roblox connection is stable.

**Reward curve is flat and very negative**
The bot is dying instantly every episode. Check that `OCEAN_FAIL_STEPS`
in `config/rl_params.json` is at least 30 -- if it's 1, the bot has no
time to learn to escape water. Defaults are tuned; only lower this if
you're trying to speed up training in known-safe environments.

---

## Verifying your setup

Before the first run, verify everything is configured:
```
python scripts/setup_check.py
```
This checks Python version, CUDA, package install, monitor count,
Tesseract presence, model weights, and config file validity.

## Contributing

Before opening a pull request, run the privacy scan to make sure your
changes don't leak personal data:
```
python scripts/privacy_scan.py
```
This catches Windows user paths, email addresses, API keys, Discord
webhooks, Roblox cookies, and IP addresses.

---

## License

MIT -- see [LICENSE](LICENSE).
