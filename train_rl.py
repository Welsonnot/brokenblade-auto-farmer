"""
Phase 2 - PPO Reinforcement Learning Fine-Tuning
=================================================
Loads the IL-pretrained backbone from models/il_baseline.pth into a PPO
policy and fine-tunes it live in the running Broken Blade game.

IL weights give the policy a head start - it already knows how to navigate,
lock on, and attack from watching your play.  PPO then discovers improvements
through self-play and the reward signal (boss HP, kills, player survival).

Prerequisites:
    pip install "stable-baselines3[extra]" gymnasium
    python recorder.py    ->  record gameplay (full loop)
    python train_il.py    ->  creates models/il_baseline.pth

Usage:
    python train_rl.py                 # new run, loads IL weights
    python train_rl.py --resume        # resume from latest checkpoint
    python train_rl.py --timesteps 500000

Outputs:
    models/rl_latest.zip               resumed every SAVE_FREQ steps
    models/rl_ckpt_NSTEPS_steps.zip   periodic checkpoint snapshots
    logs/ppo/                          TensorBoard logs
      -> tensorboard --logdir logs/ppo

Two-stage backbone training:
    Steps 0 ... FREEZE_STEPS   : backbone frozen  (only heads learn)
    Steps FREEZE_STEPS ... end  : backbone unfrozen (full end-to-end)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import threading
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env  import DummyVecEnv

from rl.rnd_model    import RNDModel
from core.screen     import ScreenGrabber
from core.input      import InputEmulator
from core.state      import StateController
from rl.game_env     import BrokenBladeEnv
from models.policy_net import (
    ILPolicyNet, MobileNetV2Extractor, FEATURE_DIM, N_ACTIONS
)

# -- Config (same file Qwen advisor tunes) -------------------------------------

def _load_params() -> dict:
    path = pathlib.Path(__file__).parent / "config" / "rl_params.json"
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

_RL_P = _load_params()

# -- Paths ---------------------------------------------------------------------
IL_WEIGHTS  = "models/il_baseline.pth"
RL_LATEST   = "models/rl_latest.zip"
CKPT_DIR    = "models/"
LOG_DIR     = "logs/ppo"

# -- Hyper-parameters ----------------------------------------------------------
# Optimized for max learning-per-step on RTX 3060 12 GB + live game at 60 Hz.
#
# Math:  n_steps=2048 -> 34 s rollout (covers navigate + fight cycle)
#        batch_size=256 -> 2048/256 = 8 minibatches per epoch
#        n_epochs=8 -> 8x8 = 64 gradient updates per rollout  (was 32)
#        Each gradient step uses 4x more data -> far less noise.
#
# VRAM:  batch 256 x 3x224x224 fp32 ~ 154 MB input + ~3 GB activations
#        + model (~14 MB) + optimizer (~28 MB) ~ 3.5 GB peak.
#        Roblox at low graphics ~ 1-2 GB.  Total ~ 5 GB / 12 GB available.

def _linear_schedule(initial_value: float):
    """Linear LR decay: starts at initial_value, reaches 0 at final timestep."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

PPO_KWARGS = dict(
    n_steps         = 2048,    # ~34 s at 60 Hz - full fight cycle in one rollout
    batch_size      = 256,     # 4x larger minibatch -> smoother gradients
    n_epochs        = 8,       # 8 passes over each rollout - GPU crunches hard
    gamma           = 0.99,
    gae_lambda      = 0.97,    # higher lambda pairs with longer rollouts for better
                               #   long-horizon credit (boss HP over 30 s)
    clip_range      = 0.15,    # tighter clip with 8 epochs prevents overshooting
                               #   and protects IL pretrained features
    ent_coef        = 0.02,    # 22 actions (was 10) -> needs more exploration
                               #   pressure so it tries Space, Shift, etc.
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    learning_rate   = _linear_schedule(1.5e-5),  # surgical - half the previous peak
                                                 # (resume from 44K; was 3e-5->0)
    tensorboard_log = LOG_DIR,
    device          = "cuda",
    verbose         = 1,
)

TOTAL_STEPS  = 500_000    # ~2.3 h at 60 Hz - converges faster with tuned params
SAVE_FREQ    = 25_000     # checkpoint every 25 K steps (~ 20 checkpoints total)
FREEZE_STEPS = 50_000     # backbone frozen for 10 % of run to stabilise heads


# -- Callbacks -----------------------------------------------------------------

class BackboneUnfreezeCallback(BaseCallback):
    """Unfreezes the MobileNetV2 backbone and lowers LR after FREEZE_STEPS."""

    def __init__(self, freeze_until: int = FREEZE_STEPS, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._freeze_until = freeze_until
        self._done         = False

    def _on_step(self) -> bool:
        if not self._done and self.num_timesteps >= self._freeze_until:
            extractor = self.model.policy.features_extractor
            extractor.unfreeze_backbone()
            # lower LR for careful end-to-end fine-tuning
            for grp in self.model.policy.optimizer.param_groups:
                grp['lr'] = 1e-5
            self._done = True
            print(f"\n[RL] -- Backbone UNFROZEN at step {self.num_timesteps:,}  "
                  f"LR -> 1e-5 [OK]")
        return True


class LatestSaveCallback(BaseCallback):
    """
    Saves a rolling 'latest' checkpoint so training can always be resumed.

    Atomic write: model.save() writes to <path>.tmp, then os.replace()
    swaps it into place.  If the process is killed mid-save (watchdog
    timeout, Ctrl+C, batch-loop restart) the existing <path> is never
    truncated -- the partial .tmp is just discarded on next run.
    """

    def __init__(self, save_path: str, freq: int = 5_000,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self._path = save_path
        self._tmp  = save_path + ".tmp"
        self._freq = freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self._freq == 0:
            try:
                self.model.save(self._tmp)
                os.replace(self._tmp, self._path)   # atomic on Windows + POSIX
            except Exception as exc:
                # Clean up partial tmp so it never accumulates
                try:
                    if os.path.exists(self._tmp):
                        os.remove(self._tmp)
                except Exception:
                    pass
                print(f"[LatestSave] save failed at step "
                      f"{self.num_timesteps}: {exc}")
        return True


class TelemetryCallback(BaseCallback):
    """
    Logs a compact JSON row to logs/telemetry.jsonl every FREQ steps.
    The Qwen advisor (qwen_advisor.py) reads this file to diagnose training
    and suggest config/rl_params.json updates - without touching any code.

    Logged fields:
        step            current timestep
        ep_rew_mean     rolling mean episode reward (from SB3 logger)
        ep_len_mean     rolling mean episode length
        attack_hit_rate fraction of M1/skill steps where boss bar was visible
        ocean_rate      fraction of steps in ocean
        fps             measured env steps/second
    """

    TELEMETRY_PATH = "logs/telemetry.jsonl"
    FREQ           = 512     # write one row per 512 steps (~ every ~8 s at 60 Hz)

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._t0            = time.time()
        self._steps_since   = 0
        # counters reset every FREQ steps
        self._attack_hits   = 0
        self._attack_total  = 0
        self._ocean_steps   = 0
        os.makedirs("logs", exist_ok=True)

    def _on_step(self) -> bool:
        self._steps_since += 1

        # Accumulate from info dict (SB3 stores last step's info in self.locals)
        info = self.locals.get("infos", [{}])[0]
        in_ocean  = info.get("in_ocean", False)
        # attack tracking is approximated from the env info
        # (game_env doesn't expose it directly - we read it from locals action)
        action = self.locals.get("actions", [[]])[0]
        if len(action) > 20:   # sanity check - action is 22-dim
            m1_pressed    = bool(action[20])
            skill_pressed = any(bool(action[i]) for i in [9, 10, 11, 12, 14])
            attacking     = m1_pressed or skill_pressed
            if attacking:
                self._attack_total += 1
                # boss_visible isn't in info, but we can proxy via ocean/state
                # A non-ocean, non-dead step with attacks is likely hitting
                if not in_ocean and info.get("game_state") == "COMBAT":
                    self._attack_hits += 1
        if in_ocean:
            self._ocean_steps += 1

        if self._steps_since >= self.FREQ:
            self._flush()
            self._steps_since  = 0
            self._attack_hits  = 0
            self._attack_total = 0
            self._ocean_steps  = 0
            self._t0           = time.time()
        return True

    def _flush(self) -> None:
        elapsed = max(time.time() - self._t0, 1e-6)
        fps     = self._steps_since / elapsed

        attack_hit_rate = (
            self._attack_hits / self._attack_total
            if self._attack_total > 0 else None
        )
        ocean_rate = self._ocean_steps / max(self._steps_since, 1)

        # Pull rolling stats from SB3's internal logger
        ep_rew_mean = None
        ep_len_mean = None
        try:
            ep_rew_mean = float(self.model.logger.name_to_value.get(
                "rollout/ep_rew_mean", float("nan")))
            ep_len_mean = float(self.model.logger.name_to_value.get(
                "rollout/ep_len_mean", float("nan")))
        except Exception:
            pass

        row = {
            "step":            self.num_timesteps,
            "ep_rew_mean":     ep_rew_mean,
            "ep_len_mean":     ep_len_mean,
            "attack_hit_rate": attack_hit_rate,
            "ocean_rate":      round(ocean_rate, 4),
            "fps":             round(fps, 1),
            "ts":              time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with open(self.TELEMETRY_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass


class RNDCallback(BaseCallback):
    """
    Trains the RND predictor after every rollout.

    Fires in _on_rollout_end() - this hook runs during the PPO update
    window (after rollout collection, before the next episode).  The
    game loop is already paused here for the PPO gradient steps, so
    training the predictor adds zero latency to env step throughput.

    Mini-batch size is intentionally small (256) so the GPU doesn't
    spike VRAM during a window already occupied by the PPO update.
    """

    _MINI_BATCH = 256

    def __init__(self, rnd_model: RNDModel) -> None:
        super().__init__(verbose=0)
        self._rnd = rnd_model

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # -- Release all virtual keys BEFORE GPU work begins ------------------
        # During PPO update (next ~15-30 s), no env.step() will fire - but the
        # last action's HOLD_KEYS (w/a/s/d/shift) are still pressed.  Without
        # this, the bot runs into walls / dashes off cliffs during every
        # update window.
        try:
            self.training_env.env_method('_release_all')
        except Exception as exc:
            print(f"[RND] _release_all failed: {exc}")

        # rollout_buffer.observations: (n_steps, n_envs, C, H, W) uint8
        obs      = self.model.rollout_buffer.observations
        obs_flat = np.ascontiguousarray(obs.reshape(-1, *obs.shape[2:]))

        total_loss, n_batches = 0.0, 0
        for start in range(0, len(obs_flat), self._MINI_BATCH):
            batch       = obs_flat[start : start + self._MINI_BATCH]
            total_loss += self._rnd.update(batch)
            n_batches  += 1

        if n_batches:
            print(f"[RND] Predictor loss: {total_loss / n_batches:.5f}  "
                  f"({len(obs_flat)} obs, {n_batches} mini-batches)")


class WatchdogCallback(BaseCallback):
    """
    Background thread that monitors env step rate.

    If no env step completes for TIMEOUT_S seconds the process is almost
    certainly stuck - the most common cause is Roblox disconnecting during
    the PPO update gap (2-3 s of no inputs -> server kick) after which the
    RecoveryGuard's 90 s timeout loops forever on the disconnect screen.

    On trigger: saves the model to RL_LATEST so --resume works, then calls
    os._exit(0).  The auto-restart launcher (launch_trainer.bat) picks it
    back up immediately.
    """

    TIMEOUT_S   = 120.0   # 2 min - longer than any normal PPO update + OCR call
    CHECK_EVERY = 15.0    # poll interval (seconds)

    def __init__(self, save_path: str) -> None:
        super().__init__(verbose=0)
        self._save_path   = save_path
        self._last_step_t = time.time()
        self._t = threading.Thread(target=self._watch, daemon=True)
        self._t.start()

    # -- background watcher ----------------------------------------------------

    def _watch(self) -> None:
        while True:
            time.sleep(self.CHECK_EVERY)
            idle = time.time() - self._last_step_t
            if idle > self.TIMEOUT_S:
                print(f"\n[Watchdog] [WARN]  No env step for {idle:.0f} s - "
                      "Roblox likely disconnected.  Saving and exiting ...")
                tmp = self._save_path + ".tmp"
                try:
                    self.model.save(tmp)
                    os.replace(tmp, self._save_path)   # atomic
                    print(f"[Watchdog] [OK]  Saved -> {self._save_path}  "
                          "(launch_trainer.bat will auto-resume)")
                except Exception as exc:
                    # Discard any partial .tmp so it doesn't accumulate
                    try:
                        if os.path.exists(tmp): os.remove(tmp)
                    except Exception:
                        pass
                    print(f"[Watchdog] Save failed: {exc}")
                # Hard exit - the main thread is blocked, can't unwind cleanly
                os._exit(0)

    # -- SB3 callback hook -----------------------------------------------------

    def _on_step(self) -> bool:
        self._last_step_t = time.time()
        return True


# -- IL weight transfer --------------------------------------------------------

def _transfer_il_weights(model: PPO, il_path: str) -> None:
    """
    Copy IL backbone + trunk -> PPO features_extractor.
    Remap IL action_head   -> PPO action_net (binary logit pairs).
    """
    if not os.path.isfile(il_path):
        print(f"[RL] No IL weights at {il_path} - "
              "policy starts from random initialisation.")
        return

    print(f"[RL] Loading IL weights from {il_path} ...")
    il = ILPolicyNet()
    il.load_state_dict(torch.load(il_path, map_location="cpu", weights_only=True))
    il.eval()

    # 1. Backbone + trunk
    extractor = model.policy.features_extractor
    missing, unexpected = extractor.load_state_dict(
        il.extractor_state_dict(), strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing}")
    print("  [OK] backbone + trunk transferred")

    # 2. Action net  (IL: 512->10  ->  SB3: 512->20 with paired logits)
    try:
        init = il.action_head_for_sb3()
        net  = model.policy.action_net
        with torch.no_grad():
            net.weight.copy_(init['weight'].to(net.weight.device))
            net.bias.copy_(  init['bias'].to(  net.bias.device))
        print("  [OK] action_net transferred")
    except Exception as exc:
        print(f"  [WARN] action_net transfer skipped: {exc}")


# -- Entry point ---------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # -- Subsystems ------------------------------------------------------------
    grabber   = ScreenGrabber(monitor_idx=2, target_fps=60)
    input_emu = InputEmulator()
    state_ctl = StateController(grabber)
    grabber.start()
    state_ctl.start()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    # -- RND curiosity model (optional) ----------------------------------------
    rnd_model: RNDModel | None = None
    if _RL_P.get("rnd_enabled", False):
        rnd_model = RNDModel(
            output_dim = int(_RL_P.get("rnd_output_dim", 64)),
            lr         = float(_RL_P.get("rnd_lr",       1e-4)),
            device     = "cuda",
        )
        print(f"[RL] RND curiosity enabled  "
              f"(beta={_RL_P.get('rnd_beta', 0.3):.3f}, "
              f"dim={_RL_P.get('rnd_output_dim', 64)})")

    # -- Environment -----------------------------------------------------------
    def _make_env():
        env = BrokenBladeEnv(input_emu, state_ctl, grabber, rnd_model=rnd_model)
        return Monitor(env, LOG_DIR)

    vec_env = DummyVecEnv([_make_env])

    # -- Model -----------------------------------------------------------------
    if args.resume and os.path.isfile(RL_LATEST):
        print(f"[RL] Resuming from {RL_LATEST}")
        # Override saved LR schedule with the current one in PPO_KWARGS so
        # tuning the schedule actually takes effect on resume.
        model = PPO.load(
            RL_LATEST, env=vec_env, device="cuda",
            custom_objects={
                "learning_rate":  PPO_KWARGS["learning_rate"],
                "lr_schedule":    PPO_KWARGS["learning_rate"],
                "clip_range":     PPO_KWARGS["clip_range"],
            },
        )
        print(f"[RL] LR schedule overridden -> 1.5e-5 -> 0 linear")
        already_trained = True
    else:
        print("[RL] Building new PPO policy ...")
        policy_kwargs = dict(
            features_extractor_class  = MobileNetV2Extractor,
            features_extractor_kwargs = dict(features_dim=FEATURE_DIM),
            # pi has NO extra MLP layers -> action_net connects directly to
            # the 512-dim feature vector, matching IL action_head shape
            net_arch                  = dict(pi=[], vf=[256]),
            normalize_images          = True,   # SB3 divides obs by 255
        )
        model = PPO(
            policy        = "CnnPolicy",
            env           = vec_env,
            policy_kwargs = policy_kwargs,
            **PPO_KWARGS,
        )
        # Freeze backbone until FREEZE_STEPS to stabilise IL features first
        model.policy.features_extractor.freeze_backbone()
        _transfer_il_weights(model, IL_WEIGHTS)
        already_trained = False

    # -- Callbacks -------------------------------------------------------------
    cb_list = [
        CheckpointCallback(
            save_freq   = SAVE_FREQ,
            save_path   = CKPT_DIR,
            name_prefix = "rl_ckpt",
        ),
        LatestSaveCallback(RL_LATEST, freq=10_000),
        BackboneUnfreezeCallback(freeze_until=FREEZE_STEPS),
        WatchdogCallback(save_path=RL_LATEST),   # kills + saves on 2-min freeze
        TelemetryCallback(),                     # feeds data to qwen_advisor.py
    ]
    if rnd_model is not None:
        cb_list.append(RNDCallback(rnd_model))  # trains predictor after each rollout
    callbacks = CallbackList(cb_list)

    # -- Train -----------------------------------------------------------------
    print(f"\n[RL] Starting PPO - {args.timesteps:,} timesteps")
    print(f"     Backbone frozen for first {FREEZE_STEPS:,} steps")
    print(f"     TensorBoard  ->  tensorboard --logdir {LOG_DIR}\n")

    try:
        model.learn(
            total_timesteps     = args.timesteps,
            callback            = callbacks,
            reset_num_timesteps = not already_trained,
            progress_bar        = True,
        )
    except KeyboardInterrupt:
        print("\n[RL] Interrupted - saving ...")
    finally:
        # Atomic save: tmp file -> rename, so Ctrl+C / crash never truncates
        # the existing rl_latest.zip.
        _tmp = RL_LATEST + ".tmp"
        try:
            model.save(_tmp)
            os.replace(_tmp, RL_LATEST)
            print(f"[RL] Saved -> {RL_LATEST}")
        except Exception as exc:
            try:
                if os.path.exists(_tmp): os.remove(_tmp)
            except Exception:
                pass
            print(f"[RL] [WARN] Final save failed: {exc}")
        input_emu.release_all()
        state_ctl.stop()
        grabber.stop()
        print("[RL] To run the trained policy:  python run_rl.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Fine-Tuning - Phase 2")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from models/rl_latest.zip")
    parser.add_argument("--timesteps",  type=int, default=TOTAL_STEPS,
                        help="Total environment steps to train for")
    main(parser.parse_args())
