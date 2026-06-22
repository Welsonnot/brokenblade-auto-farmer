"""
setup_check.py -- Pre-flight verification
==========================================
Run BEFORE the first training session to catch common setup problems.

Checks:
  1. Python version  (>=3.10)
  2. CUDA available  (GPU for PPO + RND)
  3. Required pip packages installed
  4. Monitor configuration  (Roblox must be on monitor 2 by default)
  5. Tesseract OCR  (optional -- only needed for kill counter)
  6. Model weights present  (or link to Releases)
  7. config/rl_params.json valid

Usage:
    python scripts/setup_check.py
"""

import importlib
import json
import os
import sys

# Run from project root so relative paths (models/, config/) resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)


PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"


def header(title: str) -> None:
    print(f"\n--- {title} " + "-" * (50 - len(title)))


def check_python() -> bool:
    header("Python version")
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    tag = PASS if ok else FAIL
    print(f"  {tag} Python {v.major}.{v.minor}.{v.micro}  (need >= 3.10)")
    return ok


def check_cuda() -> bool:
    header("CUDA / GPU")
    try:
        import torch
    except ImportError:
        print(f"  {FAIL} torch not installed -- run: pip install -r requirements.txt")
        return False

    if not torch.cuda.is_available():
        print(f"  {FAIL} torch.cuda.is_available() == False")
        print(f"        Install CUDA-enabled torch:")
        print(f"        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  {PASS} CUDA OK -- {name} ({vram:.1f} GB VRAM)")
    if vram < 6.0:
        print(f"  {WARN} VRAM < 6 GB -- you may need to reduce batch_size in train_rl.py")
    return True


def check_packages() -> bool:
    header("Python packages")
    required = [
        "mss", "cv2", "pynput", "pydirectinput", "PIL",
        "torch", "torchvision", "stable_baselines3", "gymnasium",
        "numpy",
    ]
    all_ok = True
    for name in required:
        try:
            importlib.import_module(name)
            print(f"  {PASS} {name}")
        except ImportError:
            print(f"  {FAIL} {name}")
            all_ok = False
    if not all_ok:
        print(f"\n  Run: pip install -r requirements.txt")
    return all_ok


def check_monitors() -> bool:
    header("Monitor configuration")
    try:
        import mss
    except ImportError:
        print(f"  {SKIP} mss not installed")
        return False
    with mss.MSS() as sct:
        mons = sct.monitors
    print(f"  Detected {len(mons)-1} monitor(s):")
    for i, m in enumerate(mons):
        label = "VIRTUAL ALL" if i == 0 else f"monitor[{i}]"
        print(f"    {label:12s} {m['width']:5d} x {m['height']:5d}  @ ({m['left']}, {m['top']})")
    if len(mons) < 3:
        print(f"\n  {WARN} Only {len(mons)-1} physical monitor(s) detected.")
        print(f"        The bot defaults to monitor index 2 (second display).")
        print(f"        If you only have ONE monitor, edit train_rl.py, main.py,")
        print(f"        and game_env.py to use monitor_idx=1 instead.")
        return False
    print(f"  {PASS} Monitor 2 available -- Roblox should run there.")
    print(f"        Tip: run obs_ocean_check.py to verify Roblox is captured.")
    return True


def check_tesseract() -> bool:
    header("Tesseract OCR (optional)")
    path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not os.path.isfile(path):
        print(f"  {WARN} Tesseract not at default path: {path}")
        print(f"        Kill-counter reward will be disabled.")
        print(f"        Download: https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = path
        version = pytesseract.get_tesseract_version()
        print(f"  {PASS} Tesseract {version} installed at default path.")
        return True
    except ImportError:
        print(f"  {WARN} pytesseract package not installed -- pip install pytesseract")
        return False


def check_weights() -> bool:
    header("Model weights")
    files = {
        "models/il_baseline.pth":      "imitation learning baseline",
        "models/rl_latest.zip":        "PPO checkpoint (latest)",
    }
    present = []
    missing = []
    for path, desc in files.items():
        if os.path.isfile(path):
            size = os.path.getsize(path) / 1e6
            present.append((path, size, desc))
            print(f"  {PASS} {path}  ({size:.1f} MB)  -- {desc}")
        else:
            missing.append((path, desc))
            print(f"  {SKIP} {path}  -- {desc}")

    if missing:
        print(f"\n  Missing weights are OPTIONAL.  You have three options:")
        print(f"    A. Train from scratch:  python recorder.py -> python train_il.py")
        print(f"    B. Download weights from Releases page on GitHub")
        print(f"    C. Use the rule-based bot:  python main.py")
    return True


def check_config() -> bool:
    header("Config file")
    path = "config/rl_params.json"
    if not os.path.isfile(path):
        print(f"  {FAIL} {path} not found.")
        return False
    try:
        with open(path) as f:
            cfg = json.load(f)
        print(f"  {PASS} {path} valid JSON ({len(cfg)} keys)")
        return True
    except json.JSONDecodeError as e:
        print(f"  {FAIL} {path} is not valid JSON: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("  Broken Blade RL Bot -- Setup Verification")
    print("=" * 60)

    results = {
        "Python":     check_python(),
        "CUDA":       check_cuda(),
        "Packages":   check_packages(),
        "Monitors":   check_monitors(),
        "Tesseract":  check_tesseract(),
        "Weights":    check_weights(),
        "Config":     check_config(),
    }

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, ok in results.items():
        tag = PASS if ok else (WARN if name in ("Tesseract", "Monitors") else FAIL)
        print(f"  {tag:8s} {name}")

    blockers = [n for n, ok in results.items()
                if not ok and n not in ("Tesseract", "Weights", "Monitors")]
    print()
    if blockers:
        print(f"  {FAIL} BLOCKERS: {', '.join(blockers)}")
        print(f"  Fix these before training.  See messages above.")
        return 1
    print(f"  {PASS} All required checks passed.  Ready to train.")
    print()
    print(f"  Next steps:")
    print(f"    1. Open Roblox on the correct monitor")
    print(f"    2. Run: python scripts/obs_ocean_check.py  (verifies frame capture)")
    print(f"    3. Run: launch_trainer.bat                 (start training)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
