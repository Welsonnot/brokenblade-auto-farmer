"""
Qwen Advisor - Training Diagnostics for Broken Blade RL
========================================================
Reads training telemetry from logs/telemetry.jsonl, asks a local Qwen model
(via Ollama) what numeric parameters look off, and PRINTS the suggestion.

It does NOT write anything automatically.
Workflow:
    1. Run this script (or --loop to keep watching)
    2. Review the printed suggestion
    3. Apply the change to config/rl_params.json if it makes sense

This keeps a human in the loop before any config changes land.

Usage:
    # Scan once and print suggestion:
    python qwen_advisor.py

    # Watch continuously, print suggestion every 5 minutes:
    python qwen_advisor.py --loop --interval 300

Qwen model used: qwen3.6-fast (create with: ollama create qwen3.6-fast -f ModelFile.txt)
                 Falls back to: qwen2.5:3b, qwen2.5:7b, llama3.2
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

# -- Paths ---------------------------------------------------------------------
_ROOT         = Path(__file__).parent.parent   # rl/ -> project root
_TELEMETRY    = _ROOT / "logs" / "telemetry.jsonl"
_PARAMS_PATH  = _ROOT / "config" / "rl_params.json"

# -- Qwen / Ollama config -------------------------------------------------------
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL_NAMES   = ["qwen3.6-fast", "qwen2.5:3b", "qwen2.5:7b", "llama3.2"]

# -- Advisor config -------------------------------------------------------------
RECENT_ROWS   = 20     # number of telemetry rows to summarize for Qwen
MIN_ROWS      = 5      # don't advise until at least this many rows exist

# -- Hard limits - Qwen must never suggest values outside these bounds ----------
# Prevents runaway changes that could destabilize training.
_BOUNDS: dict[str, tuple[float, float]] = {
    "LOST_WINDOW_S":      (15.0,   120.0),
    "OCEAN_FAIL_STEPS":   (10.0,   100.0),
    "AIR_ATTACK_GRACE":   (2.0,    20.0),
    "AIR_ATTACK_BASE":    (0.01,   0.30),
    "AIR_ATTACK_MAX_MUL": (1.0,    8.0),
    "RESET_SETTLE_S":     (1.0,    5.0),
    "OCEAN_PENALTY":      (0.5,    10.0),
    "OCEAN_FAIL_PENALTY": (10.0,   200.0),
}


# -- Telemetry reader ----------------------------------------------------------

def _read_telemetry(n: int = RECENT_ROWS) -> list[dict]:
    """Return the last `n` rows from telemetry.jsonl."""
    if not _TELEMETRY.exists():
        return []
    rows = []
    with open(_TELEMETRY) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows[-n:]


def _summarize(rows: list[dict]) -> str:
    """Turn recent telemetry rows into a compact text block for Qwen."""
    if not rows:
        return "(no telemetry yet)"

    steps       = [r.get("step",            0)     for r in rows]
    rewards     = [r.get("ep_rew_mean")             for r in rows if r.get("ep_rew_mean") is not None]
    ep_lens     = [r.get("ep_len_mean")             for r in rows if r.get("ep_len_mean") is not None]
    attack_hrs  = [r.get("attack_hit_rate")         for r in rows if r.get("attack_hit_rate") is not None]
    ocean_rates = [r.get("ocean_rate",       0.0)   for r in rows]
    fpses       = [r.get("fps",              0.0)   for r in rows]

    def _avg(lst):
        return sum(lst) / len(lst) if lst else None

    def _trend(lst):
        if len(lst) < 2:
            return "stable"
        delta = lst[-1] - lst[0]
        if abs(delta) < 0.01 * (abs(lst[0]) + 1e-6):
            return "stable"
        return "improving" if delta > 0 else "declining"

    lines = [
        f"Steps seen: {steps[0]} -> {steps[-1]}",
        f"Ep reward mean: {_avg(rewards):.3f} ({_trend(rewards)})"  if rewards     else "Ep reward mean: N/A",
        f"Ep length mean: {_avg(ep_lens):.1f}"                       if ep_lens     else "Ep length mean: N/A",
        f"Attack hit rate: {_avg(attack_hrs):.2%}"                   if attack_hrs  else "Attack hit rate: N/A (boss bar not visible while attacking, or 0 attacks)",
        f"Ocean rate: {_avg(ocean_rates):.2%} ({_trend(ocean_rates)})",
        f"FPS: {_avg(fpses):.1f}",
    ]
    return "\n".join(lines)


# -- Current params reader -----------------------------------------------------

def _read_params() -> dict:
    try:
        with open(_PARAMS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# -- Ollama caller -------------------------------------------------------------

def _call_ollama(prompt: str, model: str) -> str | None:
    """POST to local Ollama and return the response text, or None on failure."""
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }).encode()

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except urllib.error.URLError as exc:
        print(f"[Advisor] Ollama unreachable: {exc.reason}")
        return None
    except Exception as exc:
        print(f"[Advisor] Ollama error: {exc}")
        return None


def _find_model() -> str | None:
    """Return the first available Ollama model from MODEL_NAMES."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data   = json.loads(resp.read())
            avail  = {m["name"].split(":")[0] for m in data.get("models", [])}
            avail |= {m["name"]               for m in data.get("models", [])}
    except Exception:
        avail = set()

    for name in MODEL_NAMES:
        if name in avail or name.split(":")[0] in avail:
            return name
    # If we can't list models, try the first one and let it fail gracefully
    return MODEL_NAMES[0]


# -- Prompt builder ------------------------------------------------------------

def _build_prompt(telemetry_summary: str, current_params: dict) -> str:
    params_str  = json.dumps(
        {k: v for k, v in current_params.items() if not k.startswith("_")},
        indent=2
    )
    bounds_str  = "\n".join(
        f"  {k}: [{lo}, {hi}]" for k, (lo, hi) in _BOUNDS.items()
    )

    return f"""You are a reinforcement learning parameter tuning assistant for a Roblox boss-farming bot.

The bot uses PPO to learn to fight a boss in the game "Broken Blade".
It receives training telemetry and must suggest small adjustments to numeric config parameters.

## Current config/rl_params.json:
{params_str}

## Allowed parameter bounds (NEVER suggest values outside these):
{bounds_str}

## Recent training telemetry:
{telemetry_summary}

## Your task:
Analyze the telemetry and suggest parameter changes that would improve training.
Focus on the most important issue only. Do NOT suggest changes unless there is a clear problem.

Rules:
- Output ONLY a valid JSON object with the parameters you recommend changing.
- Only include parameters that actually need to change (omit unchanged ones).
- If nothing needs changing, output: {{}}
- Do NOT include explanations outside the JSON - a human will review this.
- Keep changes small: adjust by <=30% per suggestion.
- Suggest only ONE change at a time (the most important one).

Common issues and fixes:
- High ocean_rate (>15%): lower OCEAN_FAIL_STEPS (faster episode end) or raise OCEAN_PENALTY
- Attack hit rate = 0 or N/A after 5000+ steps: lower AIR_ATTACK_BASE (too harsh), raise AIR_ATTACK_GRACE
- Reward declining + low ep_len: raise AIR_ATTACK_GRACE (bot stopping too soon)
- Reward declining + high ep_len: lower LOST_WINDOW_S (boss search taking too long)
- Ep reward always negative: check AIR_ATTACK_BASE isn't too high

Output (JSON only):"""


# -- Response parser -----------------------------------------------------------

def _parse_response(text: str) -> dict:
    """Extract and validate the JSON object from Qwen's response."""
    # Find outermost { ... } block
    start = text.find('{')
    end   = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}

    # Strip comment keys and validate bounds
    cleaned = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if k not in _BOUNDS:
            print(f"[Advisor] Ignoring unknown key '{k}'")
            continue
        lo, hi = _BOUNDS[k]
        v_clamped = max(lo, min(hi, float(v)))
        if v_clamped != float(v):
            print(f"[Advisor] Clamped {k}: {v} -> {v_clamped} (bounds [{lo}, {hi}])")
        cleaned[k] = v_clamped

    return cleaned


# -- Printer (no writes - user applies the change) ----------------------------

def _print_suggestion(suggestions: dict, current: dict) -> bool:
    """
    Print the suggested changes for the user to review.
    Does NOT write anything to disk.
    Returns True when there is a real suggestion.
    """
    if not suggestions:
        print("[Advisor] No changes needed - training looks healthy.")
        return False

    changed = {k: v for k, v in suggestions.items() if current.get(k) != v}
    if not changed:
        print("[Advisor] Qwen's values match current config - no action needed.")
        return False

    print()
    print("=" * 62)
    print("  QWEN SUGGESTION")
    print("  Review and apply to config/rl_params.json if it looks right.")
    print("=" * 62)
    for k, new in changed.items():
        old = current.get(k, "?")
        print(f"  {k}: {old}  ->  {new}")
    print()
    print("  Suggested JSON:")
    print()
    print("  " + json.dumps(changed, indent=2).replace("\n", "\n  "))
    print("=" * 62)
    print()
    return True


# -- Main ----------------------------------------------------------------------

def run_once(verbose: bool = True) -> bool:
    """
    Run one advisor cycle.
    Reads telemetry, asks Qwen, prints suggestion.
    Returns True when Qwen had a suggestion to show.
    """
    rows = _read_telemetry()
    if len(rows) < MIN_ROWS:
        if verbose:
            print(f"[Advisor] Only {len(rows)}/{MIN_ROWS} telemetry rows - "
                  "waiting for more training data.")
        return False

    summary  = _summarize(rows)
    current  = _read_params()
    model    = _find_model()

    if verbose:
        print(f"[Advisor] Telemetry summary:\n{summary}\n")
        print(f"[Advisor] Asking {model} ...")

    prompt   = _build_prompt(summary, current)
    response = _call_ollama(prompt, model)

    if response is None:
        print("[Advisor] No response from Ollama - is it running?")
        return False

    if verbose:
        print(f"[Advisor] Qwen raw response:\n{response}\n")

    suggestions = _parse_response(response)
    return _print_suggestion(suggestions, current)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen advisor - reads telemetry and prints suggested config changes"
    )
    parser.add_argument("--loop",     action="store_true",
                        help="Run repeatedly instead of once")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between advisor cycles when --loop is set (default: 300)")
    parser.add_argument("--quiet",    action="store_true",
                        help="Skip printing telemetry summary and Qwen raw response")
    args = parser.parse_args()

    print("[Advisor] Qwen advisor started  (read-only - review suggestions before applying)")
    print(f"[Advisor] Telemetry: {_TELEMETRY}")
    print(f"[Advisor] Config:    {_PARAMS_PATH}")
    print()

    if args.loop:
        print(f"[Advisor] Loop mode - checking every {args.interval}s.  Ctrl+C to stop.\n")
        while True:
            try:
                run_once(verbose=not args.quiet)
            except KeyboardInterrupt:
                print("\n[Advisor] Stopped.")
                break
            except Exception as exc:
                print(f"[Advisor] Unexpected error: {exc}")
            print(f"[Advisor] Sleeping {args.interval}s ...\n")
            time.sleep(args.interval)
    else:
        run_once(verbose=not args.quiet)


if __name__ == "__main__":
    main()
