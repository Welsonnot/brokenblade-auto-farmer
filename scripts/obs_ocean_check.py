"""
obs_ocean_check.py - Two-branch ocean detection diagnostic
===========================================================
Run in each of these spots and paste ALL outputs:
  1. In ocean - daytime (sun glare)        -> [OK] TRUE  (Branch A)
  2. In ocean - night                      -> [OK] TRUE  (Branch B)
  3. On land, fighting monsters            -> [FAIL] FALSE (land+monster: bri<85)
  4. On land, old monster glow             -> [FAIL] FALSE (variance>25)
  5. On land, boss arena dark floor        -> [FAIL] FALSE (Branch A bri<85; Branch B var>8)

Usage:  python obs_ocean_check.py
"""

# Run from project root so relative paths resolve.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)


import json, os
import numpy as np
import mss
import cv2

# -- Live values from rl_params.json -------------------------------------------
_cfg = {}
try:
    with open(os.path.join(os.path.dirname(__file__), "config", "rl_params.json")) as f:
        _cfg = json.load(f)
except Exception:
    pass

# Branch A - bright/daytime ocean
BRIGHT_BRI_MIN  = float(_cfg.get("ocean_bright_bri_min",  85.0))
BRIGHT_VAR_MAX  = float(_cfg.get("ocean_bright_var_max",  35.0))
BRIGHT_BIAS_MIN = float(_cfg.get("ocean_bright_bias_min", 38.0))
BRIGHT_GR_MIN   = float(_cfg.get("ocean_bright_gr_min",   15.0))

# Branch B - dark/night ocean
DARK_BRI_MAX    = float(_cfg.get("ocean_dark_bri_max",    60.0))
DARK_VAR_MAX    = float(_cfg.get("ocean_dark_var_max",     8.0))

MONITOR_IDX = 2

# -- Grab frame ----------------------------------------------------------------
with mss.MSS() as sct:
    monitor = sct.monitors[MONITOR_IDX]
    raw_bgr = np.array(sct.grab(monitor))[:, :, :3]

print(f"Monitor {MONITOR_IDX}: {raw_bgr.shape[1]}x{raw_bgr.shape[0]}")

# -- _frame_to_obs() -----------------------------------------------------------
small = cv2.resize(raw_bgr, (224, 224), interpolation=cv2.INTER_AREA)
rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
obs   = rgb.transpose(2, 0, 1).astype(np.uint8)

# -- Compute stats - exact replica of in_ocean() -------------------------------
r_l = obs[0, 85:160,  56: 90].astype(np.float32)
g_l = obs[1, 85:160,  56: 90].astype(np.float32)
b_l = obs[2, 85:160,  56: 90].astype(np.float32)
r_r = obs[0, 85:160, 134:168].astype(np.float32)
g_r = obs[1, 85:160, 134:168].astype(np.float32)
b_r = obs[2, 85:160, 134:168].astype(np.float32)
r   = np.concatenate([r_l.ravel(), r_r.ravel()])
g   = np.concatenate([g_l.ravel(), g_r.ravel()])
b   = np.concatenate([b_l.ravel(), b_r.ravel()])
bri       = (r + b) * 0.5
mean_bri  = float(np.mean(bri))
std_bri   = float(np.std(bri))
blue_bias = float(np.mean(b) - np.mean(r))
green_red = float(np.mean(g) - np.mean(r))   # G-R: ocean water > ice crystals

# -- Branch A: Daytime / bright ocean -----------------------------------------
a_bri_pass  = mean_bri  > BRIGHT_BRI_MIN
a_var_pass  = std_bri   < BRIGHT_VAR_MAX
a_bias_pass = blue_bias > BRIGHT_BIAS_MIN
a_gr_pass   = green_red > BRIGHT_GR_MIN
bright_ocean = a_bri_pass and a_var_pass and a_bias_pass and a_gr_pass

# -- Branch B: Night / dark ocean ----------------------------------------------
b_bri_pass = mean_bri < DARK_BRI_MAX
b_var_pass = std_bri  < DARK_VAR_MAX
dark_ocean  = b_bri_pass and b_var_pass

result = bright_ocean or dark_ocean

# -- Print ---------------------------------------------------------------------
print()
print("--- Measured values (left [56:90] + right [134:168] strips, rows 85-160) --")
print(f"  Brightness (R+B)/2 : {mean_bri:6.1f}   R={np.mean(r):.1f}  G={np.mean(g):.1f}  B={np.mean(b):.1f}")
print(f"  Variance (std)     : {std_bri:6.1f}")
print(f"  Blue bias  (B-R)   : {blue_bias:6.1f}")
print(f"  Green-Red  (G-R)   : {green_red:6.1f}   <- KEY: ocean water > ice crystals")
print()

print("--- Branch A - Daytime / bright ocean (all four must pass) ----------------")
print(f"  {'[OK]' if a_bri_pass  else '[FAIL]'} bri  > {BRIGHT_BRI_MIN:.0f}   : {mean_bri:6.1f}   (land+monster~77 excluded here)")
print(f"  {'[OK]' if a_var_pass  else '[FAIL]'} std  < {BRIGHT_VAR_MAX:.0f}   : {std_bri:6.1f}   (ocean~10-28, glow monsters~50)")
print(f"  {'[OK]' if a_bias_pass else '[FAIL]'} B-R  > {BRIGHT_BIAS_MIN:.0f}   : {blue_bias:6.1f}   (ocean~40-51, land+monster~33)")
print(f"  {'[OK]' if a_gr_pass   else '[FAIL]'} G-R  > {BRIGHT_GR_MIN:.0f}   : {green_red:6.1f}   (ocean water>15, ice crystals~8.6)")
if bright_ocean:
    print("  -> [OK] Branch A PASSES")
else:
    a_failed = []
    if not a_bri_pass:  a_failed.append(f"bri {mean_bri:.1f} <= {BRIGHT_BRI_MIN}")
    if not a_var_pass:  a_failed.append(f"std {std_bri:.1f} >= {BRIGHT_VAR_MAX}")
    if not a_bias_pass: a_failed.append(f"B-R {blue_bias:.1f} <= {BRIGHT_BIAS_MIN}")
    if not a_gr_pass:   a_failed.append(f"G-R {green_red:.1f} <= {BRIGHT_GR_MIN}")
    print(f"  -> [FAIL] Branch A fails: {', '.join(a_failed)}")

print()
print("--- Branch B - Night / dark ocean (both must pass) -------------------------")
print(f"  {'[OK]' if b_bri_pass else '[FAIL]'} bri  < {DARK_BRI_MAX:.0f}   : {mean_bri:6.1f}   (very dark)")
print(f"  {'[OK]' if b_var_pass else '[FAIL]'} std  < {DARK_VAR_MAX:.0f}    : {std_bri:6.1f}   (night water~3.6; boss arena~22 rejected)")
if dark_ocean:
    print("  -> [OK] Branch B PASSES")
else:
    b_failed = []
    if not b_bri_pass: b_failed.append(f"bri {mean_bri:.1f} >= {DARK_BRI_MAX}")
    if not b_var_pass: b_failed.append(f"std {std_bri:.1f} >= {DARK_VAR_MAX}")
    print(f"  -> [FAIL] Branch B fails: {', '.join(b_failed)}")

print()
if result:
    branch = "A (bright)" if bright_ocean else "B (dark)"
    print(f"  [OK]  in_ocean() -> TRUE   (OCEAN - Branch {branch} passed)")
else:
    print(f"  [FAIL]  in_ocean() -> FALSE  (LAND)")

print()
print("  Real measurements per scenario:")
print("    Wavy ocean      : bri~ 86  std~28  B-R~+40  -> [OK] TRUE  (Branch A)")
print("    Daytime ocean   : bri~106  std~19  B-R~+51  -> [OK] TRUE  (Branch A)")
print("    Borderline ocean: bri~ 91  std~ 9  B-R~+43  -> [OK] TRUE  (Branch A)")
print("    Night ocean     : bri~ 48  std~ 4  B-R~-13  -> [OK] TRUE  (Branch B)")
print("    Land+monster    : bri~ 77  std~13  B-R~+33  -> [FAIL] FALSE (A: bri 77<85; B: bri>60)")
print("    Old glow monster: bri~ 82  std~55  B-R~+36  -> [FAIL] FALSE (A: bri 82<85; B: bri>60)")
print("    Boss arena      : bri~ 46  std~22  B-R~+47  -> [FAIL] FALSE (A: bri<85; B: std>8) + gated")

# -- Save visuals --------------------------------------------------------------
cv2.imwrite("obs_view.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
vis = rgb.copy()
lr = vis[85:160,  56: 90].astype(np.float32)
vis[85:160,  56: 90] = np.clip(lr * 0.4 + np.array([0,200,200])*0.6, 0,255).astype(np.uint8)
rr = vis[85:160, 134:168].astype(np.float32)
vis[85:160, 134:168] = np.clip(rr * 0.4 + np.array([0,200,0])*0.6, 0,255).astype(np.uint8)
cr = vis[85:160,  90:134].astype(np.float32)
vis[85:160,  90:134] = np.clip(cr * 0.7 + np.array([200,0,0])*0.3, 0,255).astype(np.uint8)
cv2.imwrite("obs_sample_region.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print()
print("  obs_view.png / obs_sample_region.png saved.")
print()
