@echo off
title Broken Blade RL Trainer
prompt [BrokenBlade]$G

:restart
echo.
echo [Launcher] ============================================================
echo [Launcher]  Starting RL training  (%date% %time%)
echo [Launcher] ============================================================
echo.

if exist "models\rl_latest.zip" (
    echo [Launcher] Found rl_latest.zip -- resuming.
    python train_rl.py --resume
) else (
    echo [Launcher] No checkpoint found -- fresh start from IL weights.
    python train_rl.py
)

echo.
echo [Launcher] Training process exited (code %errorlevel%).
echo [Launcher] Restarting in 15 seconds... (Ctrl+C to cancel)
echo.
timeout /t 15 /nobreak
goto restart
