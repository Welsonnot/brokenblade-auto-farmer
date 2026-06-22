@echo off
echo Installing dependencies...
pip install pynput pyautogui
echo.
echo Starting auto-attack script...
python "%~dp0autoattack.py"
pause
