@echo off
setlocal
cd /d "%~dp0"
python -B web\serve.py --port 0 --open
if errorlevel 1 pause
