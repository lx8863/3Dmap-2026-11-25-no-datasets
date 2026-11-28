@echo off
setlocal
cd /d "%~dp0"
set "3DMAP_PYTHON=python"
if exist "web\.vggt\venv\Scripts\python.exe" set "3DMAP_PYTHON=web\.vggt\venv\Scripts\python.exe"
if exist "web\.venv\Scripts\python.exe" set "3DMAP_PYTHON=web\.venv\Scripts\python.exe"
"%3DMAP_PYTHON%" -B web\serve.py --port 0 --open
if errorlevel 1 pause
