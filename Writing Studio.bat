@echo off
rem Starts Writing Studio. The first time (or if something is missing) it runs the installer first.
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" if exist "models\.complete" goto run
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
if errorlevel 1 (
  echo.
  echo Setup did not finish. Run "Writing Studio.bat" again to retry; it picks up where it stopped.
  pause
  exit /b 1
)
:run
start "" .venv\Scripts\pythonw.exe app.py
