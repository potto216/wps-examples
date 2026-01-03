@echo off
REM Try to activate the virtual environment from the repo, fall back to absolute path
if exist "%~dp0.venv\Scripts\Activate.bat" (
  call "%~dp0.venv\Scripts\Activate.bat"
) else (
  ECHO Add absolute path CALL to venv activation script below
)

python scripts\capture\wps_capture_cli.py ^
  --tcp-ip 127.0.0.1 ^
  --equipment X240 ^
  --sleep-time 3 ^
  --log-file capture.log ^
  --data-path "C:\\Users\\Public\\Documents\\Teledyne LeCroy Wireless\\My Capture Files" ^
  --le ^
  --bredr

  PAUSE