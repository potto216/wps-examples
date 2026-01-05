@echo off
setlocal enabledelayedexpansion

:: 1. Get the GUID of the currently active power scheme
for /f "tokens=4" %%a in ('powercfg /getactivescheme') do (
    set "original_guid=%%a"
)

echo Original Power Scheme GUID: %original_guid%

:: 2. Find the GUID for the "Always on" profile
:: We search for the line containing "Always on" and pick the 4th token (the GUID)
set "target_guid="
for /f "tokens=4" %%a in ('powercfg /list ^| findstr /i /c:"Always on"') do (
    set "target_guid=%%a"
)

:: 3. Check if target_guid was found and is not empty
if "%target_guid%"=="" (
    echo ERROR: "Always on" profile not found in 'powercfg /list'.
    pause
    exit /b
)

:: 4. Switch to "Always on"
echo Switching to "Always on" (%target_guid%)...
powercfg /setactive %target_guid%

echo.
echo The power profile is now "Always on".
echo Press any key to restore the original setting and exit...
pause >nul


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


:: 5. Restore the original power scheme
echo Restoring original scheme (%original_guid%)...
powercfg /setactive %original_guid%

echo Done.

  PAUSE