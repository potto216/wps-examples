@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: capture_example.bat
:: 
:: Description:
::   Runs the WPS capture CLI with optional power scheme management.
::   If a power scheme name is provided, the script will switch to that
::   scheme before running the capture, then restore the original scheme.
::
:: Usage:
::   capture_example.bat [power_scheme_name]
::   capture_example.bat /?
::   capture_example.bat --help
::
:: Arguments:
::   power_scheme_name  (Optional) Name of the power scheme to use during
::                      capture (e.g., "Always on", "High performance")
::                      If not provided, no power scheme change occurs.
::   /?, --help, -h     Display this help message
::
:: Examples:
::   capture_example.bat "Always on"
::   capture_example.bat
:: ============================================================================

:: Check for help flag
if "%~1"=="/?" goto :show_help
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help

:: Get the power scheme name from command line (optional)
set "power_scheme_name=%~1"
set "original_guid="
set "target_guid="

:: Only handle power scheme if one was specified
if not "%power_scheme_name%"=="" (
    echo Power scheme requested: %power_scheme_name%
    echo.
    
    :: 1. Get the GUID of the currently active power scheme
    for /f "tokens=4" %%a in ('powercfg /getactivescheme') do (
        set "original_guid=%%a"
    )
    
    echo Original Power Scheme GUID: %original_guid%
    
    :: 2. Find the GUID for the requested power profile
    for /f "tokens=4" %%a in ('powercfg /list ^| findstr /i /c:"%power_scheme_name%"') do (
        set "target_guid=%%a"
    )
    
    :: 3. Check if target_guid was found and is not empty
    if "!target_guid!"=="" (
        echo ERROR: Power scheme "%power_scheme_name%" not found in 'powercfg /list'.
        echo.
        echo Available power schemes:
        powercfg /list
        pause
        exit /b 1
    )
    
    :: 4. Switch to requested power scheme
    echo Switching to "%power_scheme_name%" ^(!target_guid!^)...
    powercfg /setactive !target_guid!
    
    echo.
    echo The power profile is now "%power_scheme_name%".
    echo.
) else (
    echo No power scheme specified - using current power settings.
    echo.
)


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


:: Restore the original power scheme if we changed it
if not "%original_guid%"=="" (
    echo.
    echo Restoring original power scheme ^(%original_guid%^)...
    powercfg /setactive %original_guid%
    echo Power scheme restored.
)

echo.
echo Done. Hit any key to exit.
PAUSE
goto :eof

:show_help
echo.
echo capture_example.bat - WPS Capture with Power Scheme Management
echo.
echo Usage:
echo   capture_example.bat [power_scheme_name]
echo   capture_example.bat /?
echo.
echo Arguments:
echo   power_scheme_name  ^(Optional^) Name of the power scheme to use during
echo                      capture ^(e.g., "Always on", "High performance"^)
echo                      If not provided, no power scheme change occurs.
echo.
echo   /?, --help, -h     Display this help message
echo.
echo Examples:
echo   capture_example.bat "Always on"
echo   capture_example.bat
echo.
echo To see available power schemes on your system, run:
echo   powercfg /list
echo.
exit /b 0