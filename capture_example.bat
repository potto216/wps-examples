@echo off
setlocal

REM Ensure the system stays awake during capture (monitor may still sleep).
powershell -NoProfile -ExecutionPolicy Bypass -Command "powercfg /change standby-timeout-ac 0; powercfg /change standby-timeout-dc 0; powercfg /change hibernate-timeout-ac 0; powercfg /change hibernate-timeout-dc 0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0capture_example.ps1"

endlocal
