@echo off
title MT5 Trading Bot Update Utility
echo ========================================================
echo   UPDATING TRADING BOT CODE FROM GITHUB...
echo ========================================================
echo.

:: Try pulling using system PATH git, fallback to default program path if needed
git pull origin main 2>nul
if %errorlevel% neq 0 (
    echo System PATH git failed. Trying default installation path...
    "C:\Program Files\Git\cmd\git.exe" pull origin main
)

echo.
echo ========================================================
echo   Update check completed!
echo ========================================================
pause
