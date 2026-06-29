@echo off
title MT5 Trading Bot Monitor
echo ========================================================
echo   STARTING PROFESSIONAL MT5 TRADING BOT...
echo ========================================================
echo.

:: Run the trading bot using the virtual environment python interpreter.
:: We use the -u flag to force unbuffered console output for real-time logs.
.venv\Scripts\python.exe -u run_trader.py

echo.
echo ========================================================
echo   Trading bot has stopped or failed to start.
echo ========================================================
pause
