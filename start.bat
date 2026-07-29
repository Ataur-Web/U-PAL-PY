@echo off
rem minimal launcher for the FastAPI backend. we use the `py` launcher
rem because pip.exe breaks when the install path contains a space, which
rem it does on this laptop (C:\Users\Ataur Rahman\...).
title U-PAL-PY backend
setlocal

rem prefer Python 3.12, fall back to 3.11. both have good wheel coverage
rem for torch and sentence-transformers. 3.13+ sometimes doesn't.
set PY_CMD=

py -3.12 -V >nul 2>&1
if not errorlevel 1 (
  set PY_CMD=py -3.12
  goto :python_found
)

py -3.11 -V >nul 2>&1
if not errorlevel 1 (
  set PY_CMD=py -3.11
  goto :python_found
)

echo [ERROR] Neither Python 3.12 nor 3.11 found via the py launcher.
echo         Install from https://www.python.org/downloads/
pause
exit /b 1

:python_found
echo Using %PY_CMD%

rem create venv on first run
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment ...
  %PY_CMD% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] venv creation failed.
    pause
    exit /b 1
  )
)

rem call the venv python directly so we don't have to activate it
set VENV_PY=.venv\Scripts\python.exe

rem install deps on first run (marker file saves time on subsequent runs)
if not exist ".venv\.installed" (
  echo [2/3] Installing dependencies (first run takes a few minutes) ...
  "%VENV_PY%" -m pip install --upgrade pip
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] pip install failed, check the output above.
    pause
    exit /b 1
  )
  echo. > .venv\.installed
)

rem copy the env template over on first run
if not exist ".env" (
  echo [!] No .env found, copying from .env.example
  copy /y .env.example .env >nul
)

echo.
echo Starting U-PAL-PY on http://localhost:3001
echo Health:  http://localhost:3001/api/health
echo Docs:    http://localhost:3001/docs
echo.
"%VENV_PY%" -m uvicorn app.main:app --reload --port 3001

endlocal
