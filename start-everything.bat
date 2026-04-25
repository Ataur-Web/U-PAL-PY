@echo off
rem one-click launcher. starts Ollama, two ngrok tunnels (Ollama + FastAPI)
rem and the backend in separate windows so each service is easy to watch.
title U-PAL-PY, All-in-One Launcher
setlocal EnableExtensions

rem we keep the ngrok token out of the repo. first we check for a
rem .env.local at the repo root, otherwise we prompt. token page is at
rem https://dashboard.ngrok.com/get-started/your-authtoken
if exist ".env.local" (
  for /f "usebackq tokens=1,* delims==" %%a in (".env.local") do (
    if /i "%%a"=="NGROK_TOKEN"    set "NGROK_TOKEN=%%b"
    if /i "%%a"=="OLLAMA_DOMAIN"  set "OLLAMA_DOMAIN=%%b"
    if /i "%%a"=="BACKEND_DOMAIN" set "BACKEND_DOMAIN=%%b"
  )
)
if "%NGROK_TOKEN%"==""    set /p "NGROK_TOKEN=Enter your ngrok authtoken: "
if "%OLLAMA_DOMAIN%"==""  set "OLLAMA_DOMAIN=ollama-upal.ngrok.app"
if "%BACKEND_DOMAIN%"=="" set "BACKEND_DOMAIN=upal-backend.ngrok.app"
set "OLLAMA_MODEL=llama3.1:8b-instruct-q5_K_M"

rem prefer Python 3.12, fall back to 3.11 (both have good wheel coverage
rem for torch / sentence-transformers)
py -3.12 -V >nul 2>&1
if not errorlevel 1 goto have_312
py -3.11 -V >nul 2>&1
if not errorlevel 1 goto have_311
echo [ERROR] Python 3.12 or 3.11 not found.
echo         Download: https://www.python.org/downloads/
pause & exit /b 1

:have_312
set "PY_CMD=py -3.12"
goto check_tools

:have_311
set "PY_CMD=py -3.11"
goto check_tools

:check_tools
where ollama >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ollama not found. Download: https://ollama.com/download
  pause & exit /b 1
)
where ngrok >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ngrok not found. Download: https://ngrok.com/download
  pause & exit /b 1
)

rem create venv on first run
if exist ".venv\Scripts\python.exe" goto check_deps
echo Creating virtual environment with %PY_CMD% ...
%PY_CMD% -m venv .venv
if errorlevel 1 (echo [ERROR] venv creation failed. & pause & exit /b 1)

:check_deps
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

rem probe for representative deps, if any are missing we reinstall.
"%VENV_PY%" -c "import langchain_anthropic, chromadb" >nul 2>&1
if not errorlevel 1 if exist ".venv\.installed" goto check_env

echo Installing / updating dependencies (takes a few minutes) ...
"%VENV_PY%" -m pip install --upgrade pip -q
"%VENV_PY%" -m pip install -r requirements.txt -q
if errorlevel 1 (echo [ERROR] pip install failed. & pause & exit /b 1)
echo.>.venv\.installed

:check_env
if not exist ".env" (
  echo Copying .env.example to .env ...
  copy /y .env.example .env >nul
)

rem [1/3] Ollama
echo.
echo [1/3] Starting Ollama server...
taskkill /F /IM ollama.exe >nul 2>&1

(
  echo @echo off
  echo title Ollama Server
  echo set OLLAMA_ORIGINS=*
  echo set OLLAMA_HOST=0.0.0.0
  echo ollama serve
) > "%TEMP%\upal_ollama.bat"
start "Ollama Server" cmd /k "%TEMP%\upal_ollama.bat"
ping -n 5 127.0.0.1 >nul

rem [2/3] ngrok tunnels. we write a temp yaml config with both tunnels,
rem then start ngrok pointing at it.
rem ref: https://ngrok.com/docs/agent/config/
echo [2/3] Starting ngrok tunnels for Ollama and Backend...
taskkill /F /IM ngrok.exe >nul 2>&1

(
  echo version: "3"
  echo agent:
  echo   authtoken: %NGROK_TOKEN%
  echo tunnels:
  echo   ollama:
  echo     addr: 11434
  echo     proto: http
  echo     domain: %OLLAMA_DOMAIN%
  echo   backend:
  echo     addr: 3001
  echo     proto: http
  echo     domain: %BACKEND_DOMAIN%
) > "%TEMP%\upal_ngrok.yml"

(
  echo @echo off
  echo title ngrok, Ollama + Backend Tunnels
  echo ngrok start --all --config "%TEMP%\upal_ngrok.yml"
) > "%TEMP%\upal_ngrok.bat"
start "ngrok, Tunnels" /MIN cmd /k "%TEMP%\upal_ngrok.bat"
ping -n 4 127.0.0.1 >nul

rem [3/3] FastAPI. we keep it in a visible window so any traceback is
rem on screen, no hunting through log files. DO NOT click inside the
rem window (Windows QuickEdit pauses the process).
echo [3/3] Starting FastAPI backend on http://localhost:3001...
if exist "backend.log" del "backend.log"

(
  echo @echo off ^& title U-PAL-PY backend
  echo color 0B
  echo cd /d "%CD%"
  echo set "PYTHONUNBUFFERED=1"
  echo set "PYTHONFAULTHANDLER=1"
  echo echo U-PAL-PY FastAPI backend, http://localhost:3001
  echo echo Public URL: https://%BACKEND_DOMAIN%
  echo echo.
  echo echo (do not click inside this window, QuickEdit pauses uvicorn^)
  echo echo.
  echo "%VENV_PY%" -m uvicorn app.main:app --host 0.0.0.0 --port 3001
  echo echo.
  echo echo ====== UVICORN EXITED, scroll up for the reason ======
  echo pause
) > "%TEMP%\upal_backend.bat"
start "U-PAL-PY backend" cmd /k "%TEMP%\upal_backend.bat"
ping -n 7 127.0.0.1 >nul

echo.
echo All services running, keep this window open
echo   Ollama:    http://localhost:11434
echo   Ollama:    https://%OLLAMA_DOMAIN%
echo   FastAPI:   http://localhost:3001
echo   FastAPI:   https://%BACKEND_DOMAIN%   (this is CHAT_BACKEND_URL)
echo   API docs:  http://localhost:3001/docs
echo.
echo In Vercel (u-pal-py project) set:
echo   CHAT_BACKEND_URL = https://%BACKEND_DOMAIN%
echo the URL is static so you only set it once.
echo.
echo Press any key to EXIT (other windows stay open)
pause >nul

endlocal
