@echo off
rem wrapper around scripts/fetch_openorca.py. we use this to pull a small
rem sample of the OpenOrca dataset into the Chroma index so the bot has
rem extra general-knowledge Q&A to fall back on.
title Fetch OpenOrca Education Q&A
setlocal

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] Virtual environment not found.
  echo         Run start-everything.bat first.
  pause
  exit /b 1
)

echo.
echo Fetching OpenOrca education Q^&A pairs into Chroma ...
echo.
echo streams the first 50,000 rows from HuggingFace and filters for
echo university-relevant content. first run downloads roughly 500 MB.
echo.
echo pass --sample 20000 for a faster smaller run.
echo pass --dry-run to preview without saving.
echo.

"%VENV_PY%" -m scripts.fetch_openorca %*

echo.
echo Done. Restart start-everything.bat to load the enriched index.
echo.
pause
endlocal
