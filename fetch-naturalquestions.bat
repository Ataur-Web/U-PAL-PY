@echo off
rem wrapper around scripts/fetch_naturalquestions.py. pulls Google's
rem Natural Questions short-answer subset into the Chroma index so the
rem bot can handle general academic questions, not just UWTSD-specific
rem ones. dataset is published by Google Research.
rem ref: https://ai.google.com/research/NaturalQuestions
title Fetch Google Natural Questions
setlocal

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] Virtual environment not found.
  echo         Run start-everything.bat first.
  pause
  exit /b 1
)

echo.
echo Fetching Natural Questions general-knowledge Q^&A pairs into Chroma ...
echo.
echo streams the nq_open subset from HuggingFace and filters for
echo academic and study-related questions. first run downloads
echo roughly 100 MB.
echo.
echo pass --sample 10000 for a faster smaller run.
echo pass --dry-run to preview without saving.
echo.

"%VENV_PY%" -m scripts.fetch_naturalquestions %*

echo.
echo Done. Restart start-everything.bat to load the enriched index.
echo.
pause
endlocal
