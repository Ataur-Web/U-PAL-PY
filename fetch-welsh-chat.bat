@echo off
rem wrapper around scripts/fetch_welsh_chat.py. pulls a Welsh-language
rem chat dataset (locailabs/nemotron-chat-welsh) into Chroma so the bot
rem has natural Welsh phrasing to ground its replies in. complements
rem fetch-welsh.bat which fetches BydTermCymru terminology only.
rem ref: https://huggingface.co/datasets/locailabs/nemotron-chat-welsh
title Fetch Welsh Chat Dataset
setlocal

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] Virtual environment not found.
  echo         Run start-everything.bat first.
  pause
  exit /b 1
)

echo.
echo Fetching Welsh chat pairs (nemotron-chat-welsh) into Chroma ...
echo.
echo streams a Welsh-language chat dataset from HuggingFace and filters
echo for high-quality Welsh Q^&A pairs. first run downloads roughly
echo 200 MB.
echo.
echo pass --sample 5000 for a faster smaller run.
echo pass --dry-run to preview without saving.
echo.

"%VENV_PY%" -m scripts.fetch_welsh_chat %*

echo.
echo Done. Restart start-everything.bat to load the enriched index.
echo.
pause
endlocal
