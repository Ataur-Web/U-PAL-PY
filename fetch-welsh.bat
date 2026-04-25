@echo off
rem wrapper around scripts/fetch_termcymru.py. pulls fresh Welsh-English
rem terminology from BydTermCymru (termau.cymru) so the language detector
rem and query augmenter have an up-to-date vocab.
title Fetch BydTermCymru Welsh Terms
setlocal

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] Virtual environment not found.
  echo         Run start-everything.bat first to set up the environment.
  pause
  exit /b 1
)

echo.
echo Fetching Welsh terminology from BydTermCymru (termau.cymru) ...
echo queries the public API for university-relevant Welsh-English pairs.
echo takes roughly 1-2 minutes.
echo.

"%VENV_PY%" -m scripts.fetch_termcymru %*

echo.
echo Done. Restart start-everything.bat to load the new terms.
echo.
pause
endlocal
