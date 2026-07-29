@echo off
rem one-off helper that rebuilds the Chroma vector index from the JSON
rem files in app/data/. we keep this as a .bat so the Windows student
rem workflow is just "double-click".
title U-PAL-PY ingest
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] No venv found. Run start.bat first.
  pause
  exit /b 1
)

set VENV_PY=.venv\Scripts\python.exe

echo.
echo Ingesting UWTSD corpus / facts / knowledge into Chroma ...
echo (first run downloads the sentence-transformers model, roughly 450 MB)
echo.

"%VENV_PY%" -m scripts.ingest %*

echo.
pause
endlocal
