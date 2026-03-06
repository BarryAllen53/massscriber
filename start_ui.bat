@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo [HATA] Once install.bat calistirilmali.
  exit /b 1
)

call .venv\Scripts\activate
python app.py
