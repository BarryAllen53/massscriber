@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
  echo [INFO] Sanal ortam bulunamadi. Once install.bat calistiriliyor...
  call install.bat
  if errorlevel 1 goto :error
)

call .venv\Scripts\activate
if errorlevel 1 goto :error

echo [INFO] Desktop build araclari yukleniyor...
python -m pip install -e ".[desktop]"
if errorlevel 1 goto :error

echo [INFO] Windows masaustu paketi olusturuluyor...
python build_desktop.py
if errorlevel 1 goto :error

echo.
echo [OK] Build tamamlandi.
echo [INFO] Cikti klasoru: dist\Massscriber
exit /b 0

:error
echo.
echo [HATA] Desktop build sirasinda bir problem olustu.
exit /b 1
