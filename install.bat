@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
  echo [INFO] Sanal ortam olusturuluyor...
  python -m venv .venv
  if errorlevel 1 goto :error
)

call .venv\Scripts\activate
if errorlevel 1 goto :error

echo [INFO] Pip guncelleniyor...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [INFO] Gerekli paketler yukleniyor...
python -m pip install -e .
if errorlevel 1 goto :error

echo.
echo [OK] Kurulum tamamlandi.
echo [INFO] Ilk calistirmada model indirilecektir.
exit /b 0

:error
echo.
echo [HATA] Kurulum sirasinda bir problem olustu.
exit /b 1
