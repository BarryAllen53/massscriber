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

where nvidia-smi >nul 2>nul
if errorlevel 1 goto :done

echo [INFO] NVIDIA GPU algilandi. CUDA runtime paketleri kontrol ediliyor...
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-nvjitlink-cu12
if errorlevel 1 (
  echo [UYARI] NVIDIA runtime paketleri otomatik yuklenemedi.
  echo [UYARI] Uygulama yine calisir; gerekirse CPU moduna otomatik gecer.
  goto :done
)

echo [OK] NVIDIA runtime paketleri hazir.

:done
echo.
echo [OK] Kurulum tamamlandi.
echo [INFO] Ilk calistirmada model indirilecektir.
exit /b 0

:error
echo.
echo [HATA] Kurulum sirasinda bir problem olustu.
exit /b 1
