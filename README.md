# Massscriber

Massscriber, `faster-whisper` uzerine kurulu yerel bir Python ses transkripsiyon uygulamasidir. Ucretsizdir, sure siniri yoktur ve internet uzerinden API kullanmadan kendi bilgisayarinda calisir.

## Neden bu yapi?

- `large-v3`: en yuksek dogruluk icin
- `turbo`: `large-v3` tabanli, cok daha hizli, kalite kaybi genelde kucuk
- `tiny` / `base`: daha zayif sistemlerde hizli denemeler icin
- `faster-whisper`: orijinal Whisper'a gore daha hizli calisir ve Windows'ta pratik kullanimi daha kolaydir
- `Gradio` arayuzu: surukle-birak mantiginda kolay kullanim

Not: "Tamamen hatasiz" transkripsiyon bugun pratikte mumkun degil. Ama yerel ve ucretsiz tarafta en guclu seceneklerden biri bu yapi.

## Desteklenen dosyalar

Arayuzde su uzantilar dogrudan tanimli:

- `.mp3`
- `.wav`
- `.m4a`
- `.flac`
- `.ogg`
- `.aac`
- `.wma`
- `.mp4`
- `.mkv`

`faster-whisper`, altta medya cozumu icin kendi paketleriyle birlikte gelir; bu nedenle klasik `ffmpeg` kurulumu cogunlukla gerekmez.

## Kurulum

### 1. Otomatik kurulum

Windows'ta:

```bat
install.bat
```

### 2. Elle kurulum

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Uygulamayi baslatma

### Arayuz

```bat
start_ui.bat
```

veya:

```powershell
.venv\Scripts\activate
python app.py
```

Arayuz varsayilan olarak `http://127.0.0.1:7860` adresinde acilir.

### Komut satiri

```powershell
.venv\Scripts\activate
python app.py transcribe "C:\sesler\ornek.mp3" --model large-v3 --formats txt srt json
```

## Onerilen ayarlar

### En yuksek kalite

- Model: `large-v3`
- Beam Size: `5`
- VAD: acik
- Word timestamps: acik

### En hizli kullanim

- Model: `turbo`
- Device: `cuda` varsa GPU
- Compute type: `float16`
- Batch size: `8` veya `16`

## Cikti dosyalari

Varsayilan olarak `outputs` klasorune su dosyalar yazilir:

- `txt`
- `srt`
- `json`

Istersen arayuzden `vtt` de ekleyebilirsin.

## GPU notu

NVIDIA GPU kullanacaksan `faster-whisper` tarafinda CUDA kutuphaneleri gerekebilir. CPU ile de calisir, sadece daha yavas olur.

`faster-whisper` resmi deposuna gore:

- CPU icin `int8` kullanim oldukca verimlidir
- GPU icin `float16` veya `int8_float16` kullanilabilir

Eger CUDA tarafinda DLL hatasi alirsan resmi `faster-whisper` README'sindeki Windows kutuphane notlarini takip et.

## Ilk calistirma

Ilk transkripsiyonda model dosyasi otomatik indirilir. `large-v3` buyuk bir model oldugu icin ilk acilista biraz zaman alabilir.

## Kaynaklar

- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [openai/whisper](https://github.com/openai/whisper)
