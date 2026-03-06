from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import gradio as gr

from massscriber.diagnostics import detect_system_status, render_system_status
from massscriber.library import (
    build_preview,
    extract_transcript_ids,
    records_to_rows,
    search_transcripts,
    update_review_status,
)
from massscriber.providers import (
    PROVIDERS,
    PROVIDER_LABELS,
    get_provider_default_model,
    get_provider_env_keys,
    get_provider_models,
    normalize_provider_name,
    provider_supports_speaker_labels,
    provider_supports_translation,
    provider_uses_remote_api,
)
from massscriber.profiles import delete_profile, get_profile, list_profile_names, save_profile
from massscriber.transcriber import SUPPORTED_MODELS, TranscriptionEngine
from massscriber.types import TranscriptionSettings
from massscriber.watcher import build_watch_rows, watch_folder

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = APP_ROOT / "outputs"
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4", ".mkv")
REVIEW_STATUS_CHOICES = ["pending", "reviewed", "needs-edit", "approved"]
PROVIDER_CHOICES = [f"{name} - {PROVIDER_LABELS[name]}" for name in PROVIDERS]

logger = logging.getLogger(__name__)


def render_provider_help(provider_name: str) -> str:
    provider = normalize_provider_name(provider_name)
    env_keys = ", ".join(get_provider_env_keys(provider)) or "yok"
    translation = "evet" if provider_supports_translation(provider) else "hayir"
    speakers = "evet" if provider_supports_speaker_labels(provider) else "hayir"
    models = ", ".join(get_provider_models(provider))
    return "\n".join(
        [
            "### Provider Bilgisi",
            f"- Secili provider: `{provider}`",
            f"- API key env: `{env_keys}`",
            f"- Translation destegi: `{translation}`",
            f"- Speaker label destegi: `{speakers}`",
            f"- Modeller: `{models}`",
        ]
    )


def update_provider_ui(provider_name: str):
    provider = normalize_provider_name(provider_name)
    model_choices = get_provider_models(provider)
    default_model = get_provider_default_model(provider)
    task_choices = ["transcribe", "translate"] if provider_supports_translation(provider) else ["transcribe"]
    task_value = "transcribe" if "translate" not in task_choices else "transcribe"
    return (
        gr.update(choices=model_choices, value=default_model),
        gr.update(choices=task_choices, value=task_value),
        render_provider_help(provider),
        gr.update(value=False, interactive=provider_supports_speaker_labels(provider)),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Massscriber",
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"),
    ) as demo:
        gr.Markdown(
            """
            # Massscriber
            Yerelde calisan, ucretsiz, sinirsiz sureli ses transkripsiyon araci.

            `large-v3`: en yuksek dogruluk
            `turbo`: dogruluktan cok az odun verip cok daha hizli

            Not: Tek ve buyuk dosyalarda ilerleme cubugu artik asama asama guncellenir.
            """
        )

        with gr.Row():
            files = gr.Files(
                label="Ses Dosyalari",
                file_count="multiple",
                file_types=list(SUPPORTED_EXTENSIONS),
                type="filepath",
            )
            preview = gr.Textbox(
                label="Transkript Onizlemesi",
                lines=20,
                max_lines=30,
                interactive=False,
            )

        with gr.Accordion("Yerel Disk Modu", open=False):
            gr.Markdown(
                """
                Browser upload yerine tam dosya yolu ya da klasor vererek dogrudan diskten okuyabilirsin.
                Buyuk dosyalarda bu mod genelde daha stabildir.
                """
            )
            local_paths = gr.Textbox(
                label="Tam Dosya Yollari",
                lines=5,
                placeholder="Her satira bir tam yol yaz.\nC:\\Kayitlar\\toplanti-1.mp3\nD:\\Arsiv\\podcast.wav",
            )
            with gr.Row():
                folder_path = gr.Textbox(
                    label="Klasor Yolu",
                    placeholder="C:\\Kayitlar\\UzunDosyalar",
                )
                scan_recursive = gr.Checkbox(
                    value=True,
                    label="Alt Klasorleri de Tara",
                )

        with gr.Row():
            provider = gr.Dropdown(
                choices=list(PROVIDERS),
                value="local",
                label="Provider",
                info="local tamamen ucretsizdir. Diger secenekler resmi transcript API provider'laridir.",
            )
            model = gr.Dropdown(
                choices=list(SUPPORTED_MODELS),
                value="large-v3",
                label="Model",
                info="Provider degistiginde model listesi otomatik guncellenir.",
            )
            language = gr.Textbox(
                value="auto",
                label="Dil",
                placeholder="auto, tr, en, de, ar, ja ...",
            )
            task = gr.Dropdown(
                choices=["transcribe", "translate"],
                value="transcribe",
                label="Gorev",
            )

        with gr.Row():
            device = gr.Dropdown(
                choices=["auto", "cuda", "cpu"],
                value="auto",
                label="Cihaz",
            )
            compute_type = gr.Dropdown(
                choices=["auto", "float16", "int8_float16", "int8", "float32"],
                value="auto",
                label="Hesap Tipi",
            )
            output_formats = gr.CheckboxGroup(
                choices=["txt", "srt", "vtt", "json"],
                value=["txt", "srt", "json"],
                label="Cikti Formatlari",
            )

        with gr.Row():
            beam_size = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=5,
                label="Beam Size",
            )
            batch_size = gr.Slider(
                minimum=1,
                maximum=32,
                step=1,
                value=8,
                label="Batch Size",
            )
            vad_filter = gr.Checkbox(
                value=True,
                label="Sessiz Bolgeleri Temizle (VAD)",
            )
            word_timestamps = gr.Checkbox(
                value=True,
                label="Kelime Timestamps",
            )

        with gr.Accordion("Gelismis Ayarlar", open=False):
            with gr.Row():
                provider_api_key = gr.Textbox(
                    value="",
                    type="password",
                    label="Provider API Key",
                    placeholder="Bos birakirsan uygun ortam degiskeni kullanilir",
                )
                provider_base_url = gr.Textbox(
                    value="",
                    label="Provider Base URL (opsiyonel)",
                    placeholder="Ozel gateway ya da proxy kullaniyorsan gir",
                )
            with gr.Row():
                provider_timeout_seconds = gr.Slider(
                    minimum=30,
                    maximum=3600,
                    step=30,
                    value=900,
                    label="API Timeout (sn)",
                )
                provider_poll_interval = gr.Slider(
                    minimum=1,
                    maximum=30,
                    step=1,
                    value=3,
                    label="Polling Araligi (sn)",
                )
                provider_smart_format = gr.Checkbox(
                    value=True,
                    label="Provider Smart Format / Formatting",
                )
            with gr.Row():
                provider_speaker_labels = gr.Checkbox(
                    value=False,
                    label="Provider Speaker Labels",
                )
                provider_keep_raw_response = gr.Checkbox(
                    value=False,
                    label="Raw API Response'u JSON'a ekle",
                )
            provider_keywords = gr.Textbox(
                value="",
                lines=3,
                label="Provider Keywords / Word Boost",
                placeholder="Her satira bir keyword yaz. Destekleyen provider'larda boost edilir.",
            )
            provider_status = gr.Markdown(render_provider_help("local"))

            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.0,
                    label="Temperature",
                )
                vad_min_silence_ms = gr.Slider(
                    minimum=250,
                    maximum=2000,
                    step=50,
                    value=500,
                    label="Min Silence (ms)",
                )
                cpu_threads = gr.Slider(
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    label="CPU Threads (0 = otomatik)",
                )

            with gr.Row():
                output_dir = gr.Textbox(
                    value=str(DEFAULT_OUTPUT_DIR),
                    label="Cikti Klasoru",
                )
                initial_prompt = gr.Textbox(
                    value="",
                    label="Initial Prompt",
                    placeholder="Terimler, ozel isimler veya alan baglami",
                )
                condition_on_previous_text = gr.Checkbox(
                    value=False,
                    label="Onceki Metni Baglam Olarak Kullan",
                )

            with gr.Row():
                subtitle_max_chars = gr.Slider(
                    minimum=20,
                    maximum=84,
                    step=2,
                    value=42,
                    label="Altyazi Maks Karakter",
                )
                subtitle_max_duration = gr.Slider(
                    minimum=2.0,
                    maximum=10.0,
                    step=0.5,
                    value=6.0,
                    label="Altyazi Maks Sure (sn)",
                )
                subtitle_pause_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=0.6,
                    label="Duraklama Esigi (sn)",
                )
                subtitle_split_on_pause = gr.Checkbox(
                    value=True,
                    label="Duraklamada Altyaziyi Bol",
                )

            with gr.Row():
                enable_diarization = gr.Checkbox(
                    value=False,
                    label="Speaker Diarization (Deneysel)",
                )
                diarization_model = gr.Textbox(
                    value="pyannote/speaker-diarization-3.1",
                    label="Diarization Modeli",
                )
                diarization_token = gr.Textbox(
                    value="",
                    type="password",
                    label="HF Token (opsiyonel)",
                    placeholder="HUGGINGFACE_HUB_TOKEN yoksa buraya girebilirsin",
                )

            glossary_text = gr.Textbox(
                value="",
                lines=4,
                label="Glossary / Duzeltme Kurallari",
                placeholder="Her satira bir kural yaz.\nOpen AI => OpenAI\nChat GPT => ChatGPT",
            )
            with gr.Row():
                glossary_case_sensitive = gr.Checkbox(
                    value=False,
                    label="Glossary Case Sensitive",
                )
                glossary_whole_word = gr.Checkbox(
                    value=True,
                    label="Sadece Tam Kelime Eslesmesi",
                )

        system_status = gr.Markdown(render_system_status(detect_system_status(DEFAULT_OUTPUT_DIR)))
        refresh_status_button = gr.Button("Sistem Durumunu Yenile")

        with gr.Row():
            run_button = gr.Button("Transcribe", variant="primary")
            clear_button = gr.Button("Temizle")

        result_table = gr.Dataframe(
            headers=[
                "Dosya",
                "Provider",
                "Dil",
                "Sure (sn)",
                "Segment",
                "Model",
                "Cihaz",
                "Ciktilar",
            ],
            datatype=["str", "str", "str", "number", "number", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="Sonuclar",
        )
        downloads = gr.Files(label="Olusan Dosyalar")
        logs = gr.Textbox(label="Log", lines=12, interactive=False)

        with gr.Accordion("Klasor Izleme Paneli", open=False):
            gr.Markdown(
                """
                Yeni gelen medya dosyalarini belirli araliklarla tarayip otomatik transcribe eder.
                Uzun sureli izleme icin `cycles` degerini buyutabilir ya da CLI `watch` komutunu kullanabilirsin.
                """
            )
            with gr.Row():
                watch_folder_path = gr.Textbox(
                    label="Izlenecek Klasor",
                    placeholder="C:\\Kayitlar\\Gelenler",
                )
                watch_archive_dir = gr.Textbox(
                    label="Arsiv Klasoru (opsiyonel)",
                    placeholder="C:\\Kayitlar\\Islenenler",
                )
            with gr.Row():
                watch_recursive = gr.Checkbox(value=True, label="Alt Klasorleri de Izle")
                watch_poll_interval = gr.Slider(
                    minimum=1,
                    maximum=60,
                    step=1,
                    value=5,
                    label="Tarama Araligi (sn)",
                )
                watch_stable_seconds = gr.Slider(
                    minimum=0,
                    maximum=120,
                    step=1,
                    value=10,
                    label="Dosya Stabil Bekleme (sn)",
                )
                watch_cycles = gr.Slider(
                    minimum=1,
                    maximum=120,
                    step=1,
                    value=6,
                    label="UI Izleme Dongusu",
                )
            with gr.Row():
                watch_button = gr.Button("Klasoru Izle", variant="secondary")
                watch_refresh_button = gr.Button("Watch Durumunu Yenile")
            watch_table = gr.Dataframe(
                headers=["Dosya", "Kaynak", "Islenme Zamani", "Formatlar"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Watch Gecmisi",
            )
            watch_logs = gr.Textbox(label="Watch Log", lines=12, interactive=False)

        with gr.Accordion("Workflow Profilleri", open=False):
            gr.Markdown(
                """
                Watch, glossary ve temel transkripsiyon ayarlarini tek isim altinda kaydedebilirsin.
                """
            )
            with gr.Row():
                saved_profile = gr.Dropdown(
                    choices=list_profile_names(),
                    value=None,
                    label="Kayitli Profiller",
                )
                profile_name = gr.Textbox(
                    value="",
                    label="Profil Adi",
                    placeholder="ornek: Podcast GPU Preseti",
                )
            with gr.Row():
                save_profile_button = gr.Button("Profili Kaydet")
                load_profile_button = gr.Button("Profili Yukle")
                delete_profile_button = gr.Button("Profili Sil")
                refresh_profiles_button = gr.Button("Profilleri Yenile")
            profile_status = gr.Textbox(label="Profil Durumu", lines=3, interactive=False)

        with gr.Accordion("Transcript Library ve Batch Review", open=False):
            gr.Markdown(
                """
                `outputs` klasorundeki transcript'leri tara, ara ve toplu review durumlari ata.
                """
            )
            with gr.Row():
                library_query = gr.Textbox(
                    value="",
                    label="Search Query",
                    placeholder="isim, marka, cumle, dil, model ...",
                )
                library_status_filter = gr.Dropdown(
                    choices=["all", *REVIEW_STATUS_CHOICES],
                    value="all",
                    label="Review Filter",
                )
            with gr.Row():
                library_search_button = gr.Button("Transcript Library'yi Tara")
                library_refresh_button = gr.Button("Sonuclari Yenile")
            library_summary = gr.Textbox(label="Library Ozet", lines=2, interactive=False)
            library_table = gr.Dataframe(
                headers=["ID", "Provider", "Dil", "Model", "Status", "Reviewed", "Snippet"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Transcript Sonuclari",
            )
            library_preview = gr.Textbox(label="Ilk Eslesen Transcript", lines=14, interactive=False)
            with gr.Row():
                review_targets = gr.Textbox(
                    value="",
                    lines=4,
                    label="Review Target ID'leri",
                    placeholder="Her satira bir transcript ID yaz.\nBos birakip gorunen sonuclari uygula secenegini acabilirsin.",
                )
                review_note = gr.Textbox(
                    value="",
                    lines=4,
                    label="Review Notu",
                    placeholder="ornegin: isimler kontrol edildi, tekrar gecilecek",
                )
            with gr.Row():
                review_status = gr.Dropdown(
                    choices=REVIEW_STATUS_CHOICES,
                    value="reviewed",
                    label="Yeni Review Status",
                )
                review_apply_visible = gr.Checkbox(
                    value=False,
                    label="Gorunen Sonuclara Uygula",
                )
                apply_review_button = gr.Button("Review Status Uygula")
            review_status_log = gr.Textbox(label="Review Log", lines=3, interactive=False)

        run_button.click(
            fn=run_batch,
            inputs=[
                files,
                local_paths,
                folder_path,
                scan_recursive,
                provider,
                model,
                language,
                task,
                device,
                compute_type,
                output_formats,
                beam_size,
                batch_size,
                vad_filter,
                word_timestamps,
                provider_api_key,
                provider_base_url,
                provider_timeout_seconds,
                provider_poll_interval,
                provider_smart_format,
                provider_speaker_labels,
                provider_keywords,
                provider_keep_raw_response,
                temperature,
                vad_min_silence_ms,
                cpu_threads,
                output_dir,
                initial_prompt,
                condition_on_previous_text,
                subtitle_max_chars,
                subtitle_max_duration,
                subtitle_pause_threshold,
                subtitle_split_on_pause,
                enable_diarization,
                diarization_model,
                diarization_token,
                glossary_text,
                glossary_case_sensitive,
                glossary_whole_word,
            ],
            outputs=[result_table, preview, downloads, logs],
        )
        clear_button.click(
            fn=clear_ui,
            outputs=[files, local_paths, folder_path, scan_recursive, result_table, preview, downloads, logs],
        )
        refresh_status_button.click(
            fn=refresh_system_status,
            inputs=[output_dir],
            outputs=[system_status],
        )
        watch_button.click(
            fn=run_watch_panel,
            inputs=[
                watch_folder_path,
                watch_archive_dir,
                watch_recursive,
                watch_poll_interval,
                watch_stable_seconds,
                watch_cycles,
                provider,
                model,
                language,
                task,
                device,
                compute_type,
                output_formats,
                beam_size,
                batch_size,
                vad_filter,
                word_timestamps,
                provider_api_key,
                provider_base_url,
                provider_timeout_seconds,
                provider_poll_interval,
                provider_smart_format,
                provider_speaker_labels,
                provider_keywords,
                provider_keep_raw_response,
                temperature,
                vad_min_silence_ms,
                cpu_threads,
                output_dir,
                initial_prompt,
                condition_on_previous_text,
                subtitle_max_chars,
                subtitle_max_duration,
                subtitle_pause_threshold,
                subtitle_split_on_pause,
                enable_diarization,
                diarization_model,
                diarization_token,
                glossary_text,
                glossary_case_sensitive,
                glossary_whole_word,
            ],
            outputs=[watch_table, watch_logs],
        )
        watch_refresh_button.click(
            fn=refresh_watch_panel,
            inputs=[output_dir],
            outputs=[watch_table, watch_logs],
        )
        save_profile_button.click(
            fn=save_current_profile,
            inputs=[
                profile_name,
                saved_profile,
                provider,
                model,
                language,
                task,
                device,
                compute_type,
                output_formats,
                beam_size,
                batch_size,
                vad_filter,
                word_timestamps,
                provider_api_key,
                provider_base_url,
                provider_timeout_seconds,
                provider_poll_interval,
                provider_smart_format,
                provider_speaker_labels,
                provider_keywords,
                provider_keep_raw_response,
                temperature,
                vad_min_silence_ms,
                cpu_threads,
                output_dir,
                initial_prompt,
                condition_on_previous_text,
                subtitle_max_chars,
                subtitle_max_duration,
                subtitle_pause_threshold,
                subtitle_split_on_pause,
                enable_diarization,
                diarization_model,
                diarization_token,
                glossary_text,
                glossary_case_sensitive,
                glossary_whole_word,
                watch_folder_path,
                watch_archive_dir,
                watch_recursive,
                watch_poll_interval,
                watch_stable_seconds,
                watch_cycles,
            ],
            outputs=[saved_profile, profile_name, profile_status],
        )
        load_profile_button.click(
            fn=load_saved_profile,
            inputs=[saved_profile],
            outputs=[
                saved_profile,
                profile_name,
                provider,
                model,
                language,
                task,
                device,
                compute_type,
                output_formats,
                beam_size,
                batch_size,
                vad_filter,
                word_timestamps,
                provider_api_key,
                provider_base_url,
                provider_timeout_seconds,
                provider_poll_interval,
                provider_smart_format,
                provider_speaker_labels,
                provider_keywords,
                provider_keep_raw_response,
                temperature,
                vad_min_silence_ms,
                cpu_threads,
                output_dir,
                initial_prompt,
                condition_on_previous_text,
                subtitle_max_chars,
                subtitle_max_duration,
                subtitle_pause_threshold,
                subtitle_split_on_pause,
                enable_diarization,
                diarization_model,
                diarization_token,
                glossary_text,
                glossary_case_sensitive,
                glossary_whole_word,
                watch_folder_path,
                watch_archive_dir,
                watch_recursive,
                watch_poll_interval,
                watch_stable_seconds,
                watch_cycles,
                provider_status,
                profile_status,
            ],
        )
        delete_profile_button.click(
            fn=delete_saved_profile,
            inputs=[saved_profile],
            outputs=[saved_profile, profile_name, profile_status],
        )
        refresh_profiles_button.click(
            fn=refresh_profile_choices,
            inputs=[saved_profile],
            outputs=[saved_profile, profile_status],
        )
        library_search_button.click(
            fn=search_library_panel,
            inputs=[output_dir, library_query, library_status_filter],
            outputs=[library_table, library_summary, library_preview],
        )
        library_refresh_button.click(
            fn=search_library_panel,
            inputs=[output_dir, library_query, library_status_filter],
            outputs=[library_table, library_summary, library_preview],
        )
        apply_review_button.click(
            fn=apply_review_panel,
            inputs=[
                output_dir,
                library_query,
                library_status_filter,
                review_targets,
                review_status,
                review_note,
                review_apply_visible,
            ],
            outputs=[library_table, library_summary, library_preview, review_status_log],
        )

        provider.change(
            fn=update_provider_ui,
            inputs=[provider],
            outputs=[model, task, provider_status, provider_speaker_labels],
        )

        demo.queue(status_update_rate=1, default_concurrency_limit=1, max_size=8)

    return demo


def render_logs(log_lines: list[str], current_status: str | None = None) -> str:
    visible_lines = list(log_lines[-20:])
    if current_status:
        visible_lines.extend(["", current_status])
    return "\n".join(visible_lines)


def strip_wrapping_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned


def is_supported_media_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_input_files(
    uploaded_files: list[str] | None,
    local_paths_text: str,
    folder_path: str,
    recursive_scan: bool,
) -> tuple[list[Path], list[str]]:
    sources: list[Path] = []
    warnings: list[str] = []
    seen: set[Path] = set()

    def add_candidate(path: Path, *, origin: str) -> None:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        if not resolved.exists():
            warnings.append(f"[UYARI] {origin}: bulunamadi -> {resolved}")
            return
        if not is_supported_media_file(resolved):
            warnings.append(f"[UYARI] {origin}: desteklenmeyen uzanti -> {resolved.name}")
            return
        seen.add(resolved)
        sources.append(resolved)

    for raw_path in uploaded_files or []:
        add_candidate(Path(str(raw_path)), origin="upload")

    for line in local_paths_text.splitlines():
        cleaned = strip_wrapping_quotes(line)
        if not cleaned:
            continue
        candidate = Path(cleaned).expanduser()
        if candidate.is_dir():
            warnings.append(f"[UYARI] manual path klasor cikti, klasor alani kullan -> {candidate}")
            continue
        add_candidate(candidate, origin="manual path")

    cleaned_folder = strip_wrapping_quotes(folder_path)
    if cleaned_folder:
        folder = Path(cleaned_folder).expanduser().resolve()
        if not folder.exists():
            warnings.append(f"[UYARI] klasor bulunamadi -> {folder}")
        elif not folder.is_dir():
            warnings.append(f"[UYARI] klasor alani dosya degil, klasor bekleniyor -> {folder}")
        else:
            iterator = folder.rglob("*") if recursive_scan else folder.glob("*")
            folder_matches = 0
            for candidate in iterator:
                if is_supported_media_file(candidate):
                    add_candidate(candidate, origin="folder scan")
                    folder_matches += 1
            if folder_matches == 0:
                warnings.append(f"[UYARI] klasorde desteklenen medya dosyasi bulunamadi -> {folder}")

    return sources, warnings


def clear_ui():
    return None, "", "", True, None, "", None, ""


def serialize_profile_payload(
    provider: str,
    model: str,
    language: str,
    task: str,
    device: str,
    compute_type: str,
    output_formats: list[str],
    beam_size: int,
    batch_size: int,
    vad_filter: bool,
    word_timestamps: bool,
    provider_api_key: str,
    provider_base_url: str,
    provider_timeout_seconds: float,
    provider_poll_interval: float,
    provider_smart_format: bool,
    provider_speaker_labels: bool,
    provider_keywords: str,
    provider_keep_raw_response: bool,
    temperature: float,
    vad_min_silence_ms: int,
    cpu_threads: int,
    output_dir: str,
    initial_prompt: str,
    condition_on_previous_text: bool,
    subtitle_max_chars: int,
    subtitle_max_duration: float,
    subtitle_pause_threshold: float,
    subtitle_split_on_pause: bool,
    enable_diarization: bool,
    diarization_model: str,
    diarization_token: str,
    glossary_text: str,
    glossary_case_sensitive: bool,
    glossary_whole_word: bool,
    watch_folder_path: str,
    watch_archive_dir: str,
    watch_recursive: bool,
    watch_poll_interval: float,
    watch_stable_seconds: float,
    watch_cycles: int,
) -> dict[str, object]:
    return {
        "provider": provider,
        "model": model,
        "language": language,
        "task": task,
        "device": device,
        "compute_type": compute_type,
        "output_formats": list(output_formats or []),
        "beam_size": int(beam_size),
        "batch_size": int(batch_size),
        "vad_filter": bool(vad_filter),
        "word_timestamps": bool(word_timestamps),
        "provider_api_key": provider_api_key,
        "provider_base_url": provider_base_url,
        "provider_timeout_seconds": float(provider_timeout_seconds),
        "provider_poll_interval": float(provider_poll_interval),
        "provider_smart_format": bool(provider_smart_format),
        "provider_speaker_labels": bool(provider_speaker_labels),
        "provider_keywords": provider_keywords,
        "provider_keep_raw_response": bool(provider_keep_raw_response),
        "temperature": float(temperature),
        "vad_min_silence_ms": int(vad_min_silence_ms),
        "cpu_threads": int(cpu_threads),
        "output_dir": output_dir,
        "initial_prompt": initial_prompt,
        "condition_on_previous_text": bool(condition_on_previous_text),
        "subtitle_max_chars": int(subtitle_max_chars),
        "subtitle_max_duration": float(subtitle_max_duration),
        "subtitle_pause_threshold": float(subtitle_pause_threshold),
        "subtitle_split_on_pause": bool(subtitle_split_on_pause),
        "enable_diarization": bool(enable_diarization),
        "diarization_model": diarization_model,
        "diarization_token": diarization_token,
        "glossary_text": glossary_text,
        "glossary_case_sensitive": bool(glossary_case_sensitive),
        "glossary_whole_word": bool(glossary_whole_word),
        "watch_folder_path": watch_folder_path,
        "watch_archive_dir": watch_archive_dir,
        "watch_recursive": bool(watch_recursive),
        "watch_poll_interval": float(watch_poll_interval),
        "watch_stable_seconds": float(watch_stable_seconds),
        "watch_cycles": int(watch_cycles),
    }


def save_current_profile(
    profile_name: str,
    selected_profile: str | None,
    provider: str,
    model: str,
    language: str,
    task: str,
    device: str,
    compute_type: str,
    output_formats: list[str],
    beam_size: int,
    batch_size: int,
    vad_filter: bool,
    word_timestamps: bool,
    provider_api_key: str,
    provider_base_url: str,
    provider_timeout_seconds: float,
    provider_poll_interval: float,
    provider_smart_format: bool,
    provider_speaker_labels: bool,
    provider_keywords: str,
    provider_keep_raw_response: bool,
    temperature: float,
    vad_min_silence_ms: int,
    cpu_threads: int,
    output_dir: str,
    initial_prompt: str,
    condition_on_previous_text: bool,
    subtitle_max_chars: int,
    subtitle_max_duration: float,
    subtitle_pause_threshold: float,
    subtitle_split_on_pause: bool,
    enable_diarization: bool,
    diarization_model: str,
    diarization_token: str,
    glossary_text: str,
    glossary_case_sensitive: bool,
    glossary_whole_word: bool,
    watch_folder_path: str,
    watch_archive_dir: str,
    watch_recursive: bool,
    watch_poll_interval: float,
    watch_stable_seconds: float,
    watch_cycles: int,
):
    effective_name = profile_name.strip() or (selected_profile or "").strip()
    if not effective_name:
        raise gr.Error("Profil kaydetmek icin bir ad girmelisin.")

    payload = serialize_profile_payload(
        provider,
        model,
        language,
        task,
        device,
        compute_type,
        output_formats,
        beam_size,
        batch_size,
        vad_filter,
        word_timestamps,
        provider_api_key,
        provider_base_url,
        provider_timeout_seconds,
        provider_poll_interval,
        provider_smart_format,
        provider_speaker_labels,
        provider_keywords,
        provider_keep_raw_response,
        temperature,
        vad_min_silence_ms,
        cpu_threads,
        output_dir,
        initial_prompt,
        condition_on_previous_text,
        subtitle_max_chars,
        subtitle_max_duration,
        subtitle_pause_threshold,
        subtitle_split_on_pause,
        enable_diarization,
        diarization_model,
        diarization_token,
        glossary_text,
        glossary_case_sensitive,
        glossary_whole_word,
        watch_folder_path,
        watch_archive_dir,
        watch_recursive,
        watch_poll_interval,
        watch_stable_seconds,
        watch_cycles,
    )
    profiles = save_profile(effective_name, payload)
    return gr.update(choices=sorted(profiles), value=effective_name), effective_name, f"[OK] Profil kaydedildi: {effective_name}"


def load_saved_profile(profile_name: str | None):
    effective_name = (profile_name or "").strip()
    if not effective_name:
        raise gr.Error("Yuklemek icin once kayitli bir profil sec.")

    payload = get_profile(effective_name)
    if not payload:
        raise gr.Error(f"Profil bulunamadi: {effective_name}")

    def value(key: str, default: object) -> object:
        return payload.get(key, default)

    return (
        gr.update(choices=list_profile_names(), value=effective_name),
        effective_name,
        value("provider", "local"),
        value("model", "large-v3"),
        value("language", "auto"),
        value("task", "transcribe"),
        value("device", "auto"),
        value("compute_type", "auto"),
        value("output_formats", ["txt", "srt", "json"]),
        value("beam_size", 5),
        value("batch_size", 8),
        value("vad_filter", True),
        value("word_timestamps", True),
        value("provider_api_key", ""),
        value("provider_base_url", ""),
        value("provider_timeout_seconds", 900),
        value("provider_poll_interval", 3),
        value("provider_smart_format", True),
        value("provider_speaker_labels", False),
        value("provider_keywords", ""),
        value("provider_keep_raw_response", False),
        value("temperature", 0.0),
        value("vad_min_silence_ms", 500),
        value("cpu_threads", 0),
        value("output_dir", str(DEFAULT_OUTPUT_DIR)),
        value("initial_prompt", ""),
        value("condition_on_previous_text", False),
        value("subtitle_max_chars", 42),
        value("subtitle_max_duration", 6.0),
        value("subtitle_pause_threshold", 0.6),
        value("subtitle_split_on_pause", True),
        value("enable_diarization", False),
        value("diarization_model", "pyannote/speaker-diarization-3.1"),
        value("diarization_token", ""),
        value("glossary_text", ""),
        value("glossary_case_sensitive", False),
        value("glossary_whole_word", True),
        value("watch_folder_path", ""),
        value("watch_archive_dir", ""),
        value("watch_recursive", True),
        value("watch_poll_interval", 5.0),
        value("watch_stable_seconds", 10.0),
        value("watch_cycles", 6),
        render_provider_help(str(value("provider", "local"))),
        f"[OK] Profil yuklendi: {effective_name}",
    )


def delete_saved_profile(profile_name: str | None):
    effective_name = (profile_name or "").strip()
    if not effective_name:
        raise gr.Error("Silmek icin once bir profil sec.")
    profiles = delete_profile(effective_name)
    return gr.update(choices=sorted(profiles), value=None), "", f"[OK] Profil silindi: {effective_name}"


def refresh_profile_choices(selected_profile: str | None):
    names = list_profile_names()
    current = selected_profile if selected_profile in names else None
    return gr.update(choices=names, value=current), f"[INFO] {len(names)} profil bulundu."


def refresh_system_status(output_dir: str) -> str:
    target = output_dir.strip() or str(DEFAULT_OUTPUT_DIR)
    return render_system_status(detect_system_status(target))


def refresh_watch_panel(output_dir: str) -> tuple[list[list[str]], str]:
    target = output_dir.strip() or str(DEFAULT_OUTPUT_DIR)
    rows = build_watch_rows(target)
    if not rows:
        return [], "[INFO] Watch gecmisi henuz bos."
    return rows, f"[INFO] {len(rows)} kayitli watch sonucu bulundu."


def search_library_panel(output_dir: str, query: str, status_filter: str):
    target = output_dir.strip() or str(DEFAULT_OUTPUT_DIR)
    records, summary = search_transcripts(target, query=query, status_filter=status_filter)
    return records_to_rows(records), summary, build_preview(records)


def apply_review_panel(
    output_dir: str,
    query: str,
    status_filter: str,
    review_targets_text: str,
    review_status: str,
    review_note: str,
    review_apply_visible: bool,
):
    target = output_dir.strip() or str(DEFAULT_OUTPUT_DIR)
    transcript_ids = extract_transcript_ids(review_targets_text)
    if not transcript_ids and review_apply_visible:
        records, _ = search_transcripts(target, query=query, status_filter=status_filter)
        transcript_ids = [record.transcript_id for record in records]

    if not transcript_ids:
        raise gr.Error("Review icin en az bir transcript ID gir ya da gorunen sonuclari uygula secenegini ac.")

    updated_count = update_review_status(
        target,
        transcript_ids,
        status=review_status,
        note=review_note,
    )
    records, summary = search_transcripts(target, query=query, status_filter=status_filter)
    message = f"[OK] {updated_count} transcript guncellendi. Yeni status: {review_status}"
    return records_to_rows(records), summary, build_preview(records), message


def run_batch(
    files: list[str] | None,
    local_paths_text: str,
    folder_path: str,
    recursive_scan: bool,
    provider: str,
    model: str,
    language: str,
    task: str,
    device: str,
    compute_type: str,
    output_formats: list[str],
    beam_size: int,
    batch_size: int,
    vad_filter: bool,
    word_timestamps: bool,
    provider_api_key: str,
    provider_base_url: str,
    provider_timeout_seconds: float,
    provider_poll_interval: float,
    provider_smart_format: bool,
    provider_speaker_labels: bool,
    provider_keywords: str,
    provider_keep_raw_response: bool,
    temperature: float,
    vad_min_silence_ms: int,
    cpu_threads: int,
    output_dir: str,
    initial_prompt: str,
    condition_on_previous_text: bool,
    subtitle_max_chars: int,
    subtitle_max_duration: float,
    subtitle_pause_threshold: float,
    subtitle_split_on_pause: bool,
    enable_diarization: bool,
    diarization_model: str,
    diarization_token: str,
    glossary_text: str,
    glossary_case_sensitive: bool,
    glossary_whole_word: bool,
    progress: gr.Progress = gr.Progress(),
):
    if not output_formats:
        raise gr.Error("En az bir cikti formati secmelisin.")

    resolved_files, warnings = collect_input_files(files, local_paths_text, folder_path, recursive_scan)
    if not resolved_files:
        raise gr.Error(
            "En az bir ses dosyasi secmelisin ya da yerel disk moduna tam dosya yolu/klasor girmelisin."
        )

    normalized_language = None if language.strip().lower() == "auto" else language.strip()
    normalized_provider = normalize_provider_name(provider)
    resolved_model = model.strip() or get_provider_default_model(normalized_provider)
    settings = TranscriptionSettings(
        provider=normalized_provider,
        model=resolved_model,
        provider_model=resolved_model,
        provider_api_key=provider_api_key.strip() or None,
        provider_base_url=provider_base_url.strip() or None,
        provider_timeout_seconds=float(provider_timeout_seconds),
        provider_poll_interval=float(provider_poll_interval),
        provider_smart_format=bool(provider_smart_format),
        provider_speaker_labels=bool(provider_speaker_labels),
        provider_keywords=provider_keywords,
        provider_keep_raw_response=bool(provider_keep_raw_response),
        language=normalized_language,
        task=task,
        device=device,
        compute_type=compute_type,
        beam_size=int(beam_size),
        batch_size=int(batch_size),
        temperature=float(temperature),
        vad_filter=bool(vad_filter),
        vad_min_silence_ms=int(vad_min_silence_ms),
        word_timestamps=bool(word_timestamps),
        condition_on_previous_text=bool(condition_on_previous_text),
        initial_prompt=initial_prompt,
        cpu_threads=int(cpu_threads) if cpu_threads else None,
        subtitle_max_chars=int(subtitle_max_chars),
        subtitle_max_duration=float(subtitle_max_duration),
        subtitle_split_on_pause=bool(subtitle_split_on_pause),
        subtitle_pause_threshold=float(subtitle_pause_threshold),
        enable_diarization=bool(enable_diarization),
        diarization_model=diarization_model.strip() or "pyannote/speaker-diarization-3.1",
        diarization_token=diarization_token.strip() or None,
        glossary_text=glossary_text,
        glossary_case_sensitive=bool(glossary_case_sensitive),
        glossary_whole_word=bool(glossary_whole_word),
        output_formats=tuple(output_formats),
    )

    engine = TranscriptionEngine()
    table_rows: list[list[object]] = []
    preview_chunks: list[str] = []
    generated_files: list[str] = []
    log_lines: list[str] = warnings + [
        f"[INFO] {len(resolved_files)} dosya siraya alindi. provider={normalized_provider}, model={resolved_model}, gorev={task}"
    ]
    sticky_messages: set[str] = set(log_lines)
    current_status = "[CALISIYOR] Kuyruk hazirlaniyor"

    yield table_rows, "", generated_files, render_logs(log_lines, current_status)

    total = len(resolved_files)
    for index, source in enumerate(resolved_files, start=1):
        current_status = f"[CALISIYOR] {source.name}: siraya alindi"
        progress((index - 1) / total, desc=f"Sirada: {source.name}")
        yield (
            table_rows,
            "\n\n".join(preview_chunks),
            generated_files,
            render_logs(log_lines, current_status),
        )

        try:
            result = None
            for file_progress, message, maybe_result in engine.stream_file(source, settings, output_dir):
                if message.startswith("[UYARI]") and message not in sticky_messages:
                    log_lines.append(message)
                    sticky_messages.add(message)
                current_status = f"[CALISIYOR] {message}"
                overall_progress = ((index - 1) + min(max(file_progress, 0.0), 1.0)) / total
                progress(overall_progress, desc=message)
                if maybe_result is not None:
                    result = maybe_result
                yield (
                    table_rows,
                    "\n\n".join(preview_chunks),
                    generated_files,
                    render_logs(log_lines, current_status),
                )
        except Exception as exc:
            logger.exception("Transcription failed for %s", source)
            log_lines.append(f"[HATA] {source.name}: {exc}")
            current_status = None
            yield (
                table_rows,
                "\n\n".join(preview_chunks),
                generated_files,
                render_logs(log_lines, current_status),
            )
            continue

        if result is None:
            log_lines.append(f"[HATA] {source.name}: sonuc olusturulamadi.")
            current_status = None
            yield (
                table_rows,
                "\n\n".join(preview_chunks),
                generated_files,
                render_logs(log_lines, current_status),
            )
            continue

        preview_chunks.append(f"## {source.name}\n{result.text.strip()}")
        generated_files.extend(str(path) for path in result.output_files.values())
        table_rows.append(
            [
                source.name,
                result.provider,
                result.language or "unknown",
                round(result.duration or 0, 2),
                len(result.segments),
                result.model,
                result.device,
                ", ".join(sorted(result.output_files.keys())),
            ]
        )
        log_lines.append(
            f"[OK] {source.name} -> dil={result.language or 'unknown'}, "
            f"segment={len(result.segments)}, cikti={len(result.output_files)}"
        )
        current_status = None
        progress(index / total, desc=f"Tamamlandi: {source.name}")
        yield (
            table_rows,
            "\n\n".join(preview_chunks),
            generated_files,
            render_logs(log_lines, current_status),
        )

    if not table_rows:
        raise gr.Error("Hicbir dosya basariyla transcribe edilemedi. Log kismini kontrol et.")

    log_lines.append("[INFO] Tum isler tamamlandi.")
    progress(1.0, desc="Tum isler tamamlandi")
    yield table_rows, "\n\n".join(preview_chunks), generated_files, render_logs(log_lines)


def run_watch_panel(
    watch_folder_path: str,
    watch_archive_dir: str,
    watch_recursive: bool,
    watch_poll_interval: float,
    watch_stable_seconds: float,
    watch_cycles: int,
    provider: str,
    model: str,
    language: str,
    task: str,
    device: str,
    compute_type: str,
    output_formats: list[str],
    beam_size: int,
    batch_size: int,
    vad_filter: bool,
    word_timestamps: bool,
    provider_api_key: str,
    provider_base_url: str,
    provider_timeout_seconds: float,
    provider_poll_interval: float,
    provider_smart_format: bool,
    provider_speaker_labels: bool,
    provider_keywords: str,
    provider_keep_raw_response: bool,
    temperature: float,
    vad_min_silence_ms: int,
    cpu_threads: int,
    output_dir: str,
    initial_prompt: str,
    condition_on_previous_text: bool,
    subtitle_max_chars: int,
    subtitle_max_duration: float,
    subtitle_pause_threshold: float,
    subtitle_split_on_pause: bool,
    enable_diarization: bool,
    diarization_model: str,
    diarization_token: str,
    glossary_text: str,
    glossary_case_sensitive: bool,
    glossary_whole_word: bool,
):
    if not watch_folder_path.strip():
        raise gr.Error("Izlenecek klasor yolunu girmelisin.")

    normalized_provider = normalize_provider_name(provider)
    resolved_model = model.strip() or get_provider_default_model(normalized_provider)
    settings = TranscriptionSettings(
        provider=normalized_provider,
        model=resolved_model,
        provider_model=resolved_model,
        provider_api_key=provider_api_key.strip() or None,
        provider_base_url=provider_base_url.strip() or None,
        provider_timeout_seconds=float(provider_timeout_seconds),
        provider_poll_interval=float(provider_poll_interval),
        provider_smart_format=bool(provider_smart_format),
        provider_speaker_labels=bool(provider_speaker_labels),
        provider_keywords=provider_keywords,
        provider_keep_raw_response=bool(provider_keep_raw_response),
        language=None if language.strip().lower() == "auto" else language.strip(),
        task=task,
        device=device,
        compute_type=compute_type,
        beam_size=int(beam_size),
        batch_size=int(batch_size),
        temperature=float(temperature),
        vad_filter=bool(vad_filter),
        vad_min_silence_ms=int(vad_min_silence_ms),
        word_timestamps=bool(word_timestamps),
        condition_on_previous_text=bool(condition_on_previous_text),
        initial_prompt=initial_prompt,
        cpu_threads=int(cpu_threads) if cpu_threads else None,
        subtitle_max_chars=int(subtitle_max_chars),
        subtitle_max_duration=float(subtitle_max_duration),
        subtitle_split_on_pause=bool(subtitle_split_on_pause),
        subtitle_pause_threshold=float(subtitle_pause_threshold),
        enable_diarization=bool(enable_diarization),
        diarization_model=diarization_model.strip() or "pyannote/speaker-diarization-3.1",
        diarization_token=diarization_token.strip() or None,
        glossary_text=glossary_text,
        glossary_case_sensitive=bool(glossary_case_sensitive),
        glossary_whole_word=bool(glossary_whole_word),
        output_formats=tuple(output_formats),
    )

    log_lines = [
        f"[INFO] Watch baslatildi: {watch_folder_path}",
        f"[INFO] Provider: {normalized_provider}, model: {resolved_model}",
        f"[INFO] UI dongu sayisi: {int(watch_cycles)}",
    ]
    target_output_dir = output_dir.strip() or str(DEFAULT_OUTPUT_DIR)
    rows = build_watch_rows(target_output_dir)
    yield rows, "\n".join(log_lines)

    for cycle_index in range(int(watch_cycles)):
        cycle_message = f"[INFO] Dongu {cycle_index + 1}/{int(watch_cycles)} calisiyor..."
        log_lines.append(cycle_message)
        yield build_watch_rows(target_output_dir), "\n".join(log_lines[-20:])
        for event in watch_folder(
            watch_folder_path,
            settings,
            target_output_dir,
            recursive=bool(watch_recursive),
            poll_interval=float(watch_poll_interval),
            stable_seconds=float(watch_stable_seconds),
            once=True,
            archive_dir=watch_archive_dir.strip() or None,
        ):
            log_lines.append(event)
            rows = build_watch_rows(target_output_dir)
            yield rows, "\n".join(log_lines[-20:])
        if cycle_index < int(watch_cycles) - 1:
            time.sleep(max(1.0, float(watch_poll_interval)))

    log_lines.append("[INFO] UI watch dongusu tamamlandi.")
    yield build_watch_rows(target_output_dir), "\n".join(log_lines[-20:])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Massscriber local transcription app")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Gradio UI")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true")

    cli_parser = subparsers.add_parser("transcribe", help="Transcribe files from the CLI")
    cli_parser.add_argument("files", nargs="+", help="Audio or video files to transcribe")
    add_common_transcription_arguments(cli_parser)

    watch_parser = subparsers.add_parser("watch", help="Watch a folder and auto-transcribe new media files")
    watch_parser.add_argument("folder", help="Folder to watch for media files")
    watch_parser.add_argument("--poll-interval", type=float, default=5.0)
    watch_parser.add_argument("--stable-seconds", type=float, default=10.0)
    watch_parser.add_argument("--once", action="store_true")
    watch_parser.add_argument("--archive-dir", default="")
    watch_parser.add_argument("--no-recursive", action="store_true")
    add_common_transcription_arguments(watch_parser)

    doctor_parser = subparsers.add_parser("doctor", help="Show system health and runtime readiness")
    doctor_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    return parser


def add_common_transcription_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="local", choices=list(PROVIDERS))
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Provider modeli. local icin desteklenen modeller: "
            + ", ".join(SUPPORTED_MODELS)
        ),
    )
    parser.add_argument("--language", default="auto")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "int8_float16", "int8", "float32"],
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--no-word-timestamps", action="store_true")
    parser.add_argument("--use-context", action="store_true")
    parser.add_argument("--vad-min-silence-ms", type=int, default=500)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["txt", "srt", "vtt", "json"],
        default=["txt", "srt", "json"],
    )
    parser.add_argument("--initial-prompt", default="")
    parser.add_argument("--subtitle-max-chars", type=int, default=42)
    parser.add_argument("--subtitle-max-duration", type=float, default=6.0)
    parser.add_argument("--subtitle-pause-threshold", type=float, default=0.6)
    parser.add_argument("--no-subtitle-pause-split", action="store_true")
    parser.add_argument("--enable-diarization", action="store_true")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--diarization-token", default="")
    parser.add_argument("--glossary-file", default="")
    parser.add_argument("--glossary-text", default="")
    parser.add_argument("--glossary-case-sensitive", action="store_true")
    parser.add_argument("--glossary-phrase-mode", action="store_true")
    parser.add_argument("--provider-api-key", default="")
    parser.add_argument("--provider-base-url", default="")
    parser.add_argument("--provider-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--provider-poll-interval", type=float, default=3.0)
    parser.add_argument("--provider-no-smart-format", action="store_true")
    parser.add_argument("--provider-speaker-labels", action="store_true")
    parser.add_argument("--provider-keywords", default="")
    parser.add_argument("--provider-keep-raw-response", action="store_true")


def build_settings_from_args(args: argparse.Namespace) -> TranscriptionSettings:
    glossary_text = getattr(args, "glossary_text", "") or ""
    glossary_file = getattr(args, "glossary_file", "") or ""
    if glossary_file:
        glossary_text = Path(glossary_file).expanduser().read_text(encoding="utf-8")
    normalized_provider = normalize_provider_name(getattr(args, "provider", "local"))
    resolved_model = (getattr(args, "model", "") or "").strip() or get_provider_default_model(normalized_provider)
    return TranscriptionSettings(
        provider=normalized_provider,
        model=resolved_model,
        provider_model=resolved_model,
        provider_api_key=getattr(args, "provider_api_key", "") or None,
        provider_base_url=getattr(args, "provider_base_url", "") or None,
        provider_timeout_seconds=float(getattr(args, "provider_timeout_seconds", 900.0)),
        provider_poll_interval=float(getattr(args, "provider_poll_interval", 3.0)),
        provider_smart_format=not bool(getattr(args, "provider_no_smart_format", False)),
        provider_speaker_labels=bool(getattr(args, "provider_speaker_labels", False)),
        provider_keywords=getattr(args, "provider_keywords", "") or "",
        provider_keep_raw_response=bool(getattr(args, "provider_keep_raw_response", False)),
        language=None if args.language.lower() == "auto" else args.language,
        task=args.task,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        temperature=args.temperature,
        vad_filter=not args.no_vad,
        vad_min_silence_ms=args.vad_min_silence_ms,
        word_timestamps=not args.no_word_timestamps,
        condition_on_previous_text=args.use_context,
        initial_prompt=args.initial_prompt,
        cpu_threads=args.cpu_threads or None,
        subtitle_max_chars=args.subtitle_max_chars,
        subtitle_max_duration=args.subtitle_max_duration,
        subtitle_split_on_pause=not args.no_subtitle_pause_split,
        subtitle_pause_threshold=args.subtitle_pause_threshold,
        enable_diarization=args.enable_diarization,
        diarization_model=args.diarization_model,
        diarization_token=args.diarization_token or None,
        glossary_text=glossary_text,
        glossary_case_sensitive=args.glossary_case_sensitive,
        glossary_whole_word=not args.glossary_phrase_mode,
        output_formats=tuple(args.formats),
    )


def run_cli(args: argparse.Namespace) -> int:
    settings = build_settings_from_args(args)

    engine = TranscriptionEngine()
    for raw_file in args.files:
        result = None
        for _, message, maybe_result in engine.stream_file(raw_file, settings, args.output_dir):
            if maybe_result is not None:
                result = maybe_result
            elif message.startswith("[UYARI]") or message.startswith("[INFO]"):
                print(message)

        if result is None:
            raise RuntimeError(f"Transcription finished without a result: {raw_file}")
        print(f"[OK] {Path(raw_file).name}")
        print(f"  Provider: {result.provider}")
        print(f"  Language: {result.language or 'unknown'}")
        print(f"  Duration: {round(result.duration or 0, 2)} sec")
        print(f"  Outputs : {', '.join(str(path) for path in result.output_files.values())}")

    return 0


def run_watch(args: argparse.Namespace) -> int:
    settings = build_settings_from_args(args)
    events = watch_folder(
        args.folder,
        settings,
        args.output_dir,
        recursive=not args.no_recursive,
        poll_interval=args.poll_interval,
        stable_seconds=args.stable_seconds,
        once=args.once,
        archive_dir=args.archive_dir or None,
    )
    for event in events:
        print(event)
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    status = detect_system_status(args.output_dir)
    print(render_system_status(status))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command in (None, "ui"):
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 7860)
        share = getattr(args, "share", False)
        demo = build_demo()
        demo.launch(
            server_name=host,
            server_port=port,
            share=share,
            inbrowser=True,
            show_error=True,
            max_file_size="2gb",
            pwa=True,
        )
        return 0

    if args.command == "transcribe":
        return run_cli(args)

    if args.command == "watch":
        return run_watch(args)

    if args.command == "doctor":
        return run_doctor(args)

    parser.print_help()
    return 1
