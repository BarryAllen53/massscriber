from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr

from massscriber.transcriber import SUPPORTED_MODELS, TranscriptionEngine
from massscriber.types import TranscriptionSettings

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = APP_ROOT / "outputs"
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4", ".mkv")

logger = logging.getLogger(__name__)


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
            model = gr.Dropdown(
                choices=list(SUPPORTED_MODELS),
                value="large-v3",
                label="Model",
                info="En dogru secim large-v3, en hizli secim turbo.",
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
            run_button = gr.Button("Transcribe", variant="primary")
            clear_button = gr.Button("Temizle")

        result_table = gr.Dataframe(
            headers=[
                "Dosya",
                "Dil",
                "Sure (sn)",
                "Segment",
                "Model",
                "Cihaz",
                "Ciktilar",
            ],
            datatype=["str", "str", "number", "number", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="Sonuclar",
        )
        downloads = gr.Files(label="Olusan Dosyalar")
        logs = gr.Textbox(label="Log", lines=12, interactive=False)

        run_button.click(
            fn=run_batch,
            inputs=[
                files,
                local_paths,
                folder_path,
                scan_recursive,
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
                temperature,
                vad_min_silence_ms,
                cpu_threads,
                output_dir,
                initial_prompt,
                condition_on_previous_text,
            ],
            outputs=[result_table, preview, downloads, logs],
        )
        clear_button.click(
            fn=clear_ui,
            outputs=[files, local_paths, folder_path, scan_recursive, result_table, preview, downloads, logs],
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


def run_batch(
    files: list[str] | None,
    local_paths_text: str,
    folder_path: str,
    recursive_scan: bool,
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
    temperature: float,
    vad_min_silence_ms: int,
    cpu_threads: int,
    output_dir: str,
    initial_prompt: str,
    condition_on_previous_text: bool,
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
    settings = TranscriptionSettings(
        model=model,
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
        output_formats=tuple(output_formats),
    )

    engine = TranscriptionEngine()
    table_rows: list[list[object]] = []
    preview_chunks: list[str] = []
    generated_files: list[str] = []
    log_lines: list[str] = warnings + [
        f"[INFO] {len(resolved_files)} dosya siraya alindi. model={model}, gorev={task}"
    ]
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Massscriber local transcription app")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Gradio UI")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true")

    cli_parser = subparsers.add_parser("transcribe", help="Transcribe files from the CLI")
    cli_parser.add_argument("files", nargs="+", help="Audio or video files to transcribe")
    cli_parser.add_argument("--model", default="large-v3", choices=SUPPORTED_MODELS)
    cli_parser.add_argument("--language", default="auto")
    cli_parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    cli_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    cli_parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "int8_float16", "int8", "float32"],
    )
    cli_parser.add_argument("--beam-size", type=int, default=5)
    cli_parser.add_argument("--batch-size", type=int, default=8)
    cli_parser.add_argument("--temperature", type=float, default=0.0)
    cli_parser.add_argument("--no-vad", action="store_true")
    cli_parser.add_argument("--no-word-timestamps", action="store_true")
    cli_parser.add_argument("--use-context", action="store_true")
    cli_parser.add_argument("--vad-min-silence-ms", type=int, default=500)
    cli_parser.add_argument("--cpu-threads", type=int, default=0)
    cli_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    cli_parser.add_argument(
        "--formats",
        nargs="+",
        choices=["txt", "srt", "vtt", "json"],
        default=["txt", "srt", "json"],
    )
    cli_parser.add_argument("--initial-prompt", default="")

    return parser


def run_cli(args: argparse.Namespace) -> int:
    settings = TranscriptionSettings(
        model=args.model,
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
        output_formats=tuple(args.formats),
    )

    engine = TranscriptionEngine()
    for raw_file in args.files:
        result = engine.transcribe_file(raw_file, settings, args.output_dir)
        print(f"[OK] {Path(raw_file).name}")
        print(f"  Language: {result.language or 'unknown'}")
        print(f"  Duration: {round(result.duration or 0, 2)} sec")
        print(f"  Outputs : {', '.join(str(path) for path in result.output_files.values())}")

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

    parser.print_help()
    return 1
