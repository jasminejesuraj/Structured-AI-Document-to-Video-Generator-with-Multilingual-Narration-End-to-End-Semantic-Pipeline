"""Local web UI for the AI Document -> Video pipeline.

One screen:
  upload document -> Generate -> storyboard + images (local) -> video (Kaggle GPU)
  -> preview + download the final video.

Run:  python webui/app.py    then open http://localhost:7860
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure sibling modules are importable when run as `python webui/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

import pipeline
from kaggle_runner import LANGUAGE_MAP, credentials_present

MAX_LOG_LINES = 500


def _kaggle_status_md() -> str:
    if credentials_present():
        return "**Kaggle:** credentials detected — video stage is available."
    return (
        "**Kaggle:** no credentials found. Set `KAGGLE_USERNAME` + `KAGGLE_KEY` "
        "(or place `kaggle.json` in `~/.kaggle/`) to enable the video stage. "
        "You can still generate the storyboard + images."
    )


def generate(input_file, language, make_video):
    logs: list[str] = []

    def log(line: str) -> str:
        logs.append(line)
        return "\n".join(logs[-MAX_LOG_LINES:])

    gallery_state = None
    video_state = None

    if not input_file:
        yield log("Please upload a PDF or DOCX document first."), gallery_state, video_state
        return

    run_dir = pipeline.make_run_dir()
    yield log(f"Run directory: {run_dir}"), gallery_state, video_state

    # Stage 1: storyboard
    yield log("\n=== Stage 1/3: Generating storyboard (Ollama) ==="), gallery_state, video_state
    for line in pipeline.run_storyboard(input_file, run_dir):
        if line.startswith("__EXIT__"):
            rc = int(line[len("__EXIT__"):])
            if rc != 0:
                yield log(f"Storyboard stage failed (exit code {rc})."), gallery_state, video_state
                return
        else:
            yield log(line), gallery_state, video_state

    scenes, _ = pipeline.storyboard_summary(run_dir)
    yield log(f"Storyboard ready: {scenes} scene(s)."), gallery_state, video_state

    # Stage 2: images
    yield log("\n=== Stage 2/3: Generating images (SDXL Turbo) ==="), gallery_state, video_state
    for line in pipeline.run_images(run_dir):
        if line.startswith("__EXIT__"):
            rc = int(line[len("__EXIT__"):])
            if rc != 0:
                yield log(f"Image stage failed (exit code {rc})."), gallery_state, video_state
                return
        else:
            yield log(line), gallery_state, video_state

    scenes, images = pipeline.storyboard_summary(run_dir)
    gallery_state = images
    yield log(f"Images ready: {len(images)} image(s)."), gallery_state, video_state

    if not make_video:
        yield log("\nDone (video stage skipped)."), gallery_state, video_state
        return

    # Stage 3: video on Kaggle
    yield log("\n=== Stage 3/3: Rendering video on Kaggle (free GPU) ==="), gallery_state, video_state
    if not credentials_present():
        yield log(
            "Kaggle credentials not found — skipping video stage. "
            "Set KAGGLE_USERNAME + KAGGLE_KEY (or kaggle.json) and try again."
        ), gallery_state, video_state
        return

    try:
        from kaggle_runner import run_video

        for line in run_video(run_dir, language):
            if line.startswith("FINAL_VIDEO::"):
                video_state = line[len("FINAL_VIDEO::"):]
            else:
                yield log(line), gallery_state, video_state
    except Exception as exc:  # surface Kaggle errors in the UI
        yield log(f"Video stage error: {exc}"), gallery_state, video_state
        return

    yield log("\nAll done! Final video is ready below."), gallery_state, video_state


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Document → Video") as demo:
        gr.Markdown(
            "# AI Document → Video\n"
            "Upload a PDF/DOCX → generate a narrated video. "
            "Storyboard + images run locally; the video renders on Kaggle's free GPU."
        )
        status_md = gr.Markdown(_kaggle_status_md())

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Input document (PDF or DOCX)",
                    file_types=[".pdf", ".docx"],
                    type="filepath",
                )
                language = gr.Dropdown(
                    choices=list(LANGUAGE_MAP.keys()),
                    value="English",
                    label="Narration language",
                )
                make_video = gr.Checkbox(
                    value=True,
                    label="Render video on Kaggle (free GPU)",
                )
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                logs = gr.Textbox(
                    label="Progress",
                    lines=20,
                    max_lines=20,
                    autoscroll=True,
                )

        gallery = gr.Gallery(label="Storyboard images", columns=4, height=260)
        video_out = gr.Video(label="Final video")

        run_btn.click(
            fn=generate,
            inputs=[input_file, language, make_video],
            outputs=[logs, gallery, video_out],
        )
    return demo


if __name__ == "__main__":
    server_port = int(os.environ.get("AIV_PORT", "7860"))
    build_ui().queue().launch(server_name="0.0.0.0", server_port=server_port)
