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

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
#hero {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 48%, #2563eb 100%);
  border-radius: 18px; padding: 26px 30px; margin-bottom: 6px;
  box-shadow: 0 10px 30px rgba(79,70,229,.25);
}
#hero h1 {color:#fff; margin:0; font-size:1.85rem; font-weight:700; letter-spacing:-.01em;}
#hero p {color:#e9e8ff; margin:.45rem 0 0; font-size:1rem;}
.status-pill {
  display:inline-flex; align-items:center; gap:.5rem; padding:.5rem .9rem;
  border-radius:999px; font-size:.92rem; font-weight:500; margin:2px 0 4px;
}
.status-ok   {background:#e7f8ef; color:#0f7a44; border:1px solid #b7ead0;}
.status-warn {background:#fff6e6; color:#a8650a; border:1px solid #f3ddae;}
.status-pill .dot {width:9px; height:9px; border-radius:50%;}
.status-ok .dot   {background:#16a34a;}
.status-warn .dot {background:#f59e0b;}
#config-card {
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: 16px; padding: 14px 16px;
}
#generate-btn {font-weight:600; font-size:1.02rem; border-radius:12px;}
#gallery-panel, #video-panel {border-radius:16px;}
.section-title {font-weight:600; font-size:1.05rem; margin:.4rem 0 .2rem; color:var(--body-text-color);}
footer {display:none !important;}
"""


def _kaggle_status_md() -> str:
    if credentials_present():
        return (
            '<div class="status-pill status-ok"><span class="dot"></span>'
            "Kaggle credentials detected — the video stage is ready.</div>"
        )
    return (
        '<div class="status-pill status-warn"><span class="dot"></span>'
        "No Kaggle credentials — set KAGGLE_USERNAME + KAGGLE_KEY or add "
        "~/.kaggle/kaggle.json to enable the video stage. Storyboard + images still work."
        "</div>"
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
        gr.HTML(
            '<div id="hero">'
            "<h1>AI Document → Video</h1>"
            "<p>Upload a PDF or DOCX → get a narrated video. Storyboard and images "
            "are generated locally; the video renders on Kaggle's free GPU.</p>"
            "</div>"
        )
        gr.HTML(_kaggle_status_md())

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_id="config-card"):
                gr.HTML('<div class="section-title">1 · Input</div>')
                input_file = gr.File(
                    label="Document (PDF or DOCX)",
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
                run_btn = gr.Button(
                    "Generate", variant="primary", elem_id="generate-btn", size="lg"
                )
            with gr.Column(scale=2):
                gr.HTML('<div class="section-title">2 · Progress</div>')
                logs = gr.Textbox(
                    label=None,
                    show_label=False,
                    lines=21,
                    max_lines=21,
                    autoscroll=True,
                )

        gr.HTML('<div class="section-title">3 · Storyboard images</div>')
        gallery = gr.Gallery(
            label=None, show_label=False, columns=4, height=260, elem_id="gallery-panel"
        )
        gr.HTML('<div class="section-title">4 · Final video</div>')
        video_out = gr.Video(label=None, show_label=False, elem_id="video-panel")

        run_btn.click(
            fn=generate,
            inputs=[input_file, language, make_video],
            outputs=[logs, gallery, video_out],
        )
    return demo


if __name__ == "__main__":
    server_port = int(os.environ.get("AIV_PORT", "7860"))
    build_ui().queue().launch(
        server_name="0.0.0.0", server_port=server_port, theme=THEME, css=CSS
    )
