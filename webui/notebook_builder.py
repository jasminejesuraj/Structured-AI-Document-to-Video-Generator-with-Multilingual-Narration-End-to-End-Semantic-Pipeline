"""Build a parameterized, non-interactive Kaggle notebook for the video stage.

The original Kaggle notebook prompted for the narration language with
``input()`` (which cannot run headlessly) and read its assets from a
hard-coded dataset path. This module regenerates an equivalent notebook with:

- the language injected as plain variables (no ``input()``),
- the dataset/asset paths pointed at the dataset slug we upload,
- the final video written to ``/kaggle/working`` so the Kaggle API can fetch it.

The image-generation / SVD / gTTS / moviepy logic mirrors the user's working
notebook so output quality is unchanged.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_notebook


def _install_cell() -> str:
    return (
        "!pip install -q gtts deep-translator imageio imageio-ffmpeg\n"
        "!apt-get update -qq\n"
        "!apt-get install -y ffmpeg\n"
        "!pip install -q moviepy==2.1.1"
    )


def _config_cell(dataset_slug: str, language_code: str, language_name: str) -> str:
        return f"""import json, os, glob
 
# Preferred location: dataset mounted at /kaggle/input/<slug>
PREFERRED_FOLDER = "/kaggle/input/{dataset_slug}"
 
# Diagnostics: show what is actually mounted under /kaggle/input.
INPUT_ROOT = "/kaggle/input"
print("Contents of", INPUT_ROOT, ":")
if os.path.isdir(INPUT_ROOT):
    for entry in sorted(os.listdir(INPUT_ROOT)):
        print("  -", entry)
else:
    print("  (/kaggle/input does not exist)")
 
# Locate storyboard.json robustly: try the preferred path first, then search
# anywhere under /kaggle/input (handles slug/name differences and nesting).
def _find_storyboard():
    preferred = os.path.join(PREFERRED_FOLDER, "storyboard.json")
    if os.path.exists(preferred):
        return preferred
    matches = glob.glob(os.path.join(INPUT_ROOT, "**", "storyboard.json"), recursive=True)
    return matches[0] if matches else None
 
JSON_PATH = _find_storyboard()
if JSON_PATH is None:
    raise FileNotFoundError(
        "storyboard.json was not found under /kaggle/input. The dataset is "
        "either not attached to this kernel or has a different layout. "
        "Mounted entries are printed above."
    )
 
# Everything else (scene images) lives next to storyboard.json.
PROJECT_FOLDER = os.path.dirname(JSON_PATH)
print("Using PROJECT_FOLDER:", PROJECT_FOLDER)


OUTPUT_DIR = "/kaggle/working/outputs"
CLIPS_DIR = f"{{OUTPUT_DIR}}/clips"
AUDIO_DIR = f"{{OUTPUT_DIR}}/audio"
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)
scenes = data["scenes"]

# Narration language is injected by the app (no interactive prompt)
LANGUAGE = "{language_code}"
LANG_NAME = "{language_name}"

print(f"Loaded {{len(scenes)}} scenes | Language: {{LANG_NAME}} ({{LANGUAGE}})")"""


def _svd_load_cell() -> str:
    return """import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from diffusers import StableVideoDiffusionPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to("cuda")
pipe.unet.enable_forward_chunking()
pipe.enable_attention_slicing()
print("SVD loaded")"""


def _clips_cell() -> str:
    # Images are flattened into the dataset root as scene_<n>.png
    return """from diffusers.utils import load_image
import imageio
import numpy as np
import torch

for scene in scenes:
    scene_num = scene["scene_number"]
    img_path = f"{PROJECT_FOLDER}/scene_{scene_num}.png"
    if not os.path.exists(img_path):
        print(f"Missing image for scene {scene_num}")
        continue

    print(f"Animating scene {scene_num}")
    image = load_image(img_path).resize((1024, 576))
    with torch.no_grad():
        frames = pipe(
            image,
            num_frames=40,
            num_inference_steps=25,
            motion_bucket_id=25,
            noise_aug_strength=0.02,
            decode_chunk_size=4,
        ).frames[0]

    video_path = f"{CLIPS_DIR}/clip_{scene_num:02d}.mp4"
    writer = imageio.get_writer(video_path, fps=7, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()

    del frames
    torch.cuda.empty_cache()
    print(f"Scene {scene_num} clip done")

print("All clips generated")"""


def _audio_cell() -> str:
    return """from gtts import gTTS
from deep_translator import GoogleTranslator

for scene in scenes:
    scene_num = scene["scene_number"]
    narration = scene["narration_text"]

    if LANGUAGE != "en":
        try:
            narration = GoogleTranslator(source="en", target=LANGUAGE).translate(narration)
        except Exception as e:
            print(f"Translation failed for scene {scene_num}: {e}")

    audio_path = f"{AUDIO_DIR}/audio_{scene_num:02d}.mp3"
    try:
        gTTS(text=narration, lang=LANGUAGE, slow=False).save(audio_path)
        scene["audio_path"] = audio_path
        print(f"Audio saved: {audio_path}")
    except Exception as e:
        print(f"Audio generation failed for scene {scene_num}: {e}")"""


def _assemble_cell() -> str:
    return """from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

final_clips = []
for scene in scenes:
    scene_num = scene["scene_number"]
    clip_path = f"{CLIPS_DIR}/clip_{scene_num:02d}.mp4"
    audio_path = f"{AUDIO_DIR}/audio_{scene_num:02d}.mp3"
    if not (os.path.exists(clip_path) and os.path.exists(audio_path)):
        continue

    audio = AudioFileClip(audio_path)
    video = VideoFileClip(clip_path)
    if audio.duration > video.duration:
        loops = int(audio.duration / video.duration) + 1
        video = concatenate_videoclips([video] * loops).subclipped(0, audio.duration)
    else:
        video = video.subclipped(0, audio.duration)
    video = video.with_audio(audio)
    final_clips.append(video)
    print(f"Scene {scene_num} ready")

if not final_clips:
    raise RuntimeError("No clips were produced; cannot assemble video.")

output_filename = f"/kaggle/working/final_video_{LANG_NAME.lower()}.mp4"
final = concatenate_videoclips(final_clips, method="compose").with_fps(24)
final.write_videofile(
    output_filename,
    codec="libx264",
    audio_codec="aac",
    preset="medium",
)
print(f"FINAL_VIDEO::{output_filename}")"""


def build_video_notebook(dataset_slug: str, language_code: str, language_name: str):
    """Return an ``nbformat`` NotebookNode for the Kaggle video stage."""
    cells = [
        new_code_cell(_install_cell()),
        new_code_cell(_config_cell(dataset_slug, language_code, language_name)),
        new_code_cell(_svd_load_cell()),
        new_code_cell(_clips_cell()),
        new_code_cell(_audio_cell()),
        new_code_cell(_assemble_cell()),
    ]
    nb = new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    return nb


def write_video_notebook(path: str, dataset_slug: str, language_code: str, language_name: str) -> str:
    """Write the generated notebook to ``path`` and return that path."""
    nb = build_video_notebook(dataset_slug, language_code, language_name)
    nbformat.validate(nb)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return path


if __name__ == "__main__":
    import tempfile

    out = write_video_notebook(
        f"{tempfile.gettempdir()}/_smoke_video_kernel.ipynb",
        dataset_slug="ai-doc-to-video-assets",
        language_code="en",
        language_name="English",
    )
    print("Wrote + validated notebook:", out)
