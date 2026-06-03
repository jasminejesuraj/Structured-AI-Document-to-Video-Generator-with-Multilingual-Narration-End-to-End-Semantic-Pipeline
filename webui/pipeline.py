"""Run the local stages (storyboard + images) as streaming subprocesses.

The existing scripts are executed unchanged except via environment variables:
- ``AIV_INPUT_DOC``  -> input document for ``prompt_generation.py``
- ``AIV_STORYBOARD`` -> storyboard.json location (shared by both stages)

Each stage runs with ``cwd`` set to a per-run working directory so the relative
``outputs/`` folder and ``storyboard.json`` land there.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "webui" / "runs"


def make_run_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    return run_dir


def _stream(cmd, cwd: Path, env: dict):
    """Yield output lines; the final line is ``__EXIT__<returncode>``."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()
    yield f"__EXIT__{proc.returncode}"


def run_storyboard(input_doc: str, run_dir: Path):
    env = os.environ.copy()
    env["AIV_INPUT_DOC"] = str(input_doc)
    env["AIV_STORYBOARD"] = str(Path(run_dir) / "storyboard.json")
    yield from _stream(
        [sys.executable, "-u", str(REPO_ROOT / "prompt_generation.py")], run_dir, env
    )


def run_images(run_dir: Path):
    env = os.environ.copy()
    env["AIV_STORYBOARD"] = str(Path(run_dir) / "storyboard.json")
    yield from _stream([sys.executable, "-u", str(REPO_ROOT / "image.py")], run_dir, env)


def storyboard_summary(run_dir: Path):
    """Return (num_scenes, list_of_image_paths) for a completed run."""
    import json

    run_dir = Path(run_dir)
    sb = run_dir / "storyboard.json"
    scenes = 0
    if sb.exists():
        with open(sb) as f:
            scenes = len(json.load(f).get("scenes", []))
    images = sorted(str(p) for p in (run_dir / "outputs").glob("scene_*.png"))
    return scenes, images
