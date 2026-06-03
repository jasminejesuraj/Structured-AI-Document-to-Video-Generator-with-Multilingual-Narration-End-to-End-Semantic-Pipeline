"""Drive the video stage on Kaggle's free GPU via the Kaggle public API.

Flow:
1. Stage a dataset folder (storyboard.json + flattened scene_*.png) and push it.
2. Generate a non-interactive notebook (language injected) + kernel metadata.
3. Push the kernel (GPU + internet enabled) and poll until it completes.
4. Download the kernel output and locate the final video.

The ``kaggle`` package authenticates on import, so it is imported lazily inside
functions (after credentials are verified) to keep the web app importable
without credentials.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

# UI language name -> (gTTS/translate code) used by the notebook
LANGUAGE_MAP = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-CN",
    "Arabic": "ar",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
}

DATASET_SLUG = "ai-doc-to-video-assets"
KERNEL_SLUG = "ai-doc-to-video-svd"


class KaggleConfigError(RuntimeError):
    """Raised when Kaggle credentials are missing or unusable."""


def credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return cfg.exists()


def get_username() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    if user:
        return user
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f)["username"]
    raise KaggleConfigError("Kaggle username not found (set KAGGLE_USERNAME or provide kaggle.json).")


def _get_api():
    if not credentials_present():
        raise KaggleConfigError(
            "Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or place kaggle.json in ~/.kaggle/."
        )
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _status_value(status) -> str:
    """Normalize kernels_status return (dict or object) to a lowercase string."""
    if isinstance(status, dict):
        val = status.get("status", "")
    else:
        val = getattr(status, "status", status)
    return str(val).lower().replace("kernelworkerstatus.", "")


def _status_message(status) -> str:
    if isinstance(status, dict):
        return status.get("failureMessage") or ""
    return getattr(status, "failureMessage", "") or ""


def stage_dataset(run_dir: Path, dataset_dir: Path, username: str) -> Path:
    """Build the dataset folder: storyboard.json + flattened scene images + metadata."""
    run_dir = Path(run_dir)
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    storyboard = run_dir / "storyboard.json"
    if not storyboard.exists():
        raise FileNotFoundError(f"storyboard.json not found in {run_dir}")
    shutil.copy2(storyboard, dataset_dir / "storyboard.json")

    outputs = run_dir / "outputs"
    images = sorted(outputs.glob("scene_*.png")) if outputs.exists() else []
    if not images:
        raise FileNotFoundError(f"No scene images found in {outputs}")
    for img in images:
        shutil.copy2(img, dataset_dir / img.name)  # flattened into dataset root

    metadata = {
        "title": "AI Doc To Video Assets",
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return dataset_dir


def build_kernel_dir(
    kernel_dir: Path,
    username: str,
    dataset_id: str,
    language_code: str,
    language_name: str,
) -> Path:
    """Write the kernel notebook + kernel-metadata.json into ``kernel_dir``."""
    from notebook_builder import write_video_notebook

    kernel_dir = Path(kernel_dir)
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    nb_name = "video_kernel.ipynb"
    write_video_notebook(
        str(kernel_dir / nb_name),
        dataset_slug=DATASET_SLUG,
        language_code=language_code,
        language_name=language_name,
    )

    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "AI Doc To Video - SVD",
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return kernel_dir


def _push_dataset(api, dataset_dir: Path):
    try:
        api.dataset_create_new(str(dataset_dir), public=False, quiet=True, convert_to_csv=False)
        return "created"
    except Exception as exc:  # already exists -> push a new version
        msg = str(exc).lower()
        if "already exists" in msg or "409" in msg or "use" in msg and "version" in msg:
            api.dataset_create_version(
                str(dataset_dir), version_notes="update from app", quiet=True, convert_to_csv=False
            )
            return "versioned"
        # Fall back to versioning on any create error if the dataset is reachable
        try:
            api.dataset_create_version(
                str(dataset_dir), version_notes="update from app", quiet=True, convert_to_csv=False
            )
            return "versioned"
        except Exception:
            raise exc


def run_video(run_dir, language_name: str, poll_seconds: int = 20, timeout_seconds: int = 5400):
    """Generator that yields human-readable log lines for the Kaggle video stage.

    The final video path (when successful) is yielded as ``FINAL_VIDEO::<path>``.
    """
    run_dir = Path(run_dir)
    language_code = LANGUAGE_MAP.get(language_name)
    if language_code is None:
        raise ValueError(f"Unsupported language: {language_name}")

    yield "Authenticating with Kaggle..."
    api = _get_api()
    username = get_username()
    dataset_id = f"{username}/{DATASET_SLUG}"
    kernel_ref = f"{username}/{KERNEL_SLUG}"

    yield "Staging dataset (storyboard.json + scene images)..."
    dataset_dir = run_dir / "kaggle_dataset"
    stage_dataset(run_dir, dataset_dir, username)

    yield f"Uploading dataset to Kaggle ({dataset_id})..."
    action = _push_dataset(api, dataset_dir)
    yield f"Dataset {action}."
    # Give Kaggle a moment to process the new dataset version before the kernel runs.
    time.sleep(10)

    yield "Building notebook + kernel metadata..."
    kernel_dir = run_dir / "kaggle_kernel"
    build_kernel_dir(kernel_dir, username, dataset_id, language_code, language_name)

    yield f"Pushing kernel ({kernel_ref}) with GPU enabled..."
    api.kernels_push(str(kernel_dir))

    yield "Kernel queued. Waiting for the GPU run to complete (this can take a while)..."
    start = time.time()
    last = None
    while True:
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Kaggle kernel did not finish within the timeout.")
        time.sleep(poll_seconds)
        try:
            status = api.kernels_status(kernel_ref)
        except Exception as exc:
            yield f"(status check retry: {exc})"
            continue
        state = _status_value(status)
        if state != last:
            yield f"Kernel status: {state}"
            last = state
        if state in ("complete", "error", "cancelacknowledged", "cancelrequested"):
            if state != "complete":
                raise RuntimeError(f"Kaggle kernel failed: {state} {_status_message(status)}")
            break

    yield "Kernel complete. Downloading output..."
    out_dir = run_dir / "kaggle_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel_ref, str(out_dir))

    videos = sorted(out_dir.rglob("final_video_*.mp4"))
    if not videos:
        videos = sorted(out_dir.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError("No video file found in Kaggle kernel output.")

    final = run_dir / videos[0].name
    shutil.copy2(videos[0], final)
    yield f"Final video downloaded: {final.name}"
    yield f"FINAL_VIDEO::{final}"


if __name__ == "__main__":
    print("credentials_present:", credentials_present())
    print("languages:", ", ".join(LANGUAGE_MAP))
