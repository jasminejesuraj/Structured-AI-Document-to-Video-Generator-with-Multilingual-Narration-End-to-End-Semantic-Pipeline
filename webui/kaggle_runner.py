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


def _config_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))

def credentials_present() -> bool:
    # Legacy key pair (env or kaggle.json)
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if (_config_dir() / "kaggle.json").exists():
        return True
    # New access-token system (env or ~/.kaggle/access_token)
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if (_config_dir() / "access_token").exists():
        return True
    return False
 
 
def get_username(api=None) -> str:
    # Legacy sources first (cheap, no network).
    user = os.environ.get("KAGGLE_USERNAME")
    if user:
        return user
    #cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    cfg = _config_dir() / "kaggle.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f)["username"]
    #raise KaggleConfigError("Kaggle username not found (set KAGGLE_USERNAME or provide kaggle.json).")
    # New access-token system: the authenticated client knows who we are.
    if api is not None:
        for attr in ("username",):
            val = getattr(api, attr, None)
            if val:
                return val
        cfg_vals = getattr(api, "config_values", None)
        if isinstance(cfg_vals, dict):
            val = cfg_vals.get("username")
            if val:
                return val
    raise KaggleConfigError(
        "Kaggle username not found. With the new access-token system, set "
        "KAGGLE_USERNAME alongside ~/.kaggle/access_token so the app can build "
        "your dataset/kernel IDs."
    )
 
 

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
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return kernel_dir
def _dataset_files(api, dataset_id: str):
    """Return the list of files Kaggle reports for ``dataset_id`` (or [] on error)."""
    try:
        result = api.dataset_list_files(dataset_id)
    except Exception:
        return []
    files = getattr(result, "files", None)
    if files is None and isinstance(result, dict):
        files = result.get("datasetFiles") or result.get("files")
    return files or []
 
 
def wait_for_dataset(api, dataset_id: str, expected_names, timeout_seconds: int = 600,
                     poll_seconds: int = 15):
    """Poll Kaggle until the dataset lists all ``expected_names`` (or time out).
 
    A brand-new dataset (or new version) is not usable in a kernel until Kaggle
    finishes processing it, which can take well over the old fixed 10s sleep.
    """
    expected = set(expected_names)
    start = time.time()
    while True:
        files = _dataset_files(api, dataset_id)
        names = {getattr(f, "name", None) or (f.get("name") if isinstance(f, dict) else None)
                 for f in files}
        names.discard(None)
        if expected.issubset(names):
            return True
        if time.time() - start > timeout_seconds:
            return False
        time.sleep(poll_seconds)
 
 
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

def _try_fetch_video(api, kernel_ref: str, out_dir: Path):
    """Download kernel output and return the final video path, or None.
 
    Used both when the kernel reports ``complete`` and as a fallback when the
    status endpoint is failing but the run may already have produced its output.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_output(kernel_ref, str(out_dir))
    except Exception:
        pass  # the files may still have been written before the error
    videos = sorted(out_dir.rglob("final_video_*.mp4"))
    if not videos:
        videos = sorted(out_dir.rglob("*.mp4"))
    return videos[0] if videos else None
 

def run_video(
    run_dir,
    language_name: str,
    poll_seconds: int = 20,
    timeout_seconds: int = int(os.environ.get("AIV_KAGGLE_TIMEOUT", "14400")),
):
    """Generator that yields human-readable log lines for the Kaggle video stage.

    The final video path (when successful) is yielded as ``FINAL_VIDEO::<path>``.
    """
    run_dir = Path(run_dir)
    language_code = LANGUAGE_MAP.get(language_name)
    if language_code is None:
        raise ValueError(f"Unsupported language: {language_name}")

    yield "Authenticating with Kaggle..."
    api = _get_api()
    username = get_username(api)


    dataset_id = f"{username}/{DATASET_SLUG}"
    kernel_ref = f"{username}/{KERNEL_SLUG}"

    yield "Staging dataset (storyboard.json + scene images)..."
    dataset_dir = run_dir / "kaggle_dataset"
    stage_dataset(run_dir, dataset_dir, username)

        # File names the kernel will need to see mounted under /kaggle/input.
    expected_names = {
        p.name for p in dataset_dir.iterdir() if p.name != "dataset-metadata.json"
    }
 

    yield f"Uploading dataset to Kaggle ({dataset_id})..."
    action = _push_dataset(api, dataset_dir)
    yield f"Dataset {action}."

    # A new dataset/version is not usable in a kernel until Kaggle finishes
    # processing it. Poll until every staged file is listed (with a safety cap)
    # instead of a fixed sleep, which was too short and let the kernel start
    # against an empty mount.
     # Kaggle's dataset_list_files reports "This dataset type is not supported
    # through the API" for these datasets, so we cannot poll for readiness.
    # The assets are tiny (a few KB) and process within ~1 minute, so wait a
    # fixed, generous margin before the kernel mounts them.
    dataset_wait = int(os.environ.get("AIV_DATASET_WAIT", "120"))
    yield f"Waiting {dataset_wait}s for Kaggle to process the dataset..."
    time.sleep(dataset_wait)
    yield "Proceeding to render."

    yield "Building notebook + kernel metadata..."
    kernel_dir = run_dir / "kaggle_kernel"
    build_kernel_dir(kernel_dir, username, dataset_id, language_code, language_name)

    yield f"Pushing kernel ({kernel_ref}) with GPU enabled..."
    api.kernels_push(str(kernel_dir))

    yield "Kernel queued. Rendering on Kaggle GPU — this can take a while..."
    out_dir = run_dir / "kaggle_output"
    start = time.time()
    last_heartbeat = 0.0
    while True:
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Kaggle kernel did not finish within the timeout.")
        time.sleep(poll_seconds)
        # Primary signal: has the kernel produced the final video yet?
        # kernels_output fetches exactly what appears in the kernel's Output tab.
        video = _try_fetch_video(api, kernel_ref, out_dir)
        if video is not None:
            final = run_dir / video.name
            shutil.copy2(video, final)
            yield f"Final video downloaded: {final.name}"
            yield f"FINAL_VIDEO::{final}"
            return
        # Secondary signal: status. It often 500s on some accounts, so treat a
        # failure as "still working" (not an error). Only an explicit failure
        # state with no video produced aborts the run.
        try:
            state = _status_value(api.kernels_status(kernel_ref))
        except Exception:
            state = "working"
        if state in ("error", "cancelacknowledged", "cancelrequested"):
            # Give the output one last chance (the video may have been written
            # just before the status flipped), then fail.
            video = _try_fetch_video(api, kernel_ref, out_dir)
            if video is not None:
                final = run_dir / video.name
                shutil.copy2(video, final)
                yield f"Final video downloaded: {final.name}"
                yield f"FINAL_VIDEO::{final}"
                return
            raise RuntimeError(f"Kaggle kernel failed: {state}")
        # Heartbeat so the UI shows progress without spamming 500 lines.
        if time.time() - last_heartbeat > 60:
            yield "Still rendering on Kaggle... (working)"
            last_heartbeat = time.time()


if __name__ == "__main__":
    print("credentials_present:", credentials_present())
    print("languages:", ", ".join(LANGUAGE_MAP))
