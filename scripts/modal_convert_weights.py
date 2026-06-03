"""
Modal script to convert Keras MobileNetV2 pretrained weights → FOMO PyTorch checkpoints.

Runs the conversion inside a Modal container that has both TensorFlow and PyTorch,
then downloads the resulting .pt files back to the local weights/ directory.

Usage (from repo root):
    modal run scripts/modal_convert_weights.py
    modal run scripts/modal_convert_weights.py --sizes s m   # subset
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("fomo-convert-weights")

REPO_ROOT = Path(__file__).resolve().parent.parent

# TF + PyTorch image — TF 2.x ships its own numpy; pin a compatible version.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        # TF 2.16 is the last release that co-installs cleanly with torch 2.x
        "uv pip install --system 'tensorflow-cpu==2.16.*' torch numpy"
    )
    .add_local_file(
        REPO_ROOT / "scripts" / "convert_keras_mobilenetv2_to_pytorch.py",
        remote_path="/convert.py",
        copy=True,
    )
    .add_local_dir(
        REPO_ROOT / "fomo",
        remote_path="/fomo_src/fomo",
        copy=True,
    )
    .add_local_file(
        REPO_ROOT / "pyproject.toml",
        remote_path="/fomo_src/pyproject.toml",
        copy=True,
    )
    .run_commands("uv pip install --system -e /fomo_src")
)

# Volume to cache the downloaded Keras .h5 files across runs
keras_cache_vol = modal.Volume.from_name("fomo-keras-cache", create_if_missing=True)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 30,
    volumes={"/keras_cache": keras_cache_vol},
)
def convert_weights(sizes: list[str]) -> dict[str, bytes]:
    """
    Run the conversion inside Modal and return a mapping of
    filename → raw bytes for each produced .pt file.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "/convert.py",
            "--sizes", *sizes,
            "--out-dir", "/tmp/weights",
            "--keras-cache", "/keras_cache",
        ],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed with exit code {result.returncode}")

    keras_cache_vol.commit()

    out_files: dict[str, bytes] = {}
    for pt_file in Path("/tmp/weights").glob("FOMO*.pt"):
        out_files[pt_file.name] = pt_file.read_bytes()

    return out_files


@app.local_entrypoint()
def main(sizes: str = "s,m,l"):
    """
    Download the converted FOMO checkpoints to <repo_root>/weights/.

    Pass --sizes s,m,l (comma-separated) to convert a subset.
    """
    size_list = [s.strip() for s in sizes.split(",")]
    valid = {"s", "m", "l"}
    bad = set(size_list) - valid
    if bad:
        raise ValueError(f"Unknown sizes: {bad}. Choose from {valid}.")

    print(f"\nStarting Modal conversion for sizes: {size_list}")
    out_files = convert_weights.remote(size_list)

    local_weights = REPO_ROOT / "weights"
    local_weights.mkdir(parents=True, exist_ok=True)

    print("\nDownloading results …")
    for name, data in sorted(out_files.items()):
        dest = local_weights / name
        dest.write_bytes(data)
        print(f"  ✓ {dest}  ({len(data) // 1024} KB)")

    print("\n✓ Done. Upload these to huggingface.co/fomo-edge-ai/FOMO:")
    for name in sorted(out_files):
        print(f"  weights/{name}")
