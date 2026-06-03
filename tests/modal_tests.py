from pathlib import Path
import modal

app = modal.App("fomo-tests")

# Calculate the root of the repository relative to this script's location
REPO_ROOT = Path(__file__).parent.parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands("uv pip install --system scipy pytest torch torchvision")
    .add_local_dir(
        REPO_ROOT / "dist",
        remote_path="/dist",
        copy=True,
    )
    .run_commands("uv pip install --system /dist/fomo_edge_ai-*.whl")
    .add_local_dir(
        REPO_ROOT / "tests",
        remote_path="/workspace/tests",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 75,
)
def run_pytest(args: list[str] | None = None):
    import subprocess
    import sys

    # If args are specified, run only those custom pytest arguments
    if args:
        import shlex
        quoted_args = " ".join(shlex.quote(arg) for arg in args)
        cmd = f"cd /workspace && python -m pytest {quoted_args}"
        print(f"Running custom tests: {cmd}", flush=True)
        r = subprocess.run(["bash", "-lc", cmd], check=True)
        return

    # Otherwise, run the default holistic test suite: unit, inference, and training
    def run(cmd_args, label):
        print(f"\n{'='*60}", flush=True)
        print(f"  {label}", flush=True)
        print(f"{'='*60}\n", flush=True)
        r = subprocess.run(
            cmd_args,
            cwd="/workspace",
            capture_output=False,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"{label} — pytest exited with code {r.returncode}")

    # 1. Unit tests (~10s, CPU)
    run(
        [sys.executable, "-m", "pytest",
         "tests/unit/test_fomo.py",
         "-v", "--tb=short", "--no-header"],
        "1/5 — FOMO unit tests",
    )

    # 2. Export / quantize unit tests (structural + config validation, CPU)
    run(
        [sys.executable, "-m", "pytest",
         "tests/unit/test_export.py",
         "-v", "--tb=short", "--no-header"],
        "2/5 — Export + quantize unit tests",
    )

    # 3. Inference e2e (pretrained checkpoint download + stable inference)
    run(
        [sys.executable, "-m", "pytest",
         "tests/e2e/test_fomo_inference.py",
         "-v", "-m", "fomo", "--tb=short", "--no-header", "-s"],
        "3/5 — FOMO inference e2e tests",
    )

    # 4. Training e2e (git-clone dataset, 2-epoch smoke train)
    run(
        [sys.executable, "-m", "pytest",
         "tests/e2e/test_fomo_training.py",
         "-v", "-m", "fomo", "--tb=short", "--no-header", "-s"],
        "4/5 — FOMO training e2e test",
    )

    # 5. Export + quantize e2e (FP32 TFLite → INT8 Vela quantization)
    run(
        [sys.executable, "-m", "pytest",
         "tests/e2e/test_fomo_export.py",
         "-v", "-m", "fomo", "--tb=short", "--no-header", "-s"],
        "5/5 — FOMO export + quantize e2e test",
    )

    print("\n✓ All FOMO tests passed\n", flush=True)


@app.local_entrypoint()
def main(*pytest_args: str):
    run_pytest.remote(list(pytest_args) if pytest_args else None)
