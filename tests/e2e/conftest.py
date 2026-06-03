"""E2E test configuration and fixtures."""

import gc
import multiprocessing
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_python_env() -> dict[str, str]:
    """Return an env that makes one-shot test scripts import local sources."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    paths = [str(_REPO_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


# Force 'spawn' multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def pytest_report_header(config):
    return None


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Skip decorators
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def cuda_cleanup():
    """Free GPU memory. Call after heavy tests."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Pre-spawned subprocess worker for CUDA isolation
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r"""
import json, os, subprocess, sys, tempfile

while True:
    line = sys.stdin.readline()
    if not line:
        break
    msg = json.loads(line)
    script_text, timeout = msg["s"], msg["t"]

    fd, path = tempfile.mkstemp(suffix=".py", prefix="ly_")
    os.write(fd, script_text.encode())
    os.close(fd)
    try:
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        resp = {"rc": r.returncode, "o": r.stdout[-4000:], "e": r.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        resp = {"rc": -1, "o": "", "e": f"Timed out after {timeout}s"}
    except Exception as exc:
        resp = {"rc": -1, "o": "", "e": str(exc)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"""


def _start_worker():
    """Start the subprocess worker. Called once at import time."""
    import atexit
    import subprocess as _sp
    import sys as _sys
    import tempfile as _tmp

    fd, path = _tmp.mkstemp(suffix=".py", prefix="ly_worker_")
    import os as _os

    _os.write(fd, _WORKER_SCRIPT.encode())
    _os.close(fd)

    proc = _sp.Popen(
        [_sys.executable, path],
        stdin=_sp.PIPE,
        stdout=_sp.PIPE,
        stderr=_sp.DEVNULL,
        cwd=str(_REPO_ROOT),
        env=_repo_python_env(),
        text=True,
    )

    def _cleanup():
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=5)
        try:
            _os.unlink(path)
        except OSError:
            pass

    atexit.register(_cleanup)
    return proc


_worker_proc = _start_worker()


def run_in_subprocess(script: str, *, timeout: int = 300) -> str:
    """Run Python code in a fresh subprocess via the pre-spawned worker."""
    import json
    import textwrap

    msg = json.dumps({"s": textwrap.dedent(script), "t": timeout})
    _worker_proc.stdin.write(msg + "\n")
    _worker_proc.stdin.flush()

    resp_line = _worker_proc.stdout.readline()
    if not resp_line:
        raise RuntimeError(
            "Subprocess worker died unexpectedly.  Check stderr for details."
        )

    resp = json.loads(resp_line)
    if resp["rc"] != 0:
        raise RuntimeError(
            f"Subprocess exited with code {resp['rc']}\n"
            f"--- stdout (last 2000 chars) ---\n{resp['o'][-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{resp['e'][-2000:]}"
        )
    return resp["o"]


def run_direct_subprocess(script: str, *, timeout: int = 300) -> str:
    """Run Python code in a one-shot subprocess."""
    import os
    import subprocess
    import sys
    import tempfile
    import textwrap

    fd, path = tempfile.mkstemp(suffix=".py", prefix="ly_direct_")
    os.write(fd, textwrap.dedent(script).encode())
    os.close(fd)
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            cwd=str(_REPO_ROOT),
            env=_repo_python_env(),
            text=True,
            timeout=timeout,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess exited with code {result.returncode}\n"
            f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cuda_device():
    """Return CUDA device string if available."""
    if has_cuda():
        return "cuda"
    pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def gpu_info():
    """Return GPU information for logging."""
    if not has_cuda():
        return {"available": False}

    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


@pytest.fixture(scope="session")
def sample_image():
    """Get a sample image for inference tests."""
    from fomo import SAMPLE_IMAGE

    return SAMPLE_IMAGE


@pytest.fixture(scope="function")
def temp_export_dir(tmp_path):
    """Create a temporary directory for export artifacts."""
    return tmp_path / "exports"


@pytest.fixture(autouse=True, scope="function")
def cleanup_gpu_memory():
    """Clear GPU memory before and after each test to prevent state corruption."""
    yield
    cuda_cleanup()


@pytest.fixture(scope="class")
def reset_gpu_state():
    """Force GPU state reset between test classes."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Model catalog — single source of truth
# ---------------------------------------------------------------------------

# (family, size, weights)
MODEL_CATALOG = [
    ("fomo", "s", "FOMOs.pt"),
    ("fomo", "m", "FOMOm.pt"),
    ("fomo", "l", "FOMOl.pt"),
]

FLAGSHIP_FAMILIES = {"fomo"}

FAMILY_MARKERS = {
    "fomo": pytest.mark.fomo,
}


def _normalize_marks(marks):
    """Normalize a mark or a collection of marks to a flat list."""
    if marks is None:
        return []
    if isinstance(marks, (list, tuple, set)):
        return [mark for mark in marks if mark is not None]
    return [marks]


def family_marks(family: str, marks=None):
    """Return pytest marks for a model family plus any extra marks."""
    return [FAMILY_MARKERS[family], *_normalize_marks(marks)]


def flagship_nightly_marks(family: str, *_):
    return pytest.mark.flagship_nightly


def general_nightly_marks(*_):
    return pytest.mark.general_nightly


def model_case(family: str, size: str, *, weights: str | None = None, marks=None):
    """Build a parametrized model case with family markers attached."""
    values = (family, size) if weights is None else (family, size, weights)
    return pytest.param(
        *values, marks=family_marks(family, marks), id=f"{family}-{size}"
    )


def model_cases(models, *, with_weights: bool = False, marks_resolver=None):
    """Attach family markers to a model matrix used in parametrized tests."""
    params = []
    for family, size, *rest in models:
        weights = rest[0] if with_weights else None
        marks = marks_resolver(family, size, *rest) if marks_resolver else None
        params.append(model_case(family, size, weights=weights, marks=marks))
    return params


ALL_MODELS = [(f, s) for f, s, _ in MODEL_CATALOG]
ALL_MODELS_WITH_WEIGHTS = MODEL_CATALOG
ALL_MODEL_WEIGHT_PARAMS = model_cases(
    ALL_MODELS_WITH_WEIGHTS,
    with_weights=True,
    marks_resolver=flagship_nightly_marks,
)


def get_model_weights(family: str, size: str) -> str:
    """Get the weight file name for a model family and size."""
    for f, s, w in MODEL_CATALOG:
        if f == family and s == size:
            return w
    raise ValueError(f"Unknown model: {family}-{size}")


@lru_cache(maxsize=None)
def _detect_local_weights_family(weights: str) -> str:
    """Detect a local checkpoint's family for skip-only environment validation."""
    from fomo import FOMO

    model = FOMO(weights)
    return model.FAMILY


@lru_cache(maxsize=None)
def _has_fomo_download_route(weights: str) -> bool:
    """Return whether a missing test weight has a canonical FOMO HF route."""
    import fomo.models  # noqa: F401
    from fomo.models.base.model import BaseModel

    filename = Path(weights).name
    for cls in BaseModel._registry:
        try:
            url = cls.get_download_url(filename)
        except Exception:
            continue
        if url and url.startswith("https://huggingface.co/"):
            return True
    return False


def require_test_weights(weights: str, expected_family: str | None = None) -> str:
    """Skip cleanly if a test depends on missing or obviously wrong local weights."""
    path = Path(weights)
    if path.parent != Path("."):
        if not path.exists():
            if _has_fomo_download_route(weights):
                return weights
            pytest.skip(f"Required local weights not found: {weights}")
        if expected_family is not None:
            try:
                detected_family = _detect_local_weights_family(str(path))
            except Exception as exc:
                pytest.skip(
                    f"Local weights are unusable for testing: {weights} ({exc})"
                )
            if detected_family != expected_family:
                pytest.skip(
                    "Local weights do not match the expected family: "
                    f"{weights} detected as '{detected_family}', expected '{expected_family}'"
                )
    return weights


def load_model(model_type: str, size: str, device: str = "cuda"):
    """Load a model by type and size."""
    from fomo import FOMO

    weights = require_test_weights(
        get_model_weights(model_type, size),
        expected_family=model_type,
    )
    return FOMO(weights, device=device)
