"""End-to-end export + quantize test for FOMO.

Mirrors the workflow from ``train_librefomo_sample_and_export_vela.py``:

  1. Load the HuggingFace point-localization dataset
     (``bdanko/sjsu-headcount-scene-1`` — same as the training e2e test)
  2. Build a minimal FOMO-s model (no pretrained weights)
  3. Export to TFLite FP32 via ``model.export()``
  4. Prepare a calibration iterator in the same format used by the notebook
  5. Quantize to INT8 via ``VelaQuantizer``
  6. Verify both flatbuffers exist and have non-trivial size

Run via pytest::

    pytest tests/e2e/test_fomo_export.py -v -m e2e

"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from .conftest import run_direct_subprocess

pytestmark = [pytest.mark.e2e, pytest.mark.fomo]

# ---------------------------------------------------------------------------
# Dataset — same HF repo used by the training e2e test
# ---------------------------------------------------------------------------

HF_REPO = "bdanko/sjsu-headcount-scene-1"
DATASET_ROOT = Path.home() / ".cache" / "fomo" / "sjsu-headcount-scene-1"

# Calibration: we only need a handful of samples for the smoke test
NUM_CALIBRATION_SAMPLES = 8
INPUT_SIZE = 96   # FOMO-s native resolution


def _download_dataset() -> None:
    if DATASET_ROOT.exists() and (DATASET_ROOT / "data.yaml").exists():
        return
    print(f"\nDownloading dataset {HF_REPO} from HuggingFace …")
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone",
            f"https://huggingface.co/datasets/{HF_REPO}",
            str(DATASET_ROOT),
        ],
        check=True,
    )
    print(f"Dataset downloaded to {DATASET_ROOT}")


def _patch_data_yaml() -> None:
    data_yaml = DATASET_ROOT / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    if data.get("path") != str(DATASET_ROOT):
        data["path"] = str(DATASET_ROOT)
        data_yaml.write_text(yaml.dump(data, default_flow_style=False))


@pytest.fixture(scope="module")
def fomo_dataset():
    """Download and patch the shared FOMO dataset."""
    _download_dataset()
    _patch_data_yaml()
    return DATASET_ROOT


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


def test_fomo_export_and_quantize(fomo_dataset, tmp_path):
    """Smoke test: FP32 export → INT8 quantization on a tiny FOMO-s model.

    Uses ``run_direct_subprocess`` so litert / ai-edge-quantizer state is
    isolated from the rest of the test suite.

    Asserts:
      - FP32 flatbuffer is written and non-empty
      - INT8 flatbuffer is written and non-empty
    """
    dataset_path = str(fomo_dataset)
    out_dir = str(tmp_path / "tflite")
    fp32_path = str(tmp_path / "tflite" / "fomo_s_fp32.tflite")
    int8_path = str(tmp_path / "tflite" / "fomo_s_int8.tflite")

    run_direct_subprocess(
        f"""
        import numpy as np
        import torch
        from pathlib import Path
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        from PIL import Image
        import json

        from fomo import FOMO
        from fomo.export import INT8Quantizer, INT8QuantizeConfig

        INPUT_SIZE = {INPUT_SIZE}
        NUM_CALIBRATION_SAMPLES = {NUM_CALIBRATION_SAMPLES}
        FOMO_DOWNSAMPLE_FACTOR = 8
        GRID_SIZE = INPUT_SIZE // FOMO_DOWNSAMPLE_FACTOR   # 12

        # ----------------------------------------------------------------
        # 1.  Build a minimal FOMO-s (no pretrained weights)
        # ----------------------------------------------------------------
        model = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")
        assert model.family == "fomo"
        assert model.input_size == INPUT_SIZE
        print("Model built OK", flush=True)

        # ----------------------------------------------------------------
        # 2.  Export to TFLite FP32
        # ----------------------------------------------------------------
        fp32_path = model.export(output_path=r"{fp32_path}")
        assert Path(fp32_path).exists(), f"FP32 file missing: {{fp32_path}}"
        assert Path(fp32_path).stat().st_size > 0, "FP32 file is empty"
        print(f"FP32 export OK: {{fp32_path}} ({{Path(fp32_path).stat().st_size}} bytes)", flush=True)

        # ----------------------------------------------------------------
        # 3.  Build calibration dataset — same approach as the notebook
        #
        #     We build a minimal PyTorch Dataset over real images from the
        #     HuggingFace dataset directory so the calibration loop mirrors
        #     get_calibration_dataset() in the training script exactly.
        # ----------------------------------------------------------------

        image_transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        dataset_root = Path(r"{dataset_path}")
        image_paths = sorted(dataset_root.rglob("*.jpg"))[:NUM_CALIBRATION_SAMPLES * 2]
        assert image_paths, f"No .jpg images found under {{dataset_root}}"

        def get_calibration_data(image_paths, num_samples):
            "Yields representative data formatted for the LiteRT graph."
            count = 0
            for img_path in image_paths:
                if count >= num_samples:
                    return
                img = Image.open(img_path).convert("RGB")
                tensor = image_transform(img).unsqueeze(0).numpy()
                yield {{"args_0": tensor}}
                count += 1

        calib_iter = get_calibration_data(image_paths, NUM_CALIBRATION_SAMPLES)

        # ----------------------------------------------------------------
        # 4.  Quantize to INT8
        # ----------------------------------------------------------------
        config = INT8QuantizeConfig(num_calibration_samples=NUM_CALIBRATION_SAMPLES)
        qt = INT8Quantizer()
        int8_path = qt.quantize(fp32_path, calib_iter, config, output_path=r"{int8_path}")

        assert Path(int8_path).exists(), f"INT8 file missing: {{int8_path}}"
        assert Path(int8_path).stat().st_size > 0, "INT8 file is empty"
        print(f"INT8 export OK: {{int8_path}} ({{Path(int8_path).stat().st_size}} bytes)", flush=True)

        print("\\n✓ FOMO export + quantize smoke test PASSED", flush=True)
        """,
        timeout=600,
    )
