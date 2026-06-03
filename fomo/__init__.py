"""FOMO — Lightweight point-localization models."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path as _Path

# Core API — always available
from .models import FOMO
from .utils.results import Results, Boxes, Masks, Keypoints, Points, Probs, OBB, Gaze

SAMPLE_IMAGE = str(_Path(__file__).parent / "assets" / "parkour.jpg")

try:
    __version__ = version("fomo-edge-ai")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


# Lazy imports for optional/heavy modules
def __getattr__(name):
    _lazy = {
        # TFLite export
        "TFLiteExporter": (".export", "TFLiteExporter"),
        # Quantization
        "INT8Quantizer": (".export", "INT8Quantizer"),
        "INT8QuantizeConfig": (".export", "INT8QuantizeConfig"),
        # Validation
        "PointValidator": (".validation", "PointValidator"),
        "ValidationConfig": (".validation", "ValidationConfig"),
        # Data utilities
        "DATASETS_DIR": (".data", "DATASETS_DIR"),
        "load_data_config": (".data", "load_data_config"),
        "check_dataset": (".data", "check_dataset"),
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name]
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module 'fomo' has no attribute '{name}'")


__all__ = [
    # Main model
    "FOMO",
    # Results
    "Results",
    "Boxes",
    "Masks",
    "Keypoints",
    "Points",
    "Probs",
    "OBB",
    "Gaze",
    # Assets
    "SAMPLE_IMAGE",
    # Export — TFLite
    "TFLiteExporter",
    # Quantization
    "INT8Quantizer",
    "INT8QuantizeConfig",
    # Validation
    "PointValidator",
    "ValidationConfig",
    # Data
    "DATASETS_DIR",
    "load_data_config",
    "check_dataset",
]
