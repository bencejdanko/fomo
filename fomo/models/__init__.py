"""
FOMO model registry and unified factory.

All model families register here via ``__init_subclass__``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseModel
from ..tasks import resolve_task
from ..utils.download import download_weights
from ..utils.logging import ensure_default_logging
from ..utils.serialization import (
    REQUIRED_CHECKPOINT_METADATA_KEYS,
    validate_checkpoint_metadata,
    load_untrusted_torch_file,
)

logger = logging.getLogger(__name__)

_METADATA_CONVERSION_HELP = (
    "FOMO checkpoints must include metadata keys: "
    f"{', '.join(REQUIRED_CHECKPOINT_METADATA_KEYS)}. "
    "inspect the file with `fomo metadata path=...`."
)

# Always-available models (importing triggers __init_subclass__ registration)
from .fomo.model import FOMO  # noqa: E402


def _resolve_weights_path(model_path: str | dict | None) -> str | dict | None:
    """Resolve bare filenames to weights/ directory."""
    if not isinstance(model_path, str):
        return model_path
    path = Path(model_path)
    if path.parent == Path(".") and not model_path.startswith(("./", "../")):
        weights_path = Path("weights") / path.name
        if weights_path.exists():
            return str(weights_path)
        if path.exists():
            return str(path)
        return str(weights_path)
    return model_path


def _unwrap_state_dict(state_dict: dict) -> dict:
    """Extract weights from nested checkpoint formats."""
    if "ema" in state_dict and isinstance(state_dict.get("ema"), dict):
        ema_data = state_dict["ema"]
        return ema_data.get("module", ema_data)
    if "model" in state_dict and isinstance(state_dict.get("model"), dict):
        return state_dict["model"]
    if "state_dict" in state_dict and isinstance(state_dict.get("state_dict"), dict):
        return state_dict["state_dict"]
    return state_dict


def _find_registered_family(family: str):
    for cls in BaseModel._registry:
        if cls.FAMILY == family:
            return cls
    return None


def _matching_model_classes(weights_dict: dict):
    return [cls for cls in BaseModel._registry if cls.can_load(weights_dict)]


def _looks_like_fomo_filename(model_path: str) -> bool:
    return Path(model_path).name.lower().startswith("libre")


def _has_any_fomo_metadata(loaded: object) -> bool:
    if not isinstance(loaded, dict):
        return False
    metadata_keys = set(REQUIRED_CHECKPOINT_METADATA_KEYS) - {"model"}
    return bool(metadata_keys & set(loaded))


# =============================================================================
# FOMO — unified factory function
# =============================================================================


def FOMO(
    model_path: str | dict | None,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
    compute_units: str = "all",
):
    """
    Unified factory that detects model family from weights and returns
    the appropriate model instance.
    """
    ensure_default_logging()
    model_path = _resolve_weights_path(model_path)

    if task is not None and isinstance(model_path, str):
        filename = Path(model_path).name
        for cls in BaseModel._registry:
            if cls.detect_size_from_filename(filename) is not None:
                resolve_task(
                    explicit_task=task,
                    default_task=cls.DEFAULT_TASK,
                    supported_tasks=cls.SUPPORTED_TASKS,
                )
                break

    # Non-PyTorch formats: delegate to inference backends
    if isinstance(model_path, str):
        if model_path.endswith(".onnx"):
            from ..backends.onnx import OnnxBackend
            return OnnxBackend(model_path, nb_classes=nb_classes or 1, device=device, task=task)

        if model_path.endswith(".torchscript"):
            from ..backends.torchscript import TorchScriptBackend
            return TorchScriptBackend(model_path, nb_classes=nb_classes, device=device, task=task)

        if model_path.endswith((".engine", ".tensorrt")):
            from ..backends.tensorrt import TensorRTBackend
            return TensorRTBackend(model_path, nb_classes=nb_classes, device=device, task=task)

        if Path(model_path).is_dir() and (Path(model_path) / "model.xml").exists():
            from ..backends.openvino import OpenVINOBackend
            return OpenVINOBackend(model_path, nb_classes=nb_classes, device=device, task=task)

        if Path(model_path).is_dir() and Path(model_path).suffix == ".mlpackage":
            from ..backends.coreml import CoreMLBackend
            return CoreMLBackend(
                model_path,
                nb_classes=nb_classes or 1,
                device=device,
                compute_units=compute_units,
                task=task,
            )

        if Path(model_path).is_dir():
            ncnn_param = Path(model_path) / "model.ncnn.param"
            ncnn_bin = Path(model_path) / "model.ncnn.bin"
            if ncnn_param.exists() and ncnn_bin.exists():
                from ..backends.ncnn import NcnnBackend
                return NcnnBackend(model_path, nb_classes=nb_classes, device=device, task=task)

        # Download if missing
        if not Path(model_path).exists():
            if size is None:
                for cls in BaseModel._registry:
                    detected = cls.detect_size_from_filename(Path(model_path).name)
                    if detected is not None:
                        size = detected
                        logger.debug("Detected size '%s' from filename", size)
                        break
                if size is None:
                    raise ValueError(
                        f"Model weights file not found: {model_path}\n"
                        f"Cannot auto-download: unable to determine size from filename.\n"
                        f"Please specify size explicitly or provide a valid weights file path."
                    )

            try:
                download_weights(model_path, size)
            except Exception as e:
                logger.warning("Auto-download failed: %s", e)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

    # Load weights once
    loaded = None
    if isinstance(model_path, str):
        try:
            if Path(model_path).suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file as load_safetensors_file
                except ImportError as e:
                    raise ImportError(
                        "Loading safetensors weights requires safetensors. "
                        "Install with: pip install safetensors"
                    ) from e

                loaded = load_safetensors_file(model_path, device="cpu")
            else:
                loaded = load_untrusted_torch_file(
                    model_path,
                    map_location="cpu",
                    context="model inspection",
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e
    elif isinstance(model_path, dict):
        loaded = model_path

    if loaded is not None:
        metadata_errors = validate_checkpoint_metadata(loaded, strict=False)
        has_v1_metadata = not metadata_errors
        has_partial_metadata = _has_any_fomo_metadata(loaded)
        is_legacy_fomo = (
            not has_v1_metadata
            and isinstance(loaded, dict)
            and (has_partial_metadata or (isinstance(model_path, str) and _looks_like_fomo_filename(model_path)))
        )
        if not has_v1_metadata:
            if is_legacy_fomo:
                logger.warning(
                    "FOMO checkpoint metadata is missing or incomplete for %s: %s. "
                    "Loading through the legacy compatibility path. %s",
                    model_path,
                    "; ".join(metadata_errors),
                    _METADATA_CONVERSION_HELP,
                )
            else:
                logger.warning(
                    "FOMO metadata was not found in %s. Loading through the "
                    "legacy architecture-detection path. This appears to be an "
                    "upstream or foreign checkpoint, not a FOMO v1.0 checkpoint. %s",
                    model_path,
                    _METADATA_CONVERSION_HELP,
                )
        weights_dict = _unwrap_state_dict(loaded)
    else:
        metadata_errors = []
        has_v1_metadata = False
        has_partial_metadata = False
        is_legacy_fomo = False
        weights_dict = None

    # Find the right model class.
    matched_cls = None
    metadata_family = (
        loaded.get("model_family")
        if isinstance(loaded, dict)
        and isinstance(loaded.get("model_family"), str)
        else None
    )
    if metadata_family:
        cls = _find_registered_family(metadata_family)
        if cls is not None and (weights_dict is None or cls.can_load(weights_dict)):
            matched_cls = cls

    if matched_cls is None and isinstance(model_path, str):
        filename = Path(model_path).name
        for cls in BaseModel._registry:
            if cls.detect_size_from_filename(filename) and (weights_dict is None or cls.can_load(weights_dict)):
                matched_cls = cls
                break

    if matched_cls is None and weights_dict is not None:
        matching_classes = _matching_model_classes(weights_dict)
        if matching_classes:
            matched_cls = matching_classes[0]

    if matched_cls is None:
        matched_cls = _find_registered_family("fomo")

    if matched_cls is None:
        raise ValueError(
            "Could not detect model architecture. No model family registered for 'fomo'."
        )

    # Auto-detect size
    if size is None:
        if weights_dict is not None:
            size = matched_cls.detect_size(weights_dict)
        if size is None and isinstance(model_path, str):
            size = matched_cls.detect_size_from_filename(Path(model_path).name)
        if size is None:
            raise ValueError(
                f"Could not automatically detect {matched_cls.__name__} model size.\n"
                f"Please specify size explicitly: FOMO('{model_path}', size='s')"
            )
        logger.debug("Auto-detected size: %s", size)

    has_metadata = has_v1_metadata or has_partial_metadata

    # Auto-detect nb_classes.
    if nb_classes is None:
        if has_metadata:
            nb_classes = 1
        else:
            if weights_dict is not None:
                nb_classes = matched_cls.detect_nb_classes(weights_dict)
            if nb_classes is None:
                nb_classes = 1

    checkpoint_task = (
        loaded.get("task")
        if isinstance(loaded, dict) and isinstance(loaded.get("task"), str)
        else None
    )

    filename_task = (
        matched_cls.detect_task_from_filename(Path(model_path).name)
        if isinstance(model_path, str)
        else None
    )
    resolved_task = resolve_task(
        explicit_task=task,
        checkpoint_task=checkpoint_task,
        filename_task=filename_task,
        default_task=matched_cls.DEFAULT_TASK,
        supported_tasks=matched_cls.SUPPORTED_TASKS,
    )

    if has_metadata:
        # Our trainer checkpoint — pass path for metadata handling
        model = matched_cls(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
        )
    else:
        # Pretrained checkpoint — pass extracted state dict
        model = matched_cls(
            model_path=weights_dict,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
        )

    model.model_path = model_path
    return model


__all__ = [
    "FOMO",
    "FOMO",
]
