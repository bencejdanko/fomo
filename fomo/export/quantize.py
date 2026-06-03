"""Post-export quantization for TFLite flatbuffers.

This module provides an :class:`INT8Quantizer` that produces INT8 static-quantized
flatbuffers using Google's ai-edge-quantizer.

Install with:
    pip install ai-edge-quantizer-nightly

Example::

    from fomo.export import INT8Quantizer, INT8QuantizeConfig

    config = INT8QuantizeConfig(num_calibration_samples=150)
    qt = INT8Quantizer()
    int8_path = qt.quantize(
        "fomom_fp32.tflite",
        calibration_data=my_iter,   # Iterable[dict[str, np.ndarray]]
        config=config,
        output_path="fomom_int8.tflite",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from ai_edge_quantizer import quantizer as _qt_module
from ai_edge_quantizer import recipe as _recipe_module

logger = logging.getLogger(__name__)


@dataclass
class INT8QuantizeConfig:
    """Configuration for INT8 static quantization.

    Attributes:
        num_calibration_samples: Maximum number of calibration samples
            consumed from the iterator (default ``150``).
        signature_key: Signature key used when building the calibration dict
            passed to the quantizer (default ``"serving_default"``).
    """

    num_calibration_samples: int = 150
    signature_key: str = "serving_default"

    def __post_init__(self) -> None:
        if self.num_calibration_samples <= 0:
            raise ValueError(
                f"num_calibration_samples must be a positive integer, "
                f"got {self.num_calibration_samples!r}"
            )


class INT8Quantizer:
    """INT8 static quantizer.

    Uses ``ai-edge-quantizer-nightly`` to apply the ``static_wi8_ai8``
    recipe (weights INT8, activations INT8) and produces a flatbuffer.
    """

    def quantize(
        self,
        fp32_tflite: Union[str, Path],
        calibration_data: Iterable[dict],
        config: INT8QuantizeConfig,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Quantize *fp32_tflite* to INT8.

        Args:
            fp32_tflite: Path to a FP32 TFLite flatbuffer.
            calibration_data: Representative inputs keyed by the flatbuffer
                input signature name (usually ``"args_0"``).
            config: Quantization settings (sample count, …).
            output_path: Output path.  Defaults to ``<stem>_int8.tflite``
                next to *fp32_tflite*.

        Returns:
            Absolute :class:`~pathlib.Path` of the INT8 flatbuffer.

        Raises:
            FileNotFoundError: If *fp32_tflite* does not exist.
            ValueError: If *calibration_data* yields no samples.
        """
        src = Path(fp32_tflite)
        if not src.exists():
            raise FileNotFoundError(f"FP32 TFLite flatbuffer not found: {src}")

        if output_path is None:
            stem = src.stem
            if stem.endswith("_fp32"):
                stem = stem[: -len("_fp32")]
            out = src.with_name(f"{stem}_int8.tflite")
        else:
            out = Path(output_path)

        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Quantizing %s → %s (samples=%d)",
            src,
            out,
            config.num_calibration_samples,
        )

        quant_recipe = _recipe_module.static_wi8_ai8()

        qt = _qt_module.Quantizer(str(src))
        qt.load_quantization_recipe(quant_recipe)

        samples = []
        for i, sample in enumerate(calibration_data):
            if i >= config.num_calibration_samples:
                break
            samples.append(sample)

        if not samples:
            raise ValueError(
                "calibration_data yielded no samples. "
                "Provide at least one representative input."
            )

        calibration_dict = {config.signature_key: samples}

        logger.info("Calibrating with %d samples…", len(samples))
        calib_result = qt.calibrate(calibration_dict)

        logger.info("Applying quantization…")
        quant_result = qt.quantize(calib_result)

        quant_result.export_model(str(out))

        logger.info("INT8 TFLite export complete: %s", out)
        return out.resolve()
