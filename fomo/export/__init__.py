"""TFLite export and quantization utilities for FOMO.

Two-step workflow::

    from fomo import FOMO
    from fomo.export import INT8Quantizer, INT8QuantizeConfig

    model = FOMO("FOMOm.pt")

    # Step 1 — FP32 TFLite via litert-torch
    fp32_path = model.export(output_path="weights/fomom_fp32.tflite")

    # Step 2 — INT8 quantization
    qt = INT8Quantizer()
    int8_path = qt.quantize(
        fp32_path,
        calibration_data=calib_iter,   # Iterable[dict[str, np.ndarray]]
        config=INT8QuantizeConfig(num_calibration_samples=150),
    )
"""

from .tflite import TFLiteExporter
from .quantize import INT8Quantizer, INT8QuantizeConfig

__all__ = [
    "TFLiteExporter",
    "INT8Quantizer",
    "INT8QuantizeConfig",
]
