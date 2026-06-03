"""TFLite export using Google LiteRT (litert-torch).

Install with:
    pip install litert-torch
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import litert_torch
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TFLiteExporter:
    """Convert a PyTorch model to a TFLite FP32 flatbuffer via litert-torch.

    Example::

        from fomo import FOMO
        from fomo.export import TFLiteExporter

        model = FOMO("FOMOm.pt")
        exporter = TFLiteExporter()
        fp32_path = exporter(model.model, model.input_size, output_path="fomom_fp32.tflite")

    Or via the model facade::

        fp32_path = model.export(output_path="fomom_fp32.tflite")
    """

    def __call__(
        self,
        model: nn.Module,
        input_size: int,
        *,
        output_path: Optional[Union[str, Path]] = None,
        batch: int = 1,
    ) -> Path:
        """Convert *model* to a TFLite FP32 flatbuffer.

        Args:
            model: PyTorch ``nn.Module`` to convert. Will be moved to CPU and
                set to eval mode internally; the caller's model is unaffected.
            input_size: Square spatial resolution in pixels (e.g. ``192``).
            output_path: Destination ``.tflite`` file path. When ``None``,
                defaults to ``model_fp32.tflite`` in the current directory.
            batch: Batch size of the dummy input tensor used for tracing
                (default ``1``).

        Returns:
            :class:`~pathlib.Path` to the written flatbuffer.
        """
        out = Path(output_path) if output_path is not None else Path("model_fp32.tflite")
        out.parent.mkdir(parents=True, exist_ok=True)

        nn_model = model.cpu().eval()
        sample_input = torch.randn(batch, 3, input_size, input_size)

        logger.info(
            "Exporting to TFLite FP32: %s (input %dx%d, batch=%d)",
            out,
            input_size,
            input_size,
            batch,
        )

        edge_model = litert_torch.convert(nn_model, (sample_input,))
        edge_model.export(str(out))

        logger.info("TFLite FP32 export complete: %s", out)
        return out.resolve()
