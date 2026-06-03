"""Unit tests for fomo.export — structural checks only.

No model weights, no GPU, no network access required.
These tests verify config validation logic and the INT8Quantizer.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# INT8QuantizeConfig
# ---------------------------------------------------------------------------


def test_int8_quantize_config_defaults():
    from fomo.export import INT8QuantizeConfig

    cfg = INT8QuantizeConfig()
    assert cfg.num_calibration_samples == 150
    assert cfg.signature_key == "serving_default"


def test_int8_quantize_config_invalid_samples():
    from fomo.export import INT8QuantizeConfig

    with pytest.raises(ValueError, match="num_calibration_samples"):
        INT8QuantizeConfig(num_calibration_samples=0)


# ---------------------------------------------------------------------------
# INT8Quantizer — file / data guards (no real litert/aiedge calls)
# ---------------------------------------------------------------------------


def test_int8_quantizer_missing_fp32_file(tmp_path):
    from fomo.export import INT8Quantizer, INT8QuantizeConfig

    qt = INT8Quantizer()
    with pytest.raises(FileNotFoundError):
        qt.quantize(
            tmp_path / "nonexistent_fp32.tflite",
            iter([]),
            INT8QuantizeConfig(),
        )


def test_int8_quantizer_empty_calibration(tmp_path):
    """Empty calibration_data must raise ValueError before calling ai-edge APIs."""
    from unittest import mock
    from fomo.export import INT8Quantizer, INT8QuantizeConfig
    import fomo.export.quantize as qmod

    fp32 = tmp_path / "model_fp32.tflite"
    fp32.write_bytes(b"\x00" * 16)

    # Patch the module-level ai_edge_quantizer objects so no real calls happen
    mock_qt_instance = mock.MagicMock()
    mock_qt_cls = mock.MagicMock(return_value=mock_qt_instance)

    with mock.patch.object(qmod._qt_module, "Quantizer", mock_qt_cls), \
         mock.patch.object(qmod._recipe_module, "static_wi8_ai8", return_value=None):
        qt = INT8Quantizer()
        with pytest.raises(ValueError, match="no samples"):
            qt.quantize(fp32, iter([]), INT8QuantizeConfig())


# ---------------------------------------------------------------------------
# INT8Quantizer — output path derivation
# ---------------------------------------------------------------------------


def test_int8_quantizer_output_path_derivation(tmp_path):
    """auto-derived int8 path replaces _fp32 suffix correctly."""
    from unittest import mock
    from fomo.export import INT8Quantizer, INT8QuantizeConfig
    import fomo.export.quantize as qmod

    fp32 = tmp_path / "fomom_fp32.tflite"
    fp32.write_bytes(b"\x00" * 16)

    mock_qt_instance = mock.MagicMock()
    mock_qt_instance.calibrate.return_value = "calib_result"
    mock_qt_instance.quantize.return_value = mock.MagicMock(
        export_model=mock.MagicMock()
    )
    mock_qt_cls = mock.MagicMock(return_value=mock_qt_instance)

    with mock.patch.object(qmod._qt_module, "Quantizer", mock_qt_cls), \
         mock.patch.object(qmod._recipe_module, "static_wi8_ai8", return_value=None):
        qt = INT8Quantizer()
        result = qt.quantize(fp32, iter([{"args_0": None}]), INT8QuantizeConfig())

    assert result.name == "fomom_int8.tflite"
