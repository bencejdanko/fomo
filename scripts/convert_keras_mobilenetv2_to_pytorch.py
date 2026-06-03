"""
Convert Keras MobileNetV2 pretrained weights → FOMO PyTorch checkpoints.

Produces three files in <repo_root>/weights/:
    FOMOs.pt  (alpha=0.35, 96×96)
    FOMOm.pt  (alpha=0.50, 192×192)
    FOMOl.pt  (alpha=1.00, 224×224)

Each file is a FOMO v1.0 metadata-wrapped checkpoint readable by the
fomo-edge-ai package (model_family='fomo', task='point').

Requirements:
    pip install tensorflow torch numpy

Usage (from repo root):
    python scripts/convert_keras_mobilenetv2_to_pytorch.py
    python scripts/convert_keras_mobilenetv2_to_pytorch.py --sizes s m   # subset
    python scripts/convert_keras_mobilenetv2_to_pytorch.py --out-dir /tmp/weights
"""

from __future__ import annotations

import argparse
import math
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "weights"

# ---------------------------------------------------------------------------
# Per-size configuration
# ---------------------------------------------------------------------------

SEED = 123
VERIFY_ATOL_MAX = 1e-3

CONFIGS: dict[str, dict] = {
    "s": {
        "alpha": 0.35,
        "imgsz": 96,
        "head_in_channels": 96,
        "keras_url": (
            "https://storage.googleapis.com/tensorflow/keras-applications/"
            "mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96_no_top.h5"
        ),
        "out": "FOMOs.pt",
    },
    "m": {
        "alpha": 0.50,
        "imgsz": 192,
        "head_in_channels": 96,
        "keras_url": (
            "https://storage.googleapis.com/tensorflow/keras-applications/"
            "mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_192_no_top.h5"
        ),
        "out": "FOMOm.pt",
    },
    "l": {
        "alpha": 1.00,
        "imgsz": 224,
        "head_in_channels": 192,
        "keras_url": (
            "https://storage.googleapis.com/tensorflow/keras-applications/"
            "mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
        ),
        "out": "FOMOl.pt",
    },
}


# ---------------------------------------------------------------------------
# PyTorch model (mirrors fomo/models/fomo/nn.py exactly)
# ---------------------------------------------------------------------------

def _make_divisible(v: float, divisor: int = 8, min_value=None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _same_pad_1d(input_size: int, kernel: int, stride: int, dilation: int = 1):
    out_size = math.ceil(float(input_size) / float(stride))
    effective_kernel = (kernel - 1) * dilation + 1
    total_pad = max((out_size - 1) * stride + effective_kernel - input_size, 0)
    before = total_pad // 2
    return before, total_pad - before


def _same_pad_2d(input_hw, kernel_hw, stride_hw, dilation_hw=(1, 1)):
    top, bottom = _same_pad_1d(input_hw[0], kernel_hw[0], stride_hw[0], dilation_hw[0])
    left, right  = _same_pad_1d(input_hw[1], kernel_hw[1], stride_hw[1], dilation_hw[1])
    return left, right, top, bottom


class StaticSamePadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, bias=False, input_hw=None):
        super().__init__()
        if input_hw is None:
            raise ValueError("StaticSamePadConv2d requires fixed input_hw")
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.pad = nn.ZeroPad2d(_same_pad_2d(input_hw, kernel_size, stride))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, input_hw):
        super().__init__(
            StaticSamePadConv2d(in_channels, out_channels, kernel_size, stride, input_hw=input_hw),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU6(inplace=True),
        )


class DepthwiseConvBNReLU6(nn.Sequential):
    def __init__(self, channels, kernel_size, stride, input_hw):
        super().__init__(
            StaticSamePadConv2d(channels, channels, kernel_size, stride,
                                groups=channels, input_hw=input_hw),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU6(inplace=True),
        )


class ProjectConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 input_hw, use_residual):
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = use_residual
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_channels, 1, 1, input_hw))
        layers.append(DepthwiseConvBNReLU6(hidden_channels, 3, stride, input_hw))
        layers.append(ProjectConvBN(hidden_channels, out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_residual else out


class FOMOBackbone(nn.Module):
    def __init__(self, size: str):
        super().__init__()
        cfg = CONFIGS[size]
        alpha, imgsz = cfg["alpha"], cfg["imgsz"]
        c0 = _make_divisible(32 * alpha, 8)
        c1 = _make_divisible(16 * alpha, 8)
        c2 = _make_divisible(24 * alpha, 8)
        c3 = _make_divisible(32 * alpha, 8)

        self.conv1 = ConvBNReLU6(3, c0, 3, 2, (imgsz, imgsz))
        hw = math.ceil(imgsz / 2)
        self.expanded_conv = InvertedResidual(c0, c1, 1, 1, (hw, hw), False)
        self.block_1 = InvertedResidual(c1, c2, 2, 6, (hw, hw), False)
        hw = math.ceil(hw / 2)
        self.block_2 = InvertedResidual(c2, c2, 1, 6, (hw, hw), True)
        self.block_3 = InvertedResidual(c2, c3, 2, 6, (hw, hw), False)
        hw = math.ceil(hw / 2)
        self.block_4 = InvertedResidual(c3, c3, 1, 6, (hw, hw), True)
        self.block_5 = InvertedResidual(c3, c3, 1, 6, (hw, hw), True)
        self.block_6_expand = ConvBNReLU6(c3, int(round(c3 * 6)), 1, 1, (hw, hw))

    def forward(self, x):
        x = self.conv1(x)
        x = self.expanded_conv(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return self.block_6_expand(x)


class FOMOModel(nn.Module):
    def __init__(self, size: str = "m", nc: int = 1, head_channels: int | None = None):
        super().__init__()
        self.size = size
        self.nc = nc
        self.head_channels = head_channels if head_channels is not None else nc + 1
        self.imgsz = CONFIGS[size]["imgsz"]
        self.backbone = FOMOBackbone(size)
        self.head = nn.Conv2d(
            CONFIGS[size]["head_in_channels"], self.head_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Weight-transfer helpers  (Keras NHWC → PyTorch NCHW)
# ---------------------------------------------------------------------------

def assign_conv2d(pt_conv: nn.Conv2d, keras_w: np.ndarray):
    """Copy a regular Keras conv kernel (H, W, in, out) → PyTorch (out, in, H, W)."""
    pt_conv.weight.data.copy_(torch.from_numpy(keras_w.transpose(3, 2, 0, 1)))


def assign_depthwise(pt_conv: nn.Conv2d, keras_w: np.ndarray):
    """Copy a depthwise Keras kernel (H, W, channels, 1) → PyTorch (channels, 1, H, W)."""
    pt_conv.weight.data.copy_(torch.from_numpy(keras_w.transpose(2, 3, 0, 1)))


def assign_bn(pt_bn: nn.BatchNorm2d, keras_bn_weights: List[np.ndarray]):
    gamma, beta, mean, var = keras_bn_weights
    pt_bn.weight.data.copy_(torch.from_numpy(gamma))
    pt_bn.bias.data.copy_(torch.from_numpy(beta))
    pt_bn.running_mean.data.copy_(torch.from_numpy(mean))
    pt_bn.running_var.data.copy_(torch.from_numpy(var))


def conv_of(block):
    """Return the Conv2d inside a StaticSamePadConv2d-wrapping block."""
    return block[0].conv


def bn_of(block):
    """Return the BatchNorm2d from a ConvBN* sequential."""
    return block[1]


# ---------------------------------------------------------------------------
# Keras → PyTorch backbone copy
# ---------------------------------------------------------------------------

def copy_keras_to_backbone(keras_model, pt_backbone: FOMOBackbone):
    def kw(layer_name: str):
        return keras_model.get_layer(layer_name).get_weights()

    # Conv1
    assign_conv2d(conv_of(pt_backbone.conv1), kw("Conv1")[0])
    assign_bn(bn_of(pt_backbone.conv1), kw("bn_Conv1"))

    # expanded_conv: depthwise → project  (expand_ratio=1, no expand conv)
    assign_depthwise(conv_of(pt_backbone.expanded_conv.conv[0]),
                     kw("expanded_conv_depthwise")[0])
    assign_bn(bn_of(pt_backbone.expanded_conv.conv[0]),
              kw("expanded_conv_depthwise_BN"))
    assign_conv2d(pt_backbone.expanded_conv.conv[1][0],
                  kw("expanded_conv_project")[0])
    assign_bn(pt_backbone.expanded_conv.conv[1][1],
              kw("expanded_conv_project_BN"))

    # block_1 … block_5: expand → depthwise → project
    for idx in range(1, 6):
        block = getattr(pt_backbone, f"block_{idx}")
        assign_conv2d(conv_of(block.conv[0]), kw(f"block_{idx}_expand")[0])
        assign_bn(bn_of(block.conv[0]),       kw(f"block_{idx}_expand_BN"))
        assign_depthwise(conv_of(block.conv[1]), kw(f"block_{idx}_depthwise")[0])
        assign_bn(bn_of(block.conv[1]),          kw(f"block_{idx}_depthwise_BN"))
        assign_conv2d(block.conv[2][0], kw(f"block_{idx}_project")[0])
        assign_bn(block.conv[2][1],     kw(f"block_{idx}_project_BN"))

    # block_6_expand only (no depthwise / project)
    assign_conv2d(conv_of(pt_backbone.block_6_expand), kw("block_6_expand")[0])
    assign_bn(bn_of(pt_backbone.block_6_expand),       kw("block_6_expand_BN"))


# ---------------------------------------------------------------------------
# Head initialisation (reproducible Kaiming uniform, same as original notebook)
# ---------------------------------------------------------------------------

def initialize_head(model: FOMOModel, seed: int = SEED):
    torch.manual_seed(seed)
    nn.init.kaiming_uniform_(model.head.weight, a=math.sqrt(5))
    if model.head.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.head.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(model.head.bias, -bound, bound)


# ---------------------------------------------------------------------------
# Verification: backbone outputs must match Keras reference within tolerance
# ---------------------------------------------------------------------------

def verify_backbone(size: str, keras_model, pt_model: FOMOModel):
    import tensorflow as tf  # local import — only needed here

    imgsz = CONFIGS[size]["imgsz"]
    keras_probe = tf.keras.Model(
        inputs=keras_model.input,
        outputs=keras_model.get_layer("block_6_expand_relu").output,
    )
    rng = np.random.default_rng(SEED + len(size) + imgsz)
    x_nhwc = rng.normal(size=(1, imgsz, imgsz, 3)).astype(np.float32)
    x_nchw = np.transpose(x_nhwc, (0, 3, 1, 2))

    keras_out = keras_probe(x_nhwc, training=False).numpy()
    torch_out = pt_model.backbone(torch.from_numpy(x_nchw)).detach().cpu().numpy()
    torch_out = np.transpose(torch_out, (0, 2, 3, 1))

    max_abs = float(np.max(np.abs(keras_out - torch_out)))
    mean_abs = float(np.mean(np.abs(keras_out - torch_out)))

    if max_abs > VERIFY_ATOL_MAX:
        raise AssertionError(
            f"Backbone similarity failed for size={size}: "
            f"max_abs={max_abs:.8f}, mean_abs={mean_abs:.8f}, tol={VERIFY_ATOL_MAX}"
        )
    print(
        f"  ✓ Keras similarity size={size}: "
        f"max_abs={max_abs:.8f}, mean_abs={mean_abs:.8f}"
    )


# ---------------------------------------------------------------------------
# Checkpoint saving using fomo's own wrap_fomo_checkpoint
# ---------------------------------------------------------------------------

def save_fomo_checkpoint(path: Path, model: FOMOModel, size: str):
    from fomo.utils.serialization import wrap_fomo_checkpoint

    state_dict = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
    )
    ckpt = wrap_fomo_checkpoint(
        state_dict,
        model_family="fomo",
        size=size,
        task="point",
        nc=model.nc,
        names={0: "person"},
        imgsz=CONFIGS[size]["imgsz"],
    )
    torch.save(ckpt, path)
    print(f"  ✓ Saved {path}")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_if_needed(url: str, path: Path):
    if path.exists():
        return
    print(f"  Downloading {path.name} …")
    urllib.request.urlretrieve(url, path)


# ---------------------------------------------------------------------------
# Per-size conversion
# ---------------------------------------------------------------------------

def convert_one_size(size: str, keras_cache_dir: Path, out_dir: Path) -> Path:
    import tensorflow as tf  # late import so the script is importable without TF

    cfg = CONFIGS[size]
    h5_path = keras_cache_dir / f"mobilenet_v2_{size}_no_top.h5"
    download_if_needed(cfg["keras_url"], h5_path)

    keras_model = tf.keras.applications.MobileNetV2(
        input_shape=(cfg["imgsz"], cfg["imgsz"], 3),
        alpha=cfg["alpha"],
        include_top=False,
        weights=str(h5_path),
    )

    model = FOMOModel(size=size, nc=1, head_channels=2)
    model.eval()

    copy_keras_to_backbone(keras_model, model.backbone)
    initialize_head(model, seed=SEED)
    model.eval()

    verify_backbone(size, keras_model, model)

    out_path = out_dir / cfg["out"]
    save_fomo_checkpoint(out_path, model, size)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Keras MobileNetV2 pretrained weights to FOMO PyTorch checkpoints"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["s", "m", "l"],
        default=["s", "m", "l"],
        help="Which model sizes to convert (default: all three)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for .pt files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--keras-cache",
        type=Path,
        default=None,
        help="Directory to cache downloaded Keras .h5 files (default: <out-dir>/keras_cache)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    keras_cache: Path = args.keras_cache or (out_dir / "keras_cache")
    keras_cache.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"\nConverting sizes: {args.sizes}")
    print(f"Output dir: {out_dir}\n")

    for size in args.sizes:
        print(f"── size={size} ──────────────────────────────────")
        convert_one_size(size, keras_cache, out_dir)

    print("\n✓ Done. Weights written:")
    for size in args.sizes:
        p = out_dir / CONFIGS[size]["out"]
        if p.exists():
            print(f"  {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
