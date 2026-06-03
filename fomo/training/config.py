"""Training configuration dataclasses for FOMO."""

import logging
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


def load_train_cfg(path) -> dict:
    """Load a training-config yaml as a dict suitable for ``model.train(**out)``.

    Args:
        path: Path to a yaml file containing training parameters.

    Returns:
        Dict of training kwargs parsed from the yaml.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
        ValueError: If the yaml content is not a mapping.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Training cfg yaml not found: {yaml_path}")
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Training cfg {yaml_path} must be a yaml mapping, "
            f"got {type(raw).__name__}."
        )
    return raw


@dataclass(kw_only=True)
class TrainConfig:
    """Base training configuration. Subclasses override defaults per model family."""

    # Model
    size: str = "s"
    num_classes: int = 80

    # Data
    data: Optional[str] = None
    data_dir: Optional[str] = None
    imgsz: int = 640

    # Training
    epochs: int = 300
    # Global batch size. Under multi-GPU DDP the per-rank batch is
    # ``batch // world_size``.
    # Set to -1 to enable automatic selection: the trainer probes GPU memory
    # at small batch sizes, fits a linear model, and picks the largest batch
    # that fits within 70 % of total VRAM.
    batch: int = 16
    # Single device or multi-device spec. Accepts:
    #   - "auto" / "" → auto-pick (cuda → mps → cpu)
    #   - "cpu", "mps", "0", "cuda:0", 0 → single device
    #   - [0, 1] or "0,1" → multi-GPU, requires torchrun launch
    device: Union[str, int, List[int]] = "auto"
    # SyncBatchNorm across ranks under DDP. Off here; per-family configs
    # override (yolo9 defaults True per upstream MultimediaTechLab). No-op
    # when not distributed.
    sync_bn: bool = False

    # Optimizer
    optimizer: str = "sgd"
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    nesterov: bool = True

    # Scheduler
    scheduler: str = "yoloxwarmcos"
    warmup_epochs: int = 5
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.05

    # Augmentation
    mosaic_prob: float = 1.0
    mixup_prob: float = 1.0
    hsv_prob: float = 1.0
    flip_prob: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.1, 2.0)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 2.0

    # Training features
    ema: bool = True
    ema_decay: float = 0.9998
    amp: bool = True
    # Nominal (effective) batch size for gradient accumulation. When set, the
    # trainer accumulates ``round(nbs / batch)`` micro-batches per optimizer
    # step so the effective batch size is ``nbs``.
    # Left as None (the default), gradient accumulation is disabled and
    # training is unchanged.
    nbs: Optional[int] = None

    # Checkpointing / output
    project: str = "runs/train"
    name: str = "exp"
    exist_ok: bool = False
    save_period: int = 10
    eval_interval: int = 10

    # System
    workers: int = 4
    patience: int = 50
    resume: bool = False
    log_interval: int = 10
    seed: int = 0
    allow_download_scripts: bool = False

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Construct config, warning on unknown keys."""
        valid = {f.name for f in fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown training config keys (ignored): {sorted(unknown)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dict with tuples converted to lists for YAML/checkpoint."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def to_yaml(self, path) -> None:
        """Serialize config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass(kw_only=True)
class FOMOConfig(TrainConfig):
    """FOMO point-localizer training defaults.

    - Adam optimizer, lr=3e-4
    - Cosine learning rate scheduler (cos, min_lr_ratio=0.05)
    - Weighted CrossEntropy: background=1.0, foreground=fg_weight (100×)
    - No mosaic/mixup/HSV/flip augmentation (plain stretch resize + [-1,1] norm)
    - Validate every epoch (best model selected by validation F1 score)
    - EMA and AMP disabled for simplicity in v1

    Pass a standard YOLO ``data.yaml`` path via ``data=`` to ``model.train()``.
    Training is gated experimental; pass ``allow_experimental=True`` to enable.
    """

    optimizer: str = "adam"
    lr0: float = 3e-4
    weight_decay: float = 0.0  # Adam without wd, matching reference

    # Foreground class weight in weighted CrossEntropyLoss
    fg_weight: float = 100.0

    # Standard LR schedule
    scheduler: str = "cos"
    warmup_epochs: int = 0
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.05

    # No augmentation — plain resize only
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0

    # FOMO doesn't use EMA or AMP in v1
    ema: bool = False
    amp: bool = False

    # Training schedule
    epochs: int = 40
    batch: int = 32
    eval_interval: int = 1  # validate every epoch for plateau stepping

    # Validation sweep parameters (mirror reference evaluate_sweep)
    conf_thresholds: Tuple[float, ...] = (0.25, 0.35, 0.50, 0.65, 0.80, 0.90)
    nms_radii: Tuple[int, ...] = (1, 2)
    distance_tolerance: float = 1.5

    name: str = "fomo_exp"
