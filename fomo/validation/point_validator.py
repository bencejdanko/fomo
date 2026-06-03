"""Point-task validator."""

from __future__ import annotations

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment as _scipy_lsa

from .base import BaseValidator
from .config import ValidationConfig

if TYPE_CHECKING:
    from fomo.models.base import BaseModel

logger = logging.getLogger(__name__)


def val_collate_fn(batch):
    """Collate validation batch: stack preprocessed images and padded targets."""
    if len(batch[0]) == 5:
        imgs, targets, img_infos, img_ids, _segments = zip(*batch)
    else:
        imgs, targets, img_infos, img_ids = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs))
    targets = torch.from_numpy(np.stack(targets))
    return imgs, targets, img_infos, img_ids


class PointValidator(BaseValidator):
    """Validator for point-localization models.

    Computes precision, recall, F1, and mean matched distance using one-to-one
    point matching. Model families own prediction decoding and target-space
    conversion; this class owns only matching and metric aggregation.
    """

    task = "point"

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)
        self.distance_tolerance = float(getattr(self.config, "point_distance_tolerance", 1.5))
        self.nms_radius = int(getattr(self.config, "point_nms_radius", 1))
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_distance = 0.0
        self._last_metric_shape = (self.config.imgsz, self.config.imgsz)
        self.nc = model.nb_classes
        self.class_names: Optional[List[str]] = None
        self.val_preproc = None

    def _resolve_imgsz(self) -> int:
        """Return the validation image size, falling back to the model native size."""
        if self.config.imgsz is not None:
            return int(self.config.imgsz)

        get_input_size = getattr(self.model, "_get_input_size", None)
        if callable(get_input_size):
            return int(get_input_size())

        return 640

    def _setup_dataloader(self) -> DataLoader:
        """Create validation dataloader from config."""
        from fomo.data import load_data_config, get_img_files, img2label_paths
        from fomo.data.dataset import YOLODataset

        actual_imgsz = self._resolve_imgsz()
        self.config.imgsz = actual_imgsz
        self._actual_imgsz = actual_imgsz
        img_size = (actual_imgsz, actual_imgsz)

        img_files = None
        label_files = None
        split_name = self.config.split
        data_cfg = None

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
            data_dir = data_cfg["root"]
            self.nc = data_cfg.get("nc", self.nc)

            names = data_cfg.get("names", None)
            if isinstance(names, dict):
                self.class_names = [names[i] for i in range(len(names))]
            else:
                self.class_names = names

            img_files_key = f"{self.config.split}_img_files"
            label_files_key = f"{self.config.split}_label_files"

            if img_files_key in data_cfg:
                img_files = data_cfg[img_files_key]
                label_files = data_cfg.get(label_files_key)
            else:
                split_path_str = data_cfg.get(
                    self.config.split, f"images/{self.config.split}"
                )

                if str(split_path_str).endswith(".txt"):
                    txt_path = Path(data_cfg["path"]) / split_path_str
                    if txt_path.exists():
                        try:
                            img_files = get_img_files(txt_path)
                            label_files = img2label_paths(img_files)
                        except (FileNotFoundError, ValueError):
                            pass
                else:
                    full_split_path = Path(data_cfg["path"]) / split_path_str

                    if full_split_path.exists():
                        img_files_list = []
                        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                            img_files_list.extend(full_split_path.glob(ext))
                            img_files_list.extend(full_split_path.glob(ext.upper()))

                        if img_files_list:
                            img_files = sorted(set(img_files_list))
                            label_files = img2label_paths(img_files)
        else:
            data_dir = self.config.data_dir
            self.class_names = None

        self.val_preproc = self.model._get_val_preprocessor(img_size=actual_imgsz)

        if img_files is not None:
            dataset = YOLODataset(
                img_files=img_files,
                label_files=label_files,
                img_size=img_size,
                preproc=self.val_preproc,
            )
        else:
            dataset = YOLODataset(
                data_dir=str(Path(data_dir)),
                split=split_name,
                img_size=img_size,
                preproc=self.val_preproc,
            )

        use_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        nw = self.config.num_workers

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=use_cuda,
            prefetch_factor=4 if nw > 0 else None,
            persistent_workers=nw > 0,
            collate_fn=val_collate_fn,
            drop_last=False,
        )

        return dataloader

    def _preprocess_batch(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        images, targets, img_info, img_ids = batch

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        images = images.float()

        if getattr(self.val_preproc, "custom_normalization", False):
            pass
        elif self.val_preproc.normalize:
            if images.max() > 1.0:
                images = images / 255.0
        else:
            if images.max() <= 1.0:
                images = images * 255.0

        if images.dim() == 3:
            images = images.unsqueeze(0)

        return images, targets, img_info, img_ids

    def _init_metrics(self) -> None:
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_distance = 0.0

    def _postprocess_predictions(self, preds: Any, batch: Any) -> List[torch.Tensor]:
        decoder = getattr(self.model, "_decode_point_predictions", None)
        if not callable(decoder):
            raise NotImplementedError(
                f"{self.model.__class__.__name__} must implement "
                "_decode_point_predictions(...) for point validation."
            )
        metric_shape_fn = getattr(self.model, "_point_metric_shape", None)
        if callable(metric_shape_fn):
            self._last_metric_shape = metric_shape_fn(preds, self.config.imgsz)
        decoded = decoder(
            preds,
            conf_thres=self.config.conf_thres,
            max_det=self.config.max_det,
            nms_radius=self.nms_radius,
        )
        return [row.detach().cpu() if isinstance(row, torch.Tensor) else torch.as_tensor(row) for row in decoded]

    def _target_points_from_boxes(
        self, target: torch.Tensor, metric_h: int, metric_w: int
    ) -> torch.Tensor:
        target_encoder = getattr(self.model, "_point_targets_from_boxes", None)
        if callable(target_encoder):
            return target_encoder(
                target,
                metric_shape=(metric_h, metric_w),
                imgsz=self.config.imgsz,
            ).float()
        valid = target[:, 2] > target[:, 0]
        target = target[valid]
        if target.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        cx = (target[:, 0] + target[:, 2]) * 0.5 * (metric_w / self.config.imgsz)
        cy = (target[:, 1] + target[:, 3]) * 0.5 * (metric_h / self.config.imgsz)
        cls = target[:, 4]
        return torch.stack((cx, cy, cls), dim=1).float()

    def _update_metrics(
        self, preds: List[torch.Tensor], targets: torch.Tensor, img_info: Any, img_ids: Any = None
    ) -> None:
        if not preds:
            return
        metric_h, metric_w = self._last_metric_shape

        for b, pred_rows in enumerate(preds):
            pred_xy = pred_rows[:, :2].float() if len(pred_rows) else torch.zeros((0, 2))
            pred_cls = pred_rows[:, 2].float() if len(pred_rows) else torch.zeros((0,))
            true_rows = self._target_points_from_boxes(targets[b].cpu(), metric_h, metric_w)
            true_xy = true_rows[:, :2]
            true_cls = true_rows[:, 2]

            if len(pred_xy) == 0 and len(true_xy) == 0:
                continue
            if len(pred_xy) == 0:
                self.total_fn += len(true_xy)
                continue
            if len(true_xy) == 0:
                self.total_fp += len(pred_xy)
                continue

            dist = torch.cdist(pred_xy, true_xy)
            class_mismatch = pred_cls[:, None] != true_cls[None, :]
            dist = dist.masked_fill(class_mismatch, float("inf"))
            finite = torch.isfinite(dist)
            if not finite.any():
                self.total_fp += len(pred_xy)
                self.total_fn += len(true_xy)
                continue

            dist_finite = torch.where(finite, dist, torch.tensor(1e9, device=dist.device))
            dist_np = dist_finite.detach().cpu().numpy()
            rows, cols = _scipy_lsa(dist_np)
            matched_preds = set()
            matched_trues = set()
            for r, c in zip(rows, cols):
                value = float(dist[r, c])
                if value <= self.distance_tolerance:
                    self.total_tp += 1
                    self.total_distance += value
                    matched_preds.add(r)
                    matched_trues.add(c)

            self.total_fp += len(pred_xy) - len(matched_preds)
            self.total_fn += len(true_xy) - len(matched_trues)

    def _run_validation_augmented(self) -> None:
        """Point-task validators do not support box-level TTA."""
        raise ValueError(
            "PointValidator does not support augment=True. "
            "Point-localization TTA is not implemented; pass augment=False."
        )

    def _compute_metrics(self) -> Dict[str, float]:
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) else 0.0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        mean_distance = self.total_distance / self.total_tp if self.total_tp else 0.0
        return {
            "metrics/precision": precision,
            "metrics/recall": recall,
            "metrics/F1": f1,
            "metrics/mean_distance": mean_distance,
            "metrics/TP": float(self.total_tp),
            "metrics/FP": float(self.total_fp),
            "metrics/FN": float(self.total_fn),
        }
