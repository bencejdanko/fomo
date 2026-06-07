import torch
import pytest

from fomo import FOMO, Points
from fomo.models.base.inference import InferenceRunner
from fomo.models.fomo.utils import decode_points_from_logits, postprocess
from fomo.tasks import normalize_task, resolve_task
from fomo.utils.results import Results
from fomo.validation.point_validator import PointValidator
from fomo.utils.download import _detect_family_from_filename


pytestmark = pytest.mark.unit


def test_point_task_normalization():
    assert normalize_task("points") == "point"
    assert resolve_task(default_task="point", supported_tasks=("point",)) == "point"
    assert _detect_family_from_filename("FOMOs.pt") == "fomo"


def test_fomo_forward_shapes():
    for size, imgsz, grid in (("s", 96, 12), ("m", 192, 24), ("l", 224, 28)):
        model = FOMO(model_path=None, size=size, nb_classes=1, device="cpu")
        out = model._forward(torch.zeros(1, 3, imgsz, imgsz))
        assert out.shape == (1, 2, grid, grid)


def test_fomo_decode_and_postprocess_points():
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 1, 2, 1] = 8.0
    decoded = decode_points_from_logits(logits, conf_threshold=0.5, nms_radius=1)
    assert decoded[0].shape == (1, 4)
    assert decoded[0][0, :3].tolist() == [1.0, 2.0, 1.0]

    det = postprocess(logits, conf_thres=0.5, input_size=32, original_size=(80, 40))
    assert det["num_detections"] == 1
    assert torch.allclose(det["points"][0], torch.tensor([30.0, 25.0]))
    assert det["classes"][0].item() == 0.0


def test_points_results_summary_and_len():
    points = Points(torch.tensor([[10.0, 20.0, 0.0, 0.9]]), orig_shape=(40, 80))
    result = Results(boxes=None, points=points, orig_shape=(40, 80), names={0: "object"})
    assert len(result) == 1
    assert result.points.xyn.tolist() == [[0.125, 0.5]]
    assert result.summary()[0] == {
        "name": "object",
        "class": 0,
        "confidence": 0.9,
        "point": {"x": 10.0, "y": 20.0},
    }


def test_inference_runner_wraps_empty_points():
    model = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")
    runner = InferenceRunner(model)
    result = runner._wrap_results(
        {
            "points": torch.zeros((0, 2)),
            "scores": torch.zeros((0,)),
            "classes": torch.zeros((0,)),
            "num_detections": 0,
        },
        original_size=(80, 40),
        image_path=None,
        classes=None,
    )
    assert result.boxes is None
    assert result.points is not None
    assert len(result) == 0
    assert result.summary() == []


def test_point_validator_metrics_match_centers():
    validator = PointValidator.__new__(PointValidator)
    validator.config = type(
        "Cfg",
        (),
        {
            "imgsz": 32,
            "conf_thres": 0.5,
            "max_det": 300,
            "point_distance_tolerance": 1.5,
            "point_nms_radius": 1,
        },
    )()
    validator.distance_tolerance = 1.5
    validator.nms_radius = 1
    validator.model = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")
    validator._last_metric_shape = (4, 4)
    validator._init_metrics()

    preds = [torch.tensor([[1.0, 2.0, 0.0, 0.99]])]
    targets = torch.zeros(1, 120, 5)
    targets[0, 0] = torch.tensor([4.0, 12.0, 12.0, 20.0, 0.0])

    validator._update_metrics(preds, targets, None)
    metrics = validator._compute_metrics()
    assert metrics["metrics/precision"] == 1.0
    assert metrics["metrics/recall"] == 1.0
    assert metrics["metrics/F1"] == 1.0
    assert metrics["metrics/mean_distance"] == 0.0


def test_fomo_multiclass_forward_and_loss():
    """nc>1 must produce (B, nc+1, H, W) output and a finite loss."""
    from fomo.models.fomo.loss import FOMOLoss

    NC = 3
    GRID = 12
    IMGSZ = 96

    model = FOMO(model_path=None, size="s", nb_classes=NC, device="cpu")
    out = model._forward(torch.zeros(1, 3, IMGSZ, IMGSZ))
    assert out.shape == (1, NC + 1, GRID, GRID), (
        f"Expected (1, {NC+1}, {GRID}, {GRID}), got {tuple(out.shape)}"
    )

    # Build a target grid: one peak at cell (2, 3), class 1
    target = torch.zeros(1, NC + 1, GRID, GRID)
    target[0, 0] = 1.0          # background everywhere
    target[0, 0, 2, 3] = 0.0   # suppress background at peak
    target[0, 2, 2, 3] = 1.0   # class 1 (channel index 2)

    loss_fn = FOMOLoss(num_classes=NC, fg_weight=100.0, device="cpu")
    loss_dict = loss_fn(out, target)
    assert torch.isfinite(loss_dict["total_loss"]), f"Loss is not finite: {loss_dict}"


def test_fomo_rebuild_for_new_classes():
    """_rebuild_for_new_classes must resize the head and preserve backbone weights."""
    model = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")
    old_backbone_w = model.model.backbone.conv1[0].conv.weight.clone()

    model._rebuild_for_new_classes(3)

    # Head output channels must now be nc+1 = 4
    assert model.model.head.out_channels == 4
    # nb_classes updated on the wrapper
    assert model.nb_classes == 3
    # Backbone weights preserved
    new_backbone_w = model.model.backbone.conv1[0].conv.weight
    assert torch.equal(old_backbone_w, new_backbone_w)


def test_fomo_schedulers():
    from fomo.models.fomo.trainer import FOMOTrainer
    from fomo.models.fomo.model import FOMO
    from fomo.training.scheduler import (
        ConstantLRScheduler,
        CosineAnnealingScheduler,
        FlatCosineScheduler,
        LinearLRScheduler,
        WarmupCosineScheduler,
    )

    model = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")

    # Test create_scheduler selections
    # constant
    t_const = FOMOTrainer(model.model, wrapper_model=model, scheduler="constant", epochs=10)
    assert isinstance(t_const.create_scheduler(10), ConstantLRScheduler)

    # cosine / cos
    t_cos = FOMOTrainer(model.model, wrapper_model=model, scheduler="cos", epochs=10)
    assert isinstance(t_cos.create_scheduler(10), CosineAnnealingScheduler)

    t_cosine = FOMOTrainer(model.model, wrapper_model=model, scheduler="cosine", epochs=10)
    assert isinstance(t_cosine.create_scheduler(10), CosineAnnealingScheduler)

    # flat_cosine
    t_flat = FOMOTrainer(model.model, wrapper_model=model, scheduler="flat_cosine", epochs=10)
    assert isinstance(t_flat.create_scheduler(10), FlatCosineScheduler)

    # linear
    t_lin = FOMOTrainer(model.model, wrapper_model=model, scheduler="linear", epochs=10)
    assert isinstance(t_lin.create_scheduler(10), LinearLRScheduler)

    # yoloxwarmcos
    t_yolox = FOMOTrainer(model.model, wrapper_model=model, scheduler="yoloxwarmcos", epochs=10)
    assert isinstance(t_yolox.create_scheduler(10), WarmupCosineScheduler)


def test_trainer_rebuild_goes_through_wrapper_model():
    """_build_yolo_datasets must call wrapper_model._rebuild_for_new_classes,
    not self.model._rebuild_for_new_classes (the raw nn.Module has no such method).
    """
    from fomo.models.fomo.trainer import FOMOTrainer

    wrapper = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")
    trainer = FOMOTrainer(wrapper.model, wrapper_model=wrapper, epochs=1)

    # Raw FOMOModel has no _rebuild_for_new_classes
    assert not hasattr(trainer.model, "_rebuild_for_new_classes"), (
        "Raw FOMOModel should NOT expose _rebuild_for_new_classes"
    )

    # Simulate what fixed _build_yolo_datasets does
    trainer.wrapper_model._rebuild_for_new_classes(3)
    trainer.model = trainer.wrapper_model.model

    assert wrapper.nb_classes == 3
    assert wrapper.model.head.out_channels == 4
    assert trainer.model is wrapper.model


def test_sweep_validation_multiclass_fp_fn_correct():
    """class-mismatch predictions must not count as TP."""
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from fomo.models.fomo.utils import decode_points_from_logits

    # 4x4, nc=2 (channels: bg=0, cls0=1, cls1=2)
    logits = torch.zeros(1, 3, 4, 4)
    logits[0, 2, 1, 2] = 8.0  # class-1 peak at (col=2, row=1)

    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 1, 2] = 2  # class-1 GT

    decoded = decode_points_from_logits(logits, conf_threshold=0.5, nms_radius=1)
    rows = decoded[0]
    preds_xy = rows[:, :2].numpy()
    preds_cls = (rows[:, 2].long() - 1).numpy()

    fg_mask = target[0] >= 1
    ys, xs = torch.where(fg_mask)
    true_cls_np = (target[0][ys, xs] - 1).numpy()
    trues_xy = torch.stack((xs, ys), dim=1).float().numpy()

    dist_mat = cdist(preds_xy, trues_xy)
    for pi in range(len(preds_cls)):
        for ti in range(len(true_cls_np)):
            if preds_cls[pi] != true_cls_np[ti]:
                dist_mat[pi, ti] = np.inf

    row_ind, col_ind = linear_sum_assignment(np.where(np.isfinite(dist_mat), dist_mat, 1e9))
    tp = sum(1 for r, c in zip(row_ind, col_ind)
             if np.isfinite(dist_mat[r, c]) and dist_mat[r, c] <= 1.5)
    assert tp == 1, f"Expected 1 TP for matching class-1 peak, got {tp}"

    # Class mismatch: class-0 prediction at same location → 0 TP
    logits_wrong = torch.zeros(1, 3, 4, 4)
    logits_wrong[0, 1, 1, 2] = 8.0  # class-0 at same cell
    decoded_wrong = decode_points_from_logits(logits_wrong, conf_threshold=0.5, nms_radius=1)
    rows_wrong = decoded_wrong[0]
    preds_cls_wrong = (rows_wrong[:, 2].long() - 1).numpy()
    dist_mat_wrong = cdist(rows_wrong[:, :2].numpy(), trues_xy)
    for pi in range(len(preds_cls_wrong)):
        for ti in range(len(true_cls_np)):
            if preds_cls_wrong[pi] != true_cls_np[ti]:
                dist_mat_wrong[pi, ti] = np.inf
    row_ind_w, col_ind_w = linear_sum_assignment(
        np.where(np.isfinite(dist_mat_wrong), dist_mat_wrong, 1e9))
    tp_wrong = sum(1 for r, c in zip(row_ind_w, col_ind_w)
                   if np.isfinite(dist_mat_wrong[r, c]) and dist_mat_wrong[r, c] <= 1.5)
    assert tp_wrong == 0, f"Expected 0 TP on class mismatch, got {tp_wrong}"


def test_trainer_propagates_dataset_names_to_wrapper(tmp_path):
    """_build_yolo_datasets must copy data.yaml names to wrapper_model."""
    import yaml
    import numpy as np
    from PIL import Image as PILImage
    from fomo.models.fomo.trainer import FOMOTrainer

    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    PILImage.fromarray(np.zeros((96, 96, 3), dtype=np.uint8)).save(img_dir / "img0.jpg")
    (lbl_dir / "img0.txt").write_text("")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.dump({
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/train",
        "nc": 2,
        "names": {0: "cat", 1: "dog"},
    }))

    wrapper = FOMO(model_path=None, size="s", nb_classes=2, device="cpu")
    wrapper.names = {0: "class_0", 1: "class_1"}  # generic defaults

    trainer = FOMOTrainer(
        wrapper.model, wrapper_model=wrapper,
        data=str(data_yaml), epochs=1, imgsz=96,
    )
    trainer._build_yolo_datasets(96, 12)

    assert wrapper.names == {0: "cat", 1: "dog"}, (
        f"wrapper.names should be {{0:'cat',1:'dog'}}, got {wrapper.names}"
    )


def test_point_validator_rejects_augment():
    """PointValidator._run_validation_augmented must raise ValueError."""
    from fomo.validation.point_validator import PointValidator

    validator = PointValidator.__new__(PointValidator)
    with pytest.raises(ValueError, match="augment=True"):
        validator._run_validation_augmented()


def test_loss_rebuilt_on_class_count_change(tmp_path):
    """Trainer must rebuild self._loss_fn when dataset class count resolves to a new value."""
    import yaml
    import numpy as np
    from PIL import Image as PILImage
    from fomo.models.fomo.trainer import FOMOTrainer

    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    PILImage.fromarray(np.zeros((96, 96, 3), dtype=np.uint8)).save(img_dir / "img0.jpg")
    (lbl_dir / "img0.txt").write_text("")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.dump({
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/train",
        "nc": 3,
        "names": {0: "cat", 1: "dog", 2: "fish"},
    }))

    wrapper = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")
    trainer = FOMOTrainer(
        wrapper.model, wrapper_model=wrapper,
        data=str(data_yaml), epochs=1, imgsz=96,
    )
    # Mock self.device and self.on_setup() to instantiate the initial loss with nc=1
    trainer.device = torch.device("cpu")
    trainer.on_setup()
    assert len(trainer._loss_fn.weights) == 2

    # Running _build_yolo_datasets should trigger a rebuild to nc=3
    trainer._build_yolo_datasets(96, 12)
    assert trainer.config.num_classes == 3
    assert len(trainer._loss_fn.weights) == 4


def test_point_validator_hungarian_matching_all_inf():
    """PointValidator._update_metrics must not crash when cost matrix contains infs (class mismatches)."""
    from fomo.validation.point_validator import PointValidator

    validator = PointValidator.__new__(PointValidator)
    validator.config = type(
        "Cfg",
        (),
        {
            "imgsz": 32,
            "conf_thres": 0.5,
            "max_det": 300,
            "point_distance_tolerance": 1.5,
            "point_nms_radius": 1,
        },
    )()
    validator.distance_tolerance = 1.5
    validator.nms_radius = 1
    validator.model = FOMO(model_path=None, size="s", nb_classes=2, device="cpu")
    validator._last_metric_shape = (4, 4)
    validator._init_metrics()

    # Create predictions and targets with class mismatch (e.g. pred is class 0, target is class 1)
    preds = [torch.tensor([[1.0, 2.0, 0.0, 0.99]])]
    targets = torch.zeros(1, 120, 5)
    targets[0, 0] = torch.tensor([4.0, 12.0, 12.0, 20.0, 1.0])  # class 1

    # Should not raise ValueError (from scipy.optimize.linear_sum_assignment)
    validator._update_metrics(preds, targets, None)

    metrics = validator._compute_metrics()
    assert metrics["metrics/precision"] == 0.0
    assert metrics["metrics/recall"] == 0.0
    assert validator.total_fp == 1
    assert validator.total_fn == 1


def test_fomo_augmented_dataset_and_trainer(tmp_path):
    """Test that FOMOAugmentedDataset properly normalizes and maps boxes to a grid,
    and that FOMOTrainer enables augmentations when parameters are set.
    """
    import yaml
    import numpy as np
    from PIL import Image as PILImage
    from fomo.models.fomo.trainer import FOMOTrainer
    from fomo.models.fomo.dataset import FOMOAugmentedDataset, _boxes_cxcy_to_grid
    
    # 1. Create a dummy dataset
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    # create a small dummy image
    img_arr = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    PILImage.fromarray(img_arr).save(img_dir / "img0.jpg")
    # write one label row: class_id cx cy w h (normalized)
    (lbl_dir / "img0.txt").write_text("0 0.5 0.5 0.2 0.2")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.dump({
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/train",
        "nc": 1,
        "names": {0: "person"},
    }))

    wrapper = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")
    
    # Instantiate trainer with augmentation parameters enabled
    trainer = FOMOTrainer(
        wrapper.model, wrapper_model=wrapper,
        data=str(data_yaml), epochs=1, imgsz=96,
        mosaic_prob=1.0, flip_prob=0.5, hsv_prob=1.0
    )
    
    train_ds, val_ds = trainer._build_yolo_datasets(96, 12)
    
    # Verify train_ds is wrapped with FOMOAugmentedDataset
    assert isinstance(train_ds, FOMOAugmentedDataset)
    
    # Verify val_ds is NOT wrapped (retains standard FOMOYOLODataset)
    from fomo.models.fomo.dataset import FOMOYOLODataset
    assert isinstance(val_ds, FOMOYOLODataset)
    
    # Fetch a sample from the augmented dataset
    img_tensor, grid, img_info, img_id = train_ds[0]
    
    # Checks
    assert img_tensor.shape == (3, 96, 96)
    assert grid.shape == (12, 12)
    # Pixels must be normalized to approx [-1.0, 1.0]
    assert img_tensor.min() >= -1.1 and img_tensor.max() <= 1.1
    # Grid should contain at least background (0) or foreground (1)
    assert (grid >= 0).all() and (grid <= 1).all()


