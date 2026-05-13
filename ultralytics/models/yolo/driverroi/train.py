from __future__ import annotations

from copy import copy
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.nn.tasks import DriverROIModel, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, RANK

DRIVER_ROI_CFG_KEYS = (
    "driver_class",
    "num_orientations",
    "roi_level",
    "roi_size",
    "driver_conf",
    "teacher_forcing",
    "teacher_forcing_epochs",
)


class DriverROITrainer(yolo.detect.DetectionTrainer):
    """Trainer for end-to-end detection plus driver ROI models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "driverroi"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> DriverROIModel:
        """Build the custom detection+ROI model and optionally load weights."""
        model_cfg = self._apply_driverroi_model_overrides(cfg)
        model = DriverROIModel(model_cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def _apply_driverroi_model_overrides(self, cfg: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
        """Merge DriverROI model overrides from training args into the model config."""
        if cfg is None:
            return None

        cfg_dict = deepcopy(cfg if isinstance(cfg, dict) else yaml_model_load(cfg))
        if cfg_dict.get("task") != "driverroi":
            return cfg_dict

        roi_cfg = deepcopy(cfg_dict.get("driver_roi", {}))
        for key in DRIVER_ROI_CFG_KEYS:
            value = getattr(self.args, key, None)
            if value is not None:
                roi_cfg[key] = value

        if getattr(self.args, "dropout", None) is not None:
            roi_cfg["dropout"] = self.args.dropout

        if roi_cfg:
            cfg_dict["driver_roi"] = roi_cfg
        return cfg_dict

    def get_validator(self):
        """Return validator with extra driver-ROI metrics."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "orient_loss", "pose_loss", "kobj_loss"
        return yolo.driverroi.DriverROIValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move optional ROI supervision tensors alongside the detection batch."""
        batch = super().preprocess_batch(batch)
        for key in ("keypoints", "orientations", "orientation", "head_orientation", "head_orientations", "roi_valid"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=self.device.type == "cuda")
        return batch
