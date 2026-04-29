from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.nn.tasks import DriverROIModel
from ultralytics.utils import DEFAULT_CFG, RANK


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
        model = DriverROIModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

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
