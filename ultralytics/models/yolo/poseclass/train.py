from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseClassModel
from ultralytics.utils import DEFAULT_CFG


class PoseClassTrainer(yolo.classify.ClassificationTrainer):
    """Trainer for single-head PoseClass models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "poseclass"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseClassModel:
        """Get pose+classification model with specified configuration and weights."""
        model = PoseClassModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of the PoseClassValidator class for validation."""
        self.loss_names = "pose_loss", "kobj_loss", "cls_loss"
        return yolo.poseclass.PoseClassValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
