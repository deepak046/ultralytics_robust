from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import PoseClassDataset, build_dataloader
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseClassModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import torch_distributed_zero_first


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

    def set_model_attributes(self):
        """Attach dataset/task metadata and args to model for loss/metrics."""
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve dataset and ensure poseclass keypoint shape exists."""
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in {self.args.data}. PoseClass requires keypoint shape metadata.")
        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build PoseClass dataset from YOLO-style pose labels."""
        return PoseClassDataset(root=img_path, args=self.args, data=self.data, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return dataloader for poseclass dataset."""
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=rank, drop_last=self.args.compile)

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move poseclass batch tensors to device."""
        batch = super().preprocess_batch(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def label_loss_items(self, loss_items: torch.Tensor | None = None, prefix: str = "train"):
        """Return labeled poseclass loss dict for logging."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        if loss_items.ndim == 0:
            vals = [round(float(loss_items), 5)]
        else:
            vals = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, vals))
