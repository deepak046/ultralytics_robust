from __future__ import annotations

from typing import Any

import torch

from ultralytics.models.yolo.detect import DetectionValidator


class DriverROIValidator(DetectionValidator):
    """Detection validator extended with lightweight driver orientation/keypoint metrics."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "driverroi"
        self.driver_class = 0
        self.driver_correct = 0
        self.driver_seen = 0
        self.pose_errors = []
        self._driver_aux = None

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize standard detection metrics plus ROI-stage accumulators."""
        super().init_metrics(model)
        self.driver_class = getattr(model, "driver_class", 0)
        self.driver_correct = 0
        self.driver_seen = 0
        self.pose_errors = []
        self._driver_aux = None

    def postprocess(self, preds):
        """Postprocess detection outputs and retain auxiliary ROI predictions for metrics."""
        if isinstance(preds, dict):
            self._driver_aux = preds.get("driver")
            preds = preds["det"]
        else:
            self._driver_aux = None
        preds = preds[0] if isinstance(preds, tuple) else preds
        return super().postprocess(preds)

    def update_metrics(self, preds, batch: dict[str, Any]) -> None:
        """Update standard detection metrics and optional driver-ROI metrics."""
        super().update_metrics(preds, batch)
        if self._driver_aux is None or "boxes" not in self._driver_aux:
            return
        orientation_key = next(
            (k for k in ("orientations", "orientation", "head_orientation", "head_orientations") if k in batch),
            None,
        )
        if orientation_key is None or "keypoints" not in batch:
            return

        batch_idx = batch["batch_idx"].view(-1)
        classes = batch["cls"].view(-1).long()
        roi_valid = batch.get("roi_valid")
        if roi_valid is not None:
            roi_valid = roi_valid.view(-1).to(batch_idx.device)
            if roi_valid.numel() == batch_idx.numel():
                roi_valid = roi_valid > 0
            elif roi_valid.numel() == int(batch["img"].shape[0]):
                roi_valid = (roi_valid > 0)[batch_idx.long()]
            else:
                roi_valid = None
        pred_img_idx = self._driver_aux["batch_indices"].view(-1).tolist()
        pred_cls = self._driver_aux.get("cls_logits", self._driver_aux.get("cls"))
        pred_cls = pred_cls.argmax(1) if pred_cls is not None and pred_cls.ndim > 1 else pred_cls
        pred_kpts = self._driver_aux["kpts"]

        for pred_i, image_idx in enumerate(pred_img_idx):
            gt_mask = (batch_idx == image_idx) & (classes == self.driver_class)
            if roi_valid is not None:
                gt_mask = gt_mask & roi_valid
            if not gt_mask.any():
                continue
            gt_candidates = torch.nonzero(gt_mask, as_tuple=False).view(-1)
            gt_kpts = batch["keypoints"][gt_candidates[0]].to(pred_kpts.device).float()
            gt_orient = batch[orientation_key].view(-1)[gt_candidates[0]].long().to(pred_kpts.device)

            if pred_cls is not None:
                self.driver_seen += 1
                self.driver_correct += int(pred_cls[pred_i].long() == gt_orient)

            vis = gt_kpts[:, 2] > 0 if gt_kpts.shape[-1] >= 3 else torch.ones(gt_kpts.shape[0], dtype=torch.bool, device=gt_kpts.device)
            if vis.any():
                diff = pred_kpts[pred_i, vis, :2] - gt_kpts[vis, :2]
                self.pose_errors.append(torch.sqrt(diff.square().sum(-1).mean()).item())

    def get_stats(self):
        """Return detection metrics plus orientation accuracy and pose RMSE when available."""
        stats = super().get_stats()
        stats["metrics/orientation_acc"] = float(self.driver_correct / self.driver_seen) if self.driver_seen else 0.0
        stats["metrics/pose_rmse"] = float(sum(self.pose_errors) / len(self.pose_errors)) if self.pose_errors else 0.0
        return stats
