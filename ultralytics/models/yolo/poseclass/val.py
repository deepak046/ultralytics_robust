from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from ultralytics.models.yolo.classify import ClassificationValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class PoseClassValidator(ClassificationValidator):
    """Validator for PoseClass models (classification + keypoints, no boxes)."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "poseclass"
        self.cls_pred = None
        self.cls_target = None
        self.pose_errors = None
        self.metrics = ClassifyMetrics()

    def get_desc(self) -> str:
        """Return a formatted string summarizing PoseClass metrics."""
        return ("%22s" + "%11s" * 3) % ("classes", "top1_acc", "top5_acc", "pose_rmse")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize class and pose metric accumulators."""
        self.names = model.names
        self.nc = len(model.names)
        self.cls_pred = []
        self.cls_target = []
        self.pose_errors = []
        self.confusion_matrix = ConfusionMatrix(names=model.names)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by moving tensors to target device and normalizing images."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def postprocess(self, preds: Any) -> dict[str, torch.Tensor]:
        """Normalize model outputs to a standard dict payload."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if not isinstance(preds, dict):
            raise TypeError("PoseClass model is expected to return a dict with cls logits/probs and keypoints.")
        return preds

    def update_metrics(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        """Update top-k classification and keypoint RMSE metrics."""
        cls_scores = preds.get("cls_logits", preds.get("cls"))
        if cls_scores is None:
            raise KeyError("PoseClass predictions must contain either 'cls_logits' or 'cls'.")
        pred_kpts = preds["kpts"]
        n5 = min(self.nc, 5)
        bs = min(batch["img"].shape[0], cls_scores.shape[0], pred_kpts.shape[0], batch["keypoints"].shape[0])
        for si in range(bs):
            target_cls = batch["cls"][si].long().view(1).cpu()
            target_kpts = batch["keypoints"][si].to(pred_kpts.device).float()
            sample_kpts = pred_kpts[si]

            self.cls_pred.append(cls_scores[si : si + 1].argsort(1, descending=True)[:, :n5].int().cpu())
            self.cls_target.append(target_cls.int())

            if target_kpts.shape[-1] == 3:
                visible = target_kpts[:, 2] > 0
            else:
                visible = torch.ones(target_kpts.shape[0], dtype=torch.bool, device=target_kpts.device)
            if visible.any():
                diff = sample_kpts[visible, :2] - target_kpts[visible, :2]
                rmse = torch.sqrt((diff.square().sum(-1)).mean()).item()
                self.pose_errors.append(rmse)

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        if RANK == 0:
            gathered_preds = [None] * dist.get_world_size()
            gathered_targets = [None] * dist.get_world_size()
            gathered_pose = [None] * dist.get_world_size()
            dist.gather_object(self.cls_pred, gathered_preds, dst=0)
            dist.gather_object(self.cls_target, gathered_targets, dst=0)
            dist.gather_object(self.pose_errors, gathered_pose, dst=0)
            self.cls_pred = [pred for rank in gathered_preds for pred in rank]
            self.cls_target = [target for rank in gathered_targets for target in rank]
            self.pose_errors = [err for rank in gathered_pose for err in rank]
        elif RANK > 0:
            dist.gather_object(self.cls_pred, None, dst=0)
            dist.gather_object(self.cls_target, None, dst=0)
            dist.gather_object(self.pose_errors, None, dst=0)

    def finalize_metrics(self) -> None:
        """Finalize confusion matrix and speed bookkeeping."""
        self.confusion_matrix.process_cls_preds(self.cls_pred, self.cls_target)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self) -> dict[str, float]:
        """Calculate and return PoseClass metrics."""
        self.metrics.process(self.cls_target, self.cls_pred)
        stats = self.metrics.results_dict
        pose_rmse = float(sum(self.pose_errors) / max(len(self.pose_errors), 1))
        stats["metrics/pose_rmse"] = pose_rmse
        stats["fitness"] = 0.7 * float(stats.get("metrics/top1_acc", 0.0)) + 0.3 * (1.0 / (1.0 + pose_rmse))
        return stats

    def print_results(self) -> None:
        """Print evaluation metrics for PoseClass."""
        pose_rmse = float(sum(self.pose_errors) / max(len(self.pose_errors), 1))
        pf = "%22s" + "%11.3g" * 3
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5, pose_rmse))

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation image samples."""
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])
        plot_images(
            labels=batch,
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch: dict[str, Any], preds: dict[str, torch.Tensor], ni: int) -> None:
        """Plot predicted classes for a validation batch."""
        cls_scores = preds.get("cls_logits", preds.get("cls"))
        if cls_scores is None:
            return
        batched_preds = dict(
            img=batch["img"],
            batch_idx=torch.arange(batch["img"].shape[0], device=batch["img"].device),
            cls=torch.argmax(cls_scores, dim=1),
            conf=torch.amax(cls_scores.softmax(1), dim=1),
        )
        plot_images(
            labels=batched_preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
