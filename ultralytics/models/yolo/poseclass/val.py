from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.models.yolo.classify import ClassificationValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import OKS_SIGMA, ClassifyMetrics, ConfusionMatrix, Metric, ap_per_class, kpt_iou
from ultralytics.utils.plotting import plot_images


class PoseClassValidator(ClassificationValidator):
    """Validator for PoseClass models (classification + keypoints, no boxes)."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "poseclass"
        self.cls_pred = None
        self.cls_target = None
        self.pose_errors = None
        self.pose_tp = None
        self.pose_conf = None
        self.pose_pred_cls = None
        self.pose_target_cls = None
        self.kpt_shape = None
        self.sigma = None
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.metrics = ClassifyMetrics()
        self.pose_metric = Metric()

    def get_desc(self) -> str:
        """Return a formatted string summarizing PoseClass metrics."""
        return ("%22s" + "%11s" * 3) % ("classes", "top1_acc", "top5_acc", "pose_rmse")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize class and pose metric accumulators."""
        self.names = model.names
        self.nc = len(model.names)
        self.pose_metric.nc = self.nc
        self.cls_pred = []
        self.cls_target = []
        self.pose_errors = []
        self.pose_tp = []
        self.pose_conf = []
        self.pose_pred_cls = []
        self.pose_target_cls = []
        self.kpt_shape = self.data["kpt_shape"]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if self.kpt_shape == [17, 3] else np.ones(nkpt) / nkpt
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
        cls_probs = cls_scores.softmax(1) if "cls_logits" in preds else cls_scores
        pred_kpts = preds["kpts"]
        n5 = min(self.nc, 5)
        bs = min(batch["img"].shape[0], cls_scores.shape[0], pred_kpts.shape[0], batch["keypoints"].shape[0])
        for si in range(bs):
            target_cls = batch["cls"][si].long().view(1).cpu()
            target_kpts = batch["keypoints"][si].to(pred_kpts.device).float()
            sample_kpts = pred_kpts[si]
            pred_conf, pred_cls = cls_probs[si].max(dim=0)

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

            # OKS-based pose matching for precision/recall/mAP(P) metrics.
            target_oks = target_kpts if target_kpts.shape[-1] >= 3 else torch.cat(
                [target_kpts, torch.ones_like(target_kpts[..., :1])], dim=-1
            )
            pred_oks = sample_kpts if sample_kpts.shape[-1] >= 3 else torch.cat(
                [sample_kpts, torch.ones_like(sample_kpts[..., :1])], dim=-1
            )
            vis = target_oks[:, 2] > 0
            area_pts = target_oks[vis, :2] if vis.any() else target_oks[:, :2]
            wh = (area_pts.max(0).values - area_pts.min(0).values).clamp(min=1e-4)
            area = (wh[0] * wh[1]).unsqueeze(0)

            oks_iou = kpt_iou(target_oks.unsqueeze(0), pred_oks.unsqueeze(0), sigma=self.sigma, area=area)
            pred_cls_t = torch.tensor([int(pred_cls.item())], device=pred_kpts.device)
            true_cls_t = torch.tensor([int(target_cls.item())], device=pred_kpts.device)
            tp_p = self.match_predictions(pred_cls_t, true_cls_t, oks_iou).cpu().numpy()

            self.pose_tp.append(tp_p)
            self.pose_conf.append(np.array([float(pred_conf.item())], dtype=np.float32))
            self.pose_pred_cls.append(np.array([int(pred_cls.item())], dtype=np.float32))
            self.pose_target_cls.append(np.array([int(target_cls.item())], dtype=np.float32))

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        if RANK == 0:
            gathered_preds = [None] * dist.get_world_size()
            gathered_targets = [None] * dist.get_world_size()
            gathered_pose = [None] * dist.get_world_size()
            gathered_tp = [None] * dist.get_world_size()
            gathered_conf = [None] * dist.get_world_size()
            gathered_pred_cls = [None] * dist.get_world_size()
            gathered_target_cls = [None] * dist.get_world_size()
            dist.gather_object(self.cls_pred, gathered_preds, dst=0)
            dist.gather_object(self.cls_target, gathered_targets, dst=0)
            dist.gather_object(self.pose_errors, gathered_pose, dst=0)
            dist.gather_object(self.pose_tp, gathered_tp, dst=0)
            dist.gather_object(self.pose_conf, gathered_conf, dst=0)
            dist.gather_object(self.pose_pred_cls, gathered_pred_cls, dst=0)
            dist.gather_object(self.pose_target_cls, gathered_target_cls, dst=0)
            self.cls_pred = [pred for rank in gathered_preds for pred in rank]
            self.cls_target = [target for rank in gathered_targets for target in rank]
            self.pose_errors = [err for rank in gathered_pose for err in rank]
            self.pose_tp = [x for rank in gathered_tp for x in rank]
            self.pose_conf = [x for rank in gathered_conf for x in rank]
            self.pose_pred_cls = [x for rank in gathered_pred_cls for x in rank]
            self.pose_target_cls = [x for rank in gathered_target_cls for x in rank]
        elif RANK > 0:
            dist.gather_object(self.cls_pred, None, dst=0)
            dist.gather_object(self.cls_target, None, dst=0)
            dist.gather_object(self.pose_errors, None, dst=0)
            dist.gather_object(self.pose_tp, None, dst=0)
            dist.gather_object(self.pose_conf, None, dst=0)
            dist.gather_object(self.pose_pred_cls, None, dst=0)
            dist.gather_object(self.pose_target_cls, None, dst=0)

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

        if len(self.pose_tp):
            tp_p = np.concatenate(self.pose_tp, 0)
            conf = np.concatenate(self.pose_conf, 0)
            pred_cls = np.concatenate(self.pose_pred_cls, 0)
            target_cls = np.concatenate(self.pose_target_cls, 0)
            pose_results = ap_per_class(
                tp_p,
                conf,
                pred_cls,
                target_cls,
                plot=self.args.plots,
                save_dir=self.save_dir,
                names=self.names,
                on_plot=self.on_plot,
                prefix="Pose",
            )[2:]
            self.pose_metric.update(pose_results)
            p_p, r_p, map50_p, map_p = self.pose_metric.mean_results()
        else:
            p_p, r_p, map50_p, map_p = 0.0, 0.0, 0.0, 0.0

        stats["metrics/precision(P)"] = float(p_p)
        stats["metrics/recall(P)"] = float(r_p)
        stats["metrics/mAP50(P)"] = float(map50_p)
        stats["metrics/mAP50-95(P)"] = float(map_p)
        stats["fitness"] = 0.5 * float(stats.get("metrics/top1_acc", 0.0)) + 0.5 * float(map_p)
        return stats

    def print_results(self) -> None:
        """Print evaluation metrics for PoseClass."""
        pose_rmse = float(sum(self.pose_errors) / max(len(self.pose_errors), 1))
        p_p, r_p, map50_p, map_p = self.pose_metric.mean_results() if len(self.pose_tp) else (0.0, 0.0, 0.0, 0.0)
        pf = "%22s" + "%11.3g" * 7
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5, pose_rmse, p_p, r_p, map50_p, map_p))

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
        """Plot predicted classes and keypoints for a validation batch."""
        cls_scores = preds.get("cls_logits", preds.get("cls"))
        pred_kpts = preds.get("kpts")
        if cls_scores is None or pred_kpts is None:
            return
        probs = cls_scores.softmax(1) if "cls_logits" in preds else cls_scores
        batched_preds = dict(
            img=batch["img"],
            batch_idx=torch.arange(batch["img"].shape[0], device=batch["img"].device),
            cls=torch.argmax(probs, dim=1),
            conf=torch.amax(probs, dim=1),
            keypoints=pred_kpts,
        )
        plot_images(
            labels=batched_preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
