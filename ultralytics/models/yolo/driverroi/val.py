from __future__ import annotations

from typing import Any

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou


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
        self.det_total_preds = 0
        self.det_images_with_preds = 0
        self.det_conf_sum = 0.0
        self.det_conf_count = 0
        self.det_class_hist = None
        self.det_debug_examples = []
        self.det_best_iou_sum = 0.0
        self.det_best_iou_count = 0
        self.det_gt_hit_count = 0
        self.det_gt_count = 0

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize standard detection metrics plus ROI-stage accumulators."""
        super().init_metrics(model)
        self.driver_class = getattr(model, "driver_class", 0)
        self.driver_correct = 0
        self.driver_seen = 0
        self.pose_errors = []
        self._driver_aux = None
        self.det_total_preds = 0
        self.det_images_with_preds = 0
        self.det_conf_sum = 0.0
        self.det_conf_count = 0
        self.det_class_hist = torch.zeros(self.nc, dtype=torch.long)
        self.det_debug_examples = []
        self.det_best_iou_sum = 0.0
        self.det_best_iou_count = 0
        self.det_gt_hit_count = 0
        self.det_gt_count = 0

    def postprocess(self, preds):
        """Postprocess detection outputs and retain auxiliary ROI predictions for metrics."""
        if isinstance(preds, dict):
            self._driver_aux = preds.get("driver")
            preds = preds["det"]
        else:
            self._driver_aux = None
        return super().postprocess(preds)

    def update_metrics(self, preds, batch: dict[str, Any]) -> None:
        """Update standard detection metrics and optional driver-ROI metrics."""
        super().update_metrics(preds, batch)
        self._update_detection_debug(preds, batch)
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
        stats["metrics/det_per_image"] = float(self.det_total_preds / max(self.seen, 1))
        stats["metrics/det_zero_frac"] = float(1.0 - (self.det_images_with_preds / max(self.seen, 1)))
        stats["metrics/det_mean_conf"] = float(self.det_conf_sum / max(self.det_conf_count, 1))
        stats["metrics/det_best_iou"] = float(self.det_best_iou_sum / max(self.det_best_iou_count, 1))
        stats["metrics/det_gt_iou50_hit_rate"] = float(self.det_gt_hit_count / max(self.det_gt_count, 1))
        return stats

    def print_results(self) -> None:
        """Print standard detection metrics plus DriverROI-specific debug summaries."""
        super().print_results()
        mean_conf = self.det_conf_sum / max(self.det_conf_count, 1)
        zero_frac = 1.0 - (self.det_images_with_preds / max(self.seen, 1))
        LOGGER.info(
            "DriverROI val debug: "
            f"det/img={self.det_total_preds / max(self.seen, 1):.3f}, "
            f"zero_det_frac={zero_frac:.3f}, "
            f"mean_conf={mean_conf:.4f}, "
            f"best_iou={self.det_best_iou_sum / max(self.det_best_iou_count, 1):.4f}, "
            f"gt_iou50_hit_rate={self.det_gt_hit_count / max(self.det_gt_count, 1):.4f}, "
            f"orientation_acc={self.driver_correct / max(self.driver_seen, 1):.4f}, "
            f"pose_rmse={sum(self.pose_errors) / max(len(self.pose_errors), 1):.4f}"
        )
        if self.det_class_hist is not None:
            class_hist = {self.names[i]: int(v) for i, v in enumerate(self.det_class_hist.tolist()) if v > 0}
            LOGGER.info(f"DriverROI val debug class_hist: {class_hist or 'no predictions'}")
        for idx, sample in enumerate(self.det_debug_examples, start=1):
            LOGGER.info(f"DriverROI val debug sample {idx}: {sample}")

    def _update_detection_debug(self, preds, batch: dict[str, Any]) -> None:
        """Collect lightweight per-epoch detection diagnostics."""
        for si, pred in enumerate(preds):
            pred_cls = pred["cls"]
            pred_conf = pred["conf"]
            pred_boxes = pred["bboxes"]
            pred_count = int(pred_cls.shape[0])
            self.det_total_preds += pred_count
            self.det_images_with_preds += int(pred_count > 0)
            pbatch = self._prepare_batch(si, batch)
            imgsz = pbatch["imgsz"]
            clip_box = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device=pred_boxes.device, dtype=pred_boxes.dtype)
            pred_boxes_clipped = pred_boxes.clone()
            pred_boxes_clipped[:, [0, 2]] = pred_boxes_clipped[:, [0, 2]].clamp_(0, clip_box[0])
            pred_boxes_clipped[:, [1, 3]] = pred_boxes_clipped[:, [1, 3]].clamp_(0, clip_box[1])

            if pred_count > 0:
                self.det_conf_sum += float(pred_conf.sum().item())
                self.det_conf_count += pred_count
                if self.det_class_hist is not None:
                    hist = torch.bincount(pred_cls.long().cpu(), minlength=self.nc)
                    self.det_class_hist += hist[: self.nc]

            gt_boxes_all = pbatch["bboxes"]
            gt_cls_all = pbatch["cls"]
            gt_count = int(gt_cls_all.shape[0])
            self.det_gt_count += gt_count
            if pred_count > 0 and gt_count > 0:
                ious = box_iou(gt_boxes_all, pred_boxes_clipped)
                best_iou_per_gt = ious.max(dim=1).values
                self.det_best_iou_sum += float(best_iou_per_gt.sum().item())
                self.det_best_iou_count += gt_count
                self.det_gt_hit_count += int((best_iou_per_gt >= 0.5).sum().item())

            if len(self.det_debug_examples) >= 3:
                continue

            gt_boxes = pbatch["bboxes"][:2].detach().cpu().tolist()
            gt_cls = pbatch["cls"][:2].detach().cpu().tolist()
            pred_boxes_sample = pred_boxes_clipped[:2].detach().cpu().tolist()
            pred_cls_sample = pred_cls[:2].detach().cpu().tolist()
            pred_conf_sample = pred_conf[:2].detach().cpu().tolist()
            best_iou_sample = []
            if pred_count > 0 and gt_count > 0:
                pairwise_iou = box_iou(pbatch["bboxes"][:2], pred_boxes_clipped[: min(5, pred_count)])
                best_iou_sample = pairwise_iou.max(dim=1).values.detach().cpu().tolist()
            self.det_debug_examples.append(
                {
                    "image": batch["im_file"][si],
                    "gt_n": int(pbatch["cls"].shape[0]),
                    "pred_n": pred_count,
                    "gt_cls": [int(x) for x in gt_cls],
                    "pred_cls": [int(x) for x in pred_cls_sample],
                    "pred_conf": [round(float(x), 4) for x in pred_conf_sample],
                    "best_iou": [round(float(x), 4) for x in best_iou_sample],
                    "gt_boxes": [[round(float(v), 2) for v in box] for box in gt_boxes],
                    "pred_boxes": [[round(float(v), 2) for v in box] for box in pred_boxes_sample],
                }
            )
