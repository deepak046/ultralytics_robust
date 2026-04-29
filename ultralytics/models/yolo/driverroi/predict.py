from __future__ import annotations

import torch

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class DriverROIPredictor(DetectionPredictor):
    """Predictor for detection plus driver ROI outputs."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "driverroi"

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Convert ROI-stage outputs into standard Results objects with extra keypoints/probabilities."""
        driver_aux = None
        if isinstance(preds, dict):
            driver_aux = preds.get("driver")
            preds = preds["det"]
        results = super().postprocess(preds[0] if isinstance(preds, tuple) else preds, img, orig_imgs, **kwargs)
        if driver_aux is None or "batch_indices" not in driver_aux:
            return results

        probs = driver_aux.get("cls")
        if probs is None and "cls_logits" in driver_aux:
            probs = driver_aux["cls_logits"].softmax(1)
        kpts = driver_aux["kpts"]
        boxes = driver_aux["boxes"]

        for aux_i, image_idx in enumerate(driver_aux["batch_indices"].view(-1).tolist()):
            result = results[image_idx]
            box = boxes[aux_i : aux_i + 1].clone()
            box = ops.scale_boxes(img.shape[2:], box, result.orig_shape).squeeze(0)
            sample_kpts = kpts[aux_i].clone()
            sample_kpts[..., 0] = box[0] + sample_kpts[..., 0] * (box[2] - box[0])
            sample_kpts[..., 1] = box[1] + sample_kpts[..., 1] * (box[3] - box[1])
            result.update(keypoints=sample_kpts.unsqueeze(0))
            result.probs = probs[aux_i] if probs is not None else None
            result.driver_box = box
        return results
