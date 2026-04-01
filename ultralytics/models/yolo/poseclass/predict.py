from __future__ import annotations

import torch

from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class PoseClassPredictor(ClassificationPredictor):
    """Predictor for PoseClass models (classification + keypoints)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "poseclass"

    def postprocess(self, preds, img, orig_imgs):
        """Process model outputs into Results objects with probs and keypoints."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        cls_scores = preds.get("cls_logits", preds.get("cls"))
        if cls_scores is None:
            raise KeyError("PoseClass predictions must contain either 'cls_logits' or 'cls'.")
        probs = cls_scores.softmax(1) if "cls_logits" in preds else cls_scores
        kpts = preds["kpts"].clone()

        results = []
        for i, (orig_img, img_path) in enumerate(zip(orig_imgs, self.batch[0])):
            sample_kpts = kpts[i]
            # Convert normalized keypoint coordinates to image space if needed.
            if sample_kpts.shape[-1] >= 2 and sample_kpts[:, :2].abs().max() <= 2.0:
                sample_kpts[:, 0] *= orig_img.shape[1]
                sample_kpts[:, 1] *= orig_img.shape[0]
            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    probs=probs[i],
                    keypoints=sample_kpts.unsqueeze(0),
                )
            )
        return results
