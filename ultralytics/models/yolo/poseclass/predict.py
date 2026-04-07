from __future__ import annotations

import torch
import torchvision.transforms as T

from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


def _poseclass_transforms(size: int | tuple[int, int] = 224) -> T.Compose:
    """Build inference transforms that match PoseClassDataset's validation pipeline.

    Uses a direct Resize (stretch to exact size) instead of Resize-shortest-edge +
    CenterCrop so that no part of the input image is discarded.
    """
    if isinstance(size, int):
        size = (size, size)
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ])


class PoseClassPredictor(ClassificationPredictor):
    """Predictor for PoseClass models (classification + keypoints)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "poseclass"

    def setup_source(self, source):
        """Set up source with transforms matching PoseClass training."""
        BasePredictor.setup_source(self, source)
        self.transforms = _poseclass_transforms(self.imgsz)

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
        xy = kpts[..., :2]
        if xy.min() < -0.5 or xy.max() > 1.5:
            kpts[..., :2] = xy.sigmoid()
        else:
            kpts[..., :2] = xy.clamp(0.0, 1.0)
        if kpts.shape[-1] >= 3:
            vis = kpts[..., 2]
            if vis.min() < -0.5 or vis.max() > 1.5:
                kpts[..., 2] = vis.sigmoid()
            else:
                kpts[..., 2] = vis.clamp(0.0, 1.0)

        results = []
        for i, (orig_img, img_path) in enumerate(zip(orig_imgs, self.batch[0])):
            sample_kpts = kpts[i].clone()
            if sample_kpts[:, :2].max() <= 1.5:
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
